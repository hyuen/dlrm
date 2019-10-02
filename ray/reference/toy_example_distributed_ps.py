from __future__ import absolute_import, division, print_function, unicode_literals

import bisect
import builtins
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from numpy import random as ra
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter
from collections import OrderedDict

import ray
from torchviz import make_dot

# One parameter server for all parameters
# Master node coordinates between epochs (sync)
# Each trainer has access to its own data (currently dummy values)
#
# master gets weights from PS
# In each epoch:
#     call all trainers to do a batch with the weights
#     update the ps with all these gradients and get updated weights


class Model(nn.Module):
    def create_mlp(self, m, n):
        layers = nn.ModuleList()

        LL = nn.Linear(m, n, bias=True)
        mean = 0.0
        std_dev = np.sqrt(2 / (m + n))
        W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
        std_dev = np.sqrt(1 / m)
        bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)

        LL.weight.data = torch.tensor(W, requires_grad=True)
        LL.bias.data = torch.tensor(bt, requires_grad=True)
        layers.append(LL)

        layers.append(nn.ReLU())
        return torch.nn.Sequential(*layers)

    def create_emb(self, m, height):
        emb_l = nn.ModuleList()

        n = height
        EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
        W = np.random.uniform(
            low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
        ).astype(np.float32)
        EE.weight.data = torch.tensor(W, requires_grad=True)
        emb_l.append(EE)
        return emb_l

    def __init__(self):
        super(Model, self).__init__()

        self.emb_l = self.create_emb(8, 20)
        self.mlp_l = self.create_mlp(1, 8)

    def forward(self, emb_row_ids, emb_offset):
        V = self.emb_l[0](emb_row_ids, emb_offset)
        res = self.mlp_l(V)
        return res

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad  # .data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                if isinstance(g, torch.Tensor):
                    p.grad = g
                else:
                    p.grad = torch.from_numpy(g)


@ray.remote
class ParameterServer(object):
    def __init__(self, lr):
        self.model = Model()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

    def reduce(self, gradients):
        if gradients[0].layout == torch.strided:
            return np.stack(gradients).sum(axis=0)
        elif gradients[0].layout == torch.sparse_coo:
            # TODO: avoid converting to dense matrix and convert back to sparse
            heights = [g.size()[0] for g in gradients]
            max_height = max(heights)
            dense_grads = [F.pad(input=g.to_dense(), pad=(0, 0, 0, max_height-h),
                                 mode='constant', value=0)
                           for g, h in zip(gradients, heights)]
            final_grad = torch.stack(dense_grads).sum(dim=0).to_sparse(1)
            return final_grad

        raise NotImplementedError

    # get n lists of gradients, one per trainer
    # each list will contain all the gradients for all the elements in the net
    def apply_gradients(self, *gradients):
        #print("there are gradients for n trainers:", len(gradients))
        summed_gradients = [self.reduce(grads)
                            for grads in zip(*gradients)]

        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()
        return self.get_weights()

    def get_weights(self):
        return self.model.state_dict()


@ray.remote
class Trainer(object):
    def __init__(self):
        self.model = Model()
        self.criterion = nn.MSELoss()
        # data loader here

    # run one batch and return gradients
    def compute_gradients(self, weights):
        self.set_weights(weights)

        # should get real inputs
        x = torch.LongTensor(np.sort(np.random.choice(20, 10, replace=False)))
        offsets = torch.LongTensor([0])
        yhat = torch.FloatTensor([[1.2]])

        self.model.zero_grad()
        y = self.model(x, offsets)
        loss = self.criterion(y, yhat)

        loss.backward()
        return self.model.get_gradients()

    def set_weights(self, weights):
        self.model.load_state_dict(weights)


def dist_train():
    ray.init()
    ps = ParameterServer.remote(0.01)
    train_actors = [Trainer.remote(), Trainer.remote()]

    current_weights = ps.get_weights.remote()

    for i in range(10):
        print("iteration", i)
        gradients = ray.get([actor.compute_gradients.remote(current_weights)
                             for actor in train_actors])
        current_weights = ps.apply_gradients.remote(*gradients)


if __name__ == "__main__":
    dist_train()
