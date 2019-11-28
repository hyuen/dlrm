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
from dlrm_toy_model import Lookup, OverArch


# Distributed parameter server implementation
#
# One embedding table per parameter server (two tables)
# Three trainers computing MLPs and using all-reduce
# Readers are mocked, readers feed trainers

# Master node coordinates between epochs (sync)
# Each trainer has access to its own data (currently dummy values)
#
# In each epoch:
#     call all trainers to do a batch with the weights
#     update the ps with all these gradients and get updated weights

@ray.remote
class ParameterServer(object):
    def __init__(self, emb_dim, emb_rows):
        self.model = Lookup(emb_dim, emb_rows)
        self.n = emb_rows
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def forward(self, inputs, lens):
        ret = self.model.forward(inputs, lens)
        return ret

    def reduce(self, gradients):
        if gradients[0].layout == torch.strided:
            return np.stack(gradients).sum(axis=0)
        elif gradients[0].layout == torch.sparse_coo:
            result = gradients[0]
            for i in range(1, len(gradients)):
                result += gradients[i]
            return result

        raise NotImplementedError

    # get n lists of gradients, one per trainer
    # each list will contain all the gradients for all the elements in the net
    def apply_gradients(self, *gradients):
        summed_gradients = [self.reduce(grads)
                            for grads in zip(*gradients)]

        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()
        return self.get_weights()

    def get_weights(self):
        return self.model.state_dict()

    def backward(self, gradient, x, offsets):
        y = self.model.forward(x, offsets)
        # print("PS", x.shape, offsets.shape, y.shape, gradient.shape)
        loss = self.criterion(y, gradient)
        self.optimizer.zero_grad()
        loss.backward()


@ray.remote
class Trainer(object):
    def __init__(self, overarch_dim, PS):
        self.model = OverArch(overarch_dim)
        self.PS = PS
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001)
        self.model.mlp[0].register_backward_hook(self.backward_pass_to_embeddings)
        self.state = False

    # data loader here
    def read_data(self):
        # Fake some inputs, should get real inputs
        #
        x1 = torch.LongTensor(np.sort(np.random.choice(20, 10, replace=False)))
        offsets1 = torch.LongTensor([0])
        
        x2 = torch.LongTensor(np.sort(np.random.choice(30, 10, replace=False)))
        offsets2 = torch.LongTensor([0])
        yhat = torch.FloatTensor([[np.random.rand()]])

        return x1, offsets1, x2, offsets2, yhat

    def backward_pass_to_embeddings(self, layer, input, output):
        emb_grad = torch.split(torch.transpose(input[2], 0, 1), 8, dim=1)
        # print("emb", emb_grad[0].shape, input[2].shape)

        x1_ref, x2_ref, offsets1_ref, offsets2_ref = self.state
        
        self.PS[0].backward.remote(emb_grad[0], x1_ref, offsets1_ref)
        self.PS[1].backward.remote(emb_grad[1], x2_ref, offsets2_ref)
        
    # run one batch and return gradients
    def forward(self): 
        #self.set_weights(weights)

        # Read a batch of data
        # Send it to PSes for embedding lookup
        # Apply MLPs forward
        # Get loss per MLP, average
        # Propagate loss to PSes
        
        x1, offsets1, x2, offsets2, yhat = self.read_data()

        x1_ref = ray.put(x1)
        x2_ref = ray.put(x2)
        offsets1_ref = ray.put(offsets1)
        offsets2_ref = ray.put(offsets2)
        
        emb1_res = self.PS[0].forward.remote(x1_ref, offsets1_ref)
        emb2_res = self.PS[1].forward.remote(x2_ref, offsets2_ref)

        y = self.model.forward(emb1_res, emb2_res)

        self.state = [x1_ref, x2_ref, offsets1_ref, offsets2_ref]
        
        loss = self.criterion(y, yhat)
        self.optimizer.zero_grad()
        loss.backward()    
        self.optimizer.step()

        return loss
    
    def get_weights(self):
        return self.model.state_dict()
    
    def set_weights(self, weights):
        self.model.load_state_dict(weights)


def dist_train():
    ray.init()

    emb_dim = 8
    overarch_dim = emb_dim * 2
    
    pses = [ParameterServer.remote(emb_dim, 20), ParameterServer.remote(emb_dim, 30)]
    trainers = [Trainer.remote(overarch_dim, pses), Trainer.remote(overarch_dim, pses)]

    for i in range(10):       
        print("iteration", i)
    

        # Do a forward pass on the MLPs and do a reduction
        # TODO: replace with ring allreduce
        ray.get([trainer.forward.remote() for trainer in trainers])

        weights = ray.get([trainer.get_weights.remote() for trainer in trainers])
        avg_weights = OrderedDict(
            [(k, (weights[0][k] + weights[1][k]) / 2) for k in weights[0]])
        # print("average weights", avg_weights)

        weight_id = ray.put(avg_weights)

        for trainer in trainers:
            trainer.set_weights.remote(weight_id)

           
        """        print("weights", weights)
        avg_weights = []
        for i in range(len(weights[0])):
            avg_weights.append(torch.mean(torch.stack([g[i] for g in weights]), dim=0))

        print(avg_weights)
        """
        #[trainer.set_gradients.remote(avg_gradients) for trainer in trainers]

        # Take the top gradient and pass it to each one of the PSes


if __name__ == "__main__":
    dist_train()
