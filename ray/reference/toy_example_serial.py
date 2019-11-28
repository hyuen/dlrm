from __future__ import absolute_import, division, print_function, unicode_literals

import bisect
import builtins
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from numpy import random as ra
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter
from collections import OrderedDict

import ray
from torchviz import make_dot


### define dlrm in PyTorch ###
class Model(nn.Module):
    def create_mlp(self, m, n):
        layers = nn.ModuleList()

        LL = nn.Linear(m, n, bias = True)
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
    
@ray.remote
class ModelActor(object):
    def __init__(self):
        self.model = Model()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

        self.dataset = False
       
    def train(self):
        for i in range(10):
            # should get real inputs
            x = torch.LongTensor(np.random.randint(20, size=10))
            offsets = torch.LongTensor([0])
            yhat = torch.FloatTensor([[1.2]])

            self.optimizer.zero_grad()
            y = self.model(x, offsets)
            loss = self.criterion(y, yhat)

            #print(loss)
            loss.backward()
            self.optimizer.step()
            return loss

    def set_weights(self, weights):
        self.model.load_state_dict(weights)
    
    def get_weights(self):
        return self.model.state_dict()


def dist_train():
    ray.init() 
    train_actors = [ModelActor.remote(), ModelActor.remote()]

    for i in range(10):
        print("master iteration", i)
    
        weights = ray.get([actor.get_weights.remote() for actor in train_actors])
        avg_weights = OrderedDict(
            [(k, (weights[0][k] + weights[1][k]) / 2) for k in weights[0]])
        print(avg_weights)

        weight_id = ray.put(avg_weights)

        for actor in train_actors:
            actor.set_weights.remote(weight_id)

        ray.get([actor.train.remote() for actor in train_actors])


def serial():
    model = Model()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    print(model)

    for i in range(10):
        # TODO: should get new data everytime
        x = torch.LongTensor(np.random.randint(20, size=10))
        offsets = torch.LongTensor([0])
        yhat = torch.FloatTensor([[1.2]])

        optimizer.zero_grad()
        y = model(x, offsets)
        loss = criterion(y, yhat)

        print(y, loss)
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    #serial()
    model = Model()
    print(model.state_dict())
    dist_train()
