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


# reference net for implementing more interesting ray primitives

# Architecture
#
# Emb(20,8)->MLP(8,1)->ReLU

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
    serial()
