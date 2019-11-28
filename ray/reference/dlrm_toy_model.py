from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import ray

class Lookup(nn.Module):
    def __init__(self, emb_dim=8, emb_rows=20):
        super(Lookup, self).__init__()
        self.emb = self.create_emb(emb_dim, emb_rows)

    def create_emb(self, dim, nrows):
        layers = nn.ModuleList()
        EE = nn.EmbeddingBag(nrows, dim, mode="sum", sparse=True)
        W = np.random.uniform(
            low=-np.sqrt(1 / nrows), high=np.sqrt(1 / nrows),
            size=(nrows, dim)).astype(np.float32)
        EE.weight.data = torch.tensor(W, requires_grad=True)
        layers.append(EE)

        return torch.nn.Sequential(*layers)

    def forward(self, emb_row_ids, emb_offset):
        res = self.emb[0](emb_row_ids, emb_offset)
        return res

    def backward(self, inputs):
        print("backward", inputs)
        self.emb.backward()
    
    
class OverArch(nn.Module):
    def __init__(self, input_dim):
        super(OverArch, self).__init__()
        self.mlp = self.create_mlp(1, input_dim)
        print("initial", self.state_dict())

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

    def forward(self, emb1, emb2):
        emb1_val, emb2_val = ray.get([emb1, emb2])
        
        T = torch.cat([emb1_val, emb2_val], dim=1)
        res = self.mlp(T)
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

    
class ModelTwoEmb(nn.Module):
    def __init__(self):
        super(ModelTwoEmb, self).__init__()

        self.emb_dim = 8
        
        self.emb_1 = self.create_emb(self.emb_dim, 20)
        self.emb_2 = self.create_emb(self.emb_dim, 30)
        self.mlp_l = self.create_mlp(1, self.emb_1[0].embedding_dim + self.emb_2[0].embedding_dim)
        
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

    def forward(self, emb_row_ids1, emb_offset1, emb_row_ids2, emb_offset2):
        V1 = self.emb_1[0](emb_row_ids1, emb_offset1)
        V2 = self.emb_2[0](emb_row_ids2, emb_offset2)
        T = torch.cat([V1, V2], dim=1)
        res = self.mlp_l(T)
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
