import math, torch_sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch, numpy as np, scipy.sparse as sp

class FALayer(nn.Module):
    def __init__(self, in_ft, out_ft, dropout, bias=True):
        super(FALayer, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(in_ft, 1)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)
        
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        
        edges = torch.nonzero(G).t()
        N = edges.size(1)
        V1, E1 = edges[0][:int(N/100)], edges[:int(N/100)]
        h2 = torch.cat([x[V1], x[E1]], dim=1)
        g = torch.tanh(self.gate(h2)).squeeze()        
        # print('g', g.size())
        G_new = torch.zeros(G.size()).cuda()
        G_new.scatter_(dim=1, index=edges, src=g.unsqueeze(0))
        
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G_new.matmul(x)
        return x

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x


class HGNN_fc(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HGNN_fc, self).__init__()
        self.fc = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        return self.fc(x)


class HGNN_embedding(nn.Module):
    def __init__(self, in_ch, n_hid, dropout=0.5):
        super(HGNN_embedding, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.hgc2(x, G))
        return x


class HGNN_classifier(nn.Module):
    def __init__(self, n_hid, n_class):
        super(HGNN_classifier, self).__init__()
        self.fc1 = nn.Linear(n_hid, n_class)

    def forward(self, x):
        x = self.fc1(x)
        return x