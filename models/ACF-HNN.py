
import math

import torch
from torch import nn
from torch.nn import init

from models import HGNN_conv, FALayer
import torch.nn.functional as F



class ACFHNNConv(nn.Module):
    def __init__(self, in_channels, out_channels, alpha=0.5, p=0.5, bias=True):
        super().__init__()

        self.low = nn.Linear(in_channels, out_channels)
        self.mid = nn.Linear(in_channels, out_channels)
        self.high = nn.Linear(in_channels, out_channels)

        self.lowalpha = nn.Parameter(torch.FloatTensor([0]))
        self.lowgamma = nn.Parameter(torch.FloatTensor([1]))
        self.highalpha = nn.Parameter(torch.FloatTensor([0]))
        self.highgamma = nn.Parameter(torch.FloatTensor([1]))
        self.midalpha = nn.Parameter(torch.FloatTensor([0]))
        self.midgamma = nn.Parameter(torch.FloatTensor([1]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.low.weight)
        nn.init.xavier_uniform_(self.mid.weight)
        nn.init.xavier_uniform_(self.high.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)



    def compute_laplacian_components(self, X, hg):
        num_nodes = X.shape[0]

        row_indices = torch.arange(num_nodes, device=X.device)
        col_indices = torch.arange(num_nodes, device=X.device)

        I = torch.sparse_coo_tensor(
            indices=torch.stack([row_indices, col_indices], dim=0),
            values=torch.ones(num_nodes, device=X.device),
            size=(num_nodes, num_nodes)
        )
        Delta = hg.L_sym
        if not Delta.is_sparse:
            Delta = Delta.to_sparse()


        return I - Delta

    def forward(self, X, hg):

        lowalpha = torch.nn.functional.hardtanh(self.lowalpha, min_val=0., max_val=1.)
        lowgamma = torch.relu(self.lowgamma)
        highalpha = torch.nn.functional.hardtanh(self.highalpha, min_val=0., max_val=1.)
        highgamma = torch.relu(self.highgamma)
        midalpha = torch.nn.functional.hardtanh(self.midalpha, min_val=0., max_val=1.)
        midgamma = torch.relu(self.midgamma)
        h = torch.matmul(self.compute_laplacian_components(X, hg), X)
        o_low = (-lowalpha * h + X) * lowgamma
        o_low = self.low(o_low)
        o_high = (highalpha * h + (1 - 2 * highalpha) * X) * highgamma
        o_high = self.high(o_high)
        o_mid = (h ** 2 - midalpha * X) * midgamma
        o_mid = self.mid(o_mid)
        out =  o_low + o_high + o_mid
        if self.bias is not None:
            out = out + self.bias
        return out


class ACFHNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, nlayers, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.nlayers = nlayers
        self.hgc = nn.ModuleList()
        if nlayers == 1:
            self.hgc.append(ACFHNNConv(in_ch, n_class))
        else:
            self.hgc.append(ACFHNNConv(in_ch, n_hid))
            for _ in range(nlayers - 2):
                self.hgc.append(ACFHNNConv(n_hid, n_hid))
            self.hgc.append(ACFHNNConv(n_hid, n_class))

    def forward(self, x, G):
        for i, layer in enumerate(self.hgc):
            if i < self.nlayers - 1:
                x = layer(x, G)
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
            else:
                x = layer(x, G)

        return F.log_softmax(x, dim=1)



