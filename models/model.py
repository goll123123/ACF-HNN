#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 21:44:10 2023

@author: amax
"""
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.optim as optim, torch.nn.functional as F

def accuracy(Z, Y):
    """
    arguments:
    Z: predictions
    Y: ground truth labels

    returns: 
    accuracy
    """
    
    predictions = Z.max(1)[1].type_as(Y)
    correct = predictions.eq(Y).double()
    correct = correct.sum()

    accuracy = correct / len(Y)
    return accuracy

class MLP(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers, dropout):
        super(MLP, self).__init__()
        if num_layers == 1:
            hidden_size = out_size

        self.pipeline = nn.Sequential(OrderedDict([
            ('layer_0', nn.Linear(in_size, hidden_size, bias=(num_layers != 1))),
            ('dropout_0', nn.Dropout(dropout)),
            ('relu_0', nn.ReLU())
        ]))

        for i in range(1, num_layers):
            if i == num_layers - 1:
                self.pipeline.add_module('layer_{}'.format(i), nn.Linear(hidden_size, out_size, bias=True))
            else:
                self.pipeline.add_module('layer_{}'.format(i), nn.Linear(hidden_size, hidden_size, bias=True))
                self.pipeline.add_module('dropout_{}'.format(i), nn.Dropout(dropout))
                self.pipeline.add_module('relu_{}'.format(i), nn.ReLU())

        self.weights_init()

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, feature):
        # print('feature', feature.size())
        return F.softmax(self.pipeline(feature), dim=1)