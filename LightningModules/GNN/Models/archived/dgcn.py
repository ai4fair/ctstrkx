#!/usr/bin/env python
# coding: utf-8

# @author: Adeel Akram + Waleed Esmail

"""DGCN Networks with the dynamic edge convolutional operator from the paper
titled "Dynamic Graph CNN for Learning on Point Clouds" [arXiv:1801.07829]"""


import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_scatter import scatter_add, scatter_mean, scatter_max

from ..gnn_base import GNNBase
from ..utils import make_mlp



# TODO: Make DGCN competible with Pipeline i.e. use GNNBase
# FIXME: Test with data from the Processing stage



# Suggested by Waleed
def mlp(channels, batch_norm=True):
    return Sequential(*[
        nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU(), nn.BatchNorm1d(channels[i]))
        for i in range(1, len(channels))
    ])


# class DGCN(GNNBase):
class DGN(nn.Module):
    def __init__(self, feature_transform=False, out_channels=2, k=20, aggr='max', num_classes=2):
        super().__init__()
        
        self.feature_transform = feature_transform
        self.out_channels = 2
        self.k = 20
        self.aggr = 'max'           # self.hparams["aggregation"]    
        self.num_classes = 2
        
        
        self.conv1 = DynamicEdgeConv(mlp([2 * 8, 64, 64, 64]), self.k, self.aggr)
        self.conv2 = DynamicEdgeConv(mlp([2 * 64, 128]), self.k, self.aggr)
        self.linear = nn.Linear(128, self.num_classes)

    def forward(self, data):
        x, batch = data.x, data.batch
        x = self.conv1(x, batch)
        x = self.conv2(x, batch)
        x = global_max_pool(x, batch)

        return self.linear(x)

