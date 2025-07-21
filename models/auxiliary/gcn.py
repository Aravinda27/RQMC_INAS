#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 17:47:50 2024

@author: user1
"""
import torch
import torch.nn as nn
import torch.optim as op
import os
from torch_geometric.nn import GCNConv
import torch.nn.functional as F



# class GCNContextualExtractor(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(GCNContextualExtractor, self).__init__()
#         self.conv1 = GCNConv(in_channels, 128)
#         self.conv2 = GCNConv(128, out_channels)

#     def forward(self, x, edge_index,device):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index).to(device)
#         return x


def gcn_contextual_extractor(x, edge_index, in_channels, out_channels):
    print(f'in_channels: {in_channels}, type: {type(in_channels)}')
    print(f'out_channels: {out_channels}, type: {type(out_channels)}')
    
    assert isinstance(in_channels, int), "in_channels must be an integer"
    assert isinstance(out_channels, int), "out_channels must be an integer"
    
    conv1 = GCNConv(in_channels, 128)
    conv2 = GCNConv(128, out_channels)
    
    x = conv1(x, edge_index)
    x = F.relu(x)
    x = conv2(x, edge_index)
    print("X",x.shape)
    
    return x

# Example usage
x = torch.randn((25, 8))  # Example feature matrix with 10 nodes and 8 features
edge_index = torch.tensor([[0, 1], [1, 0]])  # Example edge index for a graph

in_channels = 8
out_channels = 8

output = gcn_contextual_extractor(x, edge_index, in_channels, out_channels)
print(output.shape)
