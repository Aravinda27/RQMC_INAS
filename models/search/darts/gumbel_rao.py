#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:32:23 2024

@author: user1
"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable

def tempered_gumbel_softmax(logits, temp=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize."""
    gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / temp  # divide by temperature
    y_soft = gumbels.softmax(dim=-1)

    if hard:
        # Straight through.
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft

    return ret

def gumbel_rao(logits, k, temp=5.0, adaptive_sampling=False, importance_sampling=False, learnable_temp=False):
    """Enhanced GRMC estimator with tempered Gumbel-Softmax."""
    print("K values",k)
    num_classes = logits.shape[-1]
    
    if learnable_temp:
        temp = Variable(torch.ones(1) * temp, requires_grad=True)

    # Adaptive sampling logic
    if adaptive_sampling:
        variance = torch.var(logits, dim=-1, keepdim=True)
        k = int(k * torch.clamp(variance.mean(), 1, 10).item())
    
    # Importance sampling weights (example, needs proper implementation)
    if importance_sampling:
        weights = F.softmax(logits, dim=-1)
        logits = logits * weights
    
    # Sample using Tempered Gumbel-Softmax with Rao-Blackwellization
    I = torch.distributions.Categorical(logits=logits).sample()
    D = F.one_hot(I, num_classes).float()
    adjusted = logits + conditional_gumbel(logits, D, k=k)
    surrogate = tempered_gumbel_softmax(adjusted, temp=temp).mean(dim=0)
    
    return replace_gradient(D, surrogate)

def conditional_gumbel(logits, D, k=10):
    """Conditional Gumbel sampling."""
    E = torch.distributions.Exponential(rate=torch.ones_like(logits)).sample([k])
    Ei = (D * E).sum(dim=-1, keepdim=True)
    Z = logits.exp().sum(dim=-1, keepdim=True)
    adjusted = (D * (-torch.log(Ei) + torch.log(Z)) +
                (1 - D) * -torch.log(E / torch.exp(logits) + Ei / Z))
    
    return adjusted - logits

def replace_gradient(straight_through, surrogate):
    """Replace the gradient of straight_through with that of surrogate."""
    return straight_through + (surrogate - straight_through).detach()

# # Example usage:
# logits = torch.randn(10, 5)
# k = 10
# temp = -1

# # Using the enhanced GRMC estimator with tempered Gumbel-Softmax
# output = gumbel_rao(logits, k, temp=temp, adaptive_sampling=True, importance_sampling=True, learnable_temp=True)
# print(output)
