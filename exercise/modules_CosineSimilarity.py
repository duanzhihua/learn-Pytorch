# -*- coding: utf-8 -*-

import torch
from  torch import autograd
import torch.nn as nn


#torch.nn.CosineSimilarity
input1 = autograd.Variable(torch.randn(100, 128))
input2 = autograd.Variable(torch.randn(100, 128))
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
output = cos(input1, input2)
print(output)

#torch.nn.PairwiseDistance
pdist = nn.PairwiseDistance(p=2)
input1 = autograd.Variable(torch.randn(100, 128))
input2 = autograd.Variable(torch.randn(100, 128))
output = pdist(input1, input2)
print(output)




