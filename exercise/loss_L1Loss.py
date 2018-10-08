# -*- coding: utf-8 -*-

import torch
from  torch import autograd
import torch.nn as nn

loss = nn.L1Loss()
input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
target = autograd.Variable(torch.randn(3, 5))
output = loss(input, target)
output.backward()