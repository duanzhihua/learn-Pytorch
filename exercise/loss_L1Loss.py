# -*- coding: utf-8 -*-

import torch
from  torch import autograd
import torch.nn as nn


# =============================================================================
# torch.nn.L1Loss
# =============================================================================

loss = nn.L1Loss()
input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
target = autograd.Variable(torch.randn(3, 5))
output = loss(input, target)
output.backward()

# =============================================================================
# nn.yaml
# - name: l1_loss(Tensor self, Tensor target, bool size_average=true, bool reduce=True)
#   cname: AbsCriterion
#   scalar_check:
#     output: reduce || self_->isScalar()
# 
# 源码参阅：AbsCriterion.c
# 
# =============================================================================

# =============================================================================
# torch.nn.MSELoss
# =============================================================================
loss = nn.MSELoss()
input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
target = autograd.Variable(torch.randn(3, 5))
output = loss(input, target)
output.backward()