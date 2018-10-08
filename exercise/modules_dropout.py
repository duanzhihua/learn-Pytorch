# -*- coding: utf-8 -*-

import torch
from  torch import autograd
import torch.nn as nn

# =============================================================================
# Dropout
# =============================================================================

m = nn.Dropout(p=0.2)
input = autograd.Variable(torch.randn(20, 16))
output = m(input)
print(output.size())
#torch.Size([20, 16])


# =============================================================================
#  Dropout2d
#  	Shape: 
# 		- Input: math:(N, C, H, W)
# 		- Output: math:(N, C, H, W)  
# =============================================================================
        
m = nn.Dropout2d(p=0.2)
input = autograd.Variable(torch.randn(20, 16, 32, 32))
output = m(input)
print(output.size())


# =============================================================================
# Dropout3d
# =============================================================================
m = nn.Dropout3d(p=0.2)
input = autograd.Variable(torch.randn(20, 16, 4, 32, 32))
output = m(input)
print(output.size())



# =============================================================================
# torch.nn.AlphaDropout
# =============================================================================
m = nn.AlphaDropout(p=0.2)
input = autograd.Variable(torch.randn(20, 16))
output = m(input)
print(output.size())













