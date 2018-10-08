# -*- coding: utf-8 -*-

# =============================================================================
# torch.nn.Linear
#对输入数据进行线性变换: y=Ax+b
# =============================================================================
import torch
from  torch import autograd
import torch.nn as nn
 

m = nn.Linear(20, 30)
input = autograd.Variable(torch.randn(128, 20))
output = m(input)
print(output.size())
#torch.Size([128, 30])


# =============================================================================
# torch.nn.Bilinear
#对输入数据进行双线性变换: y=x1∗A∗x2+b
# =============================================================================
m = nn.Bilinear(20, 30, 40)
input1 = autograd.Variable(torch.randn(128, 20))
input2 = autograd.Variable(torch.randn(128, 30))
output = m(input1, input2)
print(output.size())
#torch.Size([128, 40])


