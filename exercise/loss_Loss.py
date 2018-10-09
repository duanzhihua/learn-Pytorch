# -*- coding: utf-8 -*-

import torch
from  torch import autograd
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# torch.nn.L1Loss loss(x,y)=1/n∑|xi−yi|
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
# torch.nn.MSELoss  loss(x,y)=1/n∑|xi−yi|2
# =============================================================================
loss = nn.MSELoss()
input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
target = autograd.Variable(torch.randn(3, 5))
output = loss(input, target)
output.backward()


# =============================================================================
# torch.nn.CrossEntropyLoss          LogSoftMax 和 NLLLoss 结合
#        loss(x, class) = -log(exp(x[class]) / (\sum_j exp(x[j])))
#                       = -x[class] + log(\sum_j exp(x[j]))
# =============================================================================


loss = nn.CrossEntropyLoss()
input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
target = autograd.Variable(torch.LongTensor(3).random_(5))
output = loss(input, target)
output.backward()


# =============================================================================
# torch.nn.NLLLoss
# loss(x, class) = -x[class]
# =============================================================================

m = nn.LogSoftmax()
loss = nn.NLLLoss()
# input is of size N x C = 3 x 5
input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
# each element in target has to have 0 <= value < C
target = autograd.Variable(torch.LongTensor([1, 0, 4]))
output = loss(m(input), target)
output.backward()


# =============================================================================
# torch.nn.PoissonNLLLoss
# target ~ Pois(input) loss(input, target) = input - target * log(input) + log(target!)
# functional.py  poisson_nll_loss
# =============================================================================

loss = nn.PoissonNLLLoss()
log_input = autograd.Variable(torch.randn(5, 2), requires_grad=True)
target = autograd.Variable(torch.randn(5, 2))
output = loss(log_input, target)
output.backward()

# =============================================================================
# torch.nn.NLLLoss2d
# =============================================================================

m = nn.Conv2d(16, 32, (3, 3)).float()
loss = nn.NLLLoss2d()
# input is of size N x C x height x width
input = autograd.Variable(torch.randn(3, 16, 10, 10))
# each element in target has to have 0 <= value < C
target = autograd.Variable(torch.LongTensor(3, 8, 8).random_(0, 4))
output = loss(m(input), target)
output.backward()


# =============================================================================
# 
# torch.nn.KLDivLoss
# loss(x,target)=1/n∑(targeti∗(log(targeti)−xi))
# =============================================================================

loss = nn.KLDivLoss(size_average=False)
batch_size = 5
log_probs1 = F.log_softmax(torch.randn(batch_size, 10), 1)
probs2 = F.softmax(torch.randn(batch_size, 10), 1)
loss(log_probs1, probs2) / batch_size



# =============================================================================
# torch.nn.BCELoss
#loss(o,t)=−1/n∑i(t[i]∗log(o[i])+(1−t[i])∗log(1−o[i]))
# =============================================================================
m = nn.Sigmoid()
loss = nn.BCELoss()
input = autograd.Variable(torch.randn(3), requires_grad=True)
target = autograd.Variable(torch.FloatTensor(3).random_(2))
output = loss(m(input), target)
output.backward()


# =============================================================================
# torch.nn.BCEWithLogitsLoss
#loss(o,t)=−1/n∑i(t[i]∗log(sigmoid(o[i]))+(1−t[i])∗log(1−sigmoid(o[i])))
#functional.py binary_cross_entropy_with_logits
# =============================================================================

loss = nn.BCEWithLogitsLoss()
input = autograd.Variable(torch.randn(3), requires_grad=True)
target = autograd.Variable(torch.FloatTensor(3).random_(2))
output = loss(input, target)
output.backward()

input = torch.randn(3, requires_grad=True)
target = torch.empty(3).random_(2)
loss = F.binary_cross_entropy_with_logits(input, target)
loss.backward()


# =============================================================================
# 
# torch.nn.MarginRankingLoss
# loss(x, y) = max(0, -y * (x1 - x2) + margin)
# loss.cpp
# =============================================================================



# =============================================================================
# 
# torch.nn.HingeEmbeddingLoss
#                 { x_i,                  if y_i ==  1
#loss(x, y) = 1/n {
#                 { max(0, margin - x_i), if y_i == -1
# =============================================================================



# =============================================================================
# 
# torch.nn.MultiLabelMarginLoss
# loss(x, y) = sum_ij(max(0, 1 - (x[y[j]] - x[i]))) / x.size(0)
# MultiLabelMarginCriterion.c
# =============================================================================


# =============================================================================
# 
# torch.nn.SmoothL1Loss
#                               { 0.5 * (x_i - y_i)^2, if |x_i - y_i| < 1
#         loss(x, y) = 1/n \sum {
#                               { |x_i - y_i| - 0.5,   otherwise
# 
# SmoothL1Criterion.c
# =============================================================================


# =============================================================================
# torch.nn.SoftMarginLoss
# loss(x, y) = sum_i (log(1 + exp(-y[i]*x[i]))) / x.nelement()
# SoftMarginCriterion.c
# 
# =============================================================================


# =============================================================================
# torch.nn.MultiLabelSoftMarginLoss
# loss(x, y) = - sum_i (y[i] * log( 1 / (1 + exp(-x[i])) )
#                   + ( (1-y[i]) * log(exp(-x[i]) / (1 + exp(-x[i])) ) )
# 
# binary_cross_entropy --->BCECriterion
# =============================================================================


# =============================================================================
# torch.nn.CosineEmbeddingLoss
#              { 1 - cos(x1, x2),              if y ==  1
# loss(x, y) = {
#              { max(0, cos(x1, x2) - margin), if y == -1
# loss.cpp cosine_embedding_loss
# =============================================================================
 

# =============================================================================
# torch.nn.MultiMarginLoss
# loss(x, y) = sum_i(max(0, (margin - x[y] + x[i]))^p) / x.size(0)
#              其中 `i == 0` 至 `x.size(0)` 并且 `i != y`.
# multi_margin_loss--->MultiMarginCriterion
# =============================================================================



# =============================================================================
# torch.nn.TripletMarginLoss
# L(a,p,n)=1N(∑i=1Nmax{d(ai,pi)−d(ai,ni)+margin,0}) 
# loss.cpp triplet_margin_loss
# =============================================================================

triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
input1 = torch.randn(100, 128, requires_grad=True)
input2 = torch.randn(100, 128, requires_grad=True)
input3 = torch.randn(100, 128, requires_grad=True)
output = triplet_loss(input1, input2, input3)
output.backward()
