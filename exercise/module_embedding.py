# -*- coding: utf-8 -*-

import torch
from  torch import autograd
from  torch.autograd import Variable
import torch.nn as nn

# =============================================================================
# 
#  torch.nn.Embedding 
#
#    Args:
#         num_embeddings (int): size of the dictionary of embeddings
#         embedding_dim (int): the size of each embedding vector
#         
#   Attributes:
#         weight (Tensor):  (num_embeddings, embedding_dim)
# 
#   Shape:
#         - Input: LongTensor of arbitrary shape containing the indices to extract
#         - Output: `(*, embedding_dim)`, where `*` is the input shape
# =============================================================================
        
embedding = nn.Embedding(10, 3)
input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
output=embedding(input)
print(output.size())


word_to_ix = {'hello': 0, 'world': 1}
embeds = nn.Embedding(2, 5)
hello_idx = torch.LongTensor([word_to_ix['hello']])
hello_idx = Variable(hello_idx)
hello_embed = embeds(hello_idx)
print(hello_embed)
 

# =============================================================================
# torch.nn.EmbeddingBag
# =============================================================================

embedding = nn.Embedding(10, 3, padding_idx=0)
input = Variable(torch.LongTensor([[0,2,0,5]]))
embedding(input)