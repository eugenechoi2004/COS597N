import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
from transformers import BertTokenizer


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(SelfAttention, self).__init__()  
        self.qkv_layer = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.linear_layer = nn.Linear(embedding_dim, embedding_dim)
        self.head_dim = embedding_dim//num_heads
        self.num_heads = num_heads

        
    def forward(self, x):
        qkv = self.qkv_layer(x)
        batch_size, max_length, qkv_dim = qkv.size()
        qkv = qkv.reshape(batch_size, max_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # batch size, num_heads, sequence length, 3 * head dimension
        q, k, v = qkv.chunk(3, dim=-1)
        scaled = torch.matmul(q, k.transpose(-1,-2))/math.sqrt(self.head_dim)
        attention = F.softmax(scaled, dim=-1)
        values = torch.matmul(attention, v)
        values = values.reshape(batch_size, max_length, self.num_heads * self.head_dim)
        values = self.linear_layer(values)
        return values
