import torch
from torch import nn
import torch.nn.functional as F
import sys
from Tokenizer import Tokenizer
from Embedding import Embedding
from SelfAttention import SelfAttention

class PositionWiseFeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout_prob):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = dropout_prob)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

