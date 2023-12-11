import torch
from torch import nn
import torch.nn.functional as F
import sys
from .Embedding import Embedding
from .SelfAttention import SelfAttention
from .PositionWiseFeedForward import PositionWiseFeedForward

class Encoder(nn.Module):
    def __init__(self, max_sequence_length, embedding_dim, hidden_dim, dropout_prob, num_heads, vocab_length):
        super(Encoder, self).__init__()
        self.attention = SelfAttention(embedding_dim = embedding_dim, num_heads = num_heads)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward = PositionWiseFeedForward(embedding_dim, hidden_dim = hidden_dim, dropout_prob = dropout_prob)
        self.dropout = nn.Dropout(p=dropout_prob)


    def forward(self, embeddings):
        values = self.attention(embeddings)
        embeddings = self.dropout(embeddings)
        new_embeddings = self.layer_norm(embeddings + values)
        output =  self.feed_forward(new_embeddings)
        output = self.dropout(output)
        output = self.layer_norm(new_embeddings + output)
        return output
        



