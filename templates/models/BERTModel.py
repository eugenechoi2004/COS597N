import torch
from torch import nn
import torch.nn.functional as F
import sys
sys.path.append('../comp')
from Encoder import Encoder

class BERTModel(nn.Module):
    def __init__(self, max_sequence_length, embedding_dim, hidden_dim, dropout_prob, num_heads, num_layers):
        super(BERTModel, self).__init__()
        self.layers = nn.ModuleList([nn.Encoder() for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Hyperparams
embedding_dim = 768 # from paper
num_encoder_layers = 6 # from paper
max_sequence_length = 500
hidden_dim = 3072 # from paper
dropout_prob = 0.2
num_heads = 12 # from paper
mlm_prob = 0.15
# 15% of rht mlm batch_size 10 ~ 11 epochs


x = ["jan nafijna jfaoja ", "gnafnk fajknfa "]
model = BERTModel(max_sequence_length, embedding_dim, hidden_dim, dropout_prob, num_heads)
model(x)
