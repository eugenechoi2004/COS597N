import torch
from torch import nn
import torch.nn.functional as F
import sys
sys.path.append('../comp')
from comp.Encoder import Encoder
from comp.Embedding import Embedding


class BERTModel(nn.Module):
    def __init__(self, max_sequence_length, embedding_dim, hidden_dim, dropout_prob, num_heads, num_layers, vocab_length):
        super(BERTModel, self).__init__()
        self.embeddings = Embedding(vocab_size = vocab_length, embedding_dim = embedding_dim, max_sequence_length = max_sequence_length, dropout = dropout_prob)
        self.layers = nn.ModuleList([Encoder(max_sequence_length, embedding_dim, hidden_dim, dropout_prob, num_heads, vocab_length) for _ in range(num_layers)])
        self.final_linear = nn.Linear(embedding_dim, vocab_length)  # Linear layer to project back to vocab space
        self.vocab_size = vocab_length

    def forward(self, x):
        x = self.embeddings(x )
        for layer in self.layers:
            x = layer(x)
        x = self.final_linear(x)  
        return x

# Hyperparams
embedding_dim = 768 # from paper
num_encoder_layers = 6 # from paper
max_sequence_length = 500
hidden_dim = 3072 # from paper
dropout_prob = 0.2
num_heads = 12 # from paper
mlm_prob = 0.15
vocab_length = 30
batch_size = 10
# 15% of rht mlm batch_size 10 ~ 11 epochs



model = BERTModel(max_sequence_length, embedding_dim, hidden_dim, dropout_prob, num_heads, num_encoder_layers, vocab_length)

