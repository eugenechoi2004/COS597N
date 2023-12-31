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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    