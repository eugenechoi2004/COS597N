import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_sequence_length, dropout):
        super(Embedding, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.positions_encodings = torch.zeros(max_sequence_length, embedding_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Populate position encodings
        for row in range(len(self.positions_encodings)):
            for col in range(0, embedding_dim, 2):
                denominator = math.pow(10000, (2 * col) / embedding_dim)
                self.positions_encodings[row, col] = math.sin(row / denominator)
                self.positions_encodings[row, col + 1] = math.cos(row / denominator)

        self.positions_encodings = nn.Parameter(self.positions_encodings, requires_grad=False)  # Mark as non-trainable
        self.positions_encodings.to(self.device)  # Move positions_encodings to specified device
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.embeddings(x)  # x will automatically be on the same device as embeddings
        output = x + self.positions_encodings[:x.size(1), :]  # Adjust positions_encodings based on x's length
        return self.dropout(output)

