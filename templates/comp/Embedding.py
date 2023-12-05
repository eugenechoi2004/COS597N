import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
from transformers import BertTokenizer


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_sequence_length, dropout):
        super(Embedding, self).__init__()  # Call to base class __init__
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.positions_encodings = torch.zeros(max_sequence_length, embedding_dim) 
        self.dropout = nn.Dropout(p=dropout)  
        for row in range(len(self.positions_encodings)):
            for col in range(len(self.positions_encodings[row])//2):
                denominator = np.power(10_000, 2*col/embedding_dim)
                self.positions_encodings[row,2*col] = np.sin(row/denominator)
                self.positions_encodings[row,2*col + 1] = np.cos(row/denominator)
        
    def forward(self, x):
        output = self.embeddings(x) + self.positions_encodings
        return self.dropout(output)

