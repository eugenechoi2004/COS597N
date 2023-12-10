import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
from tokenizers.trainers import BpeTrainer





class TokenizerBERT(nn.Module):
    def __init__(self, max_sequence_length):
        super(Tokenizer, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.vocab_size = len(self.tokenizer.vocab)
        self.max_length = max_sequence_length


    def forward(self, x):
        encoded_input = self.tokenizer(x, padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")
        input_ids = encoded_input['input_ids']
        return input_ids
