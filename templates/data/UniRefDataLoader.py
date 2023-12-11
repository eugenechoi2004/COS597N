import torch
from Bio import SeqIO
import random
import sys
sys.path.append('../comp')
from comp.TokenizerBERT import TokenizerBERT
from torch.utils.data import DataLoader, random_split

class UniRefDataLoader():
    def __init__(self, path_to_data, batch_size, tokenizer: TokenizerBERT, max_length, mask_prob=0.15, split_ratio=0.8):
        self.max_length = max_length
        self.path_to_data = path_to_data
        self.mask_prob = mask_prob
        self.batch_size = batch_size
        self.split_ratio = split_ratio  # Split ratio for training data
        self.pad_token_id = 1  # Update this as per your padding token ID
        self.mask_token_id = 2  # Update this if needed
        self.data = [tokenizer.encode(record.seq) for record in SeqIO.parse(path_to_data, "fasta")]
        self.train_loader, self.test_loader = self.prepare_data_loaders()

    def mask_sequence(self, sequence):
        masked_sequence = sequence.clone()
        mask = (torch.rand(self.max_length) < self.mask_prob)
        labels = torch.full([self.max_length], -100)
        for i in range(self.max_length):
            if mask[i]:
                masked_sequence[i] = self.mask_token_id
                labels[i] = sequence[i]
        return masked_sequence, labels

    def prepare_data_loaders(self):
        masked_data, labels = zip(*[self.mask_sequence(seq) for seq in self.data])
        dataset = torch.utils.data.TensorDataset(torch.stack(masked_data), torch.stack(labels))
        train_size = int(self.split_ratio * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader, test_loader

    def get_train_data_loader(self):
        return self.train_loader

    def get_test_data_loader(self):
        return self.test_loader
