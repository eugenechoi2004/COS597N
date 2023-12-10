import torch
from Bio import SeqIO
import random
import sys
sys.path.append('../comp')
from TokenizerBERT import TokenizerBERT
from torch.utils.data import DataLoader, random_split


class UniRefDataLoader():
    def __init__(self, path_to_data, batch_size, tokenizer:TokenizerBERT, max_length, train_val_test, mask_prob=0.15):
        self.max_length = max_length
        self.path_to_data = path_to_data
        self.mask_prob = mask_prob
        self.batch_size = batch_size
        self.pad_token_id = 0  # Update this as per your padding token ID
        self.mask_token_id = 103  # Update this if needed
        self.data = [tokenizer.encode(record.seq) for record in SeqIO.parse(path_to_data, "fasta")]
        self.data_loader = self.get_data_loader()

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


    def mask_sequence(self, sequence):
        masked_sequence = sequence.clone()
        mask = (torch.rand(self.max_length) < self.mask_prob)
        labels = torch.full([self.max_length], -100)  
        for i in range(self.max_length):
            if mask[i]:
                masked_sequence[i] = self.mask_token_id
                labels[i] = sequence[i]
        return masked_sequence, labels

    def get_data_loader(self):
        masked_data, labels = zip(*[self.mask_sequence(seq) for seq in self.data])
        dataset = torch.utils.data.TensorDataset(torch.stack(masked_data), torch.stack(labels))
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    
    def get_test_data_loader(self):
        path = os.path.join(self.path_to_data,"raw_data/MNIST_data")
        trainset = datasets.MNIST(path, download=True, train=True, transform=self.transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
        return trainloader
    
    def get_train_data_loader(self):
        path = os.path.join(self.path_to_data,"raw_data/MNIST_data")
        testset = datasets.MNIST(path, download=True, train=False, transform=self.transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
        return testloader

# Example usage
tokenizer = TokenizerBERT("/scratch/network/OPEN-CATALYST/COS597N/templates/comp/vocab.json", 500)
ul = UniRefDataLoader("raw_data/contigs-sub500_big.fasta", 64, tokenizer, 500)
for masked_sequences, labels in ul.get_data_loader():
    print(masked_sequences, labels)
    break


