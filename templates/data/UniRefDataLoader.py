import torch
from torchvision import datasets, transforms
from .CustomDataLoader import CustomDataLoader
import os
from Bio import SeqIO

class UniRefDataLoader(CustomDataLoader):

    def __init__(self, path_to_data, batch_size):
        self.path_to_data = path_to_data
        self.max_length = 200
        self.data = [record.seq for record in SeqIO.parse("raw_data/contigs-sub500_big.fasta", "fasta")]
        self.batch_size = batch_size

    def get_test_data_loader(self):
        path = os.path.join(self.path_to_data,"raw_data/contigs-sub500_big.fasta)
        trainset = self.data
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True)

    def get_train_data_loader(self):
        path = os.path.join(self.path_to_data,"raw_data/contigs-sub500_big.fasta")
        testset = self.data
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=True)
        return testloader



for record in SeqIO.parse("raw_data/contigs-sub500_big.fasta", "fasta"):
    print("ID: %s" % record.id)
    print("Sequence: %s" % record.seq)
    record
    

