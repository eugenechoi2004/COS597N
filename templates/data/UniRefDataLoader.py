import torch
from Bio import SeqIO
import random
import sys
sys.path.append('../comp')
from TokenizerBERT import TokenizerBERT

class UniRefDataLoader():
    def __init__(self, path_to_data, batch_size, tokenizer:TokenizerBERT, max_length, mask_prob=0.15):
        self.max_length = max_length
        self.path_to_data = path_to_data
        self.mask_prob = mask_prob
        self.batch_size = batch_size
        self.pad_token_id = 0  # Update this as per your padding token ID
        self.mask_token_id = 103  # Update this if needed
        self.data = [tokenizer.encode(record.seq) for record in SeqIO.parse(path_to_data, "fasta")]

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

# Example usage
tokenizer = TokenizerBERT("/scratch/network/OPEN-CATALYST/COS597N/templates/comp/vocab.json", 500)
ul = UniRefDataLoader("raw_data/contigs-sub500_big.fasta", 64, tokenizer, 500)
for masked_sequences, labels in ul.get_data_loader():
    print(masked_sequences, labels)
    break


