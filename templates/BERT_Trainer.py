from Trainer import Trainer
from models.BERTModel import BERTModel
from data.UniRefDataLoader import UniRefDataLoader
import os
import torch.optim as optim
import torch.nn.functional as F

class BERT_Trainer(Trainer):

    def __init__(self):
        data = MNISTDataLoader("./data")
        self.train_loader, self.test_loader() = data.get_train_data_loader()
        self.test_loader = data.get_test_data_loader()
        self.model = ExampleModel()
        self.device = "cpu"
        self.log_interval = 10
        self.save_checkpoints = 100
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    100, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))
    
    def test():
        return

    def save_checkpoints():
        return
    
    def save_final():
        return




trainer = Mnist_Trainer()
trainer.train()