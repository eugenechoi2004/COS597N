import torch
from torchvision import datasets, transforms
from .CustomDataLoader import CustomDataLoader
import os

class MNISTDataLoader(CustomDataLoader):

    def __init__(self, path_to_data):
        self.transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
        self.path_to_data = path_to_data

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


