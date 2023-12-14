from Trainer import Trainer
from models.BERTModel import BERTModel
from data.UniRefDataLoader import UniRefDataLoader
from comp.TokenizerBERT import TokenizerBERT
import os
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import torch
from metrics.Metrics import Metrics
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime
import json


class BERT_Trainer(Trainer):
    def __init__(self, max_sequence_length, embedding_dim, hidden_dim, dropout_prob, num_heads, num_layers, vocab_length, batch_size, epochs, checkpoint_paths, vocab_path, data_path):
        tokenizer = TokenizerBERT(vocab_path, max_sequence_length)
        data = UniRefDataLoader(data_path, batch_size, tokenizer, max_sequence_length)
        self.train_loader = data.get_train_data_loader()
        self.test_loader = data.get_test_data_loader()  # This will be used as validation set
        self.model = BERTModel(max_sequence_length, embedding_dim, hidden_dim, dropout_prob, num_heads, num_layers, vocab_length)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.log_interval = 10
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.1)
        self.pad_token_id = 1
        self.epochs = epochs
        self.checkpoint_paths = checkpoint_paths
        self.hyperparameters = {
            'max_sequence_length': max_sequence_length,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'dropout_prob': dropout_prob,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'vocab_length': vocab_length,
            'batch_size': batch_size,
            'epochs': epochs
        }
        self.current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.metric_path, self.weights_path = self.make_dirs()
        self.metrics = Metrics(self.metric_path)
        

    def make_dirs(self):
        folder_path = os.path.join(self.checkpoint_paths, self.current_time)
        os.mkdir(folder_path)
        metric_path = os.path.join(folder_path, "metrics")
        weights_path =  os.path.join(folder_path, "weights")
        config_path =  os.path.join(folder_path, "configs.json")
        os.mkdir(metric_path)
        os.mkdir(weights_path)
        with open(config_path, 'w') as file:
            json.dump(self.hyperparameters, file)

        return metric_path, weights_path

    def save_checkpoint(self, epoch, filename="checkpoint.pth.tar"):
        state = {'epoch': epoch, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
        torch.save(state, filename)

    def calculate_accuracy_and_f1(self, outputs, targets):
        predictions = torch.argmax(outputs, dim=1)
        accuracy = accuracy_score(targets.cpu(), predictions.cpu())
        f1 = f1_score(targets.cpu(), predictions.cpu(), average='weighted')
        return accuracy, f1

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss, total_accuracy, total_f1 = 0, 0, 0
        criterion = CrossEntropyLoss(ignore_index=-100)
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                output = output.view(-1, self.model.vocab_size)
                target = target.view(-1)
                loss = criterion(output, target)
                accuracy, f1 = self.calculate_accuracy_and_f1(output, target)
                total_loss += loss.item()
                total_accuracy += accuracy
                total_f1 += f1
        return total_loss / len(data_loader), total_accuracy / len(data_loader), total_f1 / len(data_loader)

    def train(self):
        print("Starting training")
        criterion = CrossEntropyLoss(ignore_index=-100)
        for epoch in range(self.epochs):
            print(f"{epoch} Epoch Ran")
            self.model.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                output = output.view(-1, self.model.vocab_size)
                target = target.view(-1)
                loss = criterion(output, target)
                loss.backward()
                self.optimizer.step()
                if batch_idx % self.log_interval == 0:
                    print(f'Train Epoch: [{batch_idx * len(data)}/{len(self.train_loader.dataset)} ({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            val_loss, val_accuracy, val_f1 = self.evaluate(self.test_loader)
            print(f'Epoch {epoch} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation F1: {val_f1:.4f}')
            self.metrics.track_loss_point(loss.item(), val_loss)
            self.metrics.track_accuracy_point(val_accuracy, val_accuracy)
            self.metrics.track_f1_point(val_f1, val_f1)
            weights = os.path.join(self.weights_path, f"checkpoint_epoch_{epoch}.pth.tar")
            self.save_checkpoint(epoch, filename=weights)
        self.metrics.save_loss_plot()
        self.metrics.save_accuracy_plot()


# Hyperparameters
embedding_dim = 768
num_encoder_layers = 6
max_sequence_length = 500
hidden_dim = 3072
dropout_prob = 0.2
num_heads = 12
vocab_length = 30
batch_size = 10
epochs = 11
checkpoing_path = "./checkpoints/"
vocab_path = "comp/vocab.json"
data_path = "data/raw_data/test.fasta"

trainer = BERT_Trainer(max_sequence_length, embedding_dim, hidden_dim, dropout_prob, num_heads, num_encoder_layers, vocab_length, batch_size, epochs,checkpoing_path, vocab_path, data_path)
trainer.train()
