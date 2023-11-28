import pickle
import os
import torch
import torch.nn as nn
from transformers import RobertaTokenizerFast, Trainer, TrainingArguments, RobertaForTokenClassification
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss, accuracy_score, matthews_corrcoef
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
import re
import tqdm
import torch.nn.functional as F
from datetime import datetime
from copy import deepcopy
from scipy.special import softmax
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, AutoModel
from tqdm import tqdm

class ProteinDegreeDataset(Dataset):

    def __init__(self, max_length, df, tokenizer, region_type):
        self.region_type = region_type # e.g. 'full'
        self.df = df
        self.seqs, self.labels, self.plddts = self.load_dataset()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_dataset(self):
        seq = list(self.df['Sequence']) # list of protein sequences
        label = list(self.df[self.region_type]) # list of list of labels
        plddts = list(self.df['pLDDT'])
        return seq, label, plddts

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq = " ".join("".join(self.seqs[idx].split()))
        seq = re.sub(r"[UZOB]", "X", seq)

        seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_length)
        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        tens = torch.tensor(self.labels[idx], dtype=torch.long)
        sample['labels'] = F.pad(tens, (0, MAX_LENGTH - len(tens)))
        plddts = torch.tensor(self.plddts[idx], dtype=torch.float)
        sample['plddts'] = F.pad(plddts, (0, MAX_LENGTH - len(plddts)))
        return sample

def precision_recall_f1_roc_convolve(name, logits, labels, convolution):
    convolved = np.convolve(np.array(logits).flatten(), np.array(convolution / np.sum(convolution)).flatten(), 'same')
    p = [(1 - i, i) for i in convolved]
    roc = [i[1] for i in p]
    roc2 = [i[0] for i in p]
    p = np.argmax(p, axis=-1)
    precision, recall, f1, support = precision_recall_fscore_support(labels, p)
    roc_auc = roc_auc_score(labels, roc)
    mcc = matthews_corrcoef(labels, p)
    return {
        f'precision_{name}':precision[1],
        f'recall_{name}':recall[1],
        f'f1_{name}':f1[1],
        f'roc_auc_{name}':roc_auc,
        f'mcc_{name}': mcc,
    }

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    # creates probabilities
    logits = softmax(logits, axis=2)
    l = []
    # concatenates all true labels for sequence into list
    for j, i in enumerate(labels):
        l = l + list(i[:len(df_val['Sequence'].iloc[j])])
    # concatenates all prob that were predicted to be disordered into list
    lg2 = []
    for k, i in enumerate(logits):
        lg2 = lg2 + [j[1] for j in i[:len(df_val['Sequence'].iloc[k])]]
    # l is list of all concatenated true labels
    # lg2 is list of all concatenated probabilities that corresponding residue in l is equal to 1  
    metrics = {}
    metrics.update(precision_recall_f1_roc_convolve('normal', lg2, l, [1]))
#     metrics.update(precision_recall_f1_roc_convolve('wa5', lg2, l, [1,1,1,1,1]))
#     metrics.update(precision_recall_f1_roc_convolve('wa9', lg2, l, [1,1,1,1,1,1,1]))
#     metrics.update(precision_recall_f1_roc_convolve('wa15', lg2, l, [1]*15))
#     metrics.update(precision_recall_f1_roc_convolve('linear5', lg2, l, [1,2,3,2,1]))
#     metrics.update(precision_recall_f1_roc_convolve('linear9', lg2, l, [1,2,3,4,5,4,3,2,1]))
#     metrics.update(precision_recall_f1_roc_convolve('linear15', lg2, l, [1,2,3,4,5,6,7,8,7,6,5,4,3,2,1]))
#     metrics.update(precision_recall_f1_roc_convolve('quad5', lg2, l, [1,3,9,3,1]))
#     metrics.update(precision_recall_f1_roc_convolve('quad9', lg2, l, [1,3,9,27,81,27,9,3,1]))
#     metrics.update(precision_recall_f1_roc_convolve('quad15', lg2, l, [1,3,9,27,81,243,729,2187,729,243,81,27,9,3,1]))
    
    logits_path = OUTPUT_DIR + '/Logits/'
    if not os.path.isdir(logits_path):
        os.mkdir(logits_path)
    new_df = deepcopy(df_val)
    new_df['Logits'] = [[i[1] for i in x] for x in list(logits)]
    pickle.dump(new_df, open(logits_path + datetime.now().strftime("%H:%M:%S"), 'wb'))
    return metrics

class CustomBERTClassifier(nn.Module):
    def __init__(self, path, num_classes, plddt = False):
        super(CustomBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(path)
        self.dropout = nn.Dropout(0.1)  # Adjust dropout rate as needed
        self.plddt = plddt
        if plddt:
            self.fc = nn.Linear(self.bert.config.hidden_size + 1, num_classes)
        else:
            self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
            

    def forward(self, input_ids, attention_mask, plddts=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state
        if plddts is not None and self.plddt:
            pooled_output = torch.concat((pooled_output, plddts.unsqueeze(-1)), dim=2)
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

# from transformers import AutoModelForTokenClassification
def model_init():
#     model = AutoModelForTokenClassification.from_pretrained(PRETRAINED_MODEL, num_labels=NUM_CLASSES)
    model = CustomBERTClassifier(PRETRAINED_MODEL, NUM_CLASSES, True)
    return model

if __name__ == '__main__':

    MAX_LENGTH = 1024
    EPOCHS = 30
    LEARNING_RATE = 2e-6
    BATCH_SIZE = 1
    TOKENIZER_PATH =  "../checkpoint-final/"
    # is this pretrained on protein sequences?
    PRETRAINED_MODEL = "../checkpoint-final/"
    NUM_CLASSES = 2
    OUTPUT_DIR = '../subset_outputs/custom_modified/'
    
    df_full = pickle.load(open('../Datasets/subset_featurized.pkl', "rb"))

    df_train = pickle.load(open('../Datasets/subset_train.pkl', "rb"))
    df_val = pickle.load(open('../Datasets/subset_val.pkl', "rb"))
    df_test = pickle.load(open('../Datasets/subset_test.pkl', "rb"))
    
    
    tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_PATH, do_lower_case=False)
    train_dataset = ProteinDegreeDataset(MAX_LENGTH, df_train, tokenizer, 'full')
    val_dataset = ProteinDegreeDataset(MAX_LENGTH, df_val, tokenizer, 'full')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = model_init()
    
    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Initialize optimizer and training parameters
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_epochs = EPOCHS
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000)
    
    best_val_loss = float('inf')
    best_model_path = OUTPUT_DIR + 'best_model.pth'
    
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        # set model in training mode
        model.train()
        
        total_loss = 0

        # for each epoch
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            plddts = batch['plddts'].to(device)
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask=attention_mask, plddts=plddts)
            outputs = outputs.view(-1, 2)
            labels = labels.view(-1)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        average_loss = total_loss / len(train_loader)
        train_losses.append(average_loss)
        print(f'Epoch {epoch+1} - Training Loss: {average_loss}')

        # Validation loop
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                plddts = batch['plddts'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, plddts=plddts)
                preds = outputs.view(-1, 2)
                labs = labels.view(-1)
                val_loss = criterion(preds, labs)
                total_val_loss += val_loss.item()
                
        # metrics = compute_metrics((outputs.cpu().numpy(), labels.cpu().numpy()))
        average_val_loss = total_val_loss / len(val_loader)
        val_losses.append(average_val_loss)
        print(f'Epoch {epoch+1} - Validation Loss: {average_val_loss}')
        
        # Save the best model
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            torch.save(model.state_dict(), best_model_path)
            print('Best model saved.')
        
    losses = {
        'train': train_losses,
        'val': val_losses
    }
    
    with open(f'{OUTPUT_DIR}losses.pkl', 'wb') as f:
        pickle.dump(losses, f)