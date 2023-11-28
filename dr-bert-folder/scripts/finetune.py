import pickle
import os
import torch
import torch.nn as nn
from transformers import RobertaTokenizerFast, Trainer, TrainingArguments
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
from transformers import AutoModelForTokenClassification


class ProteinDegreeDataset(Dataset):

    def __init__(self, max_length, df, tokenizer, region_type):
        self.region_type = region_type # e.g. 'full'
        self.df = df
        self.seqs, self.labels = self.load_dataset()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_dataset(self):
        seq = list(self.df['Sequence']) # list of protein sequences
        label = list(self.df[self.region_type]) # list of list of labels
        return seq, label

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
    print(np.array(lg2).shape)
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

def model_init():
    model = AutoModelForTokenClassification.from_pretrained(PRETRAINED_MODEL, num_labels=NUM_CLASSES)
    return model

if __name__ == '__main__':
    
    MAX_LENGTH = 1024
    EPOCHS = 10
    LEARNING_RATE = 2e-6
    BATCH_SIZE = 1
    TOKENIZER_PATH =  "../checkpoint-final/"
    # is this pretrained on protein sequences?
    PRETRAINED_MODEL = "../checkpoint-final/"
    NUM_CLASSES = 2
    SCHEDULER='cosine_with_restarts'
    
    df_train = pickle.load(open('../Datasets/train.pkl', "rb"))
    df_val = pickle.load(open('../Datasets/val.pkl', "rb"))
    df_test = pickle.load(open('../Datasets/caid.pkl', "rb"))
    df_full = pd.concat([df_train, df_val, df_test])
    
    tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_PATH, do_lower_case=False)
    
    train_dataset = ProteinDegreeDataset(MAX_LENGTH, df_train, tokenizer, 'full')
    val_dataset = ProteinDegreeDataset(MAX_LENGTH, df_val, tokenizer, 'full')
    
    OUTPUT_DIR = f'../Outputs'
    
    training_args = TrainingArguments(
        output_dir = OUTPUT_DIR + '/Checkpoints',
        num_train_epochs = EPOCHS, # 10 epochs
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 16,
        warmup_steps = 1000,
        learning_rate = LEARNING_RATE, # 2e-06
        logging_dir = OUTPUT_DIR + '/Logs',
        logging_steps = 200,
        lr_scheduler_type=SCHEDULER, # cosine with restarts
        do_train = True,
        do_eval = True,
        evaluation_strategy = 'epoch', # evaluate at every epoch
        gradient_accumulation_steps = BATCH_SIZE,
    #     fp16 = True,
    #     fp16_opt_level = '02',
        save_strategy = 'epoch',
        save_total_limit = 2,
        load_best_model_at_end = True
    )
    
    trainer = Trainer(
        model_init=model_init,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        compute_metrics = compute_metrics,
    )
    
    trainer.train()