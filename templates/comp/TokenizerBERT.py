import torch
import json

class TokenizerBERT:
    def __init__(self, vocab_file, max_length):
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)
        self.max_length = max_length
        self.pad_token_id = self.vocab['[PAD]']
        self.id_to_token = {id_: token for token, id_ in self.vocab.items()}

    def encode(self, text):
        token_ids = self.tokenize_and_pad(text)
        return torch.tensor(token_ids)

    def tokenize_and_pad(self, text):
        # Tokenize the text and truncate it to the max length
        token_ids = [self.vocab.get(token, self.vocab['[UNK]']) for token in text][:self.max_length]
        return self.pad(token_ids)

    def pad(self, token_ids):
        # Pad the token ids list to the max length
        padded = token_ids + [self.pad_token_id] * (self.max_length - len(token_ids))
        return padded

    def decode(self, token_ids):
        # Decode the token ids back to text, ignoring padding tokens
        return ''.join([self.id_to_token.get(id_, '[UNK]') for id_ in token_ids if id_ != self.pad_token_id])

