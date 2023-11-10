from transformers import pipeline
from transformers import RobertaTokenizerFast
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

class SeqEmbedder:
    
    def __init__(self, seq, model_path):
        self.num_classes = 2
        self.model_path = model_path
        self.model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=self.num_classes)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_path, do_lower_case=False)
        self.seq = seq 
    
    def get_embedding(self):
        """Returns an embedding of size len(seq) + 2 by 768."""
        input_ids = self.tokenizer.encode(self.seq, return_tensors="pt")
        # Get token embeddings
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            embeddings = outputs.hidden_states
        return embeddings[-1][0]

if __name__ == '__main__':
    protein_seq = 'VSHQPPEDGL'
    seq_embedder = SeqEmbedder(protein_seq, '/scratch/network/byw2/COS597N/dr-bert-folder/checkpoint-final')
    embedding = seq_embedder.get_embedding()
    print(len(protein_seq))
    print(embedding)
    print(embedding.shape)