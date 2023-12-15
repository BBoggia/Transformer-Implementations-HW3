import torch
from datasets import Dataset

class SummarizationDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        # Use idx to retrieve corresponding example
        return self.examples[idx]