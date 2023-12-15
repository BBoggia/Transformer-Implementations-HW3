import torch
from torch.utils.data.dataset import Dataset

class SummaryDataset(Dataset):

    def __init__(self, dataset, tokenizer, seq_len):
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        source_text = self.dataset[idx]['dialogue']
        target_text = self.dataset[idx]['summary']

        # Encoding src and tgt sentences with special tokens
        encoder_tokens = self.tokenizer.encode(source_text).ids
        decoder_tokens = self.tokenizer.encode(target_text).ids

        # Encoding dialogues and summary with the same tokenizer
        encoder_padding_size = self.seq_len - len(encoder_tokens) - 2
        decoder_padding_size = self.seq_len - len(decoder_tokens) - 1
        
        encoder_tokens = torch.cat((self.sos_token,
                                    torch.tensor(encoder_tokens, dtype=torch.int64),
                                    self.eos_token,
                                    torch.tensor([self.pad_token] * encoder_padding_size, dtype=torch.int64)), dim=0)
        
        label = torch.cat((torch.tensor(decoder_tokens, dtype=torch.int64),
                                        self.eos_token,
                                        torch.tensor([self.pad_token] * decoder_padding_size, dtype=torch.int64)), dim=0)
        
        decoder_tokens = torch.cat((self.sos_token,
                                    torch.tensor(decoder_tokens, dtype=torch.int64),
                                    torch.tensor([self.pad_token] * decoder_padding_size, dtype=torch.int64)), dim=0)

        # Masks (padding and causal mask for the decoder)
        encoder_mask = (encoder_tokens != self.pad_token).unsqueeze(0).unsqueeze(0).int()
        decoder_mask = (decoder_tokens != self.pad_token).unsqueeze(0).int()

        # Print the shapes of the tensors to make sure everything is the correct size
        assert encoder_tokens.size(0) == self.seq_len, f"encoder_tokens.size(0) = {encoder_tokens.size(0)} != {self.seq_len}"
        assert decoder_tokens.size(0) == self.seq_len, f"decoder_tokens.size(0) = {decoder_tokens.size(0)} != {self.seq_len}"
        assert label.size(0) == self.seq_len, f"label.size(0) = {label.size(0)} != {self.seq_len}"

        return {
            "encoder_input": encoder_tokens,
            "decoder_input": decoder_tokens,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask & causal_mask(decoder_tokens.size(0)),
            "label": label,
            "source_text": source_text,
            "target_text": target_text,
        }

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0