import torch
from torch.utils.data.dataset import Dataset

class LanguageDataset(Dataset):

    def __init__(self, dataset, source_tokenizer, target_tokenizer, source_lang, target_lang, seq_len):
        super().__init__()
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.dataset = dataset
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([target_tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([target_tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([target_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # source_text = self.dataset[idx]['translation'][self.source_lang]
        # target_text = self.dataset[idx]['translation'][self.target_lang]
        source_text = self.dataset[idx][self.source_lang]
        target_text = self.dataset[idx][self.target_lang]

        # Encoding src and tgt sentences with special tokens then converting to tensors
        encoder_tokens = self.source_tokenizer.encode(source_text).ids
        decoder_tokens = self.target_tokenizer.encode(target_text).ids

        # print("ENCODER 1: - ", len(encoder_tokens), encoder_tokens[0])
        # print("DECODER 1: - ", len(decoder_tokens), decoder_tokens[0])

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
        
        # Creating masks
        source_mask = (encoder_tokens != self.pad_token).unsqueeze(0).unsqueeze(0).int()
        target_mask = (decoder_tokens != self.pad_token).unsqueeze(0).int()

        # Print shapes of tensors to make sure everything is correct size
        assert encoder_tokens.size(0) == self.seq_len
        assert decoder_tokens.size(0) == self.seq_len
        assert label.size(0) == self.seq_len


        return {
            "encoder_input": encoder_tokens,
            "decoder_input": decoder_tokens,
            "encoder_mask": source_mask,
            "decoder_mask": target_mask & causal_mask(decoder_tokens.size(0)),
            "label": label,
            "source_text": source_text,
            "target_text": target_text,
        }

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0