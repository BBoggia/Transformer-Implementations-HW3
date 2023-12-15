from pathlib import Path
from tokenizers import Tokenizer, Encoding
import torch
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from datasets import load_dataset
from utils_3 import get_dir_path, get_next_model_save_path

def get_all_examples(dataset):
    for split in ['train', 'validation']:
        for example in dataset[split]:
            yield example['dialogue']
            yield example['summary']

def get_tokenizer(lang = 'en'):
    tokenizer_dir = Path(get_dir_path() + f"/{lang}_tokenizer.json")
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = tokenizer_dir / f"{lang}_tokenizer.json"

    unk_token = "<UNK>"
    special_tokens = ["<UNK>", "<SEP>", "<MASK>", "<EOS>", "<SOS>", "<PAD>"]

    if not tokenizer_path.exists():
        tokenizer = Tokenizer(BPE(unk_token=unk_token))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=75_000)
        
        dataset = load_dataset("samsum")

        sentences_generator = get_all_examples(dataset)
        tokenizer.train_from_iterator(sentences_generator, trainer=trainer)

        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def preprocess_dataset(tokenizer, examples):
    conversations = examples['dialogue']
    summaries = examples['summary']
    
    tokenized_conversations = tokenizer.encode_batch(["<SOS> " + c + " <EOS>" for c in conversations])
    tokenized_summaries = tokenizer.encode_batch(["<SOS> " + s + " <EOS>" for s in summaries])
    
    input_ids = [encoding.ids for encoding in tokenized_conversations]
    attention_mask = [[1] * len(encoding.ids) for encoding in tokenized_conversations]
    
    labels = [encoding.ids for encoding in tokenized_summaries]

    return {
        "input_ids": input_ids, 
        "attention_mask": attention_mask,
        "labels": labels
    }



tok = get_tokenizer()
dataset = load_dataset("samsum")
train_dataset = dataset['train']
val_dataset = dataset['validation']

train_dataset = train_dataset.map(
    lambda examples: preprocess_dataset(tok, examples),
    batched=True
)
val_dataset = val_dataset.map(
    lambda examples: preprocess_dataset(tok, examples),
    batched=True
)
