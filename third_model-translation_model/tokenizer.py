from pathlib import Path
from utils import get_tokenizer_save_dir
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

def get_all_examples(dataset, lang):
    for i in dataset:
        # yield i['translation'][lang]
        yield i[lang]

def get_summ_texts(dataset, split):
    dialogues = dataset[split]['dialogue']
    summary = dataset[split]['summary']
    for dialogue, highlight in zip(dialogues, summary):
        yield dialogue
        yield highlight

def get_summ_tokenizer(dataset, config, split='train'):
    tokenizer_dir = "/Users/bransonboggia/Desktop/advanced_deep_learning_HW3/summ_tokenizers"
    Path(tokenizer_dir).mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path(tokenizer_dir + "/summarization_tokenizer.json")

    unk_token = "[UNK]"
    spl_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"]

    if not tokenizer_path.exists():
        tokenizer = Tokenizer(BPE(unk_token=unk_token))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
        trainer = BpeTrainer(
            special_tokens=spl_tokens, 
            vocab_size=config['vocab_size'], 
            show_progress=True, 
            min_frequency=2
        )

        texts_generator = get_summ_texts(dataset, split)
        tokenizer.train_from_iterator(texts_generator, trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_tokenizer(dataset, lang, config):
    tokenizer_dir = get_tokenizer_save_dir(config)
    Path(tokenizer_dir).mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path(tokenizer_dir + f"/{lang}_tokenizer.json")

    unk_token = "[UNK]"
    spl_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"]

    if not tokenizer_path.exists():
        tokenizer = Tokenizer(BPE(unk_token=unk_token))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        trainer = BpeTrainer(special_tokens=spl_tokens, vocab_size=config['vocab_size'], show_progress=True, min_frequency=2)

        sentences_generator = get_all_examples(dataset, lang)
        tokenizer.train_from_iterator(sentences_generator, trainer=trainer)

        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer