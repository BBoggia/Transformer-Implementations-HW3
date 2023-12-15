from calendar import c
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from model import Transformer, WarmUpLR, buildTransformer
from torch.utils.data import DataLoader, random_split
from torch.optim import Optimizer
from data_class import LanguageDataset, causal_mask, SummaryDataset
from datasets import load_dataset
from tokenizer import get_tokenizer, get_summ_tokenizer
from torch.utils.tensorboard import SummaryWriter
from utils import get_dir_path, get_last_model_save_path, get_config, get_next_model_save_path, get_encoded_item_length, update_progress_bar
from tqdm import tqdm
import torchmetrics
import os
from multiprocessing import Pool, Manager
from threading import Thread

def get_dataset(config):
    dataset = load_dataset("opus_books", f'''{config['source_lang']}-{config['target_lang']}''', split="train")
    
    source_tokenizer = get_tokenizer(dataset, config['source_lang'], config)
    target_tokenizer = get_tokenizer(dataset, config['target_lang'], config)

    # train_dataset_length = int(len(dataset) * 0.9)
    # train_dataset, val_dataset = random_split(dataset, [train_dataset_length, len(dataset) - train_dataset_length])

    train_dataset, val_dataset = [i['translation'] for i in train_test_split(dataset, test_size=0.2, random_state=42)]
    #train_dataset, val_dataset2 = train_dataset['translation'], val_dataset['translation']
    
    train_dataset = LanguageDataset(train_dataset, source_tokenizer, target_tokenizer, config['source_lang'], config['target_lang'], config['seq_len'])
    val_dataset = LanguageDataset(val_dataset, source_tokenizer, target_tokenizer, config['source_lang'], config['target_lang'], config['seq_len'])

    print(f"Dataset length: {len(dataset)}")
    print(f"Train dataset length: {len(train_dataset)}, Val dataset length: {len(val_dataset)}")
    print(f"Source vocab size: {source_tokenizer.get_vocab_size()}, Target vocab size: {target_tokenizer.get_vocab_size()}")
    

    max_source_len = max([len(source_tokenizer.encode(item['translation'][config['source_lang']]).ids) for item in dataset])
    max_target_len = max([len(target_tokenizer.encode(item['translation'][config['target_lang']]).ids) for item in dataset])

    print(f"Max source length: {max_source_len}, Max target length: {max_target_len}")

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, source_tokenizer, target_tokenizer

def get_summ_dataset(config):
    dataset = load_dataset('samsum')

    dataset['train'] = dataset['train']

    tokenizer = get_summ_tokenizer(dataset, config)

    dataset_list = [{"dialogue": dialogue, "summary": summary} for dialogue, summary in zip(dataset["train"]["dialogue"], dataset["train"]["summary"])]

    # Split data into training and validation sets using train_test_split
    train_dataset, val_dataset = train_test_split(dataset_list, test_size=0.15, random_state=42)
    
    train_dataset = SummaryDataset(train_dataset, tokenizer, config['seq_len'])
    val_dataset = SummaryDataset(val_dataset, tokenizer, config['seq_len'])

    print(f"Train dataset length: {len(train_dataset)}, Val dataset length: {len(val_dataset)}")
    print(f"Source vocab size: {tokenizer.get_vocab_size()}")


    # Before adding multithreading this took nearly 10 minutes to run
    # Now it takes about a minute and a half
    with Manager() as manager:
        with Pool() as p:
            counter = manager.Value('i', 0)
            lock = manager.Lock()
            pbar = tqdm(total=len(dataset), desc="Calculating max length")
            # Start separate thread to update progress bar
            progress_thread = Thread(target=update_progress_bar, args=(pbar, counter, len(dataset)))
            progress_thread.start()
            # Pass tokenizer, counter, and lock to get_encoded_item_length
            source_lengths = p.starmap(get_encoded_item_length, [(item['dialogue'], tokenizer, counter, lock) for item in dataset['train']])
            target_lengths = p.starmap(get_encoded_item_length, [(item['summary'], tokenizer, counter, lock) for item in dataset['train']])
            progress_thread.join()  # wait for progress bar update thread to finish

    max_source_len = max(source_lengths)
    max_target_len = max(target_lengths)

    print(f"Max source length: {max_source_len}, Max target length: {max_target_len}")

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer

def get_model(config, source_vocab_len, target_vocab_len) -> (Transformer, Optimizer, WarmUpLR):
    model: Transformer
    optimizer: Optimizer
    lr_scheduler: WarmUpLR
    model, optimizer, lr_scheduler = buildTransformer(source_vocab_len, target_vocab_len, config['seq_len'], config['seq_len'], config['d_model'])
    return (model, optimizer, lr_scheduler)

def greedy_decode(model, source, source_mask, tokenizer_source, tokenizer_target, max_len, device):
    sos_idx = tokenizer_target.token_to_id('[SOS]')
    eos_idx = tokenizer_target.token_to_id('[EOS]')

    # Precompute encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize decoder input with sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_source, tokenizer_target, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check batch size = 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_source, tokenizer_target, max_len, device)

            source_text = batch["source_text"][0]
            target_text = batch["target_text"][0]
            model_out_text = tokenizer_target.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # Evaluate character error rate
        # Compute char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()

def train(config):
    device = config['device']
    print(f"Using device: {device}")

    train_dataloader, val_dataloader, source_tokenizer, target_tokenizer = get_dataset(config)
    
    model, optimizer, lr_scheduler = get_model(config, source_tokenizer.get_vocab_size(), target_tokenizer.get_vocab_size())
    model = model.to(device)
    writer = SummaryWriter(log_dir = get_dir_path() + f"/{config['log_dir']}/{config['experiment_name']}")

    current_epoch = 0
    global_step = 0
    best_val_loss = float('inf')

    if 'load_model' in config and config['load_model']:
        model_path = get_last_model_save_path(config)
        print(f"Loading model from {model_path}")
        state = torch.load(model_path)
        model.load_state_dict(state['model_state_dict'])
        epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = config['global_step']

    loss_func = nn.CrossEntropyLoss(ignore_index=source_tokenizer.token_to_id('[PAD]')).to(device)

    for epoch in range(current_epoch, config['num_epochs']):
        training_bar = tqdm(train_dataloader, unit="batch")
        model.train()
        for batch in training_bar:
            optimizer.zero_grad()

            encoder_input = batch['encoder_input'].to(device) # (batch_size, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (batch_size, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch_size, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (batch_size, 1, seq_len, seq_len)
            
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch_size, seq_len, d_model)
            projection_output = model.project(decoder_output) # (batch_size, seq_len, target_vocab_size)

            label = batch['label'].to(device) # (batch_size, seq_len)

            loss = loss_func(projection_output.view(-1, target_tokenizer.get_vocab_size()), label.view(-1)) # (batch_size * seq_len, target_vocab_size), (batch_size * seq_len)
            training_bar.set_description(f"Epoch {epoch:02d}, Loss: {loss.item():.4f}")
            writer.add_scalar('Loss/train', loss.item(), global_step=global_step)
            writer.flush()
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            global_step += 1
            
        run_validation(model, val_dataloader, source_tokenizer, target_tokenizer, config['seq_len'], device, lambda msg: training_bar.write(msg), global_step, writer)

        if best_val_loss > loss.item():
            best_val_loss = loss.item()
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'learning_rate': lr_scheduler.get_last_lr()[0],
                }, get_next_model_save_path(config))
                print(f"Model saved at {get_next_model_save_path(config)}")
            except:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'learning_rate': lr_scheduler.get_last_lr()[0],
                }, "/data/bboggia/hw3_adv_deep_learning/model_saves/transformer_model_0.pt")
        
def train_summ(config):
    device = config['device']
    print(f"Using device: {device}")

    source_seq_len = 1200
    learning_rate = 5e-5 

    config['seq_len'] = source_seq_len
    config['lr'] = learning_rate

    train_dataloader, val_dataloader, tokenizer = get_summ_dataset(config)

    source_vocab_size = target_vocab_size = tokenizer.get_vocab_size()
    config['vocab_size'] = source_vocab_size
    print("Loss function: CrossEntropyLoss")
    loss_func = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]')).to(device)
    print("Building model...")
    model, optimizer, lr_scheduler = buildTransformer(source_vocab_size, target_vocab_size, source_seq_len, source_seq_len, learning_rate = learning_rate, d_model=config['d_model'])
    print("Model built.")
    current_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    model.to(device)
    for epoch in range(current_epoch, config['num_epochs']):
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        model.train()
        total_loss = 0
        training_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{config['num_epochs']}", unit="batch")
        for batch in training_bar:
            optimizer.zero_grad()

            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch_size, seq_len, d_model)
            projection_output = model.project(decoder_output) # (batch_size, seq_len, target_vocab_size)

            label = batch['label'].to(device) # (batch_size, seq_len)

            loss = loss_func(projection_output.view(-1, tokenizer.get_vocab_size()), label.view(-1)) # (batch_size * seq_len, target_vocab_size), (batch_size * seq_len)
            training_bar.set_description(f"Epoch {epoch:02d}, Loss: {loss.item():.4f}")
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            global_step += 1

        avg_loss = total_loss / len(train_dataloader)
        print(f"Average Training Loss: {avg_loss:.4f}")

        run_validation(model, val_dataloader, tokenizer, tokenizer, config['seq_len'], device, lambda msg: print(msg), global_step, None)

        if best_val_loss > loss.item():
            best_val_loss = loss.item()
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'learning_rate': lr_scheduler.get_last_lr()[0],
                }, get_next_model_save_path(config))
                print(f"Model saved at {get_next_model_save_path(config)}")
            except:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'learning_rate': lr_scheduler.get_last_lr()[0],
                }, "/data/bboggia/hw3_adv_deep_learning/model_saves/transformer_model_0.pt")

        # # Validation step
        # model.eval()
        # total_val_loss = 0
        # with torch.no_grad():
        #     for batch in tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}/{config['num_epochs']}", unit="batch"):
        #         encoder_input = batch['encoder_input'].to(device)
        #         decoder_input = batch['decoder_input'].to(device)
        #         encoder_mask = batch['encoder_mask'].to(device)
        #         decoder_mask = batch['decoder_mask'].to(device)
        #         labels = batch['label'].to(device)

        #         output = model(encoder_input, encoder_mask, decoder_input, decoder_mask)
        #         logits = output.view(-1, output.size(-1))  # Flatten output
        #         batch_loss = loss_func(logits, labels.view(-1))

        #         total_val_loss += batch_loss.item()

        # avg_val_loss = total_val_loss / len(val_dataloader)
        # print(f"Average Validation Loss: {avg_val_loss:.4f}")

if __name__ == "__main__":
    config = get_config()
    # train(config)
    train_summ(config)
