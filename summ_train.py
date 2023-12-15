import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from summ_model import Transformer, WarmUpLR, buildTransformer
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from summ_data_class import causal_mask, SummaryDataset
from datasets import load_dataset
from summ_tokenizer import get_summ_tokenizer
from utils import get_config, get_next_model_save_path, get_encoded_item_length, update_progress_bar
from tqdm import tqdm
import os
from multiprocessing import Pool, Manager
from threading import Thread

def get_summ_dataset(config):
    dataset = load_dataset('samsum')

    dataset['train'] = dataset['train']

    tokenizer = get_summ_tokenizer(dataset, config)

    dataset_list = [{"dialogue": dialogue, "summary": summary} for dialogue, summary in zip(dataset["train"]["dialogue"], dataset["train"]["summary"])]

    # Split the data into training and validation sets using train_test_split
    train_dataset, val_dataset = train_test_split(dataset_list, test_size=0.15, random_state=42)
    
    train_dataset = SummaryDataset(train_dataset, tokenizer, config['seq_len'])
    val_dataset = SummaryDataset(val_dataset, tokenizer, config['seq_len'])

    print(f"Train dataset length: {len(train_dataset)}, Val dataset length: {len(val_dataset)}")
    print(f"Source vocab size: {tokenizer.get_vocab_size()}")


    # Before adding multithreading this took nearly 10 minutes to run
    # Now it takes about a minute and a half
    # with Manager() as manager:
    #     with Pool() as p:
    #         counter = manager.Value('i', 0)
    #         lock = manager.Lock()
    #         pbar = tqdm(total=len(dataset['train']), desc="Calculating max length")
    #         # Start separate thread to update progress bar
    #         progress_thread = Thread(target=update_progress_bar, args=(pbar, counter, len(dataset['train'])))
    #         progress_thread.start()
    #         # Pass tokenizer, counter, and lock to get_encoded_item_length
    #         source_lengths = p.starmap(get_encoded_item_length, [(item['dialogue'], tokenizer, counter, lock) for item in dataset['train']])
    #         target_lengths = p.starmap(get_encoded_item_length, [(item['summary'], tokenizer, counter, lock) for item in dataset['train']])
    #         progress_thread.join()  # wait for progress bar update thread to finish

    # max_source_len = max(source_lengths)
    # max_target_len = max(target_lengths)

    # print(f"Max source length: {max_source_len}, Max target length: {max_target_len}")

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer


def greedy_decode(model, source, source_mask, tokenizer, max_len, device):
    sos_idx = tokenizer.token_to_id('[SOS]')
    eos_idx = tokenizer.token_to_id('[EOS]')

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


def run_validation(model, val_dataloader, tokenizer, max_len, device, print_msg, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []
    print_msg("Running validation...2")
    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    total_val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)
            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            output = greedy_decode(model, encoder_input, encoder_mask, tokenizer, max_len, device)
            source_text = batch["source_text"][0]
            target_text = batch["target_text"][0]
            model_out_text = tokenizer.decode(output.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
            
    avg_val_loss = total_val_loss / len(val_dataloader)
    print(f"Average Validation Loss: {avg_val_loss:.4f}")

        
def train_summ(config):
    device = config['device']
    print(f"Using device: {device}")

    train_dataloader, val_dataloader, tokenizer = get_summ_dataset(config)

    source_vocab_size = target_vocab_size = tokenizer.get_vocab_size()
    config['vocab_size'] = source_vocab_size
    loss_func = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]'), label_smoothing=0.1125).to(device)
    print("Building model...")
    model, optimizer, lr_scheduler = buildTransformer(source_vocab_size, target_vocab_size, config['seq_len'], config['seq_len'], learning_rate = config['lr'], d_model=config['d_model'])
    current_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    model.to(device)
    for epoch in range(current_epoch, config['num_epochs']):
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        model.train()
        pad_token_id = tokenizer.token_to_id('[PAD]')
        total_loss = 0
        total_non_pad_elements = 0
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

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            non_pad_elements = (label != pad_token_id).sum()
            total_non_pad_elements += non_pad_elements.item()
            loss = loss_func(projection_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))
            loss_per_non_pad = loss.item() * non_pad_elements.item() 
            
            total_loss += loss_per_non_pad

            optimizer.step()
            lr_scheduler.step()

            global_step += 1

        avg_loss = total_loss / total_non_pad_elements
        print(f"Average Training Loss: {avg_loss:.4f}")
        run_validation(model, val_dataloader, tokenizer, config['seq_len'], device, lambda msg: print(msg))

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



if __name__ == "__main__":
    config = get_config()
    train_summ(config)
