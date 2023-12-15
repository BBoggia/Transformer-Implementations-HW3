import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq
from tokenizer_3 import get_tokenizer, get_dataset
from data_obj import SummarizationDataset
from model_3 import buildTransformer
from torch.nn.utils import clip_grad_norm_
from utils_3 import create_masks, evaluate, get_dir_path, get_next_model_save_path, collate_fn

def get_data_loaders(tokenizer, batch_size=32):
    collate_fn_with_tokenizer = lambda batch: collate_fn(batch, tokenizer)

    train_dataset, val_dataset = get_dataset(tokenizer)
    
    train_data = SummarizationDataset(train_dataset, tokenizer)
    val_data = SummarizationDataset(val_dataset, tokenizer)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn_with_tokenizer, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_fn_with_tokenizer)
    return train_loader, val_loader


def train(max_src_len = 1024, max_tgt_len = 128, batch_size = 16, log_interval = 100):

    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    max_src_len = max_src_len  # Max length for encoder input
    max_tgt_len = max_tgt_len  # Max length for decoder output


    train_loader, val_loader = get_data_loaders(tokenizer, batch_size)
    
    # Init model, loss func, and optimizer
    model, optimizer, scheduler = buildTransformer(vocab_size=vocab_size, max_src_len=max_src_len, max_tgt_len=max_tgt_len)
    loss_fn = nn.CrossEntropyLoss()
    
    if torch.cuda.is_available():
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    elif torch.has_mps:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    model.to(device)
    print("TRAINING START")
    
    best_accuracy = 0
    epochs = 12
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0
        train_bar = tqdm(train_loader, desc="Training")
        
        for batch in train_bar:
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            enc_mask, tgt_mask = create_masks(input_ids, labels[:,:-1], tokenizer)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, labels[:,:-1], create_masks(input_ids, labels[:,:-1], tokenizer))
            
            # Compute loss and backpropagate
            optimizer.zero_grad()

            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels[:, 1:].contiguous().view(-1)  # Shift labels for loss
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            clip_grad_norm_(model.parameters(), max_norm=1.0)

            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_bar.set_description(f"Train loss: {total_loss.item():.6f}, lr: {scheduler.get_last_lr()[0]:.2E}")
        
        metrics = evaluate(model, val_loader, tokenizer, device)
        print(f"Average Loss: {total_loss / len(train_loader)}, Validation metrics: {metrics}")

        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            # Save best model
            torch.save(model.state_dict(), get_dir_path() + '/model_saves/transformer_qa_model_best.pth')

        print(f"Epoch {epoch}, validation accuracy: {metrics['accuracy']}, best accuracy: {best_accuracy}, precision: {metrics['precision']}, recall: {metrics['recall']}, f1: {metrics['f1']}")

    # Save the final model
    torch.save(model.state_dict(), get_next_model_save_path())

if __name__ == "__main__":
    train()
    