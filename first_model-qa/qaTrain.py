from torch.utils.data import DataLoader
import torch.optim as optim
from datasets import load_dataset
from old_files.qaTokenizer import preprocess_dataset, get_tokenizer
import torch
import torch.nn as nn
from old_files.qaModel import buildTransofermer
from tqdm import tqdm

def get_train_val_datasets(tokenizer):
    datasets = load_dataset("squad_v2")
    train_dataset = datasets['train']
    val_dataset = datasets['validation']

    datasets = load_dataset("squad_v2")
    train_dataset = datasets['train']
    val_dataset = datasets['validation']
    
    # Apply the preprocessing function
    train_dataset = train_dataset.map(
        lambda examples: preprocess_dataset(examples, tokenizer),
        batched=True
    )
    val_dataset = val_dataset.map(
        lambda examples: preprocess_dataset(examples, tokenizer),
        batched=True
    )
    
    # Apply the preprocessing function
    # train_dataset = train_dataset.map(preprocess_dataset, batched=True)
    # val_dataset = val_dataset.map(preprocess_dataset, batched=True)
    
    # # Set format for pytorch
    # train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions'])
    # val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions'])
    print("DATASET DONE")
    return train_dataset, val_dataset


def encode_example(example, tokenizer):
    return preprocess_dataset(
        tokenizer,
        example['context'],
        example['question'],
        answer=example['answers']['text'][0] if example['answers']['text'] else None,
        answer_start=example['answers']['answer_start'][0] if example['answers']['answer_start'] else None,
    )    

def get_processed_dataset(tokenizer):
    raw_train_dataset, raw_val_dataset = get_train_val_datasets(tokenizer)
    
    #Applying encoding to each example
    train_dataset = raw_train_dataset # raw_train_dataset.map(encode_example, remove_columns=raw_train_dataset.column_names)
    val_dataset = raw_val_dataset # raw_val_dataset.map(encode_example, remove_columns=raw_val_dataset.column_names)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,  collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16,  collate_fn=custom_collate_fn)
    print("LOADER DONE")
    return train_loader, val_loader

def accuracy(out, labels):
    outputs = torch.argmax(out, dim=1)
    return torch.sum(outputs == labels).item()

def validate(model, val_loader, device):
    model.eval()  
    total_correct = 0
    total_elements = 0
    with torch.no_grad():  
        val_bar = tqdm(val_loader, desc="Validating")
        for batch in val_bar:
            input_ids, attention_mask, start_positions, end_positions = batch['input_ids'], batch['attention_mask'], batch['start_positions'], batch['end_positions']
            input_ids, attention_mask, start_positions, end_positions = input_ids.to(device), attention_mask.to(device), start_positions.to(device), end_positions.to(device)
            
            # Forward pass only
            start_logits, end_logits = model(input_ids, attention_mask)
            
            # Calculate accuracy
            total_correct += accuracy(start_logits, start_positions) + accuracy(end_logits, end_positions)
            total_elements += start_positions.numel() + end_positions.numel()

    return total_correct / total_elements  # Return the accuracy over the dataset

def train(max_seq_len = 1024):
    # Load and tokenize the data
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    seq_len = max_seq_len  
    
    train_loader, val_loader = get_processed_dataset(tokenizer)
    
    # Initialize model, loss function, and optimizer
    model = buildTransofermer(vocab_size=vocab_size, seq_len=seq_len)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.has_mps:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    model.to(device)
    print("TRAINING START")
    # Train
    best_accuracy = 0
    epochs = 3
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0
        train_bar = tqdm(train_loader, desc="Training")
        for batch in train_bar:
            
            # Check that all start and end positions are less than max_seq_len
            assert (batch['start_positions'] < max_seq_len).all()
            assert (batch['end_positions'] < max_seq_len).all()
            input_ids, attention_mask, start_positions, end_positions = batch['input_ids'], batch['attention_mask'], batch['start_positions'], batch['end_positions']
            input_ids, attention_mask, start_positions, end_positions = input_ids.to(device), attention_mask.to(device), start_positions.to(device), end_positions.to(device)

            # Clear out the gradients of the model's parameters
            optimizer.zero_grad()
            
            # Forward pass and Compute loss
            start_logits, end_logits = model(input_ids, attention_mask)
            start_loss = loss_function(start_logits, start_positions)
            end_loss = loss_function(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            
            # Backward pass and optimizer step
            total_loss.backward()
            optimizer.step()
            
            # Updates training bar
            train_bar.set_description(f"Train loss: {total_loss.item():.4f}")

        val_accuracy = validate(model, val_loader, device)
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            # Save the best model
            torch.save(model.state_dict(), 'transformer_qa_model_best.pth')
        
        print(f"Epoch {epoch}, validation accuracy: {val_accuracy:.2f}, best accuracy: {best_accuracy:.2f}")

    # Save the final model
    torch.save(model.state_dict(), 'transformer_qa_model.pth')

if __name__ == "__main__":
    train()