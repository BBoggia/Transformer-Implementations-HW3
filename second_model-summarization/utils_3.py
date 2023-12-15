import torch
import glob
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from torch.nn.utils.rnn import pad_sequence

def get_dir_path():
    return os.path.dirname(os.path.realpath(__file__))

def collate_fn(batch, tokenizer):
    # Extract input_ids, attention_mask, and labels from batch
    input_ids = [torch.tensor(sample['input_ids'], dtype=torch.int64) for sample in batch]
    attention_mask = [torch.tensor(sample['attention_mask'], dtype=torch.int64) for sample in batch]
    labels = [torch.tensor(sample['labels'], dtype=torch.int64) for sample in batch]
    
    # Find longest sequence in batch for padding purposes
    max_len = max([len(ids) for ids in input_ids])
    
    # Pad sequences
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.token_to_id('<PAD>'))
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100) 
    
    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'labels': labels_padded
    }

def get_next_model_save_path():
    model_save_dir = os.path.join(get_dir_path(), "model_saves")
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_paths = glob.glob(os.path.join(model_save_dir, "*.pth"))
    if len(model_save_paths) == 0:
        return os.path.join(model_save_dir, "transformer_qa_model_1.pth")
    else:
        model_save_paths.sort()
        model_save_paths = [path for path in model_save_paths if "transformer_qa_model_best.pth" not in path]
        last_model_save_path = model_save_paths[-1]
        last_model_save_path = os.path.basename(last_model_save_path)
        last_model_save_path = os.path.splitext(last_model_save_path)[0]
        last_model_save_path = last_model_save_path.split("_")[-1]
        last_model_save_path = int(last_model_save_path)
        return os.path.join(model_save_dir, f"transformer_qa_model_{last_model_save_path + 1}.pth")

def calculate_metrics(y_true, y_pred, average='binary'):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    return metrics

def evaluate(model, valid_loader, tokenizer, device):
    model.eval()
    all_true_labels = []
    all_pred_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in valid_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            # Generate output predictions using model
            predictions = model(input_ids, attention_mask)
            predictions = torch.argmax(predictions, dim=-1)

            all_true_labels.extend(labels.cpu().numpy().tolist())
            all_pred_labels.extend(predictions.cpu().numpy().tolist())

    # y_true and y_pred need to be flattened into lists of ints for accuracy, precision, etc.
    y_true = [label for sublist in all_true_labels for label in sublist if label != tokenizer.token_to_id('<PAD>')]
    y_pred = [pred for sublist in all_pred_labels for pred in sublist if pred != tokenizer.token_to_id('<PAD>')]

    metrics = calculate_metrics(y_true, y_pred, average='micro')

    return metrics

def create_masks(input_seq, target_seq, tokenizer):
    enc_pad_token = tokenizer.token_to_id('<PAD>')
    tgt_pad_token = tokenizer.token_to_id('<PAD>')

    enc_mask = (input_seq != enc_pad_token).unsqueeze(1).unsqueeze(2)
    
    tgt_mask = (target_seq != tgt_pad_token).unsqueeze(1).unsqueeze(3)
    size = target_seq.size(1)  # Get seq_len for target
    nopeak_mask = torch.ones(size, size, dtype=torch.bool).triu(1).to(input_seq.device)
    tgt_mask = tgt_mask & nopeak_mask

    return enc_mask, tgt_mask