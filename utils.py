import os
import glob
import torch
import time

def get_dir_path():
    """Returns the directory path of the current script."""
    return os.path.dirname(os.path.abspath(__file__))

def get_model_save_dir(config):
    """Returns the model save directory path specified in config."""
    model_save_dir = os.path.join(get_dir_path(), config["model_folder"])
    os.makedirs(model_save_dir, exist_ok=True)
    return model_save_dir

def get_next_model_save_path(config):
    """Returns the next model save path based on the number of existing models."""
    model_save_dir = get_model_save_dir(config)
    model_base_name = config["model_base_name"]
    run_count = get_model_run_count(config)
    new_model_name = f"{model_base_name}_{run_count + 1}.pt"
    return os.path.join(model_save_dir, new_model_name)

def get_last_model_save_path(config):
    """Returns the last model save path based on the number of existing models."""
    model_save_dir = get_model_save_dir(config)
    model_base_name = config["model_base_name"]
    run_count = get_model_run_count(config)
    new_model_name = f"{model_base_name}_{run_count}.pt"
    return os.path.join(model_save_dir, new_model_name)
    
def get_model_run_count(config):
    """Returns the count of how many model runs have been saved."""
    model_save_dir = get_model_save_dir(config)
    model_files = glob.glob(os.path.join(model_save_dir, f"{config['model_base_name']}*.pt"))
    model_files = [path for path in model_files if "best" not in os.path.basename(path)]
    return len(model_files)

def get_tokenizer_save_dir(config):
    """Returns the tokenizer save directory path specified in config."""
    tokenizer_save_dir = os.path.join(get_dir_path(), config["tokenizer_folder"])
    os.makedirs(tokenizer_save_dir, exist_ok=True)
    return tokenizer_save_dir

def get_config(source_lang = "de", target_lang = "en", seq_len = 1150, vocab_size = 30_000, d_model = 512, batch_size = 8, num_epochs = 15, lr = 7e-5, model_folder = "model_saves", model_base_name = "transformer_model", log_dir = "log_dir", experiment_name = None, device = None):
    """Returns a configuration dictionary with default values and dynamic model run count."""
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:3")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    config = {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lr": lr,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "d_model": d_model,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "model_folder": model_folder,
        "tokenizer_folder": "tokenizers",
        "model_base_name": model_base_name,
        "device": device,
        "log_dir": log_dir
    }
    config["experiment_name"] = experiment_name if experiment_name is not None else f"{config['model_base_name']}_{get_model_run_count(config)}"
    return config

def update_progress_bar(pbar, counter, total):
    while counter.value < total:
        pbar.update(counter.value - pbar.n)
        time.sleep(0.1)  # update every 0.1 seconds
    pbar.close()

def get_encoded_item_length(item, tokenizer, counter, lock):
    length = len(tokenizer.encode(item).ids)
    with lock:
        counter.value += 1
    return length