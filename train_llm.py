import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import random # Import random module
from model import TinyTransformerLM # Import the model from model.py
from convert_spelling import convert_text_to_au_english # Import the conversion function
from datasets import load_dataset # For Hugging Face datasets
from tokenizer import CustomTokenizer # Import the custom tokenizer
from torch.utils.data import IterableDataset, DataLoader

# === 1. CONFIGURATION ===
# --- Dataset Settings ---
# Set to 'local' to use a file from your drive, or 'huggingface' to stream from Hugging Face.
DATA_SOURCE_TYPE = 'local' # 'local' or 'huggingface'
LOCAL_DATASET_PATH = "training_dataset.txt" # Path for local dataset
HUGGINGFACE_DATASET_NAME = "tatsu-lab/alpaca" # Example: "tatsu-lab/alpaca", "wikitext"
HUGGINGFACE_DATASET_TEXT_FIELD = "text" # Field containing text in Hugging Face dataset
TOKENIZER_DIR = "./tokenizer_data" # Directory to save/load tokenizer data

HUGGINGFACE_TOKENIZER_SAMPLE_SIZE = 1000000 # Number of characters to sample for tokenizer training from HF dataset

CHECKPOINT_NAME = 'gpt_checkpoint.pth'
CHECKPOINT_DIR = "./checkpoints" # Dedicated directory for checkpoints

# --- Model Hyperparameters ---
# Adjusted to target approximately 10 million parameters
N_EMBD = 256       # The embedding dimension for each token.
N_HEAD = 8         # The number of attention heads.
N_LAYER = 8        # The number of transformer blocks (layers).
BLOCK_SIZE = 256   # The maximum context length for predictions.
DROPOUT = 0.1      # The dropout rate for regularization.

# --- Training Hyperparameters ---
MAX_ITERS = 5000         # Total training iterations.
LEARNING_RATE = 3e-4   # The learning rate for the optimizer.
BATCH_SIZE = 64        # How many sequences to process in parallel.
EVAL_INTERVAL = 500    # How often to evaluate the model and save a checkpoint.
EVAL_ITERS = 50        # Number of batches to use for loss estimation
GRAD_CLIP = 1.0        # Gradient clipping norm
NUM_WORKERS = min(2, os.cpu_count() // 2) if os.cpu_count() else 0 # Safer default for Colab

# --- Setup ---
torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_AMP = (device == 'cuda') # Gate AMP on CUDA

# --- Custom Iterable Dataset for Hugging Face Streaming ---
class StreamingTextDataset(IterableDataset):
    def __init__(self, dataset_name, text_field, tokenizer, block_size, split='train', au_convert=True, seed=None):
        self.dataset_name = dataset_name
        self.text_field = text_field
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.au_convert = au_convert
        self.seed = seed
        self.dataset = load_dataset(dataset_name, split=split, streaming=True)
        self.buffer = []

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        # Use a unique seed for each worker to ensure different data portions
        current_seed = self.seed + worker_id if self.seed is not None else worker_id
        rng = random.Random(current_seed)

        # Skip a deterministic number of initial samples based on worker_id and seed
        # This is a heuristic for streaming datasets without explicit sharding
        # The skip amount should be large enough to ensure workers get different data, but not too large to exhaust the stream quickly.
        skip_amount = (current_seed * 1000) % 10000 # Example: skip up to 10000 samples, based on seed
        worker_dataset = self.dataset.skip(skip_amount)
        
        for item in worker_dataset:
            if self.text_field in item:
                text = item[self.text_field]
                if self.au_convert:
                    text = convert_text_to_au_english(text)
                
                # Tokenize and extend buffer
                ids = self.tokenizer.encode(text)
                self.buffer.extend(ids)

                # Yield chunks of block_size
                while len(self.buffer) >= self.block_size + 1:
                    chunk = self.buffer[:self.block_size + 1]
                    self.buffer = self.buffer[self.block_size + 1:]
                    yield torch.tensor(chunk, dtype=torch.long)

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data():
    print(f"Loading data from {DATA_SOURCE_TYPE}...")
    
    # --- Tokenizer Training/Loading ---
    tokenizer = CustomTokenizer()
    tokenizer_save_path = os.path.join(CHECKPOINT_DIR, TOKENIZER_DIR)
    if not os.path.exists(tokenizer_save_path) or not os.path.exists(os.path.join(tokenizer_save_path, "vocab.json")):
        print("Tokenizer not found, training new one...")
        # For tokenizer training, we need a representative sample of text
        if DATA_SOURCE_TYPE == 'local':
            if not os.path.exists(LOCAL_DATASET_PATH):
                print(f"Error: Local dataset file '{LOCAL_DATASET_PATH}' not found for tokenizer training.")
                exit()
            tokenizer.train(files=[LOCAL_DATASET_PATH], save_path=tokenizer_save_path)
        elif DATA_SOURCE_TYPE == 'huggingface':
            print(f"Collecting {HUGGINGFACE_TOKENIZER_SAMPLE_SIZE} characters for tokenizer training from Hugging Face dataset...")
            hf_tokenizer_train_text = ""
            hf_dataset_sample = load_dataset(HUGGINGFACE_DATASET_NAME, split='train', streaming=True)
            current_length = 0
            for item in hf_dataset_sample:
                if HUGGINGFACE_DATASET_TEXT_FIELD in item:
                    processed_text = convert_text_to_au_english(item[HUGGINGFACE_DATASET_TEXT_FIELD])
                    hf_tokenizer_train_text += processed_text + "\n"
                    current_length += len(processed_text)
                    if current_length >= HUGGINGFACE_TOKENIZER_SAMPLE_SIZE:
                        break
            # Write to a temporary file for tokenizer training, then delete
            temp_tokenizer_train_file = "hf_tokenizer_train_data.txt"
            with open(temp_tokenizer_train_file, "w", encoding="utf-8") as f_temp:
                f_temp.write(hf_tokenizer_train_text)
            tokenizer.train(files=[temp_tokenizer_train_file], save_path=tokenizer_save_path)
            os.remove(temp_tokenizer_train_file) # Clean up temp file
    else:
        print("Loading existing tokenizer...")
        tokenizer.load(load_path=tokenizer_save_path)

    vocab_size = tokenizer.get_vocab_size
    
    # --- Data Loading for Model Training ---
    if DATA_SOURCE_TYPE == 'local':
        try:
            with open(LOCAL_DATASET_PATH, 'r', encoding='utf-8') as f:
                text = f.read()
            print(f"Successfully loaded local dataset from {LOCAL_DATASET_PATH} for model training.")
        except FileNotFoundError:
            print(f"Error: Local dataset file '{LOCAL_DATASET_PATH}' not found for model training.")
            exit()
        except Exception as e:
            print(f"An error occurred while reading the local file for model training: {e}")
            exit()
        
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        n = int(0.9*len(data))
        train_dataset = torch.utils.data.TensorDataset(data[:n])
        val_dataset = torch.utils.data.TensorDataset(data[n:])

    elif DATA_SOURCE_TYPE == 'huggingface':
        train_dataset = StreamingTextDataset(HUGGINGFACE_DATASET_NAME, HUGGINGFACE_DATASET_TEXT_FIELD, tokenizer, BLOCK_SIZE, split='train', au_convert=True, seed=1337)
        try:
            val_dataset = StreamingTextDataset(HUGGINGFACE_DATASET_NAME, HUGGINGFACE_DATASET_TEXT_FIELD, tokenizer, BLOCK_SIZE, split='validation', au_convert=True, seed=1338)
        except Exception as e:
            print(f"Warning: Validation split not found for {HUGGINGFACE_DATASET_NAME}. Falling back to using a portion of the training data for validation. Error: {e}")
            # Fallback: use a portion of the training data for validation, with a different seed
            val_dataset = StreamingTextDataset(HUGGINGFACE_DATASET_NAME, HUGGINGFACE_DATASET_TEXT_FIELD, tokenizer, BLOCK_SIZE, split='train', au_convert=True, seed=1338)

    else:
        print("Invalid DATA_SOURCE_TYPE. Must be 'local' or 'huggingface'.")
        exit()

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    return vocab_size, tokenizer, train_loader, val_loader

vocab_size, tokenizer_instance, train_loader, val_loader = load_and_preprocess_data()

# --- Training Class ---
class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader, vocab_size, tokenizer_instance):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer_instance
        self.start_iter = 0
        self.best_val_loss = float('inf') # For saving best checkpoint

        os.makedirs(CHECKPOINT_DIR, exist_ok=True) # Ensure checkpoint directory exists
        checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)

        try:
            print(f"Checking for checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # Check for hyperparameter mismatch before loading state_dict
            if (
                N_EMBD != checkpoint['n_embd'] or 
                N_HEAD != checkpoint['n_head'] or 
                N_LAYER != checkpoint['n_layer'] or
                BLOCK_SIZE != checkpoint['block_size'] or
                DROPOUT != checkpoint['dropout'] or
                vocab_size != checkpoint['vocab_size'] # Also check vocab_size
            ):
                raise ValueError("Hyperparameter or vocab_size mismatch. Starting from scratch.")

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_iter = checkpoint['iter']
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"Resuming training from iteration {self.start_iter} with best_val_loss {self.best_val_loss:.4f}")
        except FileNotFoundError:
            print("No checkpoint found. Starting training from scratch.")
        except Exception as e:
            print(f"Checkpoint could not be loaded. Starting from scratch. Error: {e}")

    def get_batch(self, split):
        # DataLoaders handle batching and iteration for us
        if split == 'train':
            try:
                batch_data = next(self.train_iter)
            except (StopIteration, AttributeError):
                self.train_iter = iter(self.train_loader)
                batch_data = next(self.train_iter)
        else: # 'val'
            try:
                batch_data = next(self.val_iter)
            except (StopIteration, AttributeError):
                self.val_iter = iter(self.val_loader)
                batch_data = next(self.val_iter)
        
        # For TensorDataset, batch_data is a list/tuple containing the tensor
        # For StreamingTextDataset, batch_data is already the tensor
        if isinstance(batch_data, (list, tuple)):
            batch_data = batch_data[0] # Extract the tensor

        x, y = batch_data[:, :-1].to(device), batch_data[:, 1:].to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(EVAL_ITERS)
            loader = self.train_loader if split == 'train' else self.val_loader
            for k, batch in enumerate(loader):
                if k >= EVAL_ITERS: break
                x, y = batch[:, :-1].to(device), batch[:, 1:].to(device)
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    logits, loss = self.model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def get_lr(self, it):
        # 200 warmup steps, then cosine decay
        if it < 200:
            return LEARNING_RATE * it / 200
        if it > MAX_ITERS:
            return 1e-4 # Minimum learning rate
        progress = (it - 200) / (MAX_ITERS - 200)
        cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        return 1e-4 + (LEARNING_RATE - 1e-4) * cosine

    def run(self):
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
        start_time = time.time()
        
        # Initialize iterators for DataLoaders
        self.train_iter = iter(self.train_loader)
        self.val_iter = iter(self.val_loader)

        for iter in range(self.start_iter, MAX_ITERS):
            # Learning rate schedule
            lr = self.get_lr(iter)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            if iter > 0 and iter % EVAL_INTERVAL == 0:
                debug_loss = self.estimate_loss()
                train_loss = debug_loss['train']
                val_loss = debug_loss['val']
                elapsed_time = time.time() - start_time
                etc_seconds = (elapsed_time / (iter - self.start_iter + 1)) * (MAX_ITERS - iter)
                print(f"[{elapsed_time:.0f}s] step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f} (ETC: {etc_seconds/60:.2f} min)")
                
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'iter': iter,
                    'vocab_size': self.vocab_size,
                    'tokenizer_special_tokens': self.tokenizer.get_special_tokens, # Store special tokens for inference
                    'n_embd': N_EMBD,
                    'n_head': N_HEAD,
                    'n_layer': N_LAYER,
                    'block_size': BLOCK_SIZE,
                    'dropout': DROPOUT,
                    'best_val_loss': self.best_val_loss # Save best val loss
                }
                torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME))

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_{CHECKPOINT_NAME}")
                    torch.save(checkpoint, best_checkpoint_path)
                    print(f"New best validation loss: {self.best_val_loss:.4f}. Saved best model to {best_checkpoint_path}")

            xb, yb = self.get_batch('train')
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP) # Gradient clipping
            scaler.step(self.optimizer)
            scaler.update()

        print(f"Training finished at iteration {MAX_ITERS}.")
        final_checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iter': MAX_ITERS,
            'vocab_size': self.vocab_size,
            'tokenizer_special_tokens': self.tokenizer.get_special_tokens,
            'n_embd': N_EMBD,
            'n_head': N_HEAD,
            'n_layer': N_LAYER,
            'block_size': BLOCK_SIZE,
            'dropout': DROPOUT,
            'best_val_loss': self.best_val_loss
        }
        torch.save(final_checkpoint, os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME))

# --- Execution ---
model = TinyTransformerLM(vocab_size, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, DROPOUT, device).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
trainer = Trainer(model, optimizer, train_loader, val_loader, vocab_size, tokenizer_instance)
trainer.run()
