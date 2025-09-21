import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import random # Import random module
from model import TinyTransformerLM # Import the model from model.py
from convert_spelling import convert_text_to_au_english # Import the conversion function

from tokenizer import CustomTokenizer # Import the custom tokenizer
from torch.utils.data import IterableDataset, DataLoader

# === 1. CONFIGURATION ===
# This section defines all the configurable parameters for the training script,
# including dataset paths, model hyperparameters, and training process settings.

# --- Dataset Settings ---
# LOCAL_DATASET_PATH: Specifies the path to the local text file used for training and tokenizer creation.
#                     This file should contain the raw text data.
LOCAL_DATASET_PATH = "training_dataset.txt" 

# TOKENIZER_DIR: The directory where the trained tokenizer's vocabulary and merge files will be saved
#                and loaded from. This path is made absolute to ensure consistent location
#                regardless of the script's execution directory.
TOKENIZER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer_data") 

# LOCAL_DATASET_CHUNK_SIZE_CHARS: Defines the size of text chunks (in characters) read from the
#                                 local dataset at a time. This is crucial for handling large
#                                 datasets that might not fit entirely into memory, allowing
#                                 for efficient processing and tokenization.
LOCAL_DATASET_CHUNK_SIZE_CHARS = 1024*1024 # 1MB chunks for processing large local files

# CHECKPOINT_NAME: The base filename for saving model and optimizer states.
#                  Used for both regular and 'best' validation loss checkpoints.
CHECKPOINT_NAME = 'gpt_checkpoint.pth'

# CHECKPOINT_DIR: The dedicated directory where all training checkpoints (model weights,
#                 optimizer state, training progress) will be stored.
CHECKPOINT_DIR = "./checkpoints" 

# --- Model Hyperparameters ---
# These parameters define the architecture of the TinyTransformerLM.
# They are adjusted to target approximately 10 million parameters for a small-scale model.

# N_EMBD: The dimensionality of the token and positional embeddings. This also determines
#         the internal dimension of the transformer blocks.
N_EMBD = 256       

# N_HEAD: The number of attention heads in the MultiheadAttention mechanism.
#         N_EMBD must be divisible by N_HEAD.
N_HEAD = 8         

# N_LAYER: The number of transformer decoder blocks stacked in the model.
N_LAYER = 8        

# BLOCK_SIZE: The maximum context length (sequence length) the model can process.
#             This defines how many previous tokens the model considers when making a prediction.
BLOCK_SIZE = 256   

# DROPOUT: The dropout rate applied to various layers for regularization,
#          helping to prevent overfitting.
DROPOUT = 0.1      

# --- Training Hyperparameters ---
# These parameters control the training process itself.

# MAX_ITERS: The total number of training iterations (batches processed) to perform.
MAX_ITERS = 5000         

# LEARNING_RATE: The initial learning rate for the AdamW optimizer.
#                This value is dynamically adjusted by a learning rate scheduler.
LEARNING_RATE = 3e-4   

# BATCH_SIZE: The number of sequences processed in parallel during each training step.
#             Larger batch sizes can lead to faster training but require more memory.
BATCH_SIZE = 64        

# EVAL_INTERVAL: How often (in training iterations) the model's performance
#                is evaluated on the validation set and a checkpoint is saved.
EVAL_INTERVAL = 500    

# EVAL_ITERS: The number of batches to use for estimating the loss on
#             both the training and validation sets during evaluation.
EVAL_ITERS = 50        

# GRAD_CLIP: The maximum norm for gradient clipping. This helps to prevent
#            exploding gradients during training, improving stability.
GRAD_CLIP = 1.0        

# NUM_WORKERS: The number of subprocesses to use for data loading.
#              A higher number can speed up data loading but consumes more CPU resources.
#              Defaults to a safe value for various environments (e.g., Colab).
NUM_WORKERS = min(2, os.cpu_count() // 2) if os.cpu_count() else 0 

# --- Helper for Chunking Local Datasets ---
# This generator function reads a large text file in smaller, manageable chunks.
# This is crucial for processing datasets that are too large to fit into RAM entirely,
# preventing OutOfMemory errors and allowing for efficient, sequential processing.
def read_chunks(file_path, chunk_size_chars):
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            # Read a chunk of the specified size.
            chunk = f.read(chunk_size_chars)
            # If the chunk is empty, we've reached the end of the file.
            if not chunk:
                break
            yield chunk

# --- Custom Iterable Dataset for Chunked Local Data ---
# This custom PyTorch IterableDataset is designed to efficiently serve tokenized data
# from large local files, processed in chunks. It handles distributing these chunks
# among multiple data loading workers.
class ChunkedLocalDataset(IterableDataset):
    def __init__(self, tokenized_data_list, block_size, seed=None):
        # tokenized_data_list: A list of lists, where each inner list contains token IDs
        #                      representing a processed chunk of the original text file.
        self.tokenized_data_list = tokenized_data_list 
        # block_size: The maximum context length for sequences yielded by the dataset.
        #             Each yielded sample will be a sequence of block_size + 1 tokens.
        self.block_size = block_size
        self.seed = seed

    def __iter__(self):
        # This method is called once per worker process when using DataLoader with num_workers > 0.
        # It sets up the data stream for that specific worker.
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        # Initialize a worker-specific random number generator for reproducibility
        # and to ensure different workers get different shuffling patterns.
        current_seed = self.seed + worker_id if self.seed is not None else worker_id
        rng = random.Random(current_seed)

        # Worker Distribution Logic:
        # The tokenized_data_list (list of chunks) is divided among the workers.
        # This is a simple contiguous split. Each worker processes a distinct slice of the chunks.
        # Note: While the overall all_tokenized_chunks list is shuffled once, this simple sharding
        # does not guarantee perfect global shuffling of individual samples when num_workers > 1.
        total_chunks = len(self.tokenized_data_list)
        start_idx = (total_chunks // num_workers) * worker_id
        end_idx = (total_chunks // num_workers) * (worker_id + 1)
        # The last worker takes any remaining chunks to ensure all data is processed.
        if worker_id == num_workers - 1: 
            end_idx = total_chunks
        
        # Select the chunks assigned to the current worker.
        worker_chunks = self.tokenized_data_list[start_idx:end_idx]

        # Further shuffle the worker's assigned chunks using its specific RNG.
        # This adds another layer of randomness to the data order.
        rng.shuffle(worker_chunks)

        # Concatenate Worker's Chunks and Yield Sequences:
        # All token IDs from the worker's assigned chunks are combined into a single stream.
        full_worker_data = []
        for chunk_ids in worker_chunks:
            full_worker_data.extend(chunk_ids)
        
        # Implement a shuffling buffer for individual samples within the worker's stream.
        # This helps to break sequential patterns that might remain after chunk-level shuffling.
        # The buffer size is chosen to be a multiple of block_size, ensuring full sequences can be formed.
        SHUFFLE_BUFFER_SIZE = 10 * self.block_size # Example buffer size, can be tuned.
        shuffling_buffer = []
        
        # Buffer Management:
        # Accumulate token IDs into the shuffling buffer.
        for token_id in full_worker_data:
            shuffling_buffer.append(token_id)
            # If the buffer is full, randomly sample a sequence from it.
            while len(shuffling_buffer) >= SHUFFLE_BUFFER_SIZE + self.block_size + 1:
                # Randomly select a starting point for a sequence within the buffer.
                start_index = rng.randint(0, len(shuffling_buffer) - (self.block_size + 1))
                chunk = shuffling_buffer[start_index : start_index + self.block_size + 1]
                
                # Remove the sampled sequence from the buffer to avoid re-sampling it too soon.
                # This is a simple removal; more sophisticated methods might involve a FIFO buffer.
                del shuffling_buffer[start_index : start_index + self.block_size + 1]
                yield torch.tensor(chunk, dtype=torch.long)
        
        # After exhausting full_worker_data, yield any remaining sequences from the buffer.
        # These will not be fully shuffled, but ensures all data is processed.
        buffer = []
        for token_id in shuffling_buffer:
            buffer.append(token_id)
            while len(buffer) >= self.block_size + 1:
                chunk = buffer[:self.block_size + 1]
                buffer = buffer[self.block_size + 1:]
                yield torch.tensor(chunk, dtype=torch.long)


# --- Setup ---
torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_AMP = (device == 'cuda') # Gate AMP on CUDA



# --- Data Loading and Preprocessing ---
# This function orchestrates the loading, tokenization, and preparation of the dataset
# for training and validation. It handles tokenizer management and data chunking.
def load_and_preprocess_data():
    print("\n[Data Preprocessing] Starting data loading and preprocessing...")
    
    # --- Tokenizer Training/Loading ---
    # Initializes the custom tokenizer and either trains a new one or loads an existing one.
    # The tokenizer is crucial for converting raw text into numerical IDs that the model can process.
    tokenizer = CustomTokenizer()
    tokenizer_save_path = TOKENIZER_DIR
    # Check if tokenizer files (vocab.json and merges.txt) already exist.
    if not os.path.exists(tokenizer_save_path) or not os.path.exists(os.path.join(tokenizer_save_path, "vocab.json")):
        print(f"[Tokenizer] Tokenizer files not found at {tokenizer_save_path}. Training a new tokenizer...")
        # Ensure the local dataset exists before attempting to train the tokenizer.
        if not os.path.exists(LOCAL_DATASET_PATH):
            print(f"[Error] Local dataset file '{LOCAL_DATASET_PATH}' not found for tokenizer training.")
            raise FileNotFoundError(f"Local dataset file '{LOCAL_DATASET_PATH}' not found for tokenizer training.")
        # Train the tokenizer on the local dataset. The tokenizer learns the vocabulary and BPE merges.
        tokenizer.train(files=[LOCAL_DATASET_PATH], save_path=tokenizer_save_path)
        print(f"[Tokenizer] New tokenizer trained and saved to {tokenizer_save_path}.")
    else:
        print(f"[Tokenizer] Loading existing tokenizer from {tokenizer_save_path}...")
        # Load the pre-trained tokenizer from the specified path.
        tokenizer.load(load_path=tokenizer_save_path)
        print("[Tokenizer] Tokenizer loaded successfully.")

    # Retrieve the vocabulary size from the loaded/trained tokenizer.
    vocab_size = tokenizer.get_vocab_size()
    print(f"[Tokenizer] Vocabulary size: {vocab_size}")
    
    # --- Data Loading for Model Training (Chunked Local) ---
    # This section processes the local dataset by reading it in chunks, applying spelling conversion,
    # tokenizing each chunk, and storing the tokenized chunks.
    print(f"\n[Dataset] Processing local dataset from '{LOCAL_DATASET_PATH}' with chunking (chunk size: {LOCAL_DATASET_CHUNK_SIZE_CHARS} chars)...")
    
    all_tokenized_chunks = []
    try:
        # Iterate through chunks of the local dataset.
        for i, chunk_text in enumerate(read_chunks(LOCAL_DATASET_PATH, LOCAL_DATASET_CHUNK_SIZE_CHARS)):
            if i % 100 == 0: 
                print(f"  [Dataset] Processing chunk {i}...")
            # Apply Australian English spelling conversion to the text chunk.
            processed_text = convert_text_to_au_english(chunk_text)
            # Encode the processed text chunk into token IDs and add to the list.
            all_tokenized_chunks.append(tokenizer.encode(processed_text))
        print(f"[Dataset] Successfully processed {len(all_tokenized_chunks)} chunks from {LOCAL_DATASET_PATH}. Total tokens will be determined by concatenation.")

    # Shuffle all tokenized chunks to ensure better global mixing before splitting.
    # This helps prevent any ordering biases from the original file affecting train/val splits.
    random.seed(1337) # Use a fixed seed for reproducibility of the shuffle.
    random.shuffle(all_tokenized_chunks)
    print("[Dataset] All tokenized chunks shuffled for better data distribution.")
    except FileNotFoundError:
        print(f"[Error] Local dataset file '{LOCAL_DATASET_PATH}' not found for model training.")
        raise FileNotFoundError(f"Local dataset file '{LOCAL_DATASET_PATH}' not found for model training.")
    except Exception as e:
        print(f"[Error] An error occurred while reading or processing local file chunks for model training: {e}. Exiting.")
        raise RuntimeError(f"Error during local file chunk processing: {e}")

    # --- Data Splitting and DataLoader Creation ---
    # The tokenized chunks are split into training and validation sets.
    # ChunkedLocalDataset and DataLoader are then used to efficiently feed data to the model.
    print("\n[Dataset] Splitting tokenized data into training and validation sets (90/10 split)...")
    n = int(0.9 * len(all_tokenized_chunks))
    # Create IterableDatasets for training and validation. These datasets will yield sequences of BLOCK_SIZE + 1 tokens.
    train_dataset = ChunkedLocalDataset(all_tokenized_chunks[:n], BLOCK_SIZE, seed=1337)
    val_dataset = ChunkedLocalDataset(all_tokenized_chunks[n:], BLOCK_SIZE, seed=1338)
    print(f"[Dataset] Training dataset created with {n} chunks. Validation dataset created with {len(all_tokenized_chunks) - n} chunks.")

    # Create DataLoaders for efficient batching and parallel data loading.
    # shuffle=True for IterableDataset shuffles the order of workers, not individual samples within a worker's stream.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    print(f"[DataLoader] Training DataLoader created with batch_size={BATCH_SIZE}, num_workers={NUM_WORKERS}.")
    print(f"[DataLoader] Validation DataLoader created with batch_size={BATCH_SIZE}, num_workers={NUM_WORKERS}.")

    print("[Data Preprocessing] Data loading and preprocessing complete.")
    return vocab_size, tokenizer, train_loader, val_loader

vocab_size, tokenizer_instance, train_loader, val_loader = load_and_preprocess_data()

# --- Training Class ---
class Trainer:
    # The Trainer class encapsulates the entire training loop, including model initialization,
    # checkpoint management, data iteration, loss calculation, and optimization.
    def __init__(self, model, optimizer, train_loader, val_loader, vocab_size, tokenizer_instance):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer_instance
        self.start_iter = 0
        self.best_val_loss = float('inf') # Tracks the best validation loss achieved for saving the best model.

        # Ensure the checkpoint directory exists to store model states.
        os.makedirs(CHECKPOINT_DIR, exist_ok=True) 
        checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)

        # Attempt to load a previous checkpoint to resume training.
        try:
            print(f"\n[Checkpoint] Checking for existing checkpoint at '{checkpoint_path}'...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Critical check: Ensure that the hyperparameters of the loaded model match
            # the current configuration. Mismatched hyperparameters (e.g., different embedding
            # dimensions or number of layers) would lead to incompatible model states.
            if (
                N_EMBD != checkpoint['n_embd'] or 
                N_HEAD != checkpoint['n_head'] or 
                N_LAYER != checkpoint['n_layer'] or
                BLOCK_SIZE != checkpoint['block_size'] or
                DROPOUT != checkpoint['dropout'] or
                vocab_size != checkpoint['vocab_size'] 
            ):
                # If hyperparameters don't match, it's safer to start training from scratch.
                print("[Checkpoint] Hyperparameter or vocabulary size mismatch detected. Starting training from scratch.")
                raise ValueError("Hyperparameter or vocab_size mismatch. Starting from scratch.")

            # Load model and optimizer states, and resume training progress.
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_iter = checkpoint['iter']
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"[Checkpoint] Resuming training from iteration {self.start_iter} with best validation loss {self.best_val_loss:.4f}.")
        except FileNotFoundError:
            print("[Checkpoint] No checkpoint found. Starting training from scratch.")
        except Exception as e:
            # Catch any other exceptions during checkpoint loading (e.g., corrupted file).
            print(f"[Checkpoint] Checkpoint could not be loaded due to an error: {e}. Starting training from scratch.")

    def get_batch(self, split):
        # Retrieves a batch of data (input sequences x and target sequences y) for training or validation.
        # DataLoaders handle the actual batching and iteration over the dataset.
        if split == 'train':
            try:
                # Attempt to get the next batch from the training DataLoader.
                batch_data = next(self.train_iter)
            except (StopIteration, AttributeError):
                # If the iterator is exhausted or not yet initialized, re-initialize it.
                self.train_iter = iter(self.train_loader)
                batch_data = next(self.train_iter)
        else: # 'val'
            try:
                # Attempt to get the next batch from the validation DataLoader.
                batch_data = next(self.val_iter)
            except (StopIteration, AttributeError):
                # If the iterator is exhausted or not yet initialized, re-initialize it.
                self.val_iter = iter(self.val_loader)
                batch_data = next(self.val_iter)
        
        # The ChunkedLocalDataset yields tensors directly. However, if a different dataset type
        # (e.g., TensorDataset) were used, the DataLoader might wrap the tensor in a list/tuple.
        # This check ensures compatibility.
        if isinstance(batch_data, (list, tuple)):
            batch_data = batch_data[0] # Extract the tensor if it's wrapped.

        # Split the batch into input sequences (x) and target sequences (y).
        # x contains tokens from 0 to BLOCK_SIZE-1.
        # y contains tokens from 1 to BLOCK_SIZE (the next token for each x token).
        x, y = batch_data[:, :-1].to(device), batch_data[:, 1:].to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self):
        # Estimates the loss on the training and validation sets.
        # This function is decorated with @torch.no_grad() to disable gradient calculations,
        # as we are only interested in evaluating performance, not training.
        out = {}
        # Set the model to evaluation mode. This disables dropout and batch normalization (if present).
        self.model.eval()
        print(f"\n[Evaluation] Starting loss estimation over {EVAL_ITERS} batches...")
        for split in ['train', 'val']:
            losses = torch.zeros(EVAL_ITERS)
            # Create a fresh iterator for evaluation to avoid exhausting the main DataLoader iterators.
            eval_loader_iter = iter(self.train_loader) if split == 'train' else iter(self.val_loader)
            for k, batch in enumerate(eval_loader_iter):
                if k >= EVAL_ITERS: break # Limit evaluation to EVAL_ITERS batches.
                # Prepare batch for device (CPU/GPU).
                x, y = batch[:, :-1].to(device), batch[:, 1:].to(device)
                # Use Automatic Mixed Precision (AMP) if enabled for faster evaluation on GPU.
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    logits, loss = self.model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean()
            print(f"[Evaluation] {split.capitalize()} loss estimated: {out[split]:.4f}")
        # Set the model back to training mode.
        self.model.train()
        print("[Evaluation] Loss estimation complete.")
        return out

    def get_lr(self, it):
        # Implements a learning rate schedule with a warmup phase followed by cosine decay.
        # This helps stabilize training at the beginning and allows for fine-tuning later.
        
        # Warmup phase: Linearly increases learning rate from 0 to LEARNING_RATE over 200 steps.
        if it < 200:
            return LEARNING_RATE * it / 200
        
        # Cosine decay phase: After warmup, the learning rate decays following a cosine curve.
        # It decays from LEARNING_RATE down to a minimum of 1e-4.
        if it > MAX_ITERS:
            return 1e-4 # Minimum learning rate after MAX_ITERS.
        
        # Calculate the progress within the cosine decay phase.
        progress = (it - 200) / (MAX_ITERS - 200)
        # Apply cosine function to scale the learning rate.
        cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        # Scale the learning rate between the minimum and the initial LEARNING_RATE.
        return 1e-4 + (LEARNING_RATE - 1e-4) * cosine

    def run(self):
        # Initializes the gradient scaler for Automatic Mixed Precision (AMP) if enabled.
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
        start_time = time.time()
        
        # Initialize iterators for DataLoaders. These will be re-initialized when exhausted.
        self.train_iter = iter(self.train_loader)
        self.val_iter = iter(self.val_loader)

        print(f"\n[Training] Starting training loop for {MAX_ITERS} iterations...")
        for iter in range(self.start_iter, MAX_ITERS):
            # --- Learning Rate Schedule Update ---
            # Adjust the learning rate according to the defined schedule (warmup + cosine decay).
            lr = self.get_lr(iter)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # --- Evaluation and Checkpointing ---
            # Periodically evaluate the model's performance and save checkpoints.
            if iter > 0 and iter % EVAL_INTERVAL == 0:
                print(f"\n[Training] Iteration {iter}: Performing evaluation and checkpointing (current LR: {lr:.6f})...")
                debug_loss = self.estimate_loss()
                train_loss = debug_loss['train']
                val_loss = debug_loss['val']
                elapsed_time = time.time() - start_time
                
                # Calculate Estimated Time to Completion (ETC).
                etc_seconds = 0
                if (iter - self.start_iter + 1) > 0:
                    etc_seconds = (elapsed_time / (iter - self.start_iter + 1)) * (MAX_ITERS - iter)
                
                print(f"[Progress] Iter {iter}/{MAX_ITERS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Elapsed: {elapsed_time:.0f}s | ETC: {etc_seconds/60:.2f} min")
                
                # Prepare checkpoint dictionary with model state, optimizer state, and training metadata.
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'iter': iter,
                    'vocab_size': self.vocab_size,
                    'tokenizer_special_tokens': self.tokenizer.get_special_tokens(), # Store special tokens for inference
                    'n_embd': N_EMBD,
                    'n_head': N_HEAD,
                    'n_layer': N_LAYER,
                    'block_size': BLOCK_SIZE,
                    'dropout': DROPOUT,
                    'best_val_loss': self.best_val_loss # Save best val loss
                }
                current_checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)
                torch.save(checkpoint, current_checkpoint_path)
                print(f"[Checkpoint] Checkpoint saved to '{current_checkpoint_path}'.")

                # Save the best model based on validation loss.
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_{CHECKPOINT_NAME}")
                    torch.save(checkpoint, best_checkpoint_path)
                    print(f"[Checkpoint] New best validation loss: {self.best_val_loss:.4f}. Best model saved to '{best_checkpoint_path}'.")

            # --- Training Step ---
            # Get a batch of training data.
            xb, yb = self.get_batch('train')
            
            # Perform forward pass and loss calculation with AMP if enabled.
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits, loss = self.model(xb, yb)
            
            # Zero gradients, scale loss, perform backward pass, unscale gradients, clip gradients, and update optimizer.
            self.optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)  # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP) # Apply gradient clipping.
            scaler.step(self.optimizer)
            scaler.update()

        print(f"\n[Training] Training finished after {MAX_ITERS} iterations.")
        # Save a final checkpoint after training completion.
        final_checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iter': MAX_ITERS,
            'vocab_size': self.vocab_size,
            'tokenizer_special_tokens': self.tokenizer.get_special_tokens(),
            'n_embd': N_EMBD,
            'n_head': N_HEAD,
            'n_layer': N_LAYER,
            'block_size': BLOCK_SIZE,
            'dropout': DROPOUT,
            'best_val_loss': self.best_val_loss
        }
        final_checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)
        torch.save(final_checkpoint, final_checkpoint_path)
        print(f"[Checkpoint] Final checkpoint saved to '{final_checkpoint_path}'.")
