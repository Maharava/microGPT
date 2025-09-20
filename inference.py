import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from model import TinyTransformerLM # Import the model from model.py
from tokenizer import CustomTokenizer # Import the custom tokenizer

# ---
# This script is a standalone version for running inference on a trained quantaTinyLLM model.
# It loads a self-contained checkpoint file to generate text.
# ---

# --- Main Execution ---

if __name__ == "__main__":
    # --- Configuration ---
    CHECKPOINT_DIR = "./checkpoints"
    CHECKPOINT_NAME = 'gpt_checkpoint.pth' # Use the correct checkpoint file
    MAX_NEW_TOKENS = 200 # Max tokens for a faster, more interactive response
    TEMPERATURE = 0.8    # Sampling temperature
    TOP_K = None         # Top-k sampling (set to None for no top-k)
    TOP_P = None         # Top-p (nucleus) sampling (set to None for no top-p)

    # --- Prerequisite Checks ---
    checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' not found.")
        print("Please run the training script or ensure the checkpoint is correctly named and located.")
        exit()

    # --- Model Loading ---
    print(f"Loading checkpoint '{checkpoint_path}'...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load the tokenizer
    tokenizer = CustomTokenizer()
    tokenizer_load_path = os.path.join(CHECKPOINT_DIR, "tokenizer_data")
    tokenizer.load(load_path=tokenizer_load_path)
    encode = tokenizer.encode
    decode = tokenizer.decode
    eot_token_id = tokenizer.tokenizer.token_to_id("<EOT>")

    # Instantiate the model with hyperparameters from the checkpoint
    model = TinyTransformerLM(
        vocab_size=checkpoint['vocab_size'],
        n_embd=checkpoint['n_embd'],
        n_head=checkpoint['n_head'],
        n_layer=checkpoint['n_layer'],
        block_size=checkpoint['block_size'],
        dropout=checkpoint['dropout'],
        device=device,
    )
    m = model.to(device)
    m.load_state_dict(checkpoint['model_state_dict'])
    m.eval()
    print("Model loaded successfully.")
    print("---")
    print("Enter a prompt to start generating text. Type 'exit' or 'quit' to end.")
    print("---")

    @torch.no_grad()
    def generate_with_sampling(model, idx, block_size, max_new_tokens, temperature=1.0, top_k=None, top_p=None, stop_token_id=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] # take the last time step

            # Apply temperature
            logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1) # apply softmax to get probabilities
            idx_next = torch.multinomial(probs, num_samples=1) # sample from the distribution
            
            # Stop if EOT token is generated
            if stop_token_id is not None and idx_next.item() == stop_token_id:
                break

            idx = torch.cat((idx, idx_next), dim=1) # append sampled index to the running sequence
        return idx

    # --- Interactive Chat Loop ---
    print("Start chatting with the model. Use <USR> and <ASSIST> for role-playing. Type 'exit' or 'quit' to end.")
    while True:
        user_input = input("> ")
        if user_input.lower() in ['exit', 'quit']:
            break

        # Format the prompt with the user tag
        formatted_prompt = f"{tokenizer.get_special_tokens[0]}{user_input}{tokenizer.get_special_tokens[2]}{tokenizer.get_special_tokens[1]}"
        context = torch.tensor(encode(formatted_prompt), dtype=torch.long, device=device).unsqueeze(0)

        # Generate text
        print("...generating...")
        generated_tokens = generate_with_sampling(
            m, context, checkpoint['block_size'], MAX_NEW_TOKENS,
            temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P, stop_token_id=eot_token_id
        )[0].tolist()

        # Decode and print the generated text
        # Find the first occurrence of EOT and truncate
        try:
            eot_index = generated_tokens.index(eot_token_id)
            generated_tokens_without_eot = generated_tokens[:eot_index]
        except ValueError:
            generated_tokens_without_eot = generated_tokens # EOT not found, print full generated text

        # Remove the prompt part from the generated tokens for cleaner output
        prompt_token_ids = encode(formatted_prompt)
        if len(generated_tokens_without_eot) > len(prompt_token_ids):
            generated_text = decode(generated_tokens_without_eot[len(prompt_token_ids):])
        else:
            generated_text = decode(generated_tokens_without_eot)

        print(generated_text.strip())
        print("---")
