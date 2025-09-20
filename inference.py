import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# ---
# This script is a standalone version for running inference on a trained quantaTinyLLM model.
# It loads a self-contained checkpoint file to generate text.
# ---

# --- Model Definition ---
# These classes must match the architecture of the trained model.

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head, block_size, dropout, device):
        super().__init__()
        self.device = device
        self.sa = nn.MultiheadAttention(n_embd, n_head, bias=False, device=device, batch_first=True)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.ln1(x)
        y, _ = self.sa(y, y, y, attn_mask=None, need_weights=False, is_causal=True)
        x = x + y
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout, device):
        super().__init__()
        self.device = device
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout, device) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, block_size, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- Main Execution ---

if __name__ == "__main__":
    # --- Configuration ---
    CHECKPOINT_FILE = 'gpt_checkpoint.pth' # Use the correct checkpoint file
    MAX_NEW_TOKENS = 200 # Max tokens for a faster, more interactive response

    # --- Prerequisite Checks ---
    if not os.path.exists(CHECKPOINT_FILE):
        print(f"Error: Checkpoint file '{CHECKPOINT_FILE}' not found.")
        print("Please run the training notebook or ensure the checkpoint is correctly named.")
        exit()

    # --- Model Loading ---
    print(f"Loading checkpoint '{CHECKPOINT_FILE}'...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=device)

    # Rebuild the tokenizer from the checkpoint
    itos = checkpoint['itos']
    stoi = { ch: i for i, ch in itos.items() }
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: ''.join([itos.get(i, '') for i in l])

    # Instantiate the model with hyperparameters from the checkpoint
    model = BigramLanguageModel(
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

    # --- Interactive Chat Loop ---
    while True:
        prompt = input("> ")
        if prompt.lower() in ['exit', 'quit']:
            break

        # Encode the prompt and create the starting context tensor
        context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

        # Generate text
        print("...generating...")
        generated_tokens = m.generate(context, block_size=checkpoint['block_size'], max_new_tokens=MAX_NEW_TOKENS)[0].tolist()

        # Print the generated text (the full sequence)
        print(decode(generated_tokens))
        print("---")
