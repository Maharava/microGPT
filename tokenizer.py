from tokenizers import ByteLevelBPETokenizer
import os

class CustomTokenizer:
    def __init__(self, vocab_size=50000, min_frequency=2, special_tokens=None):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens if special_tokens is not None else [
            "<USR>", "<ASSIST>", "<EOT>", "<PAD>"
        ]
        self.tokenizer = ByteLevelBPETokenizer()

    def train(self, files, output_dir="./tokenizer_data", save_path=None):
        if save_path is None:
            save_path = output_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        print(f"Training tokenizer on {len(files)} files...")
        self.tokenizer.train(
            files=files,
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens
        )
        self.tokenizer.save_model(save_path)
        print(f"Tokenizer trained and saved to {save_path}")

    def load(self, output_dir="./tokenizer_data", load_path=None):
        if load_path is None:
            load_path = output_dir
        vocab_path = os.path.join(load_path, "vocab.json")
        merges_path = os.path.join(load_path, "merges.txt")
        if not (os.path.exists(vocab_path) and os.path.exists(merges_path)):
            raise FileNotFoundError(f"Tokenizer files not found in {load_path}. Please train the tokenizer first.")
        
        self.tokenizer = ByteLevelBPETokenizer(
            vocab_file=vocab_path,
            merges_file=merges_path
        )
        print(f"Tokenizer loaded from {output_dir}")

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    @property
    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    @property
    def get_special_tokens(self):
        return self.special_tokens

    @property
    def get_pad_token_id(self):
        return self.tokenizer.token_to_id("<PAD>")

if __name__ == "__main__":
    # Example usage:
    # Create a dummy text file for training
    dummy_text = "Hello world! This is a test. This is another sentence. User: How are you? Assistant: I am fine. <EOT>"
    with open("dummy_text.txt", "w", encoding="utf-8") as f:
        f.write(dummy_text)

    # Train the tokenizer
    tokenizer_instance = CustomTokenizer()
    tokenizer_instance.train(files=["dummy_text.txt"], output_dir="./tokenizer_data")

    # Load the tokenizer
    loaded_tokenizer = CustomTokenizer()
    loaded_tokenizer.load(output_dir="./tokenizer_data")

    # Test encoding and decoding
    text_to_encode = "User: What is the capital of Australia? <EOT>"
    encoded_ids = loaded_tokenizer.encode(text_to_encode)
    decoded_text = loaded_tokenizer.decode(encoded_ids)

    print(f"Original text: {text_to_encode}")
    print(f"Encoded IDs: {encoded_ids}")
    print(f"Decoded text: {decoded_text}")
    print(f"Vocab size: {loaded_tokenizer.get_vocab_size}")
    print(f"Special tokens: {loaded_tokenizer.get_special_tokens}")
    print(f"PAD token ID: {loaded_tokenizer.get_pad_token_id}")

    # Clean up dummy file and directory
    os.remove("dummy_text.txt")
    # os.rmdir("./tokenizer_data") # This will fail if not empty, handle with shutil.rmtree if needed
    print("\nExample usage complete.")
