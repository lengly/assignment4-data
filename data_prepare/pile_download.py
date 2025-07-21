import os
from datasets import load_dataset
import tiktoken
import numpy as np

# https://huggingface.co/datasets/monology/pile-uncopyrighted

# Target number of tokens (5B)
TARGET_TOKENS = 5_000_000_000
# Output directory for sampled data
OUTPUT_DIR = "/workspace/dataset/pile"
# Target file size in bytes (1GB)
TARGET_FILE_SIZE = 1 * 1024 * 1024 * 1024

# Create output directory if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use GPT-2 tokenizer for token counting and encoding
tokenizer = tiktoken.get_encoding("gpt2")
END_OF_TEXT = "<|endoftext|>"
END_OF_TEXT_TOKENS = tokenizer.encode(END_OF_TEXT, allowed_special={'<|endoftext|>'})

def sample_pile(target_tokens=TARGET_TOKENS):
    """
    Sample text from the Pile dataset until reaching the target token count.
    Save the sampled text to multiple text files and tokenized .npy files, each about 1GB.
    """
    print(f"Sampling from the Pile dataset...")
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    file_idx = 0
    total_tokens = 0
    sample_count = 0
    text_buffer = []
    token_buffer = []
    token_size = 0
    def get_text_path(idx):
        return os.path.join(OUTPUT_DIR, f"pile_{idx:03d}.txt")
    def get_token_path(idx):
        return os.path.join(OUTPUT_DIR, f"pile_{idx:03d}_tokens.npy")
    def flush_buffers(idx, text_buffer, token_buffer):
        # Write text
        with open(get_text_path(idx), "w") as f:
            for line in text_buffer:
                f.write(line)
        # Write tokens
        np.save(get_token_path(idx), np.array(token_buffer, dtype=np.uint16))
    for sample in ds:
        content = sample.get("text", "")
        if not content:
            continue
        # Append <|endoftext|> to each sample
        content_with_eot = content + END_OF_TEXT + "\n"
        tokens = tokenizer.encode(content_with_eot, allowed_special={'<|endoftext|>'})
        if len(tokens) == 0:
            continue
        text_buffer.append(content_with_eot)
        token_buffer.extend(tokens)
        token_size += len(tokens) * 2 # 2 bytes per token
        total_tokens += len(tokens)
        sample_count += 1
        if sample_count % 1000 == 0:
            print(f"{sample_count} samples, {total_tokens} tokens so far...")
        # If buffer exceeds 1GB, flush to disk
        if token_size >= TARGET_FILE_SIZE:
            flush_buffers(file_idx, text_buffer, token_buffer)
            print(f"Flushed file {file_idx:03d}: {token_size/1e9:.2f}GB tokens.")
            file_idx += 1
            text_buffer = []
            token_buffer = []
            token_size = 0
        if total_tokens >= target_tokens:
            print(f"Reached target: {total_tokens} tokens.")
            break
    # Flush any remaining data
    if text_buffer or token_buffer:
        flush_buffers(file_idx, text_buffer, token_buffer)
        print(f"Flushed final file {file_idx:03d}: {token_size/1e9:.2f}GB tokens.")
    return total_tokens, sample_count

def main():
    total_tokens, sample_count = sample_pile(TARGET_TOKENS)
    print(f"Finished sampling: {total_tokens} tokens, {sample_count} samples.")
    if total_tokens < TARGET_TOKENS:
        print("Warning: Total tokens less than target. Consider increasing batch size or checking dataset availability.")

if __name__ == "__main__":
    main()
