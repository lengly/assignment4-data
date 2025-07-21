import os
from datasets import load_dataset
import tiktoken
import numpy as np

# https://huggingface.co/datasets/bigcode/the-stack

# List of programming languages to sample from The Stack
target_languages = ["python", "javascript", "java", "c++", "c"]  # You can modify this list

# Target number of tokens (5B)
TARGET_TOKENS = 5_000_000_000
# Output directory for sampled code
OUTPUT_DIR = "/workspace/dataset/thestack"
# Target file size in bytes (1GB)
TARGET_FILE_SIZE = 1 * 1024 * 1024 * 1024

# Create output directory if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use GPT-2 tokenizer for token counting and encoding
tokenizer = tiktoken.get_encoding("gpt2")
END_OF_TEXT = "<|endoftext|>"
END_OF_TEXT_TOKENS = tokenizer.encode(END_OF_TEXT, allowed_special={'<|endoftext|>'})

def sample_language(lang, target_tokens_per_lang):
    """
    Sample code from The Stack for a specific language until reaching the target token count.
    Save the sampled code to multiple text files and tokenized .npy files, each about 1GB.
    """
    print(f"Sampling language: {lang}")
    # Load the dataset with the 'default' config and filter by language
    ds = load_dataset("bigcode/the-stack", data_dir="data/" + lang, split="train", streaming=True)
    file_idx = 0
    total_tokens = 0
    sample_count = 0
    text_buffer = []
    token_buffer = []
    text_size = 0
    token_size = 0
    def get_text_path(idx):
        return os.path.join(OUTPUT_DIR, f"the_stack_{lang}_{idx:03d}.txt")
    def get_token_path(idx):
        return os.path.join(OUTPUT_DIR, f"the_stack_{lang}_{idx:03d}_tokens.npy")
    def flush_buffers(idx, text_buffer, token_buffer):
        # Write text
        with open(get_text_path(idx), "w") as f:
            for line in text_buffer:
                f.write(line)
        # Write tokens
        np.save(get_token_path(idx), np.array(token_buffer, dtype=np.uint16))
    print(f"Sampling {lang}...")
    for sample in ds:
        content = sample.get("content", "")
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
            print(f"{lang}: {sample_count} samples, {total_tokens} tokens so far...")
        # If either buffer exceeds 1GB, flush to disk
        if token_size >= TARGET_FILE_SIZE:
            flush_buffers(file_idx, text_buffer, token_buffer)
            print(f"Flushed file {file_idx:03d} for {lang}: {token_size/1e9:.2f}GB tokens.")
            file_idx += 1
            text_buffer = []
            token_buffer = []
            token_size = 0
        if total_tokens >= target_tokens_per_lang:
            print(f"Reached target for {lang}: {total_tokens} tokens.")
            break
    # Flush any remaining data
    if text_buffer or token_buffer:
        flush_buffers(file_idx, text_buffer, token_buffer)
        print(f"Flushed final file {file_idx:03d} for {lang}: {token_size/1e9:.2f}GB tokens.")
    return total_tokens, sample_count

def main():
    """
    Main function to sample from multiple languages until reaching the total target tokens.
    """
    tokens_per_lang = TARGET_TOKENS // len(target_languages)
    total_tokens_all = 0
    for lang in target_languages:
        lang_tokens, lang_samples = sample_language(lang, tokens_per_lang)
        total_tokens_all += lang_tokens
        print(f"Finished {lang}: {lang_tokens} tokens, {lang_samples} samples.")
    print(f"Total tokens sampled: {total_tokens_all}")
    if total_tokens_all < TARGET_TOKENS:
        print("Warning: Total tokens less than target. Consider increasing batch size or adding more languages.")

if __name__ == "__main__":
    main()
