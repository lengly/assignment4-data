import os
from datasets import load_dataset
import tiktoken
import numpy as np

# 配置
OUTPUT_DIR = "/workspace/dataset/paloma_c4_100_domains"
TARGET_FILE_SIZE = 1 * 1024 * 1024 * 1024  # 1GB
os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer = tiktoken.get_encoding("gpt2")
END_OF_TEXT = "<|endoftext|>"

def main():
    print("Loading allenai/paloma c4_100_domains/val ...")
    ds = load_dataset("allenai/paloma", data_dir="c4_100_domains", split="validation", streaming=True)
    file_idx = 0
    sample_count = 0
    text_buffer = []
    token_buffer = []
    token_size = 0
    def get_text_path(idx):
        return os.path.join(OUTPUT_DIR, f"paloma_c4_100_domains_validation_{idx:03d}.txt")
    def get_token_path(idx):
        return os.path.join(OUTPUT_DIR, f"paloma_c4_100_domains_validation_{idx:03d}_tokens.npy")
    def flush_buffers(idx, text_buffer, token_buffer):
        with open(get_text_path(idx), "w") as f:
            for line in text_buffer:
                f.write(line)
        np.save(get_token_path(idx), np.array(token_buffer, dtype=np.uint16))
    for sample in ds:
        content = sample.get("text", "")
        if not content:
            continue
        content_with_eot = content + END_OF_TEXT + "\n"
        tokens = tokenizer.encode(content_with_eot, allowed_special={'<|endoftext|>'})
        if len(tokens) == 0:
            continue
        text_buffer.append(content_with_eot)
        token_buffer.extend(tokens)
        token_size += len(tokens) * 2
        sample_count += 1
        if sample_count % 1000 == 0:
            print(f"{sample_count} samples so far...")
        if token_size >= TARGET_FILE_SIZE:
            flush_buffers(file_idx, text_buffer, token_buffer)
            print(f"Flushed file {file_idx:03d}: {token_size/1e9:.2f}GB tokens.")
            file_idx += 1
            text_buffer = []
            token_buffer = []
            token_size = 0
    if text_buffer or token_buffer:
        flush_buffers(file_idx, text_buffer, token_buffer)
        print(f"Flushed final file {file_idx:03d}: {token_size/1e9:.2f}GB tokens.")
    print(f"Finished: {sample_count} samples.")

if __name__ == "__main__":
    main() 