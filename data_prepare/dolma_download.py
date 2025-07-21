import os
import gzip
import json
import tiktoken
import numpy as np
from tqdm import tqdm
import random
import subprocess

# Path to the url list file (already cloned)
URL_LIST_PATH = "/workspace/dataset/dolma/git/urls/v1_7.txt"
# Directory where the .json.gz files are downloaded
DATA_DIR = "/workspace/dataset/dolma/raw"
# Output directory for processed data
OUTPUT_DIR = "/workspace/dataset/dolma/processed"
# Target number of tokens
TARGET_TOKENS = 5_000_000_000
# Target file size in bytes (1GB)
TARGET_FILE_SIZE = 1 * 1024 * 1024 * 1024
# Number of parallel downloads
PARALLEL_DOWNLOADS = 8

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_with_wget(max_files_per_type=3):
    """
    Download only the first N files per type (non-c4/cc) in parallel using wget.
    """
    with open(URL_LIST_PATH, "r") as f:
        urls = [line.strip() for line in f if line.strip()]
    urls = [u for u in urls if ("c4" not in u and "cc" not in u)]
    # Group urls by type
    type_to_urls = {}
    for url in urls:
        parts = url.split('/')
        for i, part in enumerate(parts):
            if part.endswith(".json.gz"):
                type_name = parts[-2]
                break
        else:
            continue
        type_to_urls.setdefault(type_name, []).append(url)
    # For each type, keep only the first N urls
    selected_urls = []
    for url_list in type_to_urls.values():
        url_list.sort()  # ensure order
        selected_urls.extend(url_list[:max_files_per_type])
    print(f"Total files to download: {len(selected_urls)}")
    url_tmp = "dolma_download_urls.txt"
    with open(url_tmp, "w") as f:
        for u in selected_urls:
            f.write(u + "\n")
    print(f"Starting parallel wget with {PARALLEL_DOWNLOADS} workers...")
    cmd = [
        "xargs", "-n", "1", "-P", str(PARALLEL_DOWNLOADS),
        "wget", "-q", "-P", DATA_DIR, "-nc"
    ]
    with open(url_tmp, "r") as f:
        subprocess.run(cmd, stdin=f, check=True)
    print("All downloads finished.")
    os.remove(url_tmp)

def get_data_files():
    """
    Read the url list, filter out c4 and cc, and return local file paths grouped by type.
    """
    with open(URL_LIST_PATH, "r") as f:
        urls = [line.strip() for line in f if line.strip()]
    # Filter out c4 and cc
    urls = [u for u in urls if ("c4" not in u and "cc" not in u)]
    # Group by type (e.g., books, github, etc.)
    type_to_files = {}
    for url in urls:
        # Example: .../books/books-0000.json.gz
        parts = url.split('/')
        for i, part in enumerate(parts):
            if part.endswith(".json.gz"):
                type_name = parts[-2]  # e.g., books
                break
        else:
            continue
        local_path = os.path.join(DATA_DIR, part)
        if not os.path.exists(local_path):
            continue  # skip missing files
        type_to_files.setdefault(type_name, []).append(local_path)
    # Shuffle file lists for each type
    for v in type_to_files.values():
        random.shuffle(v)
    return type_to_files

def sample_dolma(type_to_files, target_tokens=TARGET_TOKENS):
    """
    Evenly sample text from all types, tokenize, and save in 1GB chunks.
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    END_OF_TEXT = "<|endoftext|>"
    types = list(type_to_files.keys())
    num_types = len(types)
    file_idx = 0
    total_tokens = 0
    sample_count = 0
    text_buffer = []
    token_buffer = []
    token_size = 0
    file_pointers = {t: iter(type_to_files[t]) for t in types}
    file_handles = {t: None for t in types}
    json_iterators = {t: None for t in types}
    def get_text_path(idx):
        return os.path.join(OUTPUT_DIR, f"dolma_{idx:03d}.txt")
    def get_token_path(idx):
        return os.path.join(OUTPUT_DIR, f"dolma_{idx:03d}_tokens.npy")
    def flush_buffers(idx, text_buffer, token_buffer):
        with open(get_text_path(idx), "w") as f:
            for line in text_buffer:
                f.write(line)
        np.save(get_token_path(idx), np.array(token_buffer, dtype=np.uint16))
    # Prepare round-robin sampling
    exhausted_types = set()
    while total_tokens < target_tokens and len(exhausted_types) < num_types:
        for t in types:
            if t in exhausted_types:
                continue
            # Open next file if needed
            if json_iterators[t] is None:
                try:
                    next_file = next(file_pointers[t])
                except StopIteration:
                    exhausted_types.add(t)
                    continue
                f = gzip.open(next_file, 'rt', encoding='utf-8')
                file_handles[t] = f
                json_iterators[t] = (json.loads(line) for line in f)
            # Try to get next sample
            try:
                sample = next(json_iterators[t])
            except StopIteration:
                file_handles[t].close()
                file_handles[t] = None
                json_iterators[t] = None
                continue
            except Exception:
                continue  # skip broken lines
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
            total_tokens += len(tokens)
            sample_count += 1
            if sample_count % 1000 == 0:
                print(f"{sample_count} samples, {total_tokens} tokens so far...")
            if token_size >= TARGET_FILE_SIZE:
                flush_buffers(file_idx, text_buffer, token_buffer)
                print(f"Flushed file {file_idx:03d}: {token_size/1e9:.2f}GB tokens.")
                file_idx += 1
                text_buffer = []
                token_buffer = []
                token_size = 0
            if total_tokens >= target_tokens:
                break
    # Close all open files
    for f in file_handles.values():
        if f is not None:
            f.close()
    if text_buffer or token_buffer:
        flush_buffers(file_idx, text_buffer, token_buffer)
        print(f"Flushed final file {file_idx:03d}: {token_size/1e9:.2f}GB tokens.")
    print(f"Finished sampling: {total_tokens} tokens, {sample_count} samples.")

def main():
    print("=== Step 1: Download Dolma files with wget ===")
    download_with_wget(max_files_per_type=3)
    print("=== Step 2: Collect data files by type ===")
    type_to_files = get_data_files()
    print(f"Types found: {list(type_to_files.keys())}")
    print("=== Step 3: Evenly sample and tokenize ===")
    sample_dolma(type_to_files)

if __name__ == "__main__":
    main()
