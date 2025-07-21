#!/usr/bin/env python3
"""
Use GPT-2 tokenizer to tokenize files in CommonCrawl/dedup_text directory
Save as npy format and output total token count
"""

import os
import glob
import numpy as np
import tiktoken
import argparse
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def tokenize_file(file_path, tokenizer, max_length=None):
    """
    Tokenize a single file
    
    Args:
        file_path: Path to the file
        tokenizer: GPT-2 tokenizer
        max_length: Maximum token length per line, None means no limit
    
    Returns:
        tokens: List of tokens
        token_count: Number of tokens
    """
    tokens = []
    token_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                # Use GPT-2 tokenizer to tokenize the line
                line_tokens = tokenizer.encode(line, allowed_special={'<|endoftext|>'})
                
                if max_length and len(line_tokens) > max_length:
                    # Truncate if exceeds max length
                    line_tokens = line_tokens[:max_length]
                
                tokens.extend(line_tokens)
                token_count += len(line_tokens)
                
                # Output progress every 1000 lines
                if line_num % 1000 == 0:
                    logger.info(f"Processed {line_num} lines, current token count: {token_count}")
    
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return [], 0
    
    return tokens, token_count

def main():
    parser = argparse.ArgumentParser(description='Use GPT-2 tokenizer to tokenize CommonCrawl files')
    parser.add_argument('--input_dir', type=str, default='/workspace/dataset/CommonCrawl/dedup_text',
                       help='Input directory path')
    parser.add_argument('--output_dir', type=str, default='/workspace/dataset/CommonCrawl/token',
                       help='Output directory path')
    parser.add_argument('--max_length', type=int, default=None,
                       help='Maximum token length per line, None means no limit')
    parser.add_argument('--target_size_gb', type=float, default=1.0,
                       help='Target size in GB for each output file')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize GPT-2 tokenizer
    logger.info("Loading GPT-2 tokenizer...")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Get all txt files
    file_pattern = os.path.join(args.input_dir, "*.txt")
    files = glob.glob(file_pattern)
    files.sort()  # Sort by filename
    
    if not files:
        logger.error(f"No txt files found in directory {args.input_dir}")
        return
    
    logger.info(f"Found {len(files)} files")
    
    # Calculate target tokens per file (assuming 2 bytes per token for int16)
    target_size_bytes = args.target_size_gb * 1024 * 1024 * 1024
    target_tokens_per_file = int(target_size_bytes / 2)  # 2 bytes per int16 token
    logger.info(f"Target tokens per file: {target_tokens_per_file:,}")
    
    all_tokens = []
    total_token_count = 0
    current_file_tokens = []
    current_file_size = 0
    file_counter = 0
    
    # Process each file
    for file_path in tqdm(files, desc="Processing files"):
        logger.info(f"Processing file: {os.path.basename(file_path)}")
        
        # Tokenize the file
        tokens, token_count = tokenize_file(file_path, tokenizer, args.max_length)
        
        if tokens:
            total_token_count += token_count
            logger.info(f"File {os.path.basename(file_path)} completed, token count: {token_count}")
            
            # Add tokens to current file
            current_file_tokens.extend(tokens)
            current_file_size += len(tokens) * 2  # 2 bytes per token
            
            # Check if current file size exceeds target size
            if current_file_size >= target_size_bytes:
                # Save current file
                file_counter += 1
                output_filename = f"tokens_{file_counter:04d}.npy"
                output_path = os.path.join(args.output_dir, output_filename)
                np.save(output_path, np.array(current_file_tokens, dtype=np.uint16))
                
                logger.info(f"Saved file: {output_filename}, token count: {len(current_file_tokens):,}, "
                          f"size: {current_file_size / (1024*1024*1024):.2f} GB")
                
                # Reset for next file
                current_file_tokens = []
                current_file_size = 0
    
    # Save remaining tokens as the last file
    if current_file_tokens:
        file_counter += 1
        output_filename = f"tokens_{file_counter:04d}.npy"
        output_path = os.path.join(args.output_dir, output_filename)
        np.save(output_path, np.array(current_file_tokens, dtype=np.uint16))
        
        logger.info(f"Saved final file: {output_filename}, token count: {len(current_file_tokens):,}, "
                  f"size: {current_file_size / (1024*1024*1024):.2f} GB")
    
    # Save statistics
    stats = {
        'total_tokens': total_token_count,
        'total_files': len(files),
        'vocab_size': len(tokenizer),
        'output_files': file_counter,
        'target_size_gb': args.target_size_gb
    }
    
    stats_path = os.path.join(args.output_dir, 'tokenization_stats.txt')
    with open(stats_path, 'w') as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
    
    logger.info("=" * 50)
    logger.info("TOKENIZATION COMPLETED!")
    logger.info(f"Total token count: {total_token_count:,}")
    logger.info(f"Files processed: {len(files)}")
    logger.info(f"Vocabulary size: {tokenizer.vocab_size}")
    logger.info(f"Output files: {file_counter}")
    logger.info(f"Target size per file: {args.target_size_gb} GB")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
