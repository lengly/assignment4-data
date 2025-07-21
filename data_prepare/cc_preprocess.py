from collections import defaultdict
import pathlib
import sys
import os
import glob
from pathlib import Path
import concurrent.futures
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH
import argparse

# Add the cs336_data directory to the path to import utils and deduplicate
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "cs336_data"))

from cs336_data.utils import (
    gopher_quality_filter,
    mask_emails,
    mask_phone_numbers,
    mask_ips,
    identify_language,
    classify_nsfw,
    classify_toxic_speech,
    classify_quality,
)
from cs336_data.deduplicate import (
    normalize_text,
    get_ngrams,
    compute_minhash_signature,
    lsh_candidate_pairs,
    compute_jaccard_similarity,
)
from filter import detect_internal_repetition
from fastwarc.warc import ArchiveIterator

def process_wet_file(wet_path: str) -> list[str]:
    """
    Process a single WET file and return filtered text documents.
    Reused logic from quality_classifier.py
    """
    documents = []
    count = 0
    
    try:
        with open(wet_path, "rb") as stream:
            for record in ArchiveIterator(stream):
                if record.headers.get('WARC-Type') == 'conversion':
                    count += 1
                    # if count % 100 == 0:
                    #     print(f"  Processed {count} records from {os.path.basename(wet_path)}")
                    
                    try:
                        # WET files contain plain text, not HTML
                        text_bytes = record.reader.read()
                        if len(text_bytes) > 1e8:  # Skip very large documents
                            continue
                        
                        # Decode text directly (no HTML extraction needed)
                        try:
                            text = text_bytes.decode("utf-8")
                        except UnicodeDecodeError:
                            text = text_bytes.decode("utf-8", errors="replace")
                        
                        text = text.replace('\n', ' ').strip()
                        
                        # Skip if text is too short
                        if len(text) < 100:
                            continue
                        
                        # Language filtering - only keep English and Chinese
                        lang, _ = identify_language(text)
                        if lang != "en" and lang != "zh":
                            continue
                        
                        # Mask sensitive information
                        text, _ = mask_emails(text)
                        text, _ = mask_phone_numbers(text)
                        text, _ = mask_ips(text)
                        
                        # Content filtering
                        nsfw_label, nsfw_score = classify_nsfw(text)
                        if nsfw_label == "nsfw":
                            continue
                        # print(f"nsfw_label: {nsfw_label}, nsfw_score: {nsfw_score}")
                        toxic_label, toxic_score = classify_toxic_speech(text)
                        if toxic_label == "toxic":
                            continue
                        # print(f"toxic_label: {toxic_label}, toxic_score: {toxic_score}")
                        
                        # print(text)
                        # Quality filtering using gopher_quality_filter
                        if not gopher_quality_filter(text):
                            # print(f"gopher_quality_filter: {gopher_quality_filter(text)}")
                            continue
                        
                        # Additional quality classification
                        quality_label, quality_score = classify_quality(text)
                        if quality_label != "pos":  # Only keep positive quality documents
                            continue
                        # print(f"quality_label: {quality_label}, quality_score: {quality_score}")
                        
                        # Internal repetition filtering
                        if detect_internal_repetition(text, ngram_size=3, repetition_threshold=0.3):
                            continue
                        
                        documents.append(text)
                            
                    except Exception as e:
                        print(f"    Error processing record in {wet_path}: {e}")
                        continue
                        
    except Exception as e:
        print(f"Error processing file {wet_path}: {e}")
        return []
    
    return documents


def deduplicate_documents(documents: list[str], num_perm: int = 128, threshold: float = 0.8) -> list[str]:
    """
    First deduplicate by hash, then use datasketch MinHash+LSH for near-duplicate removal.
    """
    if len(documents) <= 1:
        return documents

    print(f"  [Fast Hash] Number of documents before deduplication: {len(documents)}")
    # 1. Exact hash deduplication
    hash_dict = {}
    for doc in documents:
        h = hash(doc)
        if h not in hash_dict:
            hash_dict[h] = doc
    unique_docs = list(hash_dict.values())
    print(f"  [Fast Hash] Number of documents after deduplication: {len(unique_docs)}")

    # 2. MinHash + LSH near-duplicate removal
    print(f"  [MinHashLSH] Starting near-duplicate removal...")
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = []
    for i, doc in enumerate(unique_docs):
        m = MinHash(num_perm=num_perm)
        for token in doc.split():
            m.update(token.encode('utf8'))
        lsh.insert(str(i), m)
        minhashes.append(m)
    seen = set()
    deduped = []
    for i, m in enumerate(minhashes):
        if i in seen:
            continue
        dup = lsh.query(m)
        for idx in dup:
            seen.add(int(idx))
        deduped.append(unique_docs[i])
    print(f"  [MinHashLSH] Number of documents after near-duplicate removal: {len(deduped)}")
    return deduped


def process_single_wet_file(wet_filepath: str, output_dir: str, batch_size: int = 10000) -> tuple[str, int, list[str]]:
    """
    Process a single WET file and save documents in batches.
    Returns the filename, number of documents processed, and the documents list.
    """
    try:
        print(f"Processing file: {os.path.basename(wet_filepath)}")
        documents = process_wet_file(wet_filepath)
        
        if not documents:
            return os.path.basename(wet_filepath), 0, []
        
        # Deduplicate documents within this file
        deduplicated_docs = deduplicate_documents(documents)
        
        if not deduplicated_docs:
            return os.path.basename(wet_filepath), 0, []
        
        # Save documents in batches
        base_filename = str(pathlib.Path(wet_filepath).stem)
        batch_num = 0
        
        for i in range(0, len(deduplicated_docs), batch_size):
            batch_docs = deduplicated_docs[i:i + batch_size]
            batch_filename = f"{base_filename}_batch_{batch_num:03d}.txt"
            batch_path = os.path.join(output_dir, batch_filename)
            
            with open(batch_path, "w", encoding="utf-8") as f:
                for doc in batch_docs:
                    f.write(doc)
                    f.write('<|endoftext|>')
                    f.write("\n")
            
            print(f"  Saved batch {batch_num} with {len(batch_docs)} documents to {batch_filename}")
            batch_num += 1
        
        return os.path.basename(wet_filepath), len(deduplicated_docs), deduplicated_docs
        
    except Exception as e:
        print(f"Error processing file {wet_filepath}: {e}")
        return os.path.basename(wet_filepath), 0, []


def save_documents_in_batches(documents: list[str], output_dir: str, base_filename: str, batch_size: int = 10000):
    """
    Save documents in batches with the given base filename.
    """
    batch_num = 0
    
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_filename = f"{base_filename}_batch_{batch_num:03d}.txt"
        batch_path = os.path.join(output_dir, batch_filename)
        
        with open(batch_path, "w", encoding="utf-8") as f:
            for doc in batch_docs:
                f.write(doc)
                f.write('<|endoftext|>')
                f.write("\n")
        
        print(f"  Saved batch {batch_num} with {len(batch_docs)} documents to {batch_filename}")
        batch_num += 1


def read_all_txt_documents(input_dir: str) -> list[str]:
    """
    Read all .txt files in the input_dir and return a list of documents.
    Each document is separated by <|endoftext|> per line.
    """
    documents = []
    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    print(f"Found {len(txt_files)} txt files in {input_dir}")
    for txt_file in tqdm(txt_files, desc="Reading txt files"):
        file_path = os.path.join(input_dir, txt_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = line.strip()
                if doc.endswith('<|endoftext|>'):
                    doc = doc[:-len('<|endoftext|>')].strip()
                if doc:
                    documents.append(doc)
    print(f"Total documents read from txt files: {len(documents)}")
    return documents


def main():
    """Main function to process WET files with parallel processing and global deduplication"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dedup-only', action='store_true', help='Only perform global deduplication from /workspace/CommonCrawl/text')
    args = parser.parse_args()

    wet_dir = "/workspace/CommonCrawl/wet"
    output_dir = "/workspace/CommonCrawl/text"
    num_files_to_process = 5000  # Process more files in parallel
    batch_size = 10000  # Save every 10,000 documents as a separate file

    os.makedirs(output_dir, exist_ok=True)

    if args.dedup_only:
        # Only perform global deduplication from /workspace/CommonCrawl/text
        print("[MODE] Only performing global deduplication from /workspace/CommonCrawl/text ...")
        all_documents = read_all_txt_documents(output_dir)
        print(f"\nPerforming global deduplication on {len(all_documents)} documents...")
        globally_deduplicated_docs = deduplicate_documents(all_documents)
        print(f"After global deduplication: {len(globally_deduplicated_docs)} documents")
        print(f"Removed {len(all_documents) - len(globally_deduplicated_docs)} duplicate documents across all files")
        print(f"\nSaving globally deduplicated documents in batches...")
        save_documents_in_batches(
            globally_deduplicated_docs,
            output_dir,
            "globally_deduplicated",
            batch_size
        )
        print("\nProcessing complete!")
        print(f"  Total documents before global deduplication: {len(all_documents)}")
        print(f"  Total documents after global deduplication: {len(globally_deduplicated_docs)}")
        print(f"  Duplicate documents removed: {len(all_documents) - len(globally_deduplicated_docs)}")
        print(f"  Output directory: {output_dir}")
        return

    # Find all WET files
    pattern = os.path.join(wet_dir, "*.warc.wet.gz")
    wet_files = glob.glob(pattern)
    
    if not wet_files:
        print(f"No WET files found in {wet_dir}")
        return
    
    print(f"Found {len(wet_files)} WET files in {wet_dir}")
    print(f"Processing first {num_files_to_process} files in parallel...")
    
    # Set up parallel processing
    num_cpus = len(os.sched_getaffinity(0))
    max_workers = min(num_cpus, num_files_to_process)
    print(f"Using {max_workers} workers for parallel processing")
    
    # Process files in parallel
    all_results = []
    all_documents = []
    total_documents_before_global_dedup = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        futures = []
        for wet_file in wet_files[:num_files_to_process]:
            future = executor.submit(
                process_single_wet_file,
                wet_file,
                output_dir,
                batch_size
            )
            futures.append(future)
        
        # Process completed futures with progress bar
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing WET files"
        ):
            try:
                filename, doc_count, documents = future.result()
                all_results.append((filename, doc_count))
                all_documents.extend(documents)
                total_documents_before_global_dedup += doc_count
                print(f"Completed: {filename} - {doc_count} documents")
            except Exception as e:
                print(f"Error in future: {e}")
    
    print(f"\nAll files processed. Total documents before global deduplication: {total_documents_before_global_dedup}")
    
    # Perform global deduplication on all documents
    print(f"\nPerforming global deduplication on all {len(all_documents)} documents...")
    globally_deduplicated_docs = deduplicate_documents(all_documents)
    
    print(f"After global deduplication: {len(globally_deduplicated_docs)} documents")
    print(f"Removed {len(all_documents) - len(globally_deduplicated_docs)} duplicate documents across all files")
    
    # Save globally deduplicated documents in batches
    print(f"\nSaving globally deduplicated documents in batches...")
    save_documents_in_batches(
        globally_deduplicated_docs, 
        output_dir, 
        "globally_deduplicated", 
        batch_size
    )
    
    # Print final statistics
    print(f"\nProcessing complete!")
    print(f"Final statistics:")
    print(f"  Files processed: {len(all_results)}")
    print(f"  Total documents before global deduplication: {total_documents_before_global_dedup}")
    print(f"  Total documents after global deduplication: {len(globally_deduplicated_docs)}")
    print(f"  Duplicate documents removed: {total_documents_before_global_dedup - len(globally_deduplicated_docs)}")
    print(f"  Output directory: {output_dir}")
    
    # Print individual file results
    print(f"\nIndividual file results:")
    for filename, doc_count in all_results:
        print(f"  {filename}: {doc_count} documents")


if __name__ == "__main__":
    main()
