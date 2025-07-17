import os
import re
import random
import hashlib
from collections import defaultdict
from pathlib import Path
import unicodedata
from typing import List, Set, Tuple, Dict

def exact_line_deduplication(input_files: list[os.PathLike], output_directory: os.PathLike):
    hash_freq = defaultdict(int)
    for input_file in input_files:
        with open(input_file, "r") as f:
            for line in f:
                hash_freq[hash(line)] += 1
    for input_file in input_files:
        output_file = Path(output_directory) / Path(input_file).name
        open(output_file, "w").close()
        with open(input_file, "r") as f:
            for line in f:
                if hash_freq[hash(line)] == 1:
                    with open(output_file, "a") as fout:
                        fout.write(line)


def normalize_text(text: str) -> str:
    """
    Normalize text by lowercasing, removing punctuation, normalizing whitespaces,
    removing accents, and applying NFD unicode normalization.
    """
    # Apply NFD unicode normalization
    text = unicodedata.normalize('NFD', text)
    
    # Remove accents (combining characters)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation (keep only alphanumeric and whitespace)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Normalize whitespace (replace multiple spaces with single space)
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def get_ngrams(text: str, n: int) -> Set[str]:
    """
    Extract n-grams (word sequences of length n) from normalized text.
    """
    words = text.split()
    if len(words) < n:
        return set()
    
    ngrams = set()
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.add(ngram)
    
    return ngrams


def compute_minhash_signature(ngrams: Set[str], num_hashes: int) -> List[int]:
    """
    Compute minhash signature for a set of n-grams.
    """
    if not ngrams:
        return [2**32 - 1] * num_hashes  # Use max int instead of infinity
    
    # Create hash functions using different seeds
    hash_functions = []
    for i in range(num_hashes):
        def make_hash_function(seed):
            def hash_func(x):
                return hashlib.md5(f"{seed}:{x}".encode()).hexdigest()
            return hash_func
        hash_functions.append(make_hash_function(i))
    
    # Initialize signature with max int
    signature = [2**32 - 1] * num_hashes
    
    # For each n-gram, update the signature
    for ngram in ngrams:
        for i, hash_func in enumerate(hash_functions):
            hash_value = int(hash_func(ngram), 16)
            signature[i] = min(signature[i], hash_value)
    
    return signature


def lsh_candidate_pairs(signatures: Dict[int, List[int]], num_bands: int, num_hashes: int) -> Set[Tuple[int, int]]:
    """
    Use Locality Sensitive Hashing to find candidate duplicate pairs.
    """
    if num_hashes % num_bands != 0:
        raise ValueError("num_hashes must be evenly divisible by num_bands")
    
    rows_per_band = num_hashes // num_bands
    candidate_pairs = set()
    
    # For each band
    for band in range(num_bands):
        band_hashes = defaultdict(list)
        
        # For each document
        for doc_id, signature in signatures.items():
            # Extract the hash values for this band
            start_idx = band * rows_per_band
            end_idx = start_idx + rows_per_band
            band_signature = tuple(signature[start_idx:end_idx])
            
            # Hash the band signature to create buckets
            band_hash = hash(band_signature)
            band_hashes[band_hash].append(doc_id)
        
        # Add all pairs within each bucket as candidates
        for bucket in band_hashes.values():
            if len(bucket) > 1:
                for i in range(len(bucket)):
                    for j in range(i + 1, len(bucket)):
                        candidate_pairs.add((min(bucket[i], bucket[j]), max(bucket[i], bucket[j])))
    
    return candidate_pairs


def compute_jaccard_similarity(ngrams1: Set[str], ngrams2: Set[str]) -> float:
    """
    Compute Jaccard similarity between two sets of n-grams.
    """
    if not ngrams1 and not ngrams2:
        return 1.0  # Both empty sets are considered identical
    if not ngrams1 or not ngrams2:
        return 0.0  # One empty set means no similarity
    
    intersection = len(ngrams1.intersection(ngrams2))
    union = len(ngrams1.union(ngrams2))
    
    return intersection / union


def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike
):
    """
    Perform fuzzy document deduplication using minhash and LSH.
    
    Args:
        input_files: List of paths to input files
        num_hashes: Number of hashes to use for computing minhash signatures
        num_bands: Number of bands to use for LSH
        ngrams: N-gram length (in words) for computing minhash signatures
        jaccard_threshold: Threshold for Jaccard similarity to consider documents as duplicates
        output_directory: Directory to write deduplicated files
    """
    # Ensure output directory exists
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read and normalize all documents
    documents = []
    doc_ids = []
    
    for input_file in input_files:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
            normalized_content = normalize_text(content)
            documents.append((content, normalized_content))
            doc_ids.append(input_file)
    
    # Compute n-grams and minhash signatures for all documents
    signatures = {}
    ngram_sets = {}
    
    for i, (original_content, normalized_content) in enumerate(documents):
        ngram_set = get_ngrams(normalized_content, ngrams)
        ngram_sets[i] = ngram_set
        signature = compute_minhash_signature(ngram_set, num_hashes)
        signatures[i] = signature
    
    # Use LSH to find candidate duplicate pairs
    candidate_pairs = lsh_candidate_pairs(signatures, num_bands, num_hashes)
    
    # Compute true Jaccard similarity for candidate pairs and identify duplicates
    duplicate_groups = defaultdict(set)
    processed_pairs = set()
    
    for doc1_id, doc2_id in candidate_pairs:
        if (doc1_id, doc2_id) in processed_pairs:
            continue
        
        similarity = compute_jaccard_similarity(ngram_sets[doc1_id], ngram_sets[doc2_id])
        
        if similarity >= jaccard_threshold:
            # Find existing group or create new one
            found_group = None
            for group_id, group in duplicate_groups.items():
                if doc1_id in group or doc2_id in group:
                    found_group = group_id
                    break
            
            if found_group is not None:
                duplicate_groups[found_group].add(doc1_id)
                duplicate_groups[found_group].add(doc2_id)
            else:
                new_group_id = len(duplicate_groups)
                duplicate_groups[new_group_id].add(doc1_id)
                duplicate_groups[new_group_id].add(doc2_id)
        
        processed_pairs.add((doc1_id, doc2_id))
    
    # Determine which documents to keep
    documents_to_keep = set(range(len(documents)))
    
    for group in duplicate_groups.values():
        if len(group) > 1:
            # Randomly select one document from each duplicate group to keep
            keep_doc = random.choice(list(group))
            documents_to_remove = group - {keep_doc}
            documents_to_keep -= documents_to_remove
    
    # Write deduplicated files
    for i, (original_content, _) in enumerate(documents):
        if i in documents_to_keep:
            input_file = doc_ids[i]
            output_file = output_dir / Path(input_file).name
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(original_content)


