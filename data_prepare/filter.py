from collections import defaultdict, Counter
from typing import Any, List

from cs336_data.deduplicate import (
    normalize_text,
)

def get_ngrams_with_list(text: str, n: int) -> List[Any]:
    """
    Extract n-grams (word sequences of length n) from normalized text.
    """
    words = text.split()
    if len(words) < n:
        return []
    
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)
    
    return ngrams

def detect_internal_repetition(text: str, ngram_size: int = 3, repetition_threshold: float = 0.3) -> bool:
    """
    Detect if a document has too much internal repetition using n-grams.
    
    Args:
        text: The text to analyze
        ngram_size: Size of n-grams to use for repetition detection
        repetition_threshold: Threshold for considering text as repetitive (0.0 to 1.0)
    
    Returns:
        True if the text has too much repetition, False otherwise
    """
    # Normalize text
    normalized_text = normalize_text(text)
    if len(normalized_text) < 10:
        return False
    
    # Get n-grams
    ngrams = get_ngrams_with_list(normalized_text, ngram_size)
       
    # Count n-gram frequencies
    ngram_counter = Counter(ngrams)
    
    # Calculate repetition ratio
    total_ngrams = len(ngrams)
    unique_ngrams = len(ngram_counter)
    
    if total_ngrams == 0:
        return False
    
    # Calculate how much of the text is repetitive
    # A lower unique/total ratio means more repetition
    unique_ratio = unique_ngrams / total_ngrams
    
    # Also check for very frequent n-grams (indicating excessive repetition)
    max_frequency = max(ngram_counter.values()) if ngram_counter else 0
    max_frequency_ratio = max_frequency / total_ngrams
    
    # Text is considered repetitive if:
    # 1. Unique ratio is too low (too much repetition)
    # 2. Any single n-gram appears too frequently
    is_repetitive = (unique_ratio < (1.0 - repetition_threshold)) or (max_frequency_ratio > repetition_threshold)
    return is_repetitive

if __name__ == "__main__":
    text = "a b c a b c a b c a b c a b c a b c a b c a b c a b c a b c a b c a b c a b c a b c a b c a b c a b c a b c a b c a b c a b c a b c a b c a b c"
    assert detect_internal_repetition(text)