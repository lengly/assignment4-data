import sys
import os
import random
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), "cs336_data"))

from cs336_data.utils import (
    extract_text_from_html_bytes,
    gopher_quality_filter,
    mask_emails,
    mask_phone_numbers,
    mask_ips,
    identify_language,
)
from fastwarc.warc import ArchiveIterator
import fasttext

def process_warc(warc_path, label, max_samples, out_file):
    count = 0
    saved_count = 0
    with open(warc_path, "rb") as stream, open(out_file, "a") as fout:
        for record in ArchiveIterator(stream):
            if record.headers.get('WARC-Type') == 'response':
                count += 1
                if count % 100 == 0:
                    print(f"{label}: {count} samples, written {saved_count}/{max_samples} samples")
                html_bytes = record.reader.read()
                if len(html_bytes) > 1e8:
                    continue
                text = extract_text_from_html_bytes(html_bytes)
                text = text.replace('\n', ' ').strip()
                lang, _ = identify_language(text)
                if lang != "en" and lang != "zh":
                    continue
                text, _ = mask_emails(text)
                text, _ = mask_phone_numbers(text)
                text, _ = mask_ips(text)
                if gopher_quality_filter(text):
                    fout.write(f"__label__{label} {text}\n")
                    saved_count += 1
                if saved_count >= max_samples:
                    break

def process_warc_file_with_exception_handling(warc_path, label, max_samples, out_file):
    """Process a single WARC file with exception handling"""
    count = 0
    saved_count = 0
    
    try:
        with open(warc_path, "rb") as stream, open(out_file, "a") as fout:
            for record in ArchiveIterator(stream):
                if record.headers.get('WARC-Type') == 'response':
                    count += 1
                    if count % 100 == 0:
                        print(f"{label}: {count} samples, written {saved_count}/{max_samples} samples")
                    
                    try:
                        html_bytes = record.reader.read()
                        if len(html_bytes) > 1e8:
                            continue
                        text = extract_text_from_html_bytes(html_bytes)
                        text = text.replace('\n', ' ').strip()
                        lang, _ = identify_language(text)
                        if lang != "en" and lang != "zh":
                            continue
                        text, _ = mask_emails(text)
                        text, _ = mask_phone_numbers(text)
                        text, _ = mask_ips(text)
                        if gopher_quality_filter(text):
                            fout.write(f"__label__{label} {text}\n")
                            saved_count += 1
                        if saved_count >= max_samples:
                            break
                    except Exception as e:
                        print(f"Error processing record in {warc_path}: {e}")
                        continue
                        
    except Exception as e:
        print(f"Error processing file {warc_path}: {e}")
        return False
    
    return True

def process_all_subsampled_files(data_dir, label, max_samples_per_file, out_file):
    """Process all subsampled_pos_*.warc.gz files in the data directory"""
    
    # Find all subsampled_pos_*.warc.gz files
    pattern = os.path.join(data_dir, "subsampled_pos_*.warc.gz")
    warc_files = glob.glob(pattern)
    
    print(f"Found {len(warc_files)} subsampled_pos_*.warc.gz files")
    
    processed_count = 0
    skipped_count = 0
    
    for i, warc_file in enumerate(warc_files):
        if i % 100 == 0:
            print(f"\nProcessing file {i+1}/{len(warc_files)}: {os.path.basename(warc_file)}")
        
        success = process_warc_file_with_exception_handling(
            warc_path=warc_file,
            label=label,
            max_samples=max_samples_per_file,
            out_file=out_file
        )
        
        if success:
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"✓ Successfully processed {processed_count} files")
        else:
            skipped_count += 1
            if skipped_count % 100 == 0:
                print(f"✗ Skipped {skipped_count} files")
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count} files")
    print(f"Skipped due to errors: {skipped_count} files")

if __name__ == "__main__":
    pos_data_dir = "data"
    neg_warc = "data/CC-MAIN-20250417135010-20250417165010-00065.warc.gz"
    raw_file = "quality_train_raw.txt"
    final_file = "quality_train.txt"
    val_file = "quality_val.txt"

    open(raw_file, "w").close()

    print("Processing positive samples from all subsampled files...")
    process_all_subsampled_files(
        data_dir=pos_data_dir,
        label="pos", 
        max_samples_per_file=100,
        out_file=raw_file
    )
    
    print("Processing negative samples...")
    process_warc(neg_warc, "neg", max_samples=10000, out_file=raw_file)

    print("Shuffling and splitting data into training and validation sets...")
    with open(raw_file, "r") as fin:
        lines = fin.readlines()
    random.seed(42)
    random.shuffle(lines)
    
    # Split data: 90% for training, 10% for validation
    split_idx = int(len(lines) * 0.9)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]
    
    print(f"Total samples: {len(lines)}")
    print(f"Training samples: {len(train_lines)} (90%)")
    print(f"Validation samples: {len(val_lines)} (10%)")
    
    # Write training file
    with open(final_file, "w") as fout:
        for i, line in enumerate(train_lines):
            fout.write(line)
            if i % 1000 == 0:
                print(f"write {i}/{len(train_lines)} lines to {final_file}...")
    
    # Write validation file
    with open(val_file, "w") as fout:
        for i, line in enumerate(val_lines):
            fout.write(line)
            if i % 1000 == 0:
                print(f"write {i}/{len(val_lines)} lines to {val_file}...")

    print("Training fastText model...")
    model = fasttext.train_supervised(final_file, epoch=20, lr=1, wordNgrams=2, verbose=2)
    model.save_model("cs336_data/fasttext_quality_model.bin")
    print("Save to fasttext_quality_model.bin...")
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    result = model.test(val_file)
    print(f"Validation accuracy: {result[1]:.4f}")
    print(f"Validation samples: {result[0]}")