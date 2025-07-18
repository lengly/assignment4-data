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


def process_wet_file_with_exception_handling(wet_path, label, max_samples, out_file):
    """Process a single WET file with exception handling"""
    count = 0
    saved_count = 0
    
    try:
        with open(wet_path, "rb") as stream, open(out_file, "a") as fout:
            for record in ArchiveIterator(stream):
                if record.headers.get('WARC-Type') == 'conversion':
                    count += 1
                    if count % 100 == 0:
                        print(f"{label}: {count} samples, written {saved_count}/{max_samples} samples")
                    
                    try:
                        # WET files contain plain text, not HTML
                        text_bytes = record.reader.read()
                        if len(text_bytes) > 1e8:
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
                        print(f"Error processing record in {wet_path}: {e}")
                        continue
                        
    except Exception as e:
        print(f"Error processing file {wet_path}: {e}")
        return False
    
    return True

def process_random_wet_files(wet_dir, label, max_samples, out_file, num_files=10):
    """Process randomly selected WET files from the CommonCrawl directory"""
    
    # Find all WET files in the directory
    pattern = os.path.join(wet_dir, "*.warc.wet.gz")
    wet_files = glob.glob(pattern)
    
    if not wet_files:
        print(f"No WET files found in {wet_dir}")
        return
    
    print(f"Found {len(wet_files)} WET files in {wet_dir}")
    
    # Randomly select files
    random.seed(42)  # For reproducibility
    selected_files = random.sample(wet_files, min(num_files, len(wet_files)))
    
    print(f"Randomly selected {len(selected_files)} WET files for processing")
    
    samples_per_file = max_samples // len(selected_files)
    remaining_samples = max_samples % len(selected_files)
    
    for i, wet_file in enumerate(selected_files):
        print(f"\nProcessing file {i+1}/{len(selected_files)}: {os.path.basename(wet_file)}")
        
        # Distribute remaining samples among first few files
        current_max_samples = samples_per_file
        if i < remaining_samples:
            current_max_samples += 1
        
        success = process_wet_file_with_exception_handling(
            wet_path=wet_file,
            label=label,
            max_samples=current_max_samples,
            out_file=out_file
        )
        
        if success:
            print(f"✓ Successfully processed {os.path.basename(wet_file)}")
        else:
            print(f"✗ Failed to process {os.path.basename(wet_file)}")
    
    print(f"\nProcessing complete! Total samples processed: {max_samples}")

def process_warc_file_with_exception_handling(warc_path, label, max_samples, out_file):
    """Process a single WARC file with exception handling"""
    global count, saved_count
    
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
    
    global count, saved_count
    count = 0
    saved_count = 0
    
    for i, warc_file in enumerate(warc_files):
        if saved_count % 100 == 0:
            print(f"✓ Successfully processed {saved_count}/{max_samples_per_file} files")
        
        success = process_warc_file_with_exception_handling(
            warc_path=warc_file,
            label=label,
            max_samples=max_samples_per_file,
            out_file=out_file
        )
    
    print(f"\nProcessing complete!")

if __name__ == "__main__":
    pos_data_dir = "data"
    wet_dir = "/workspace/CommonCrawl"  # Directory containing WET files
    raw_file = "quality_train_raw.txt"
    final_file = "quality_train.txt"
    val_file = "quality_val.txt"

    # open(raw_file, "w").close()

    # print("Processing positive samples from all subsampled files...")
    # process_all_subsampled_files(
    #     data_dir=pos_data_dir,
    #     label="pos", 
    #     max_samples_per_file=100,
    #     out_file=raw_file
    # )
    
    print("Processing negative samples from randomly selected WET files...")
    process_random_wet_files(
        wet_dir=wet_dir,
        label="neg",
        max_samples=10000,
        out_file=raw_file,
        num_files=10  # Process 10 randomly selected WET files
    )

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