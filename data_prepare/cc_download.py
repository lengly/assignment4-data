import os
import random
import requests
import gzip

# 配置
CRAWL_ID = "CC-MAIN-2025-26"
WET_PATHS_URL = f"https://data.commoncrawl.org/crawl-data/{CRAWL_ID}/wet.paths.gz"
DOWNLOAD_DIR = "/workspace/CommonCrawl"
SAMPLE_SIZE = 5000

def download_wet_paths(url, local_path):
    print(f"Downloading {url} ...")
    r = requests.get(url)
    with open(local_path, "wb") as f:
        f.write(r.content)

def extract_paths(gz_path):
    with gzip.open(gz_path, "rt") as f:
        return [line.strip() for line in f]

def download_file(url, dest):
    if os.path.exists(dest):
        print(f"File {dest} already exists, skipping.")
        return
    print(f"Downloading {url} ...")
    r = requests.get(url, stream=True)
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    wet_paths_gz = "wet.paths.gz"
    download_wet_paths(WET_PATHS_URL, wet_paths_gz)
    all_paths = extract_paths(wet_paths_gz)
    print(f"Total WET files: {len(all_paths)}")
    sample_paths = random.sample(all_paths, SAMPLE_SIZE)
    for i, path in enumerate(sample_paths, 1):
        url = f"https://data.commoncrawl.org/{path}"
        filename = os.path.join(DOWNLOAD_DIR, os.path.basename(path))
        print(f"[{i}/{SAMPLE_SIZE}] Downloading {filename}")
        download_file(url, filename)

if __name__ == "__main__":
    main()
