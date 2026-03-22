"""Download and prepare a 1M-sample metadata subset from laion/relaion2B-en-research-safe.

Usage:
  python "scripts/download_laion.py" --out_dir data/laion_1M --n_samples 1000000 --preview 100000

This script streams the HF dataset metadata and saves a parquet of selected samples and a URLs file
for use with `img2dataset` to download images separately.
"""
import argparse
from datasets import load_dataset
import os
import random
import pandas as pd


def main(out_dir, n_samples, seed, preview_urls):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Streaming dataset and collecting {n_samples} samples (this may take a while)...")
    ds = load_dataset("laion/relaion2B-en-research-safe", split="train", streaming=True)

    samples = []
    for i, ex in enumerate(ds):
        samples.append({"URL": ex.get("URL"), "TEXT": ex.get("TEXT"), "id": i})
        if (i + 1) >= n_samples:
            break

    random.seed(seed)
    random.shuffle(samples)

    meta_path = os.path.join(out_dir, "metadata.parquet")
    print(f"Saving metadata to {meta_path} ({len(samples)} rows)")
    pd.DataFrame(samples).to_parquet(meta_path)

    # Write a preview URL list (first preview_urls) for img2dataset testing
    urls = [s['URL'] for s in samples[:preview_urls] if s['URL']]
    urls_path = os.path.join(out_dir, "urls_preview.txt")
    with open(urls_path, "w", encoding="utf-8") as f:
        f.write("\n".join(urls))
    print(f"Wrote {len(urls)} preview URLs to {urls_path}. Use img2dataset to download images from this list.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/laion_1M")
    parser.add_argument("--n_samples", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview", type=int, default=100000)
    args = parser.parse_args()
    main(args.out_dir, args.n_samples, args.seed, args.preview)
