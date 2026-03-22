"""Pruning dataloader and pruner utilities.

Provides simple strategies to create a pruned metadata parquet from metadata + surrogate scores.

Usage example:
  python scripts/pruning_dataloader.py --metadata data/laion_1M/metadata.parquet --surrogate_scores scores.npy --out data/laion_pruned.parquet --keep_ratio 0.7 --strategy surrogate
"""
import argparse
import pandas as pd
import numpy as np
import os


def load_metadata(path):
    return pd.read_parquet(path)


def prune_by_surrogate(df, scores, keep_ratio):
    # scores: ndarray aligned with df
    n_keep = int(len(df) * keep_ratio)
    idx_sorted = np.argsort(scores)  # assume lower score = more desirable (e.g., lower energy)
    keep_idx = idx_sorted[:n_keep]
    return df.iloc[keep_idx]


def prune_random(df, keep_ratio, seed=42):
    n_keep = int(len(df) * keep_ratio)
    return df.sample(n=n_keep, random_state=seed)


def main(args):
    df = load_metadata(args.metadata)
    if args.strategy == 'random':
        pruned = prune_random(df, args.keep_ratio, seed=args.seed)
    elif args.strategy == 'surrogate':
        if not args.scores:
            raise ValueError('surrogate strategy requires --scores path')
        scores = np.load(args.scores)
        pruned = prune_by_surrogate(df, scores, args.keep_ratio)
    else:
        raise NotImplementedError(f"Unknown strategy {args.strategy}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pruned.to_parquet(args.out)
    print(f"Wrote pruned metadata ({len(pruned)}) to {args.out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=str, required=True)
    parser.add_argument('--scores', type=str, default=None)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--keep_ratio', type=float, default=0.7)
    parser.add_argument('--strategy', type=str, default='surrogate', choices=['surrogate', 'random'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
