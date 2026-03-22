"""Generate a synthetic carbon intensity trace (Delhi-tuned) and save as numpy array.

Usage:
  python scripts/generate_carbon_trace.py --out_path data/carbon_trace.npy --length 1000000
"""
import argparse
import numpy as np
import os


def main(out_path, length, seed):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.random.seed(seed)
    # Simulate hourly variation around ~0.72 kgCO2/kWh with small noise
    carbon_trace = np.clip(np.random.normal(0.72, 0.05, length), 0.65, 0.80)
    np.save(out_path, carbon_trace)
    print(f"Saved carbon trace ({length}) to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", type=str, default="data/carbon_trace.npy")
    parser.add_argument("--length", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.out_path, args.length, args.seed)
