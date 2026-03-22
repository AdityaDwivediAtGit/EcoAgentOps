# EcoAgentOps — Experiment scaffolding

This repository contains scaffolding and scripts to run the EcoAgentOps experiments (Phase 0–7).

Quick start (assumes conda):

```bash
conda create -n ecoagentops python=3.11 -y
conda activate ecoagentops
pip install -r requirements.txt
```

Phase 1 example (metadata only):

```bash
python scripts/download_laion.py --out_dir data/laion_1M --n_samples 1000000 --preview 100000
python scripts/generate_carbon_trace.py --out_path data/carbon_trace.npy --length 1000000
```

To download images, use `img2dataset` on `data/laion_1M/urls_preview.txt` or the full list you construct.

Next steps: adapt `scripts/train_surrogate.py`, `scripts/train_ppo.py` and integrate the CodeCarbon wrapper
as described in the experiment notebook `experiments.ipynb`.

New scripts added:

- `scripts/train_surrogate.py`: CLIP (vision) + TinyBERT fusion surrogate training. Use `--wandb` and `--codecarbon` flags to enable instrumentation.
- `scripts/pruning_dataloader.py`: create pruned metadata using `surrogate` or `random` strategies.
- `scripts/trl_finetune.py`: wrapper to run TRL's `vsft_llava` example with CodeCarbon tracking.

Quick example workflow:

```bash
# 1) Train surrogate on a preview of metadata
python scripts/train_surrogate.py --metadata data/laion_1M/metadata_preview.parquet --limit 5000 --epochs 1 --batch_size 8 --wandb --codecarbon

# 2) (Hypothetical) produce surrogate scores (example: run a script to infer and save scores.npy)
# 3) Prune metadata using surrogate rankings
python scripts/pruning_dataloader.py --metadata data/laion_1M/metadata.parquet --scores scores.npy --out data/laion_pruned.parquet --keep_ratio 0.7 --strategy surrogate

# 4) Run TRL finetune (wraps TRL example script)
python scripts/trl_finetune.py --model llava-hf/llava-1.5-7b-hf --dataset data/laion_pruned_parquet --lora_r 16 --lora_alpha 32 --bf16 --codecarbon
```
