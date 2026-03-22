"""TRL finetune wrapper that calls a TRL/TRL.examples script (vsft_llava) and wraps with CodeCarbon.

This is a lightweight launcher: it constructs a command for the official TRL example script and
executes it while tracking emissions. Adjust `trl_script` path and args as needed for your setup.
"""
import argparse
import subprocess
import shlex
import os

try:
    from codecarbon import OfflineEmissionsTracker
except Exception:
    OfflineEmissionsTracker = None


def main(args):
    cmd = (
        f"python -m trl.examples.scripts.vsft_llava --model_name_or_path {shlex.quote(args.model)} "
        f"--dataset_name {shlex.quote(args.dataset)} --lora_enable True --lora_r {args.lora_r} --lora_alpha {args.lora_alpha} "
        f"--bf16 {str(args.bf16).lower()} --output_dir {shlex.quote(args.output_dir)} --num_train_epochs {args.epochs} "
        f"--per_device_train_batch_size {args.batch_size}"
    )

    tracker = None
    if args.codecarbon and OfflineEmissionsTracker is not None:
        tracker = OfflineEmissionsTracker(project_name="EcoAgentOps_TRL", output_dir=args.codecarbon_dir)
        tracker.start()

    print("Running finetune command:\n", cmd)
    proc = subprocess.run(cmd, shell=True)

    if tracker is not None:
        tracker.stop()

    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--output_dir', type=str, default='checkpoints/eco_llava')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--codecarbon', action='store_true')
    parser.add_argument('--codecarbon_dir', type=str, default='codecarbon_logs')
    args = parser.parse_args()
    main(args)
