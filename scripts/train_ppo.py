"""Train a PPO policy using stable-baselines3 against the PruningEnv.

This script is a small example to get started. Replace the env with an offline batch-based
vectorized environment that serves real surrogate outputs for full experiments.
"""
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from pruning_env import PruningEnv

try:
    import wandb
except Exception:
    wandb = None

try:
    from codecarbon import OfflineEmissionsTracker
except Exception:
    OfflineEmissionsTracker = None


def make_env():
    return PruningEnv()


def main(total_timesteps, out_path, use_wandb=False, use_codecarbon=False, codecarbon_dir="codecarbon_logs"):
    env = DummyVecEnv([make_env])
    model = PPO("MlpPolicy", env, verbose=1)

    tracker = None
    if use_codecarbon and OfflineEmissionsTracker is not None:
        tracker = OfflineEmissionsTracker(project_name="EcoAgentOps_PPO", output_dir=codecarbon_dir)
        tracker.start()

    if use_wandb and wandb is not None:
        wandb.init(project="ecoagentops-ppo")

    model.learn(total_timesteps=total_timesteps)
    model.save(out_path)
    print(f"Saved PPO policy to {out_path}")

    if tracker is not None:
        tracker.stop()

    if use_wandb and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--out", type=str, default="checkpoints/ppo_policy")
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--codecarbon", action='store_true')
    parser.add_argument("--codecarbon_dir", type=str, default="codecarbon_logs")
    args = parser.parse_args()
    main(args.steps, args.out, use_wandb=args.wandb, use_codecarbon=args.codecarbon, codecarbon_dir=args.codecarbon_dir)
