#!/usr/bin/env python
"""
Train a PPO model on the SimplePokemonMoveEnv.

Usage:
    python scripts/train_ppo.py [--timesteps N] [--seed S]

Saves the model to models/ppo_move_selection_v1.zip.
"""

import argparse
import sys
from pathlib import Path

from stable_baselines3 import PPO

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from showdownrl.simple_env import SimplePokemonMoveEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on Pokemon move selection")
    parser.add_argument(
        "--timesteps", type=int, default=100_000,
        help="Total timesteps to train (default: 100000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Training PPO for {args.timesteps} timesteps...")

    env = SimplePokemonMoveEnv(seed=args.seed)
    model = PPO("MlpPolicy", env, verbose=1, seed=args.seed)

    model.learn(total_timesteps=args.timesteps)

    # Save model
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / "ppo_move_selection_v1.zip"
    model.save(str(save_path))

    print(f"Model saved to {save_path}")
    env.close()


if __name__ == "__main__":
    main()
