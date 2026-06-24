#!/usr/bin/env python
"""
Train a PPO model on the SimplePokemonMoveEnv.

Usage:
    python scripts/train_ppo.py [--timesteps N] [--seed S] [--output PATH]

Saves the model to the requested output path.
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
    parser.add_argument(
        "--opponent-policy",
        choices=["random", "max_damage", "type_aware"],
        default="random",
        help="Opponent policy used during training.",
    )
    parser.add_argument(
        "--mechanics",
        choices=["toy", "typed"],
        default="typed",
        help="Environment mechanics used during training.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/ppo_move_selection_v2.zip"),
        help="Output model path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Training PPO for {args.timesteps} timesteps...")

    env = SimplePokemonMoveEnv(seed=args.seed, opponent_policy=args.opponent_policy, mechanics=args.mechanics)
    model = PPO("MlpPolicy", env, verbose=1, seed=args.seed)

    model.learn(total_timesteps=args.timesteps)

    save_path = args.output
    if not save_path.is_absolute():
        save_path = Path(__file__).resolve().parent.parent / save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))

    print(f"Model saved to {save_path}")
    env.close()


if __name__ == "__main__":
    main()
