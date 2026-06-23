#!/usr/bin/env python
"""
Evaluate all policies (and optionally a trained PPO model) on SimplePokemonMoveEnv.

Usage:
    python scripts/evaluate_model.py [--episodes N] [--seed S]

Evaluates:
    - RandomPolicy
    - MaxDamagePolicy
    - TypeAwarePolicy
    - PPOPolicy (if models/ppo_move_selection_v1.zip exists)

Saves results to results/evaluation.csv.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from showdownrl.simple_env import SimplePokemonMoveEnv
from showdownrl.policies import random_policy, max_damage_policy, type_aware_policy


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Pokemon move policies")
    parser.add_argument(
        "--episodes", type=int, default=100,
        help="Number of episodes per policy (default: 100)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    return parser.parse_args()


def evaluate_policy(env_cls, policy_fn, n_episodes, seed):
    """Run n_episodes and return stats dict."""
    np.random.seed(seed + hash(policy_fn.__name__) % 10000)

    wins = 0
    losses = 0
    rewards = []
    turns_list = []

    for ep in range(n_episodes):
        env = env_cls(seed=seed + ep)
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        turns = 0

        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            turns += 1
            done = terminated or truncated

        rewards.append(ep_reward)
        turns_list.append(turns)

        if ep_reward > 0:
            wins += 1
        elif ep_reward < 0:
            losses += 1

        env.close()

    total = n_episodes
    return {
        "policy": policy_fn.__name__,
        "episodes": total,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / total,
        "average_reward": np.mean(rewards),
        "average_turns": np.mean(turns_list),
    }


def main():
    args = parse_args()

    print(f"Evaluating policies over {args.episodes} episodes each...\n")

    results = []

    # Baseline policies
    for policy_fn in [random_policy, max_damage_policy, type_aware_policy]:
        print(f"  {policy_fn.__name__}...")
        stats = evaluate_policy(SimplePokemonMoveEnv, policy_fn, args.episodes, args.seed)
        results.append(stats)
        print(f"    win_rate={stats['win_rate']:.2%}, avg_reward={stats['average_reward']:.3f}")

    # PPO model (if available)
    model_path = (
        Path(__file__).resolve().parent.parent / "models" / "ppo_move_selection_v1.zip"
    )
    if model_path.exists():
        print("\n  PPOPolicy...")
        model = PPO.load(str(model_path))

        def ppo_policy(obs):
            action, _ = model.predict(obs, deterministic=True)
            return int(action)

        ppo_policy.__name__ = "ppo_policy"
        stats = evaluate_policy(SimplePokemonMoveEnv, ppo_policy, args.episodes, args.seed)
        results.append(stats)
        print(f"    win_rate={stats['win_rate']:.2%}, avg_reward={stats['average_reward']:.3f}")
    else:
        print(f"\n  PPO model not found at {model_path} — skipping.")

    # Save results
    df = pd.DataFrame(results)
    print("\nEvaluation Results:")
    print(df.to_string(index=False))

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "evaluation.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
