#!/usr/bin/env python
"""
Evaluate all policies (and optionally a trained PPO model) on SimplePokemonMoveEnv.

Usage:
    python scripts/evaluate_model.py [--episodes N] [--seed S] [--model PATH ...]

Evaluates:
    - RandomPolicy
    - MaxDamagePolicy
    - TypeAwarePolicy
    - PPO models passed with --model, or models/ppo_move_selection_v1.zip if present

Saves results to results/evaluation.csv.
"""

import argparse
import hashlib
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
    parser.add_argument(
        "--opponent-policy",
        choices=["random", "max_damage", "type_aware"],
        default="random",
        help="Opponent policy used during evaluation.",
    )
    parser.add_argument(
        "--mechanics",
        choices=["toy", "typed", "rich"],
        default="toy",
        help="Environment mechanics used during evaluation.",
    )
    parser.add_argument(
        "--observation-mode",
        choices=["simple", "rich", "auto"],
        default="auto",
        help="Observation vector for built-in policies; PPO models use their trained shape when auto.",
    )
    parser.add_argument(
        "--model",
        action="append",
        type=Path,
        default=[],
        help="PPO model zip to evaluate. Can be passed more than once.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/evaluation.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def evaluate_policy(env_cls, policy_fn, n_episodes, seed, opponent_policy, mechanics, observation_mode):
    """Run n_episodes and return stats dict."""
    policy_seed = int(hashlib.sha256(policy_fn.__name__.encode()).hexdigest()[:8], 16)
    np.random.seed(seed + policy_seed % 10000)

    wins = 0
    losses = 0
    rewards = []
    turns_list = []

    for ep in range(n_episodes):
        env = env_cls(
            seed=seed + ep,
            opponent_policy=opponent_policy,
            mechanics=mechanics,
            observation_mode=observation_mode,
        )
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
    root = Path(__file__).resolve().parent.parent

    print(
        f"Evaluating policies over {args.episodes} episodes each "
        f"({args.mechanics} mechanics, {args.opponent_policy} opponent)...\n"
    )

    results = []

    # Baseline policies
    for policy_fn in [random_policy, max_damage_policy, type_aware_policy]:
        print(f"  {policy_fn.__name__}...")
        stats = evaluate_policy(
            SimplePokemonMoveEnv,
            policy_fn,
            args.episodes,
            args.seed,
            args.opponent_policy,
            args.mechanics,
            "rich" if args.observation_mode == "rich" else "simple",
        )
        results.append(stats)
        print(f"    win_rate={stats['win_rate']:.2%}, avg_reward={stats['average_reward']:.3f}")

    model_paths = args.model or [root / "models" / "ppo_move_selection_v1.zip"]
    for raw_path in model_paths:
        model_path = raw_path if raw_path.is_absolute() else root / raw_path
        if not model_path.exists():
            print(f"\n  PPO model not found at {model_path} — skipping.")
            continue

        policy_name = model_path.stem
        print(f"\n  {policy_name}...")
        model = PPO.load(str(model_path))
        model_shape = getattr(model.observation_space, "shape", ())
        model_observation_mode = "rich" if model_shape and int(model_shape[0]) > 14 else "simple"

        def ppo_policy(obs):
            action, _ = model.predict(obs, deterministic=True)
            return int(action)

        ppo_policy.__name__ = policy_name
        stats = evaluate_policy(
            SimplePokemonMoveEnv,
            ppo_policy,
            args.episodes,
            args.seed,
            args.opponent_policy,
            args.mechanics,
            model_observation_mode if args.observation_mode == "auto" else args.observation_mode,
        )
        results.append(stats)
        print(f"    win_rate={stats['win_rate']:.2%}, avg_reward={stats['average_reward']:.3f}")

    # Save results
    df = pd.DataFrame(results)
    print("\nEvaluation Results:")
    print(df.to_string(index=False))

    csv_path = args.output if args.output.is_absolute() else root / args.output
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
