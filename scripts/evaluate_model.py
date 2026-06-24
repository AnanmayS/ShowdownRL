#!/usr/bin/env python
"""
Evaluate all policies (and optionally a trained PPO model) on SimplePokemonMoveEnv.

Usage:
    python scripts/evaluate_model.py [--episodes N] [--seed S] [--model PATH ...]

Evaluates:
    - RandomPolicy
    - MaxDamagePolicy
    - TypeAwarePolicy
    - PPO models passed with --model, or the default packaged PPO model if present

Saves results to results/evaluation.csv.
"""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from showdownrl.simple_env import SimplePokemonMoveEnv
from showdownrl.policies import random_policy, max_damage_policy, type_aware_policy
from showdownrl.policy_bridge import default_model_path


def git_sha(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


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
        choices=["random", "max_damage", "type_aware", "mixed"],
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


def evaluate_policy(
    env_cls,
    policy_fn,
    n_episodes,
    seed,
    opponent_policy,
    mechanics,
    observation_mode,
    *,
    pass_env: bool = False,
):
    """Run n_episodes and return stats dict."""
    policy_seed = int(hashlib.sha256(policy_fn.__name__.encode()).hexdigest()[:8], 16)
    np.random.seed(seed + policy_seed % 10000)

    wins = 0
    losses = 0
    draws = 0
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
        final_info = {}

        while not done:
            action = policy_fn(obs, env) if pass_env else policy_fn(obs)
            obs, reward, terminated, truncated, final_info = env.step(action)
            ep_reward += reward
            turns += 1
            done = terminated or truncated

        rewards.append(ep_reward)
        turns_list.append(turns)

        opponent_hp = float(final_info.get("opponent_hp", 0.0))
        own_hp = float(final_info.get("own_hp", 0.0))
        if opponent_hp <= 0.0 and own_hp > 0.0:
            wins += 1
        elif own_hp <= 0.0 and opponent_hp > 0.0:
            losses += 1
        else:
            draws += 1

        env.close()

    total = n_episodes
    return {
        "policy": policy_fn.__name__,
        "episodes": total,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / total,
        "non_loss_rate": (wins + draws) / total,
        "average_reward": np.mean(rewards),
        "average_turns": np.mean(turns_list),
    }


def load_rl_model(model_path: Path):
    if "maskable" in model_path.stem:
        try:
            from sb3_contrib import MaskablePPO
        except ImportError as exc:
            raise RuntimeError(
                f"{model_path.name} is a MaskablePPO model, but sb3-contrib is not installed."
            ) from exc
        return MaskablePPO.load(str(model_path)), "maskable_ppo"

    try:
        return PPO.load(str(model_path)), "ppo"
    except Exception as ppo_error:
        try:
            from sb3_contrib import MaskablePPO
        except ImportError as exc:
            raise RuntimeError(
                f"Could not load {model_path.name} as PPO, and sb3-contrib is not installed for MaskablePPO."
            ) from exc
        try:
            return MaskablePPO.load(str(model_path)), "maskable_ppo"
        except Exception as maskable_error:
            raise RuntimeError(
                f"Could not load {model_path.name} as PPO ({ppo_error}) or MaskablePPO ({maskable_error})."
            ) from maskable_error


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    scenario = f"{args.mechanics}_{args.opponent_policy}_seed{args.seed}"
    current_git_sha = git_sha(root)

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
        stats.update(
            {
                "scenario": scenario,
                "seed": args.seed,
                "mechanics": args.mechanics,
                "opponent_policy": args.opponent_policy,
                "observation_mode": "rich" if args.observation_mode == "rich" else "simple",
                "model_path": "",
                "git_sha": current_git_sha,
            }
        )
        results.append(stats)
        print(f"    win_rate={stats['win_rate']:.2%}, avg_reward={stats['average_reward']:.3f}")

    model_paths = args.model or [default_model_path()]
    for raw_path in model_paths:
        model_path = raw_path if raw_path.is_absolute() else root / raw_path
        if not model_path.exists():
            print(f"\n  PPO model not found at {model_path} — skipping.")
            continue

        policy_name = model_path.stem
        print(f"\n  {policy_name}...")
        model, model_kind = load_rl_model(model_path)
        model_shape = getattr(model.observation_space, "shape", ())
        model_observation_mode = "rich" if model_shape and int(model_shape[0]) > 14 else "simple"

        def ppo_policy(obs, env):
            predict_kwargs = {"deterministic": True}
            if model_kind == "maskable_ppo":
                predict_kwargs["action_masks"] = env.action_masks()
            action, _ = model.predict(obs, **predict_kwargs)
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
            pass_env=True,
        )
        stats.update(
            {
                "scenario": scenario,
                "seed": args.seed,
                "mechanics": args.mechanics,
                "opponent_policy": args.opponent_policy,
                "observation_mode": model_observation_mode if args.observation_mode == "auto" else args.observation_mode,
                "model_path": str(model_path.relative_to(root) if model_path.is_relative_to(root) else model_path),
                "git_sha": current_git_sha,
            }
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
