#!/usr/bin/env python
"""
Train a PPO model on the SimplePokemonMoveEnv.

Usage:
    python scripts/train_ppo.py [--timesteps N] [--seed S] [--output PATH]

Saves the model to the requested output path.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from showdownrl.simple_env import SimplePokemonMoveEnv


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_net_arch(value: str) -> list[int]:
    try:
        layers = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--net-arch must be comma-separated integers") from exc
    if not layers or any(layer <= 0 for layer in layers):
        raise argparse.ArgumentTypeError("--net-arch must contain positive layer sizes")
    return layers


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
        choices=["random", "max_damage", "type_aware", "mixed"],
        default="random",
        help="Opponent policy used during training.",
    )
    parser.add_argument(
        "--mechanics",
        choices=["toy", "typed", "rich"],
        default="typed",
        help="Environment mechanics used during training.",
    )
    parser.add_argument(
        "--observation-mode",
        choices=["simple", "rich"],
        default="simple",
        help="Observation vector used during training.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/ppo_move_selection_v2.zip"),
        help="Output model path.",
    )
    parser.add_argument("--resume-from", type=Path, help="Existing PPO model to continue training.")
    parser.add_argument("--n-envs", type=positive_int, default=8, help="Parallel env copies for PPO rollouts.")
    parser.add_argument("--n-steps", type=positive_int, default=256, help="Rollout steps per env before each update.")
    parser.add_argument("--batch-size", type=positive_int, default=256, help="PPO minibatch size.")
    parser.add_argument("--n-epochs", type=positive_int, default=8, help="Optimization epochs per rollout.")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="PPO learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE bias/variance trade-off.")
    parser.add_argument("--clip-range", type=float, default=0.20, help="PPO clipping range.")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy bonus coefficient.")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value-function loss coefficient.")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Gradient clipping norm.")
    parser.add_argument("--target-kl", type=float, default=0.03, help="Early-stop updates above this KL; use 0 to disable.")
    parser.add_argument("--net-arch", type=parse_net_arch, default=[128, 128], help="Comma-separated MLP layer sizes.")
    parser.add_argument("--ortho-init", action=argparse.BooleanOptionalAction, default=False, help="Use orthogonal init.")
    parser.add_argument("--eval-frequency", type=positive_int, default=10_000, help="Evaluate every N timesteps; use 0 to disable.")
    parser.add_argument("--eval-episodes", type=positive_int, default=20, help="Episodes per periodic evaluation.")
    return parser.parse_args()


def make_env(args: argparse.Namespace, seed_offset: int = 0):
    def _init():
        return Monitor(
            SimplePokemonMoveEnv(
                seed=args.seed + seed_offset,
                opponent_policy=args.opponent_policy,
                mechanics=args.mechanics,
                observation_mode=args.observation_mode,
            )
        )

    return _init


def build_ppo_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    target_kl = args.target_kl if args.target_kl and args.target_kl > 0 else None
    return {
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "target_kl": target_kl,
        "policy_kwargs": {
            "net_arch": args.net_arch,
            "ortho_init": args.ortho_init,
        },
    }


def write_metadata(args: argparse.Namespace, save_path: Path, best_model_dir: Path | None) -> Path:
    metadata_path = save_path.with_suffix(".metadata.json")
    metadata = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model_path": str(save_path),
        "resume_from": str(args.resume_from or ""),
        "best_model_dir": str(best_model_dir) if best_model_dir else "",
        "timesteps": args.timesteps,
        "seed": args.seed,
        "n_envs": args.n_envs,
        "mechanics": args.mechanics,
        "observation_mode": args.observation_mode,
        "opponent_policy": args.opponent_policy,
        "ppo": build_ppo_kwargs(args),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata_path


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    save_path = args.output if args.output.is_absolute() else root / args.output
    save_path.parent.mkdir(parents=True, exist_ok=True)

    rollout_size = args.n_envs * args.n_steps
    if rollout_size % args.batch_size != 0:
        print(
            f"Warning: n_envs*n_steps={rollout_size} is not divisible by batch_size={args.batch_size}.",
            flush=True,
        )

    print(
        f"Training PPO for {args.timesteps} timesteps "
        f"({args.n_envs} envs, {args.mechanics}/{args.observation_mode}, {args.opponent_policy} opponent)...",
        flush=True,
    )

    env = DummyVecEnv([make_env(args, rank) for rank in range(args.n_envs)])
    eval_env = DummyVecEnv([make_env(args, 10_000)])
    if args.resume_from:
        resume_path = args.resume_from if args.resume_from.is_absolute() else root / args.resume_from
        print(f"Resuming PPO model from {resume_path}", flush=True)
        model = PPO.load(str(resume_path), env=env, seed=args.seed, verbose=1)
    else:
        model = PPO("MlpPolicy", env, verbose=1, seed=args.seed, **build_ppo_kwargs(args))

    callbacks = []
    best_model_dir: Path | None = None
    if args.eval_frequency:
        best_model_dir = save_path.parent / f"{save_path.stem}_best"
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(best_model_dir),
                log_path=str(root / "results" / "training_eval"),
                eval_freq=max(1, args.eval_frequency // args.n_envs),
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
            )
        )

    model.learn(total_timesteps=args.timesteps, callback=callbacks or None)

    model.save(str(save_path))
    metadata_path = write_metadata(args, save_path, best_model_dir)

    print(f"Model saved to {save_path}")
    print(f"Training metadata saved to {metadata_path}")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
