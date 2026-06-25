#!/usr/bin/env python
"""
Train a PPO model on the SimplePokemonMoveEnv.

Usage:
    python scripts/train_ppo.py [--timesteps N] [--seed S] [--output PATH]

Saves the model to the requested output path.
"""

import argparse
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from showdownrl.simple_env import SimplePokemonMoveEnv


LEAGUE_STEP_RE = re.compile(r"^league_step_(\d+)\.zip$")
ACTIVATION_FNS = {
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "elu": torch.nn.ELU,
}
_LEAGUE_MODEL_CACHE: dict[Path, PPO] = {}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def parse_net_arch(value: str) -> list[int]:
    if value == "residual":
        return [256, 256, 256, 256]
    try:
        layers = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--net-arch must be comma-separated integers") from exc
    if not layers or any(layer <= 0 for layer in layers):
        raise argparse.ArgumentTypeError("--net-arch must contain positive layer sizes")
    return layers


class LeaguePool:
    def __init__(self, pool_dir: Path, max_size: int):
        self.pool_dir = Path(pool_dir)
        self.max_size = max_size
        self.pool_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model, step: int) -> Path:
        path = self.pool_dir / f"league_step_{step}.zip"
        model.save(str(path))
        checkpoints = self.list_checkpoints()
        while len(checkpoints) > self.max_size:
            _, oldest_path = checkpoints.pop(0)
            oldest_path.unlink(missing_ok=True)
            _LEAGUE_MODEL_CACHE.pop(oldest_path.resolve(), None)
        return path

    def sample(self) -> Path | None:
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        return random.choice(checkpoints)[1]

    def list_checkpoints(self) -> list[tuple[int, Path]]:
        checkpoints = []
        for path in self.pool_dir.glob("league_step_*.zip"):
            match = LEAGUE_STEP_RE.match(path.name)
            if match:
                checkpoints.append((int(match.group(1)), path))
        return sorted(checkpoints, key=lambda item: item[0])


class OpponentWrapper:
    def __init__(self, env: SimplePokemonMoveEnv, model, fallback_opponent_action, self_play_prob: float):
        self.env = env
        self.model = model
        self.fallback_opponent_action = fallback_opponent_action
        self.self_play_prob = self_play_prob

    def _opponent_obs(self):
        env = self.env
        swapped = {
            "own_hp": env.own_hp,
            "opponent_hp": env.opponent_hp,
            "moves": env.moves,
            "opponent_moves": env.opponent_moves,
            "own_types": env.own_types,
            "opponent_types": env.opponent_types,
            "own_attack_boost": env.own_attack_boost,
            "opponent_attack_boost": env.opponent_attack_boost,
            "bench": env.bench,
            "opponent_bench": env.opponent_bench,
            "active_pokemon": env.active_pokemon,
            "opponent_active": env.opponent_active,
        }
        try:
            env.own_hp, env.opponent_hp = env.opponent_hp, env.own_hp
            env.moves, env.opponent_moves = env.opponent_moves or env.moves, env.moves
            env.own_types, env.opponent_types = env.opponent_types, env.own_types
            env.own_attack_boost, env.opponent_attack_boost = (
                env.opponent_attack_boost,
                env.own_attack_boost,
            )
            env.bench, env.opponent_bench = env.opponent_bench, env.bench
            env.active_pokemon, env.opponent_active = env.opponent_active, env.active_pokemon
            return env._get_obs()
        finally:
            for name, value in swapped.items():
                setattr(env, name, value)

    def _opponent_action(self) -> int:
        if self.env.rng.random() >= self.self_play_prob:
            return self.fallback_opponent_action()
        action, _ = self.model.predict(self._opponent_obs(), deterministic=False)
        return int(action)


class LeagueSnapshotCallback(BaseCallback):
    def __init__(self, pool: LeaguePool, update_freq: int):
        super().__init__()
        self.pool = pool
        self.update_freq = update_freq
        self.last_saved_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_saved_step >= self.update_freq:
            self.pool.save(self.model, self.num_timesteps)
            self.last_saved_step = self.num_timesteps
        return True


def load_league_model(model_path: Path | None):
    if model_path is None:
        return None
    cache_key = model_path.resolve()
    if cache_key not in _LEAGUE_MODEL_CACHE:
        _LEAGUE_MODEL_CACHE[cache_key] = PPO.load(str(model_path))
    return _LEAGUE_MODEL_CACHE[cache_key]


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
    parser.add_argument(
        "--algorithm",
        choices=["ppo", "maskable_ppo"],
        default="ppo",
        help="RL algorithm to train.",
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
    parser.add_argument("--net-arch", type=parse_net_arch, default=[512, 256, 128], help="Comma-separated MLP layer sizes or 'residual'.")
    parser.add_argument("--activation-fn", choices=["relu", "tanh", "elu"], default="relu", help="Policy activation function.")
    parser.add_argument("--ortho-init", action=argparse.BooleanOptionalAction, default=False, help="Use orthogonal init.")
    parser.add_argument("--eval-frequency", type=non_negative_int, default=10_000, help="Evaluate every N timesteps; use 0 to disable.")
    parser.add_argument("--eval-episodes", type=positive_int, default=20, help="Episodes per periodic evaluation.")
    parser.add_argument("--self-play", action="store_true", default=False, help="Train against sampled past league checkpoints.")
    parser.add_argument("--league-dir", type=Path, default=Path("models/league"), help="Directory for league checkpoints.")
    parser.add_argument("--league-update-freq", type=positive_int, default=50_000, help="Timesteps between league snapshots.")
    parser.add_argument("--self-play-prob", type=float, default=0.5, help="Probability of using the league opponent for each opponent action.")
    parser.add_argument("--league-pool-size", type=positive_int, default=10, help="Maximum league checkpoints to keep.")
    return parser.parse_args()


def make_env(args: argparse.Namespace, seed_offset: int = 0, league_model_path: Path | None = None):
    def _init():
        env = SimplePokemonMoveEnv(
            seed=args.seed + seed_offset,
            opponent_policy=args.opponent_policy,
            mechanics=args.mechanics,
            observation_mode=args.observation_mode,
        )
        league_model = load_league_model(league_model_path)
        if league_model is not None:
            wrapper = OpponentWrapper(env, league_model, env._opponent_action, args.self_play_prob)
            env._opponent_action = wrapper._opponent_action
        return Monitor(env)

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
            "activation_fn": ACTIVATION_FNS[getattr(args, "activation_fn", "relu")],
            "ortho_init": args.ortho_init,
        },
    }


def resolve_algorithm(algorithm: str):
    if algorithm == "ppo":
        return PPO, EvalCallback
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    except ImportError as exc:
        raise SystemExit(
            "MaskablePPO requires sb3-contrib. Install RL dependencies with `pip install -e '.[rl]'`."
        ) from exc
    return MaskablePPO, MaskableEvalCallback


def write_metadata(args: argparse.Namespace, save_path: Path, best_model_dir: Path | None) -> Path:
    metadata_path = save_path.with_suffix(".metadata.json")
    metadata = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "algorithm": args.algorithm,
        "model_path": str(save_path),
        "resume_from": str(args.resume_from or ""),
        "best_model_dir": str(best_model_dir) if best_model_dir else "",
        "timesteps": args.timesteps,
        "seed": args.seed,
        "n_envs": args.n_envs,
        "mechanics": args.mechanics,
        "observation_mode": args.observation_mode,
        "opponent_policy": args.opponent_policy,
        "ppo": {
            **build_ppo_kwargs(args),
            "policy_kwargs": {
                **build_ppo_kwargs(args)["policy_kwargs"],
                "activation_fn": args.activation_fn,
            },
        },
        "self_play": args.self_play,
        "league_dir": str(args.league_dir),
        "league_update_freq": args.league_update_freq,
        "self_play_prob": args.self_play_prob,
        "league_pool_size": args.league_pool_size,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata_path


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    save_path = args.output if args.output.is_absolute() else root / args.output
    save_path.parent.mkdir(parents=True, exist_ok=True)
    args.league_dir = args.league_dir if args.league_dir.is_absolute() else root / args.league_dir

    rollout_size = args.n_envs * args.n_steps
    if rollout_size % args.batch_size != 0:
        print(
            f"Warning: n_envs*n_steps={rollout_size} is not divisible by batch_size={args.batch_size}.",
            flush=True,
        )

    print(
        f"Training {args.algorithm} for {args.timesteps} timesteps "
        f"({args.n_envs} envs, {args.mechanics}/{args.observation_mode}, {args.opponent_policy} opponent)...",
        flush=True,
    )

    algorithm_class, eval_callback_class = resolve_algorithm(args.algorithm)
    league_pool = None
    league_model_path = None
    if args.self_play:
        league_pool = LeaguePool(args.league_dir, args.league_pool_size)
        league_model_path = league_pool.sample()
        if league_model_path is None:
            print(f"Self-play enabled; no league checkpoints found in {args.league_dir}", flush=True)
        else:
            print(f"Self-play enabled; sampled league opponent {league_model_path}", flush=True)

    env = DummyVecEnv([make_env(args, rank, league_model_path) for rank in range(args.n_envs)])
    eval_env = DummyVecEnv([make_env(args, 10_000)])
    if args.resume_from:
        resume_path = args.resume_from if args.resume_from.is_absolute() else root / args.resume_from
        print(f"Resuming {args.algorithm} model from {resume_path}", flush=True)
        model = algorithm_class.load(str(resume_path), env=env, seed=args.seed, verbose=1)
    else:
        model = algorithm_class("MlpPolicy", env, verbose=1, seed=args.seed, **build_ppo_kwargs(args))

    callbacks = []
    best_model_dir: Path | None = None
    if args.eval_frequency:
        best_model_dir = save_path.parent / f"{save_path.stem}_best"
        callbacks.append(
            eval_callback_class(
                eval_env,
                best_model_save_path=str(best_model_dir),
                log_path=str(root / "results" / "training_eval"),
                eval_freq=max(1, args.eval_frequency // args.n_envs),
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
            )
        )
    if league_pool is not None:
        callbacks.append(LeagueSnapshotCallback(league_pool, args.league_update_freq))

    model.learn(total_timesteps=args.timesteps, callback=callbacks or None)

    model.save(str(save_path))
    if league_pool is not None:
        league_path = league_pool.save(model, model.num_timesteps)
        print(f"League checkpoint saved to {league_path}")
    metadata_path = write_metadata(args, save_path, best_model_dir)

    print(f"Model saved to {save_path}")
    print(f"Training metadata saved to {metadata_path}")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
