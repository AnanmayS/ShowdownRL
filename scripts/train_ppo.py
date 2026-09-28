#!/usr/bin/env python
"""
Train a PPO model on the SimplePokemonMoveEnv.

Usage:
    python scripts/train_ppo.py [--timesteps N] [--seed S] [--output PATH]

Saves the model to the requested output path.
"""

import argparse
import json
import math
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
_LEAGUE_MODEL_CACHE: dict[Path, Any] = {}


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

    def sample(self, temperature: float = 0.7) -> Path | None:
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        if temperature <= 0:
            return checkpoints[-1][1]
        scores = [index / max(1, len(checkpoints) - 1) for index in range(len(checkpoints))]
        weights = [math.exp(score / temperature) for score in scores]
        return random.choices([path for _, path in checkpoints], weights=weights, k=1)[0]

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

    def set_model(self, model) -> None:
        self.model = model

    def set_self_play_prob(self, value: float) -> None:
        self.self_play_prob = max(0.0, min(1.0, float(value)))

    def _opponent_obs_and_masks(self):
        return self.env.get_opponent_observation(), self.env.opponent_action_masks()

    def _opponent_action(self) -> int:
        if self.model is None or self.env.rng.random() >= self.self_play_prob:
            return self.fallback_opponent_action()
        obs, action_masks = self._opponent_obs_and_masks()
        predict_kwargs = {"deterministic": False}
        if "sb3_contrib" in type(self.model).__module__:
            predict_kwargs["action_masks"] = action_masks
        action, _ = self.model.predict(obs, **predict_kwargs)
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


class SelfPlayCurriculumCallback(BaseCallback):
    def __init__(
        self,
        pool: LeaguePool,
        update_freq: int,
        initial_prob: float,
        final_prob: float,
        total_timesteps: int,
        sampling_temperature: float,
    ):
        super().__init__()
        self.pool = pool
        self.update_freq = update_freq
        self.initial_prob = max(0.0, min(1.0, initial_prob))
        self.final_prob = max(0.0, min(1.0, final_prob))
        self.total_timesteps = max(1, total_timesteps)
        self.sampling_temperature = sampling_temperature
        self.last_update_step = -update_freq

    def _wrappers(self):
        for env in getattr(self.training_env, "envs", []):
            base_env = getattr(env, "env", env)
            wrapper = getattr(base_env, "self_play_opponent_wrapper", None)
            if wrapper is not None:
                yield wrapper

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.total_timesteps)
        prob = self.initial_prob + (self.final_prob - self.initial_prob) * progress
        for wrapper in self._wrappers():
            wrapper.set_self_play_prob(prob)

        if self.num_timesteps - self.last_update_step >= self.update_freq:
            model_path = self.pool.sample(self.sampling_temperature)
            model = load_league_model(model_path)
            if model is not None:
                for wrapper in self._wrappers():
                    wrapper.set_model(model)
            self.last_update_step = self.num_timesteps
        return True


def load_league_model(model_path: Path | None):
    if model_path is None:
        return None
    cache_key = model_path.resolve()
    if cache_key not in _LEAGUE_MODEL_CACHE:
        stem = model_path.stem
        if "recurrent" in stem:
            from showdownrl.recurrent_maskable import RecurrentMaskablePPO

            _LEAGUE_MODEL_CACHE[cache_key] = RecurrentMaskablePPO.load(str(model_path))
        elif "maskable" in stem:
            from sb3_contrib import MaskablePPO

            _LEAGUE_MODEL_CACHE[cache_key] = MaskablePPO.load(str(model_path))
        else:
            # Try MaskablePPO first (common for league checkpoints), then fall back to PPO
            try:
                from sb3_contrib import MaskablePPO
                _LEAGUE_MODEL_CACHE[cache_key] = MaskablePPO.load(str(model_path))
            except Exception:
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
        choices=["toy", "typed", "rich", "advanced"],
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
        choices=["ppo", "maskable_ppo", "recurrent_maskable_ppo"],
        default="ppo",
        help="RL algorithm to train.",
    )
    parser.add_argument("--recurrent", action="store_true", default=False,
                        help="Shortcut: train with RecurrentMaskablePPO (GRU).")
    parser.add_argument("--lstm-hidden-size", type=positive_int, default=256,
                        help="LSTM hidden state size for recurrent policy.")
    parser.add_argument("--n-lstm-layers", type=positive_int, default=1,
                        help="Number of LSTM layers for recurrent policy.")
    parser.add_argument("--shared-lstm", action="store_true", default=False,
                        help="Share LSTM between actor and critic.")
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
    parser.add_argument("--self-play-final-prob", type=float, default=0.25, help="Final self-play probability after linear curriculum decay.")
    parser.add_argument("--league-sampling-temperature", type=float, default=0.7, help="Recency-weighted league sampling temperature; 0 picks the newest checkpoint.")
    parser.add_argument("--league-pool-size", type=positive_int, default=10, help="Maximum league checkpoints to keep.")
    return parser.parse_args()


def make_env(
    args: argparse.Namespace,
    seed_offset: int = 0,
    league_model_path: Path | None = None,
    enable_self_play: bool = False,
):
    def _init():
        env = SimplePokemonMoveEnv(
            seed=args.seed + seed_offset,
            opponent_policy=args.opponent_policy,
            mechanics=args.mechanics,
            observation_mode=args.observation_mode,
        )
        league_model = load_league_model(league_model_path)
        if enable_self_play:
            wrapper = OpponentWrapper(env, league_model, env._opponent_action, args.self_play_prob)
            env.self_play_opponent_wrapper = wrapper
            env._opponent_action = wrapper._opponent_action
        return Monitor(env)

    return _init


def build_ppo_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    target_kl = args.target_kl if args.target_kl and args.target_kl > 0 else None
    policy_kwargs: dict[str, Any] = {
        "net_arch": args.net_arch,
        "activation_fn": ACTIVATION_FNS[getattr(args, "activation_fn", "relu")],
        "ortho_init": args.ortho_init,
    }
    if (
        getattr(args, "algorithm", "ppo") == "recurrent_maskable_ppo"
        or getattr(args, "recurrent", False)
    ):
        policy_kwargs.update({
            "lstm_hidden_size": args.lstm_hidden_size,
            "n_lstm_layers": args.n_lstm_layers,
            "shared_lstm": args.shared_lstm,
            "enable_critic_lstm": not args.shared_lstm,
        })
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
        "policy_kwargs": policy_kwargs,
    }


def resolve_algorithm(algorithm: str):
    if algorithm == "ppo":
        return PPO, EvalCallback
    if algorithm == "maskable_ppo":
        try:
            from sb3_contrib import MaskablePPO
            from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
        except ImportError as exc:
            raise SystemExit(
                "MaskablePPO requires sb3-contrib. Install RL dependencies with `pip install -e '.[rl]'`."
            ) from exc
        return MaskablePPO, MaskableEvalCallback
    if algorithm == "recurrent_maskable_ppo":
        try:
            from sb3_contrib import RecurrentPPO
            from showdownrl.recurrent_maskable import (
                RecurrentMaskablePPO,
                RecurrentMaskableActorCriticPolicy,
            )
        except ImportError as exc:
            raise SystemExit(
                "RecurrentMaskablePPO requires sb3-contrib. "
                "Install RL dependencies with `pip install -e '.[rl]'`."
            ) from exc
        return RecurrentMaskablePPO, EvalCallback
    raise ValueError(f"Unknown algorithm: {algorithm}")


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
        "self_play_final_prob": args.self_play_final_prob,
        "league_sampling_temperature": args.league_sampling_temperature,
        "league_pool_size": args.league_pool_size,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata_path


def main():
    args = parse_args()
    if args.recurrent:
        args.algorithm = "recurrent_maskable_ppo"
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
        league_model_path = league_pool.sample(args.league_sampling_temperature)
        if league_model_path is None:
            print(f"Self-play enabled; no league checkpoints found in {args.league_dir}", flush=True)
        else:
            print(f"Self-play enabled; sampled league opponent {league_model_path}", flush=True)

    # Select the right policy class name
    is_recurrent = args.algorithm == "recurrent_maskable_ppo" or args.recurrent
    policy_name = "MlpLstmPolicy" if is_recurrent else "MlpPolicy"

    env = DummyVecEnv([make_env(args, rank, league_model_path, args.self_play) for rank in range(args.n_envs)])
    eval_env = DummyVecEnv([make_env(args, 10_000)])
    if args.resume_from:
        resume_path = args.resume_from if args.resume_from.is_absolute() else root / args.resume_from
        print(f"Resuming {args.algorithm} model from {resume_path}", flush=True)
        model = algorithm_class.load(str(resume_path), env=env, seed=args.seed, verbose=1)
    else:
        model = algorithm_class(policy_name, env, verbose=1, seed=args.seed, **build_ppo_kwargs(args))

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
        callbacks.append(
            SelfPlayCurriculumCallback(
                league_pool,
                args.league_update_freq,
                args.self_play_prob,
                args.self_play_final_prob,
                args.timesteps,
                args.league_sampling_temperature,
            )
        )

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
