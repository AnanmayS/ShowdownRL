#!/usr/bin/env python
"""Collect expert demonstrations from a trained model.

Runs the model for N timesteps, saving (obs, action, action_mask) tuples.
Output: ~/.hermes/cron/output/expert_data_{seed}.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from showdownrl.simple_env import SimplePokemonMoveEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Collect expert demonstrations")
    parser.add_argument("--model", type=str, required=True, help="Path to model zip")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total timesteps to collect")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mechanics", default="advanced", choices=["toy", "typed", "rich", "advanced"])
    parser.add_argument("--opponent-policy", default="mixed")
    parser.add_argument("--observation-mode", default="simple")
    parser.add_argument("--output", type=str, default=None, help="Output .npz path")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = root / model_path

    print(f"Loading model from {model_path}...", flush=True)
    stem = model_path.stem
    if "maskable" in stem:
        from sb3_contrib import MaskablePPO
        model_cls = MaskablePPO
    else:
        from sb3_contrib import MaskablePPO
        model_cls = MaskablePPO
    model = model_cls.load(str(model_path))

    env = SimplePokemonMoveEnv(
        seed=args.seed,
        mechanics=args.mechanics,
        observation_mode=args.observation_mode,
        opponent_policy=args.opponent_policy,
    )

    obs, _ = env.reset()
    obs_list = []
    action_list = []
    mask_list = []
    step = 0

    while step < args.timesteps:
        masks = env.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=masks)
        obs_list.append(obs.copy())
        action_list.append(int(action))
        mask_list.append(masks.copy())

        obs, reward, terminated, truncated, info = env.step(int(action))
        step += 1

        if terminated or truncated:
            obs, _ = env.reset()

        if step % 20000 == 0:
            print(f"  Collected {step}/{args.timesteps} timesteps", flush=True)

    env.close()

    obs_arr = np.array(obs_list, dtype=np.float32)
    action_arr = np.array(action_list, dtype=np.int64)
    mask_arr = np.array(mask_list, dtype=bool)

    output_path = args.output or str(root / "models" / f"expert_data_{args.seed}.npz")
    np.savez_compressed(output_path, obs=obs_arr, actions=action_arr, masks=mask_arr)

    obs_dim = obs_arr.shape[1]
    n_actions = mask_arr.shape[1]
    print(f"\nSaved {len(obs_list)} timesteps to {output_path}")
    print(f"  Obs dim: {obs_dim}, Actions: {n_actions}")
    print(f"  Action distribution: {np.bincount(action_arr, minlength=n_actions)}")
    print(f"  Unique obs shapes ok: {obs_arr.shape}")
    print(f"  File size: {Path(output_path).stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
