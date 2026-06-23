#!/usr/bin/env python
"""
Smoke test for ShowdownRL.

Verifies:
    1. SimplePokemonMoveEnv can run a full random episode.
    2. PPO from stable_baselines3 can be instantiated and trained
       for 256 timesteps.

Prints PASS if everything works.
"""

import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from showdownrl.simple_env import SimplePokemonMoveEnv


def main():
    print("Smoke testing ShowdownRL...")

    # --- 1. Random episode ---
    print("  1. Running random episode...")
    env = SimplePokemonMoveEnv(seed=42)
    obs, info = env.reset()
    assert obs.shape == (14,), f"Expected obs shape (14,), got {obs.shape}"
    assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"

    done = False
    total_reward = 0.0
    steps = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    print(f"     Episode finished: {steps} steps, total_reward={total_reward:.3f}")
    env.close()

    # --- 2. PPO instantiation + brief training ---
    print("  2. Training PPO for 256 timesteps...")
    env = SimplePokemonMoveEnv(seed=42)
    model = PPO("MlpPolicy", env, verbose=0, seed=42)
    model.learn(total_timesteps=256)
    env.close()

    print("\nPASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFAIL: {e}")
        sys.exit(1)
