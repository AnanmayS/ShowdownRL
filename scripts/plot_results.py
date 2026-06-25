#!/usr/bin/env python
"""
Generate evaluation plots from results/evaluation.csv.

Produces:
    results/win_rate_by_policy.png
    results/average_reward_by_policy.png
    results/average_turns_by_policy.png

Usage:
    python scripts/plot_results.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd


def short_label(policy: str) -> str:
    """Return a compact display label for a policy."""
    return {
        "random_policy": "Random",
        "max_damage_policy": "Max Damage",
        "type_aware_policy": "Type Aware",
        "maskable_ppo_v11_conservative_3M": "Maskable PPO v11",
        "maskable_ppo_v12_selfplay": "Maskable PPO v12",
    }.get(policy, policy.replace("_", " ").title())


def main():
    results_dir = Path(__file__).resolve().parent.parent / "results"
    csv_path = results_dir / "evaluation.csv"

    if not csv_path.exists():
        print(f"ERROR: {csv_path} does not exist.")
        print("Run evaluate_model.py first:")
        print("  python scripts/evaluate_model.py")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    policies = df["policy"].tolist()
    labels = [short_label(p) for p in policies]

    # --- Win Rate ---
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, df["win_rate"], color="steelblue", edgecolor="white")
    ax.set_title("Win Rate by Policy", fontsize=14)
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    for bar, val in zip(bars, df["win_rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.1%}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    fig.savefig(results_dir / "win_rate_by_policy.png", dpi=150)
    plt.close(fig)
    print(f"Saved {results_dir / 'win_rate_by_policy.png'}")

    # --- Average Reward ---
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, df["average_reward"], color="darkorange", edgecolor="white")
    ax.set_title("Average Reward by Policy", fontsize=14)
    ax.set_ylabel("Average Reward")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    for bar, val in zip(bars, df["average_reward"]):
        y_pos = bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.08
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{val:.3f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=10)
    plt.tight_layout()
    fig.savefig(results_dir / "average_reward_by_policy.png", dpi=150)
    plt.close(fig)
    print(f"Saved {results_dir / 'average_reward_by_policy.png'}")

    # --- Average Turns ---
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, df["average_turns"], color="seagreen", edgecolor="white")
    ax.set_title("Average Turns by Policy", fontsize=14)
    ax.set_ylabel("Average Turns")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    for bar, val in zip(bars, df["average_turns"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    fig.savefig(results_dir / "average_turns_by_policy.png", dpi=150)
    plt.close(fig)
    print(f"Saved {results_dir / 'average_turns_by_policy.png'}")

    print("\nAll plots generated.")


if __name__ == "__main__":
    main()
