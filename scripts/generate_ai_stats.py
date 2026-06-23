#!/usr/bin/env python
"""
Generate a GitHub-friendly AI benchmark report from results/evaluation.csv.

Outputs:
    docs/ai_stats.md
    docs/benchmarks/current_evaluation.csv
    docs/assets/ai_policy_comparison.png

Usage:
    python scripts/generate_ai_stats.py
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = ROOT / "results" / "evaluation.csv"
MODEL_PATH = ROOT / "models" / "ppo_move_selection_v1.zip"
DOCS_DIR = ROOT / "docs"
ASSETS_DIR = DOCS_DIR / "assets"
BENCHMARKS_DIR = DOCS_DIR / "benchmarks"
REPORT_PATH = DOCS_DIR / "ai_stats.md"
CHART_PATH = ASSETS_DIR / "ai_policy_comparison.png"
PUBLISHED_CSV = BENCHMARKS_DIR / "current_evaluation.csv"


@dataclass(frozen=True)
class PolicyStats:
    policy: str
    episodes: int
    wins: int
    losses: int
    win_rate: float
    average_reward: float
    average_turns: float

    @property
    def reward_per_turn(self) -> float:
        if self.average_turns <= 0:
            return 0.0
        return self.average_reward / self.average_turns


def policy_label(policy: str) -> str:
    labels = {
        "random_policy": "Random",
        "max_damage_policy": "Max damage",
        "type_aware_policy": "Type aware",
        "ppo_policy": "Trained PPO",
    }
    return labels.get(policy, policy.replace("_", " ").title())


def load_stats() -> list[PolicyStats]:
    if not RESULTS_CSV.exists():
        raise SystemExit(
            f"Missing {RESULTS_CSV}. Run: python scripts/evaluate_model.py --episodes 50"
        )

    with RESULTS_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    stats: list[PolicyStats] = []
    for row in rows:
        stats.append(
            PolicyStats(
                policy=row["policy"],
                episodes=int(row["episodes"]),
                wins=int(row["wins"]),
                losses=int(row["losses"]),
                win_rate=float(row["win_rate"]),
                average_reward=float(row["average_reward"]),
                average_turns=float(row["average_turns"]),
            )
        )
    return stats


def pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def format_table(stats: list[PolicyStats]) -> str:
    rows = [
        "| Policy | Episodes | Record | Win rate | Avg reward | Avg turns | Reward / turn |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    ordered = sorted(stats, key=lambda item: item.win_rate, reverse=True)
    for item in ordered:
        rows.append(
            "| "
            + " | ".join(
                [
                    policy_label(item.policy),
                    str(item.episodes),
                    f"{item.wins}-{item.losses}",
                    pct(item.win_rate),
                    signed(item.average_reward),
                    f"{item.average_turns:.2f}",
                    signed(item.reward_per_turn, 4),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def make_chart(stats: list[PolicyStats]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(stats, key=lambda item: item.win_rate, reverse=True)
    labels = [policy_label(item.policy) for item in ordered]
    win_rates = [item.win_rate * 100 for item in ordered]
    rewards = [item.average_reward for item in ordered]
    turns = [item.average_turns for item in ordered]

    colors = ["#2f6f73", "#d28b36", "#6d5a9c", "#61737f"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    fig.patch.set_facecolor("#f8faf9")

    for ax in axes:
        ax.set_facecolor("#f8faf9")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#d7dedb", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)

    bars = axes[0].bar(labels, win_rates, color=colors, edgecolor="#f8faf9")
    axes[0].set_title("Win rate")
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Percent")
    for bar, value in zip(bars, win_rates):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.0f}%",
            ha="center",
            fontsize=9,
        )

    bars = axes[1].bar(labels, rewards, color=colors, edgecolor="#f8faf9")
    axes[1].set_title("Average reward")
    axes[1].axhline(0, color="#6b7280", linewidth=0.9)
    for bar, value in zip(bars, rewards):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.035,
            signed(value),
            ha="center",
            fontsize=9,
        )

    bars = axes[2].bar(labels, turns, color=colors, edgecolor="#f8faf9")
    axes[2].set_title("Average turns")
    axes[2].set_ylabel("Turns")
    for bar, value in zip(bars, turns):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.15,
            f"{value:.2f}",
            ha="center",
            fontsize=9,
        )

    for ax in axes:
        ax.tick_params(axis="x", rotation=25)

    fig.suptitle("ShowdownRL policy benchmark", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART_PATH, dpi=170)
    plt.close(fig)


def build_report(stats: list[PolicyStats]) -> str:
    best = max(stats, key=lambda item: item.win_rate)
    ppo = next((item for item in stats if item.policy == "ppo_policy"), None)
    model_note = "not found"
    if MODEL_PATH.exists():
        mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        model_note = f"{MODEL_PATH.name} ({mb:.1f} MB, local artifact)"

    csv_date = datetime.fromtimestamp(RESULTS_CSV.stat().st_mtime).strftime("%Y-%m-%d")
    ppo_note = (
        f"The trained PPO policy won {ppo.wins} of {ppo.episodes} episodes "
        f"({pct(ppo.win_rate)}) with an average reward of {ppo.average_reward:.3f}."
        if ppo
        else "No PPO row was present in the current evaluation CSV."
    )

    return f"""# AI Stats

This report summarizes the published benchmark in
`docs/benchmarks/current_evaluation.csv`, generated from
`results/evaluation.csv` and `SimplePokemonMoveEnv`.

![ShowdownRL policy benchmark](assets/ai_policy_comparison.png)

## Current Result

{ppo_note}

The strongest policy in this run was **{policy_label(best.policy)}** at
{pct(best.win_rate)} win rate. That means the trained model is available and
working, but the simple hand-written type-aware baseline is still the bar to
beat in this simulator.

## Benchmark Table

{format_table(stats)}

## What These Stats Mean

- **Win rate** is the clearest headline metric.
- **Average reward** captures how decisively the policy wins or loses in the
  simulator's reward function.
- **Average turns** is useful as an efficiency signal, but lower is only better
  when win rate and reward remain strong.
- **Reward / turn** is a compact tie-breaker for policies with similar win
  rates.

## Reproducing

```bash
pip install -e ".[rl]"
python scripts/evaluate_model.py --episodes 50
python scripts/generate_ai_stats.py
```

Model artifact: `{model_note}`  
Evaluation CSV timestamp: `{csv_date}`

These numbers benchmark the experimental simulator policy. The live browser
player currently uses a lightweight move-scoring policy for the official
Pokemon Showdown website, so live ladder performance should be tracked
separately once battle logging is added.
"""


def main() -> None:
    stats = load_stats()
    make_chart(stats)
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    PUBLISHED_CSV.write_text(RESULTS_CSV.read_text(encoding="utf-8"), encoding="utf-8")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(stats), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {PUBLISHED_CSV}")
    print(f"Wrote {CHART_PATH}")


if __name__ == "__main__":
    main()
