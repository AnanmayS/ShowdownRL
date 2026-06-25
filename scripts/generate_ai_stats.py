#!/usr/bin/env python
"""Generate docs and chart assets from the published benchmark CSV."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
ASSETS_DIR = DOCS_DIR / "assets"
BENCHMARKS_DIR = DOCS_DIR / "benchmarks"
REPORT_PATH = DOCS_DIR / "ai_stats.md"
CHART_PATH = ASSETS_DIR / "ai_policy_comparison.png"
PUBLISHED_CSV = BENCHMARKS_DIR / "current_evaluation.csv"


@dataclass(frozen=True)
class PolicyStats:
    scenario: str
    policy: str
    episodes: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    non_loss_rate: float
    average_reward: float
    average_turns: float
    git_sha: str


@dataclass(frozen=True)
class AggregateStats:
    policy: str
    episodes: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    non_loss_rate: float
    average_reward: float
    average_turns: float


def policy_label(policy: str) -> str:
    labels = {
        "random_policy": "Random",
        "max_damage_policy": "Max damage",
        "type_aware_policy": "Type aware",
        "ppo_move_selection_v2_typed": "PPO v2 typed",
        "ppo_move_selection_v3_rich": "PPO v3 rich",
        "ppo_move_selection_v4_rich": "PPO v4 rich",
        "ppo_move_selection_v5_rich_finetuned": "PPO v5 fine-tuned",
        "maskable_ppo_move_selection_v6_rich": "Maskable PPO v6",
        "maskable_ppo_v7_selfplay_1M": "Maskable PPO v7 self-play",
        "maskable_ppo_v8_typeaware_1M": "Maskable PPO v8 type-aware",
        "maskable_ppo_v9_3M": "Maskable PPO v9",
        "maskable_ppo_v10_conservative": "Maskable PPO v10 conservative",
        "maskable_ppo_v11_conservative_3M": "Maskable PPO v11 (bench simulator)",
        "maskable_ppo_v12_selfplay": "Maskable PPO v12 self-play",
    }
    return labels.get(policy, policy.replace("_", " ").title())


def scenario_label(scenario: str) -> str:
    if scenario == "rich_type_aware_seed42":
        return "Rich/type-aware seed 42"
    if scenario == "rich_type_aware_seed99":
        return "Rich/type-aware seed 99"
    return scenario.replace("_", " ")


def load_stats() -> list[PolicyStats]:
    if not PUBLISHED_CSV.exists():
        raise SystemExit(
            f"Missing {PUBLISHED_CSV}. Run: python scripts/regenerate_benchmarks.py"
        )

    with PUBLISHED_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    stats: list[PolicyStats] = []
    for row in rows:
        stats.append(
            PolicyStats(
                scenario=row["scenario"],
                policy=row["policy"],
                episodes=int(row["episodes"]),
                wins=int(row["wins"]),
                losses=int(row["losses"]),
                draws=int(row["draws"]),
                win_rate=float(row["win_rate"]),
                non_loss_rate=float(row["non_loss_rate"]),
                average_reward=float(row["average_reward"]),
                average_turns=float(row["average_turns"]),
                git_sha=row.get("git_sha") or "unknown",
            )
        )
    return stats


def aggregate_stats(stats: list[PolicyStats]) -> list[AggregateStats]:
    grouped: dict[str, list[PolicyStats]] = {}
    for item in stats:
        grouped.setdefault(item.policy, []).append(item)

    aggregates: list[AggregateStats] = []
    for policy, rows in grouped.items():
        episodes = sum(row.episodes for row in rows)
        wins = sum(row.wins for row in rows)
        losses = sum(row.losses for row in rows)
        draws = sum(row.draws for row in rows)
        aggregates.append(
            AggregateStats(
                policy=policy,
                episodes=episodes,
                wins=wins,
                losses=losses,
                draws=draws,
                win_rate=wins / episodes,
                non_loss_rate=(wins + draws) / episodes,
                average_reward=sum(row.average_reward * row.episodes for row in rows) / episodes,
                average_turns=sum(row.average_turns * row.episodes for row in rows) / episodes,
            )
        )
    return sorted(aggregates, key=lambda item: item.win_rate, reverse=True)


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def format_table(stats: list[PolicyStats]) -> str:
    rows = [
        "| Scenario | Policy | Episodes | Record (W-L-D) | Win rate | Non-loss | Avg reward | Avg turns |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    ordered = sorted(stats, key=lambda item: (item.scenario, -item.win_rate))
    for item in ordered:
        rows.append(
            "| "
            + " | ".join(
                [
                    scenario_label(item.scenario),
                    policy_label(item.policy),
                    str(item.episodes),
                    f"{item.wins}-{item.losses}-{item.draws}",
                    pct(item.win_rate),
                    pct(item.non_loss_rate),
                    signed(item.average_reward),
                    f"{item.average_turns:.2f}",
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def make_chart(stats: list[PolicyStats]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = aggregate_stats(stats)[:8]
    labels = [policy_label(item.policy) for item in ordered]
    win_rates = [item.win_rate * 100 for item in ordered]
    non_loss = [item.non_loss_rate * 100 for item in ordered]
    rewards = [item.average_reward for item in ordered]

    colors = [
        "#2f6f73",
        "#d28b36",
        "#6d5a9c",
        "#61737f",
        "#b0574f",
        "#507c55",
        "#8a6f3d",
        "#486b9a",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    fig.patch.set_facecolor("#f8faf9")

    for ax in axes:
        ax.set_facecolor("#f8faf9")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#d7dedb", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)

    bars = axes[0].bar(labels, win_rates, color=colors, edgecolor="#f8faf9")
    axes[0].set_title("Win rate")
    axes[0].set_ylim(0, max(65, max(win_rates) + 8))
    axes[0].set_ylabel("Percent")
    for bar, value in zip(bars, win_rates):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 1.3, f"{value:.1f}%", ha="center", fontsize=8)

    bars = axes[1].bar(labels, non_loss, color=colors, edgecolor="#f8faf9")
    axes[1].set_title("Non-loss rate")
    axes[1].set_ylim(0, max(65, max(non_loss) + 8))
    axes[1].set_ylabel("Percent")
    for bar, value in zip(bars, non_loss):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 1.3, f"{value:.1f}%", ha="center", fontsize=8)

    bars = axes[2].bar(labels, rewards, color=colors, edgecolor="#f8faf9")
    axes[2].set_title("Average reward")
    axes[2].axhline(0, color="#6b7280", linewidth=0.9)
    for bar, value in zip(bars, rewards):
        y = value + 0.03 if value >= 0 else value - 0.05
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            y,
            signed(value),
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )

    for ax in axes:
        ax.tick_params(axis="x", rotation=25, labelsize=8)

    fig.suptitle("ShowdownRL corrected-simulator benchmark", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART_PATH, dpi=170)
    plt.close(fig)


def build_report(stats: list[PolicyStats]) -> str:
    aggregates = aggregate_stats(stats)
    best = aggregates[0]
    default = next(
        item for item in aggregates if item.policy == "maskable_ppo_v11_conservative_3M"
    )
    csv_date = datetime.fromtimestamp(PUBLISHED_CSV.stat().st_mtime).strftime("%Y-%m-%d")
    git_shas = sorted({item.git_sha for item in stats})

    return f"""# AI Stats

This report summarizes the published benchmark in
`docs/benchmarks/current_evaluation.csv`, generated from `SimplePokemonMoveEnv`.

![ShowdownRL policy benchmark](assets/ai_policy_comparison.png)

## Current Result

The default trained bench-simulator policy is `maskable_ppo_v11_conservative_3M.zip`.
Across the fixed rich/type-aware evaluation seeds it recorded
{default.wins}-{default.losses}-{default.draws} (W-L-D) over {default.episodes} episodes:
{pct(default.win_rate)} win rate, {pct(default.non_loss_rate)} non-loss rate,
and {signed(default.average_reward)} average reward.

The strongest policy in the corrected simulator benchmark is
**{policy_label(best.policy)}** at {pct(best.win_rate)} win rate.

## Benchmark Table

{format_table(stats)}

## What These Stats Mean

- **Win rate** is the clearest headline metric.
- **Non-loss rate** counts wins plus simulator draws.
- **Average reward** captures how decisively the policy wins or loses in the
  simulator's reward function.
- **Average turns** is useful as an efficiency signal, but lower is only better
  when win rate and reward remain strong.

## Reproducing

```bash
pip install -e ".[rl]"
python scripts/regenerate_benchmarks.py
python scripts/generate_ai_stats.py
```

Benchmark Git SHA(s): `{", ".join(git_shas)}`
Evaluation CSV timestamp: `{csv_date}`

These numbers benchmark the experimental simulator policy. The live browser
player can use the trained model with `showdownrl live --policy ppo`, with a
heuristic fallback if the checkpoint is missing or selects an unavailable move.
Live ladder performance should still be tracked separately because real
Pokemon Showdown battles include mechanics beyond this move-selection simulator.
"""


def main() -> None:
    stats = load_stats()
    make_chart(stats)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(stats), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {CHART_PATH}")


if __name__ == "__main__":
    main()
