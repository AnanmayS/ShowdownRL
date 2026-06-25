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


def chart_label(policy: str) -> str:
    labels = {
        "maskable_ppo_v11_conservative_3M": "Maskable PPO v11",
        "type_aware_policy": "Type aware",
        "max_damage_policy": "Max damage",
        "random_policy": "Random",
    }
    return labels.get(policy, policy_label(policy))


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
    import matplotlib.patheffects as path_effects

    ordered = aggregate_stats(stats)[:6]
    ordered = list(reversed(ordered))
    labels = [chart_label(item.policy) for item in ordered]
    win_rates = [item.win_rate * 100 for item in ordered]
    non_loss_rates = [item.non_loss_rate * 100 for item in ordered]
    rewards = [item.average_reward for item in ordered]
    turns = [item.average_turns for item in ordered]

    fig = plt.figure(figsize=(15.5, 8.8), facecolor="#08111f")
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.72, 1.0],
        left=0.15,
        right=0.965,
        top=0.79,
        bottom=0.12,
        wspace=0.08,
    )
    ax = fig.add_subplot(grid[0, 0])
    panel = fig.add_subplot(grid[0, 1])

    fig.text(
        0.07,
        0.935,
        "ShowdownRL bench simulator benchmark",
        color="#f7fbff",
        fontsize=28,
        fontweight="bold",
    )
    fig.text(
        0.071,
        0.89,
        "1,000 episodes per seed | rich mechanics | type-aware opponent | aggregate over seeds 42 and 99",
        color="#96a6ba",
        fontsize=12,
    )
    fig.text(
        0.071,
        0.852,
        "V11 is the new default: 7-action MaskablePPO with bench switching and 106 observation features.",
        color="#d8e6f3",
        fontsize=12,
    )

    for axis in (ax, panel):
        axis.set_facecolor("#0d1b2d")
        for spine in axis.spines.values():
            spine.set_visible(False)

    y_positions = list(range(len(ordered)))
    bar_height = 0.34
    win_color = "#45d2c9"
    non_loss_color = "#f8b84e"
    muted_text = "#aebdd0"
    strong_text = "#edf6ff"
    grid_color = "#22344e"

    ax.barh(
        [pos + bar_height / 2 for pos in y_positions],
        win_rates,
        height=bar_height,
        color=win_color,
        alpha=0.92,
        label="Win rate",
    )
    ax.barh(
        [pos - bar_height / 2 for pos in y_positions],
        non_loss_rates,
        height=bar_height,
        color=non_loss_color,
        alpha=0.92,
        label="Non-loss",
    )
    ax.set_xlim(0, 90)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, color=strong_text, fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", colors=muted_text, labelsize=10)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=grid_color, linewidth=1.0, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("Rate (%)", color=muted_text, fontsize=11)
    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=11,
        labelcolor=strong_text,
        handlelength=1.6,
    )

    for pos, win_rate, non_loss_rate, item in zip(y_positions, win_rates, non_loss_rates, ordered):
        is_default = item.policy == "maskable_ppo_v11_conservative_3M"
        label_color = "#ffffff" if is_default else "#dbe7f2"
        glow = [
            path_effects.Stroke(linewidth=3.5, foreground="#08111f"),
            path_effects.Normal(),
        ]
        ax.text(
            win_rate + 1.1,
            pos + bar_height / 2,
            f"{win_rate:.1f}%",
            va="center",
            ha="left",
            color=label_color,
            fontsize=11,
            fontweight="bold" if is_default else "normal",
            path_effects=glow,
        )
        ax.text(
            non_loss_rate + 1.1,
            pos - bar_height / 2,
            f"{non_loss_rate:.1f}%",
            va="center",
            ha="left",
            color=label_color,
            fontsize=11,
            fontweight="bold" if is_default else "normal",
            path_effects=glow,
        )
        if is_default:
            ax.text(
                2,
                pos + 0.57,
                "DEFAULT",
                color="#07111f",
                fontsize=9,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.28,rounding_size=0.14",
                    "facecolor": "#e9f871",
                    "edgecolor": "#e9f871",
                },
            )

    panel.set_xticks([])
    panel.set_yticks([])
    panel.text(
        0.07,
        0.93,
        "Aggregate scoreboard",
        transform=panel.transAxes,
        color=strong_text,
        fontsize=17,
        fontweight="bold",
    )
    panel.text(
        0.07,
        0.875,
        "Record is wins-losses-draws across 2,000 episodes.",
        transform=panel.transAxes,
        color=muted_text,
        fontsize=10,
    )

    row_top = 0.78
    row_gap = 0.145
    for index, item in enumerate(reversed(ordered)):
        y = row_top - index * row_gap
        is_default = item.policy == "maskable_ppo_v11_conservative_3M"
        accent = "#e9f871" if is_default else "#344966"
        panel.text(
            0.07,
            y,
            chart_label(item.policy),
            transform=panel.transAxes,
            color=strong_text,
            fontsize=11,
            fontweight="bold",
        )
        panel.plot(
            [0.07, 0.93],
            [y - 0.024, y - 0.024],
            transform=panel.transAxes,
            color=accent,
            linewidth=2.2 if is_default else 0.8,
            alpha=0.95 if is_default else 0.45,
        )
        panel.text(
            0.07,
            y - 0.065,
            f"{item.wins}-{item.losses}-{item.draws}",
            transform=panel.transAxes,
            color="#f7fbff",
            fontsize=15,
            fontweight="bold",
        )
        panel.text(
            0.48,
            y - 0.058,
            f"reward {signed(item.average_reward)}",
            transform=panel.transAxes,
            color="#b8c7d9",
            fontsize=10,
        )
        panel.text(
            0.73,
            y - 0.058,
            f"{item.average_turns:.1f} turns",
            transform=panel.transAxes,
            color="#b8c7d9",
            fontsize=10,
        )

    best = max(aggregate_stats(stats), key=lambda item: item.win_rate)
    type_aware = next(item for item in aggregate_stats(stats) if item.policy == "type_aware_policy")
    delta = (best.win_rate - type_aware.win_rate) * 100
    fig.text(
        0.77,
        0.852,
        f"+{delta:.1f} pts over Type aware",
        color="#08111f",
        fontsize=13,
        fontweight="bold",
        ha="center",
        bbox={
            "boxstyle": "round,pad=0.5,rounding_size=0.18",
            "facecolor": "#e9f871",
            "edgecolor": "#e9f871",
        },
    )

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART_PATH, dpi=180, facecolor=fig.get_facecolor())
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
