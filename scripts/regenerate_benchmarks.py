#!/usr/bin/env python
"""Regenerate the published benchmark CSV and leaderboard docs."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EVALUATE = ROOT / "scripts" / "evaluate_model.py"
PUBLISHED_CSV = ROOT / "docs" / "benchmarks" / "current_evaluation.csv"
LEADERBOARD = ROOT / "docs" / "model_leaderboard.md"
DEFAULT_MODELS = [
    ROOT / "models" / "ppo_move_selection_v2_typed.zip",
    ROOT / "models" / "ppo_move_selection_v3_rich.zip",
    ROOT / "models" / "ppo_move_selection_v4_rich.zip",
]
DEFAULT_SEEDS = [42, 99]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate ShowdownRL benchmark docs.")
    parser.add_argument("--episodes", type=int, default=1000, help="Episodes per policy and seed.")
    parser.add_argument("--seed", action="append", type=int, dest="seeds", help="Evaluation seed. Repeatable.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running RL evaluation.")
    return parser.parse_args()


def evaluate_command(seed: int, episodes: int, output: Path) -> list[str]:
    command = [
        sys.executable,
        str(EVALUATE),
        "--episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--mechanics",
        "rich",
        "--observation-mode",
        "auto",
        "--opponent-policy",
        "type_aware",
    ]
    for model in DEFAULT_MODELS:
        command.extend(["--model", str(model.relative_to(ROOT))])
    command.extend(["--output", str(output.relative_to(ROOT))])
    return command


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "policy",
        "episodes",
        "wins",
        "losses",
        "draws",
        "win_rate",
        "non_loss_rate",
        "average_reward",
        "average_turns",
        "seed",
        "mechanics",
        "opponent_policy",
        "observation_mode",
        "model_path",
        "git_sha",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pct(value: str) -> str:
    return f"{float(value) * 100:.1f}%"


def signed(value: str) -> str:
    return f"{float(value):+.3f}"


def policy_label(policy: str) -> str:
    labels = {
        "random_policy": "Random",
        "max_damage_policy": "Max damage",
        "type_aware_policy": "Type aware",
        "ppo_move_selection_v2_typed": "PPO v2 typed",
        "ppo_move_selection_v3_rich": "PPO v3 rich",
        "ppo_move_selection_v4_rich": "PPO v4 rich",
    }
    return labels.get(policy, policy.replace("_", " ").title())


def write_leaderboard(rows: list[dict[str, str]], path: Path) -> None:
    ordered = sorted(rows, key=lambda row: (row["scenario"], -float(row["win_rate"])))
    lines = [
        "# Model Leaderboard",
        "",
        "Generated from `docs/benchmarks/current_evaluation.csv`.",
        "",
        "| Scenario | Policy | Episodes | Record | Win rate | Non-loss | Avg reward | Avg turns | Git SHA |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in ordered:
        record = f"{row['wins']}-{row['losses']}-{row['draws']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["scenario"],
                    policy_label(row["policy"]),
                    row["episodes"],
                    record,
                    pct(row["win_rate"]),
                    pct(row["non_loss_rate"]),
                    signed(row["average_reward"]),
                    f"{float(row['average_turns']):.2f}",
                    row.get("git_sha") or "unknown",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("Full benchmark refresh:")
    lines.append("")
    lines.append("```bash")
    lines.append("pip install -e \".[rl]\"")
    lines.append("python scripts/regenerate_benchmarks.py")
    lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    seeds = args.seeds or DEFAULT_SEEDS
    output_paths = [ROOT / "results" / f"evaluation_seed{seed}.csv" for seed in seeds]
    commands = [evaluate_command(seed, args.episodes, output) for seed, output in zip(seeds, output_paths)]

    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return 0

    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)

    rows = load_rows(output_paths)
    write_csv(rows, PUBLISHED_CSV)
    write_leaderboard(rows, LEADERBOARD)
    print(f"Wrote {PUBLISHED_CSV}")
    print(f"Wrote {LEADERBOARD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
