#!/usr/bin/env python
"""Train, evaluate, and rank candidate RL policies against the current default."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "scripts" / "train_ppo.py"
EVALUATE = ROOT / "scripts" / "evaluate_model.py"
DEFAULT_EVAL_SEEDS = [42, 99]
DEFAULT_CANDIDATE_SPECS = [
    "maskable_mixed_v7:maskable_ppo:mixed:347",
    "maskable_type_aware_v7:maskable_ppo:type_aware:348",
]
ALGORITHMS = {"ppo", "maskable_ppo"}
OPPONENT_POLICIES = {"random", "max_damage", "type_aware", "mixed"}


@dataclass(frozen=True)
class Candidate:
    name: str
    algorithm: str
    opponent_policy: str
    seed: int


@dataclass(frozen=True)
class Aggregate:
    policy: str
    episodes: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    non_loss_rate: float
    average_reward: float
    average_turns: float
    recommendation: str


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_candidate(value: str) -> Candidate:
    parts = value.split(":")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "candidate must use name:algorithm:opponent_policy:seed"
        )
    name, algorithm, opponent_policy, seed_raw = parts
    if not name or any(char in name for char in "/\\:"):
        raise argparse.ArgumentTypeError("candidate name must be a simple file-safe name")
    if algorithm not in ALGORITHMS:
        raise argparse.ArgumentTypeError(f"algorithm must be one of {sorted(ALGORITHMS)}")
    if opponent_policy not in OPPONENT_POLICIES:
        raise argparse.ArgumentTypeError(
            f"opponent_policy must be one of {sorted(OPPONENT_POLICIES)}"
        )
    try:
        seed = int(seed_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("candidate seed must be an integer") from exc
    return Candidate(name=name, algorithm=algorithm, opponent_policy=opponent_policy, seed=seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train candidate RL policies and rank them against the current default."
    )
    parser.add_argument(
        "--candidate",
        action="append",
        type=parse_candidate,
        help=(
            "Candidate as name:algorithm:opponent_policy:seed. "
            "Repeatable. Defaults to two v7 MaskablePPO candidates."
        ),
    )
    parser.add_argument("--timesteps", type=positive_int, default=100_000)
    parser.add_argument("--episodes", type=positive_int, default=500)
    parser.add_argument("--eval-seed", action="append", type=int, dest="eval_seeds")
    parser.add_argument(
        "--eval-opponent-policy",
        choices=sorted(OPPONENT_POLICIES),
        default="type_aware",
        help="Opponent policy for candidate comparison.",
    )
    parser.add_argument(
        "--mechanics",
        choices=["toy", "typed", "rich"],
        default="rich",
        help="Training and evaluation mechanics.",
    )
    parser.add_argument(
        "--observation-mode",
        choices=["simple", "rich"],
        default="rich",
        help="Candidate training observation mode. Evaluation uses auto-detected model shape.",
    )
    parser.add_argument("--n-envs", type=positive_int, default=8)
    parser.add_argument("--n-steps", type=positive_int, default=256)
    parser.add_argument("--batch-size", type=positive_int, default=256)
    parser.add_argument("--n-epochs", type=positive_int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument(
        "--current-model",
        type=Path,
        help="Default model to compare against. Defaults to showdownrl.policy_bridge.default_model_path().",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Experiment output directory. Defaults to results/experiments/<timestamp>.",
    )
    parser.add_argument("--skip-training", action="store_true", help="Evaluate existing candidate zips.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def experiment_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return ROOT / "results" / "experiments" / stamp


def current_model_path(args: argparse.Namespace) -> Path:
    if args.current_model:
        return args.current_model if args.current_model.is_absolute() else ROOT / args.current_model
    sys.path.insert(0, str(ROOT))
    from showdownrl.policy_bridge import default_model_path

    return default_model_path()


def candidate_model_path(output_dir: Path, candidate: Candidate) -> Path:
    return output_dir / "models" / f"{candidate.name}.zip"


def train_command(args: argparse.Namespace, candidate: Candidate, model_path: Path) -> list[str]:
    return [
        sys.executable,
        str(TRAIN),
        "--algorithm",
        candidate.algorithm,
        "--timesteps",
        str(args.timesteps),
        "--seed",
        str(candidate.seed),
        "--mechanics",
        args.mechanics,
        "--observation-mode",
        args.observation_mode,
        "--opponent-policy",
        candidate.opponent_policy,
        "--output",
        relative(model_path),
        "--n-envs",
        str(args.n_envs),
        "--n-steps",
        str(args.n_steps),
        "--batch-size",
        str(args.batch_size),
        "--n-epochs",
        str(args.n_epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--ent-coef",
        str(args.ent_coef),
        "--target-kl",
        str(args.target_kl),
    ]


def evaluate_command(
    args: argparse.Namespace,
    current_model: Path,
    candidate_paths: list[Path],
    seed: int,
    output_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(EVALUATE),
        "--episodes",
        str(args.episodes),
        "--seed",
        str(seed),
        "--mechanics",
        args.mechanics,
        "--observation-mode",
        "auto",
        "--opponent-policy",
        args.eval_opponent_policy,
        "--model",
        relative(current_model),
    ]
    for model_path in candidate_paths:
        command.extend(["--model", relative(model_path)])
    command.extend(["--output", relative(output_path)])
    return command


def run(command: list[str], *, dry_run: bool) -> None:
    printable = " ".join(command)
    if dry_run:
        print(printable)
        return
    print(printable, flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def aggregate_rows(rows: list[dict[str, str]], policies: set[str], baseline_policy: str) -> list[Aggregate]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        policy = row["policy"]
        if policy in policies:
            grouped.setdefault(policy, []).append(row)

    if baseline_policy not in grouped:
        raise SystemExit(f"Baseline policy {baseline_policy!r} was not present in evaluation output.")

    aggregates: list[Aggregate] = []
    for policy, policy_rows in grouped.items():
        episodes = sum(int(row["episodes"]) for row in policy_rows)
        wins = sum(int(row["wins"]) for row in policy_rows)
        losses = sum(int(row["losses"]) for row in policy_rows)
        draws = sum(int(row["draws"]) for row in policy_rows)
        win_rate = wins / episodes
        non_loss_rate = (wins + draws) / episodes
        average_reward = sum(float(row["average_reward"]) for row in policy_rows) / len(policy_rows)
        average_turns = sum(float(row["average_turns"]) for row in policy_rows) / len(policy_rows)
        aggregates.append(
            Aggregate(
                policy=policy,
                episodes=episodes,
                wins=wins,
                losses=losses,
                draws=draws,
                win_rate=win_rate,
                non_loss_rate=non_loss_rate,
                average_reward=average_reward,
                average_turns=average_turns,
                recommendation="baseline" if policy == baseline_policy else "",
            )
        )

    baseline = next(item for item in aggregates if item.policy == baseline_policy)
    ranked: list[Aggregate] = []
    for item in aggregates:
        if item.policy == baseline_policy:
            ranked.append(item)
            continue
        recommendation = (
            "promote"
            if item.win_rate > baseline.win_rate
            and item.non_loss_rate > baseline.non_loss_rate
            and item.average_reward > baseline.average_reward
            else "do-not-promote"
        )
        ranked.append(
            Aggregate(
                policy=item.policy,
                episodes=item.episodes,
                wins=item.wins,
                losses=item.losses,
                draws=item.draws,
                win_rate=item.win_rate,
                non_loss_rate=item.non_loss_rate,
                average_reward=item.average_reward,
                average_turns=item.average_turns,
                recommendation=recommendation,
            )
        )
    return sorted(ranked, key=lambda item: (item.recommendation != "baseline", -item.win_rate))


def write_summary_csv(aggregates: list[Aggregate], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "policy",
                "episodes",
                "wins",
                "losses",
                "draws",
                "win_rate",
                "non_loss_rate",
                "average_reward",
                "average_turns",
                "recommendation",
            ],
        )
        writer.writeheader()
        for item in aggregates:
            writer.writerow(
                {
                    "policy": item.policy,
                    "episodes": item.episodes,
                    "wins": item.wins,
                    "losses": item.losses,
                    "draws": item.draws,
                    "win_rate": f"{item.win_rate:.6f}",
                    "non_loss_rate": f"{item.non_loss_rate:.6f}",
                    "average_reward": f"{item.average_reward:.6f}",
                    "average_turns": f"{item.average_turns:.3f}",
                    "recommendation": item.recommendation,
                }
            )


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def signed(value: float) -> str:
    return f"{value:+.3f}"


def write_report(
    args: argparse.Namespace,
    candidates: list[Candidate],
    current_model: Path,
    aggregates: list[Aggregate],
    path: Path,
) -> None:
    lines = [
        "# RL Experiment Report",
        "",
        f"Current model: `{relative(current_model)}`",
        f"Timesteps per candidate: `{args.timesteps}`",
        f"Episodes per evaluation seed: `{args.episodes}`",
        f"Evaluation opponent: `{args.eval_opponent_policy}`",
        "",
        "## Candidates",
        "",
        "| Name | Algorithm | Train opponent | Train seed |",
        "| --- | --- | --- | ---: |",
    ]
    for candidate in candidates:
        lines.append(
            f"| `{candidate.name}` | `{candidate.algorithm}` | `{candidate.opponent_policy}` | {candidate.seed} |"
        )

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Policy | Episodes | Record (W-L-D) | Win rate | Non-loss | Avg reward | Avg turns | Recommendation |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in sorted(aggregates, key=lambda entry: entry.win_rate, reverse=True):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item.policy}`",
                    str(item.episodes),
                    f"{item.wins}-{item.losses}-{item.draws}",
                    pct(item.win_rate),
                    pct(item.non_loss_rate),
                    signed(item.average_reward),
                    f"{item.average_turns:.2f}",
                    item.recommendation,
                ]
            )
            + " |"
        )

    promotable = [item.policy for item in aggregates if item.recommendation == "promote"]
    lines.extend(["", "## Promotion Gate", ""])
    if promotable:
        lines.append(
            "Promotion candidate(s): "
            + ", ".join(f"`{policy}`" for policy in promotable)
            + ". These beat the current default on win rate, non-loss rate, and average reward."
        )
    else:
        lines.append(
            "No candidate beat the current default on win rate, non-loss rate, and average reward."
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    candidates = args.candidate or [parse_candidate(spec) for spec in DEFAULT_CANDIDATE_SPECS]
    output_dir = experiment_dir(args)
    model_paths = [candidate_model_path(output_dir, candidate) for candidate in candidates]
    current_model = current_model_path(args)
    eval_seeds = args.eval_seeds or DEFAULT_EVAL_SEEDS

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)

    if not current_model.exists():
        raise SystemExit(f"Current model does not exist: {current_model}")

    if not args.skip_training:
        for candidate, model_path in zip(candidates, model_paths):
            run(train_command(args, candidate, model_path), dry_run=args.dry_run)

    eval_paths: list[Path] = []
    for seed in eval_seeds:
        output_path = output_dir / f"evaluation_seed{seed}.csv"
        eval_paths.append(output_path)
        run(evaluate_command(args, current_model, model_paths, seed, output_path), dry_run=args.dry_run)

    if args.dry_run:
        return 0

    rows = load_rows(eval_paths)
    policies = {current_model.stem, *(path.stem for path in model_paths)}
    aggregates = aggregate_rows(rows, policies, current_model.stem)
    summary_path = output_dir / "experiment_summary.csv"
    report_path = output_dir / "report.md"
    write_summary_csv(aggregates, summary_path)
    write_report(args, candidates, current_model, aggregates, report_path)
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
