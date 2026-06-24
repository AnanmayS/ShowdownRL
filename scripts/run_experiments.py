#!/usr/bin/env python
"""Train, evaluate, and rank candidate RL policies against the current default."""

from __future__ import annotations

import argparse
import csv
import math
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
    learning_rate: float | None = None
    ent_coef: float | None = None
    n_steps: int | None = None
    batch_size: int | None = None
    n_epochs: int | None = None
    target_kl: float | None = None


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
    seed_count: int
    seed_passes: int
    seed_pass_rate: float
    win_rate_stderr: float
    non_loss_rate_stderr: float
    average_reward_stderr: float
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


def generate_tuning_candidates(args: argparse.Namespace) -> list[Candidate]:
    if args.tune_trials <= 0:
        return []
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "Optuna tuning requires optuna. Install RL dependencies with `pip install -e '.[rl]'`."
        ) from exc

    sampler = optuna.samplers.TPESampler(seed=args.tune_seed)
    study = optuna.create_study(
        study_name=args.tune_study_name,
        direction="maximize",
        sampler=sampler,
    )
    n_step_choices = [
        steps
        for steps in [16, 32, 64, 128, 256, 512]
        if steps * args.n_envs <= args.timesteps
    ]
    if not n_step_choices:
        n_step_choices = [max(1, args.timesteps // args.n_envs)]
    candidates: list[Candidate] = []
    for index in range(args.tune_trials):
        trial = study.ask()
        n_steps = trial.suggest_categorical("n_steps", n_step_choices)
        rollout_size = n_steps * args.n_envs
        batch_choices = [
            batch
            for batch in [16, 32, 64, 128, 256, 512]
            if batch <= rollout_size and rollout_size % batch == 0
        ]
        if not batch_choices:
            batch_choices = [rollout_size]
        batch_size = trial.suggest_categorical("batch_size", batch_choices)
        candidates.append(
            Candidate(
                name=f"{args.tune_base_name}_trial_{index:03d}",
                algorithm=args.tune_algorithm,
                opponent_policy=args.tune_opponent_policy,
                seed=args.tune_seed + index,
                learning_rate=trial.suggest_float("learning_rate", 5e-5, 8e-4, log=True),
                ent_coef=trial.suggest_float("ent_coef", 0.0, 0.04),
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=trial.suggest_int("n_epochs", 4, 12),
                target_kl=trial.suggest_float("target_kl", 0.015, 0.06),
            )
        )
    return candidates


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
        "--min-seed-pass-rate",
        type=float,
        default=0.70,
        help="Fraction of evaluation seeds where a candidate must beat the baseline on all gate metrics.",
    )
    parser.add_argument(
        "--tune-trials",
        type=int,
        default=0,
        help="Generate this many Optuna-sampled candidate hyperparameter trials.",
    )
    parser.add_argument("--tune-seed", type=int, default=2026)
    parser.add_argument("--tune-base-name", default="optuna_v7")
    parser.add_argument("--tune-study-name", default="showdownrl")
    parser.add_argument(
        "--tune-algorithm",
        choices=sorted(ALGORITHMS),
        default="maskable_ppo",
    )
    parser.add_argument(
        "--tune-opponent-policy",
        choices=sorted(OPPONENT_POLICIES),
        default="mixed",
    )
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


def candidate_value(candidate_value: object | None, default_value: object) -> str:
    return str(default_value if candidate_value is None else candidate_value)


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
        candidate_value(candidate.n_steps, args.n_steps),
        "--batch-size",
        candidate_value(candidate.batch_size, args.batch_size),
        "--n-epochs",
        candidate_value(candidate.n_epochs, args.n_epochs),
        "--learning-rate",
        candidate_value(candidate.learning_rate, args.learning_rate),
        "--ent-coef",
        candidate_value(candidate.ent_coef, args.ent_coef),
        "--target-kl",
        candidate_value(candidate.target_kl, args.target_kl),
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


def seed_key(row: dict[str, str], fallback: int) -> str:
    return row.get("seed") or row.get("scenario") or str(fallback)


def standard_error(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance) / math.sqrt(len(values))


def policy_seed_metrics(policy_rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for index, row in enumerate(policy_rows):
        key = seed_key(row, index)
        episodes = int(row["episodes"])
        wins = int(row["wins"])
        draws = int(row["draws"])
        metrics[key] = {
            "episodes": float(episodes),
            "win_rate": wins / episodes,
            "non_loss_rate": (wins + draws) / episodes,
            "average_reward": float(row["average_reward"]),
        }
    return metrics


def aggregate_rows(
    rows: list[dict[str, str]],
    policies: set[str],
    baseline_policy: str,
    *,
    min_seed_pass_rate: float = 0.70,
) -> list[Aggregate]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        policy = row["policy"]
        if policy in policies:
            grouped.setdefault(policy, []).append(row)

    if baseline_policy not in grouped:
        raise SystemExit(f"Baseline policy {baseline_policy!r} was not present in evaluation output.")

    aggregates: list[Aggregate] = []
    seed_metrics = {policy: policy_seed_metrics(policy_rows) for policy, policy_rows in grouped.items()}
    for policy, policy_rows in grouped.items():
        episodes = sum(int(row["episodes"]) for row in policy_rows)
        wins = sum(int(row["wins"]) for row in policy_rows)
        losses = sum(int(row["losses"]) for row in policy_rows)
        draws = sum(int(row["draws"]) for row in policy_rows)
        win_rate = wins / episodes
        non_loss_rate = (wins + draws) / episodes
        average_reward = (
            sum(float(row["average_reward"]) * int(row["episodes"]) for row in policy_rows) / episodes
        )
        average_turns = (
            sum(float(row["average_turns"]) * int(row["episodes"]) for row in policy_rows) / episodes
        )
        policy_metrics = seed_metrics[policy]
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
                seed_count=len(policy_metrics),
                seed_passes=len(policy_metrics) if policy == baseline_policy else 0,
                seed_pass_rate=1.0 if policy == baseline_policy else 0.0,
                win_rate_stderr=standard_error([item["win_rate"] for item in policy_metrics.values()]),
                non_loss_rate_stderr=standard_error(
                    [item["non_loss_rate"] for item in policy_metrics.values()]
                ),
                average_reward_stderr=standard_error(
                    [item["average_reward"] for item in policy_metrics.values()]
                ),
                recommendation="baseline" if policy == baseline_policy else "",
            )
        )

    baseline = next(item for item in aggregates if item.policy == baseline_policy)
    baseline_seed_metrics = seed_metrics[baseline_policy]
    ranked: list[Aggregate] = []
    for item in aggregates:
        if item.policy == baseline_policy:
            ranked.append(item)
            continue
        candidate_seed_metrics = seed_metrics[item.policy]
        comparable_seeds = sorted(set(candidate_seed_metrics) & set(baseline_seed_metrics))
        seed_passes = sum(
            1
            for seed in comparable_seeds
            if candidate_seed_metrics[seed]["win_rate"] > baseline_seed_metrics[seed]["win_rate"]
            and candidate_seed_metrics[seed]["non_loss_rate"]
            > baseline_seed_metrics[seed]["non_loss_rate"]
            and candidate_seed_metrics[seed]["average_reward"]
            > baseline_seed_metrics[seed]["average_reward"]
        )
        seed_pass_rate = seed_passes / len(comparable_seeds) if comparable_seeds else 0.0
        recommendation = (
            "promote"
            if item.win_rate > baseline.win_rate
            and item.non_loss_rate > baseline.non_loss_rate
            and item.average_reward > baseline.average_reward
            and seed_pass_rate >= min_seed_pass_rate
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
                seed_count=len(comparable_seeds),
                seed_passes=seed_passes,
                seed_pass_rate=seed_pass_rate,
                win_rate_stderr=item.win_rate_stderr,
                non_loss_rate_stderr=item.non_loss_rate_stderr,
                average_reward_stderr=item.average_reward_stderr,
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
                "seed_count",
                "seed_passes",
                "seed_pass_rate",
                "win_rate_stderr",
                "non_loss_rate_stderr",
                "average_reward_stderr",
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
                    "seed_count": item.seed_count,
                    "seed_passes": item.seed_passes,
                    "seed_pass_rate": f"{item.seed_pass_rate:.6f}",
                    "win_rate_stderr": f"{item.win_rate_stderr:.6f}",
                    "non_loss_rate_stderr": f"{item.non_loss_rate_stderr:.6f}",
                    "average_reward_stderr": f"{item.average_reward_stderr:.6f}",
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
        "| Name | Algorithm | Train opponent | Train seed | LR | Entropy | Steps | Batch | Epochs | Target KL |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in candidates:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{candidate.name}`",
                    f"`{candidate.algorithm}`",
                    f"`{candidate.opponent_policy}`",
                    str(candidate.seed),
                    f"{candidate.learning_rate:.6g}" if candidate.learning_rate is not None else "default",
                    f"{candidate.ent_coef:.4g}" if candidate.ent_coef is not None else "default",
                    str(candidate.n_steps) if candidate.n_steps is not None else "default",
                    str(candidate.batch_size) if candidate.batch_size is not None else "default",
                    str(candidate.n_epochs) if candidate.n_epochs is not None else "default",
                    f"{candidate.target_kl:.4g}" if candidate.target_kl is not None else "default",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Policy | Episodes | Record (W-L-D) | Win rate | Non-loss | Avg reward | Seed pass | Recommendation |",
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
                    f"{item.seed_passes}/{item.seed_count} ({pct(item.seed_pass_rate)})",
                    item.recommendation,
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Across-Seed Uncertainty",
            "",
            "| Policy | Win rate SE | Non-loss SE | Avg reward SE |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for item in sorted(aggregates, key=lambda entry: entry.win_rate, reverse=True):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item.policy}`",
                    pct(item.win_rate_stderr),
                    pct(item.non_loss_rate_stderr),
                    signed(item.average_reward_stderr),
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
            + ". These beat the current default on aggregate win rate, non-loss rate, average reward, "
            + f"and at least {pct(args.min_seed_pass_rate)} of comparable evaluation seeds."
        )
    else:
        lines.append(
            "No candidate beat the current default on aggregate win rate, non-loss rate, "
            f"average reward, and at least {pct(args.min_seed_pass_rate)} of comparable evaluation seeds."
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.min_seed_pass_rate <= 1.0:
        raise SystemExit("--min-seed-pass-rate must be between 0 and 1.")
    if args.tune_trials < 0:
        raise SystemExit("--tune-trials must be zero or greater.")

    candidates = args.candidate or [parse_candidate(spec) for spec in DEFAULT_CANDIDATE_SPECS]
    candidates = [*candidates, *generate_tuning_candidates(args)]
    if not candidates:
        raise SystemExit("No candidates were provided or generated.")

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
    aggregates = aggregate_rows(
        rows,
        policies,
        current_model.stem,
        min_seed_pass_rate=args.min_seed_pass_rate,
    )
    summary_path = output_dir / "experiment_summary.csv"
    report_path = output_dir / "report.md"
    write_summary_csv(aggregates, summary_path)
    write_report(args, candidates, current_model, aggregates, report_path)
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
