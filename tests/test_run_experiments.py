from __future__ import annotations

import argparse
import importlib.util
import unittest
from pathlib import Path

from scripts.run_experiments import (
    Candidate,
    aggregate_rows,
    evaluate_command,
    generate_tuning_candidates,
    parse_candidate,
    train_command,
)


class RunExperimentsTests(unittest.TestCase):
    def test_parse_candidate_requires_full_spec(self) -> None:
        candidate = parse_candidate("mixed_v7:maskable_ppo:mixed:347")

        self.assertEqual(candidate.name, "mixed_v7")
        self.assertEqual(candidate.algorithm, "maskable_ppo")
        self.assertEqual(candidate.opponent_policy, "mixed")
        self.assertEqual(candidate.seed, 347)
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_candidate("missing:parts")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_candidate("bad/name:ppo:mixed:1")

    def test_train_command_uses_candidate_algorithm_and_training_opponent(self) -> None:
        args = argparse.Namespace(
            timesteps=32,
            mechanics="rich",
            observation_mode="rich",
            n_envs=2,
            n_steps=16,
            batch_size=16,
            n_epochs=2,
            learning_rate=2.5e-4,
            ent_coef=0.01,
            target_kl=0.03,
        )
        candidate = Candidate("mixed_v7", "maskable_ppo", "mixed", 347)

        command = train_command(args, candidate, Path("results/experiments/x/models/mixed_v7.zip"))

        self.assertIn("--algorithm", command)
        self.assertIn("maskable_ppo", command)
        self.assertIn("--opponent-policy", command)
        self.assertIn("mixed", command)
        self.assertIn("--seed", command)
        self.assertIn("347", command)

    def test_evaluate_command_compares_current_and_candidate_models(self) -> None:
        args = argparse.Namespace(
            episodes=4,
            mechanics="rich",
            eval_opponent_policy="type_aware",
        )

        command = evaluate_command(
            args,
            Path("models/current.zip"),
            [Path("results/experiments/x/models/mixed_v7.zip")],
            42,
            Path("results/experiments/x/evaluation_seed42.csv"),
        )

        self.assertEqual(command.count("--model"), 2)
        self.assertIn("models/current.zip", command)
        self.assertIn("results/experiments/x/models/mixed_v7.zip", command)
        self.assertIn("42", command)

    def test_aggregate_rows_recommends_only_full_metric_winner(self) -> None:
        rows = [
            {
                "policy": "current",
                "seed": "42",
                "episodes": "100",
                "wins": "30",
                "losses": "40",
                "draws": "30",
                "win_rate": "0.30",
                "non_loss_rate": "0.60",
                "average_reward": "0.40",
                "average_turns": "8.0",
            },
            {
                "policy": "winner",
                "seed": "42",
                "episodes": "100",
                "wins": "35",
                "losses": "35",
                "draws": "30",
                "win_rate": "0.35",
                "non_loss_rate": "0.65",
                "average_reward": "0.50",
                "average_turns": "8.2",
            },
            {
                "policy": "win_rate_only",
                "seed": "42",
                "episodes": "100",
                "wins": "36",
                "losses": "45",
                "draws": "19",
                "win_rate": "0.36",
                "non_loss_rate": "0.55",
                "average_reward": "0.60",
                "average_turns": "7.9",
            },
        ]

        aggregates = aggregate_rows(rows, {"current", "winner", "win_rate_only"}, "current")
        recommendations = {item.policy: item.recommendation for item in aggregates}

        self.assertEqual(recommendations["current"], "baseline")
        self.assertEqual(recommendations["winner"], "promote")
        self.assertEqual(recommendations["win_rate_only"], "do-not-promote")

    def test_seed_gate_blocks_candidate_that_only_wins_one_seed(self) -> None:
        rows = [
            {
                "policy": "current",
                "seed": "42",
                "episodes": "100",
                "wins": "30",
                "losses": "40",
                "draws": "30",
                "average_reward": "0.40",
                "average_turns": "8.0",
            },
            {
                "policy": "current",
                "seed": "99",
                "episodes": "100",
                "wins": "30",
                "losses": "40",
                "draws": "30",
                "average_reward": "0.40",
                "average_turns": "8.0",
            },
            {
                "policy": "spiky",
                "seed": "42",
                "episodes": "100",
                "wins": "80",
                "losses": "10",
                "draws": "10",
                "average_reward": "1.20",
                "average_turns": "7.0",
            },
            {
                "policy": "spiky",
                "seed": "99",
                "episodes": "100",
                "wins": "10",
                "losses": "70",
                "draws": "20",
                "average_reward": "-0.20",
                "average_turns": "7.0",
            },
        ]

        aggregates = aggregate_rows(
            rows,
            {"current", "spiky"},
            "current",
            min_seed_pass_rate=0.70,
        )
        spiky = next(item for item in aggregates if item.policy == "spiky")

        self.assertEqual(spiky.seed_passes, 1)
        self.assertEqual(spiky.seed_count, 2)
        self.assertEqual(spiky.recommendation, "do-not-promote")

    def test_generate_tuning_candidates_uses_optuna_ranges(self) -> None:
        if importlib.util.find_spec("optuna") is None:
            self.skipTest("optional Optuna dependency is missing")
        args = argparse.Namespace(
            tune_trials=2,
            tune_seed=2026,
            tune_study_name="test-study",
            tune_base_name="trial",
            tune_algorithm="maskable_ppo",
            tune_opponent_policy="mixed",
            timesteps=2048,
            n_envs=8,
        )

        candidates = generate_tuning_candidates(args)

        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].name, "trial_trial_000")
        self.assertEqual(candidates[0].algorithm, "maskable_ppo")
        self.assertEqual(candidates[0].opponent_policy, "mixed")
        self.assertIsNotNone(candidates[0].learning_rate)
        self.assertIn(candidates[0].n_steps, {16, 32, 64, 128, 256})


if __name__ == "__main__":
    unittest.main()
