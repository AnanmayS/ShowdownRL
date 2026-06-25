from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.regenerate_benchmarks import ROOT, evaluate_command, write_csv, write_leaderboard


class BenchmarkScriptTests(unittest.TestCase):
    def test_evaluate_command_includes_standard_rich_benchmark_flags(self) -> None:
        command = evaluate_command(42, 2, ROOT / "results" / "evaluation_seed42.csv")
        joined = " ".join(command)

        self.assertIn("--mechanics rich", joined)
        self.assertIn("--opponent-policy type_aware", joined)
        self.assertIn("--model models/maskable_ppo_v11_conservative_3M.zip", joined)

    def test_writes_csv_and_leaderboard_with_metadata(self) -> None:
        rows = [
            {
                "scenario": "rich_type_aware_seed42",
                "policy": "maskable_ppo_v11_conservative_3M",
                "episodes": "10",
                "wins": "4",
                "losses": "3",
                "draws": "3",
                "win_rate": "0.4",
                "non_loss_rate": "0.7",
                "average_reward": "0.123",
                "average_turns": "7.5",
                "seed": "42",
                "mechanics": "rich",
                "opponent_policy": "type_aware",
                "observation_mode": "rich",
                "model_path": "models/maskable_ppo_v11_conservative_3M.zip",
                "git_sha": "abc123",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            csv_path = out_dir / "benchmarks.csv"
            leaderboard_path = out_dir / "leaderboard.md"

            write_csv(rows, csv_path)
            write_leaderboard(rows, leaderboard_path)

            self.assertIn("git_sha", csv_path.read_text(encoding="utf-8").splitlines()[0])
            leaderboard = leaderboard_path.read_text(encoding="utf-8")
            self.assertIn("Maskable PPO v11 (bench simulator)", leaderboard)
            self.assertIn("abc123", leaderboard)


if __name__ == "__main__":
    unittest.main()
