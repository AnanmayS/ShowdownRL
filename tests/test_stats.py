from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from showdownrl.stats import (
    append_battle_record,
    filter_records,
    load_battle_records,
    parse_since,
    summarize_records,
    terminal_summary,
    trend_summary,
    write_html_report,
)


class StatsTests(unittest.TestCase):
    def test_summary_handles_results_moves_and_corrupt_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stats_dir = Path(tmp)
            append_battle_record(
                {
                    "started_at": "2026-06-23T10:00:00+00:00",
                    "result": "win",
                    "turns": 4,
                    "forced_switches": 1,
                    "end_rating": 1200,
                    "format": "Random Battle",
                    "selected_moves": [{"name": "Thunderbolt"}, {"name": "Thunderbolt"}],
                },
                stats_dir,
            )
            append_battle_record(
                {
                    "started_at": "2026-06-24T10:00:00+00:00",
                    "result": "loss",
                    "turns": 8,
                    "forced_switches": 0,
                    "end_rating": 1185,
                    "format": "Random Battle",
                    "selected_moves": [{"name": "Surf"}],
                },
                stats_dir,
            )
            with (stats_dir / "battles.jsonl").open("a", encoding="utf-8") as f:
                f.write("{bad json\n")

            records, corrupt = load_battle_records(stats_dir)
            summary = summarize_records(records)

            self.assertEqual(corrupt, 1)
            self.assertEqual(summary["total"], 2)
            self.assertEqual(summary["wins"], 1)
            self.assertEqual(summary["losses"], 1)
            self.assertEqual(summary["win_rate"], 0.5)
            self.assertEqual(summary["average_turns"], 6.0)
            self.assertEqual(summary["forced_switches"], 1)
            self.assertEqual(summary["current_rating"], 1185)
            self.assertEqual(summary["rating_delta"], -15)
            self.assertEqual(summary["streak_result"], "loss")
            self.assertEqual(summary["streak_count"], 1)
            self.assertEqual(summary["most_used_moves"][0], ("Thunderbolt", 2))

            text = terminal_summary(records, corrupt_count=corrupt, stats_dir=stats_dir)
            self.assertIn("Total battles: 2", text)
            self.assertIn("Current rating: 1185 (-15)", text)
            self.assertIn("Skipped corrupt log lines: 1", text)

            trend = trend_summary(records)
            self.assertIn("2026-06-23: 1-0", trend)
            self.assertIn("2026-06-24: 0-1", trend)

    def test_filters_by_since_and_format(self) -> None:
        records = [
            {"started_at": "2026-06-20T10:00:00+00:00", "format": "Random Battle"},
            {"started_at": "2026-06-24T10:00:00+00:00", "format": "OU"},
            {"started_at": "2026-06-25T10:00:00+00:00", "format": "Random Battle"},
        ]

        filtered = filter_records(records, since=parse_since("2026-06-23"), format_name="random")

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["started_at"], "2026-06-25T10:00:00+00:00")

    def test_html_report_excludes_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stats_dir = Path(tmp)
            records = [
                {
                    "started_at": "2026-06-23T10:00:00+00:00",
                    "result": "win",
                    "turns": 5,
                    "format": "Random Battle",
                    "selected_moves": [{"name": "Flamethrower"}],
                    "password": "should-not-appear",
                }
            ]

            report = write_html_report(records, stats_dir=stats_dir)
            html = report.read_text(encoding="utf-8")

            self.assertIn("ShowdownRL Stats", html)
            self.assertIn("Flamethrower", html)
            self.assertNotIn("should-not-appear", html)
            self.assertTrue(report.exists())


if __name__ == "__main__":
    unittest.main()
