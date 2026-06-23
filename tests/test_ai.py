from __future__ import annotations

import unittest

from showdownrl.ai import ranked_moves, score_move


class AiPolicyTests(unittest.TestCase):
    def test_attacking_move_beats_status_move(self) -> None:
        moves = [
            {"name": "Swords Dance", "type": "Normal", "category": "Status", "text": "Swords Dance"},
            {"name": "Flamethrower", "type": "Fire", "category": "Special", "text": "Flamethrower Power 90"},
        ]

        best, score = ranked_moves(moves)[0]

        self.assertEqual(best["name"], "Flamethrower")
        self.assertGreater(score, score_move(moves[0]))

    def test_disabled_and_immune_moves_are_avoided(self) -> None:
        self.assertLess(score_move({"name": "Earthquake", "text": "Doesn't affect the target"}), 0)
        self.assertLess(score_move({"name": "Surf", "text": "Disabled"}), 0)

    def test_stab_and_power_raise_score(self) -> None:
        neutral = score_move({"name": "Tackle", "type": "Normal", "text": "Power 40"})
        stab = score_move(
            {
                "name": "Thunderbolt",
                "type": "Electric",
                "category": "Special",
                "text": "Power 90",
                "active_types": ["Electric"],
            }
        )

        self.assertGreater(stab, neutral)


if __name__ == "__main__":
    unittest.main()
