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

    def test_turn_state_boosts_finishing_moves(self) -> None:
        move = {"name": "Surf", "type": "Water", "category": "Special", "text": "Power 90"}

        healthy_score = score_move(move, {"opponent": {"hp_percent": 80}})
        finishing_score = score_move(move, {"opponent": {"hp_percent": 20}})

        self.assertGreater(finishing_score, healthy_score)

    def test_low_hp_recovery_beats_setup(self) -> None:
        turn_state = {"active": {"hp_percent": 25}, "opponent": {"hp_percent": 90}}
        moves = [
            {"name": "Swords Dance", "type": "Normal", "category": "Status", "text": "Swords Dance"},
            {"name": "Recover", "type": "Normal", "category": "Status", "text": "Recover"},
        ]

        best, _ = ranked_moves(moves, turn_state)[0]

        self.assertEqual(best["name"], "Recover")

    def test_status_move_penalized_when_opponent_already_has_status(self) -> None:
        move = {"name": "Thunder Wave", "type": "Electric", "category": "Status", "text": "Thunder Wave"}

        normal_score = score_move(move, {"opponent": {"status": ""}})
        redundant_score = score_move(move, {"opponent": {"status": "PAR"}})

        self.assertLess(redundant_score, normal_score)


if __name__ == "__main__":
    unittest.main()
