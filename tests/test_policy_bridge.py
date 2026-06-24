from __future__ import annotations

import unittest

from showdownrl.policy_bridge import PPOMovePolicy, turn_state_to_observation, type_effectiveness


class FakeModel:
    def __init__(self, action: int | Exception):
        self.action = action

    def predict(self, observation, deterministic=True):  # noqa: ANN001
        if isinstance(self.action, Exception):
            raise self.action
        return self.action, None


class PolicyBridgeTests(unittest.TestCase):
    def test_turn_state_to_observation_projects_live_payload(self) -> None:
        obs = turn_state_to_observation(
            {
                "active": {"hp_percent": 75},
                "opponent": {"hp_percent": 20, "types": ["Water", "Flying"]},
            },
            [
                {"name": "Thunderbolt", "type": "Electric", "text": "Power 90 Accuracy 100"},
                {"name": "Recover", "type": "Normal", "category": "Status", "text": "Recover"},
            ],
        )

        self.assertEqual(len(obs), 14)
        self.assertEqual(obs[0], 0.75)
        self.assertEqual(obs[1], 0.2)
        self.assertEqual(obs[2], 0.9)
        self.assertEqual(obs[3], 1.0)
        self.assertEqual(obs[4], 4.0)
        self.assertEqual(obs[5], 0.0)

    def test_type_effectiveness_prefers_showdown_button_text(self) -> None:
        self.assertEqual(type_effectiveness({"type": "Ground", "text": "Doesn't affect the target"}, {}), 0.0)
        self.assertEqual(type_effectiveness({"type": "Fire", "text": "Super effective"}, {}), 2.0)

    def test_ppo_policy_selects_predicted_available_move(self) -> None:
        policy = PPOMovePolicy(model=FakeModel(1))
        moves = [
            {"index": 0, "name": "Tackle", "type": "Normal", "text": "Power 40"},
            {"index": 1, "name": "Surf", "type": "Water", "text": "Power 90"},
        ]

        choice = policy.choose(moves, {"active": {"hp_percent": 100}, "opponent": {"hp_percent": 100}})

        self.assertEqual(choice.source, "ppo")
        self.assertEqual(choice.ranked[0][0]["name"], "Surf")

    def test_ppo_policy_falls_back_on_unavailable_action(self) -> None:
        policy = PPOMovePolicy(model=FakeModel(3))
        moves = [{"index": 0, "name": "Surf", "type": "Water", "text": "Power 90"}]

        choice = policy.choose(moves, {})

        self.assertEqual(choice.source, "heuristic")
        self.assertIn("unavailable action", choice.fallback_reason)
        self.assertEqual(choice.ranked[0][0]["name"], "Surf")


if __name__ == "__main__":
    unittest.main()
