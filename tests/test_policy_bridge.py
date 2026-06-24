from __future__ import annotations

import sys
import tomllib
import unittest
from pathlib import Path

from showdownrl.policy_bridge import (
    RICH_MODEL_FILENAME,
    RICH_OBS_SIZE,
    PPOMovePolicy,
    ranked_switches,
    model_search_paths,
    turn_state_to_observation,
    turn_state_to_rich_observation,
    type_effectiveness,
)


class FakeModel:
    def __init__(self, action: int | Exception):
        self.action = action

    def predict(self, observation, deterministic=True):  # noqa: ANN001
        if isinstance(self.action, Exception):
            raise self.action
        return self.action, None


class PolicyBridgeTests(unittest.TestCase):
    def test_default_model_is_declared_as_install_data(self) -> None:
        pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        model_files = data["tool"]["setuptools"]["data-files"]["models"]

        self.assertIn(f"models/{RICH_MODEL_FILENAME}", model_files)

    def test_model_search_paths_include_installed_data_dir(self) -> None:
        paths = model_search_paths(RICH_MODEL_FILENAME)

        self.assertIn(Path(sys.prefix) / "models" / RICH_MODEL_FILENAME, paths)

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

    def test_ranked_switches_prefers_healthy_type_resist(self) -> None:
        choices = [
            {"index": 0, "name": "Scizor", "hp_percent": 35, "status": "BRN", "types": ["Bug", "Steel"]},
            {"index": 1, "name": "Gyarados", "hp_percent": 90, "status": "", "types": ["Water", "Flying"]},
        ]
        ranked = ranked_switches(choices, {"opponent": {"types": ["Fire"]}})

        self.assertEqual(ranked[0][0]["name"], "Gyarados")
        self.assertGreater(ranked[0][1], ranked[1][1])

    def test_ranked_switches_avoids_fainted_options(self) -> None:
        ranked = ranked_switches(
            [
                {"index": 0, "name": "Fainted mon", "text": "Fainted mon 0%"},
                {"index": 1, "name": "Backup", "hp_percent": 12},
            ]
        )

        self.assertEqual(ranked[0][0]["name"], "Backup")

    def test_rich_observation_adds_move_context(self) -> None:
        obs = turn_state_to_rich_observation(
            {
                "active": {"hp_percent": 25, "types": ["Water"]},
                "opponent": {"hp_percent": 20, "types": ["Fire"]},
            },
            [
                {"name": "Surf", "type": "Water", "category": "Special", "text": "Power 90 Accuracy 100"},
                {"name": "Recover", "type": "Normal", "category": "Status", "text": "Recover"},
            ],
        )

        self.assertEqual(len(obs), RICH_OBS_SIZE)
        self.assertEqual(obs[14 + 1], 1.0)  # STAB
        self.assertEqual(obs[14 + 2], 1.0)  # super effective
        self.assertEqual(obs[14 + 4], 1.0)  # expected damage can finish
        self.assertEqual(obs[14 + 8 + 5], 1.0)  # recovery flag on second move

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
