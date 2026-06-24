from __future__ import annotations

import unittest

try:
    from showdownrl.simple_env import RICH_OBS_SIZE
    from showdownrl.simple_env import SimplePokemonMoveEnv
except ImportError as exc:  # pragma: no cover - depends on optional rl extras
    SimplePokemonMoveEnv = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(SimplePokemonMoveEnv is None, f"optional RL dependencies are missing: {IMPORT_ERROR}")
class SimpleEnvTests(unittest.TestCase):
    def test_type_aware_opponent_policy_selects_best_expected_damage(self) -> None:
        env = SimplePokemonMoveEnv(opponent_policy="type_aware", seed=1)
        env.moves = [
            (0.4, 1.0, 1.0),
            (0.5, 1.0, 2.0),
            (1.0, 0.5, 1.0),
            (0.3, 1.0, 1.0),
        ]

        self.assertEqual(env._opponent_action(), 1)

    def test_typed_mechanics_samples_types_and_damage_multipliers(self) -> None:
        env = SimplePokemonMoveEnv(mechanics="typed", seed=2)
        obs, info = env.reset()

        self.assertEqual(info["mechanics"], "typed")
        self.assertTrue(info["own_types"])
        self.assertTrue(info["opponent_types"])
        self.assertEqual(len(env.moves), 4)
        self.assertTrue(any(obs[2 + index * 3 + 2] != 1.0 for index in range(4)))

    def test_rich_observation_includes_support_move_flags(self) -> None:
        for seed in range(1, 20):
            env = SimplePokemonMoveEnv(mechanics="rich", observation_mode="rich", seed=seed)
            obs, info = env.reset()
            if any(
                obs[14 + index * 8 + 5] or obs[14 + index * 8 + 6] or obs[14 + index * 8 + 7]
                for index in range(4)
            ):
                self.assertEqual(info["observation_mode"], "rich")
                self.assertEqual(len(obs), RICH_OBS_SIZE)
                return
        self.fail("rich mechanics did not generate a support move in deterministic sample")


if __name__ == "__main__":
    unittest.main()
