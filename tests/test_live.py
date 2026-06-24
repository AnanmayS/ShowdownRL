from __future__ import annotations

import unittest

from showdownrl.live import LiveOptions, debug_turn_snapshot, infer_result


class LiveResultTests(unittest.TestCase):
    def test_infers_result_without_storing_credentials(self) -> None:
        self.assertEqual(infer_result("aquerro won the battle!", "aquerro"), "win")
        self.assertEqual(infer_result("aquerro forfeited.", "aquerro"), "loss")
        self.assertEqual(infer_result("Opponent forfeited.", "aquerro"), "win")
        self.assertEqual(infer_result("The battle ended.", "aquerro"), "unknown")

    def test_debug_turn_snapshot_is_redacted(self) -> None:
        options = LiveOptions(
            username="aquerro",
            password="secret-password",
            guest=False,
            site="https://play.pokemonshowdown.com/",
            format_name="Random Battle",
        )
        snapshot = debug_turn_snapshot(
            options=options,
            battle_number=1,
            turn=2,
            turn_state={
                "active": {"name": "Pikachu", "hp_percent": 40},
                "opponent": {"name": "Gyarados", "hp_percent": 20},
                "available_moves": [{"name": "Thunderbolt"}],
                "switch_options": [],
                "battle_log_tail": ["Turn 2"],
            },
            ranked=[({"name": "Thunderbolt", "type": "Electric", "text": "Power 90"}, 3.2)],
        )

        self.assertEqual(snapshot["username_mode"], "account")
        self.assertNotIn("username", snapshot)
        self.assertNotIn("password", snapshot)
        self.assertNotIn("secret-password", repr(snapshot))


if __name__ == "__main__":
    unittest.main()
