from __future__ import annotations

import unittest

from showdownrl.live import LiveOptions, SelectorHealth, debug_turn_snapshot, infer_result, selector_health_lines, switch_options_signature


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

    def test_switch_options_signature_is_stable_for_same_menu(self) -> None:
        options = [
            {"index": 0, "name": "Pikachu"},
            {"index": 1, "name": "Charizard"},
        ]

        self.assertEqual(
            switch_options_signature(options),
            switch_options_signature([{"text": "Pikachu"}, {"text": "Charizard"}]),
        )
        self.assertEqual(switch_options_signature([], fallback_count=2), ("count:2",))

    def test_selector_health_lines_mark_required_failures(self) -> None:
        lines = selector_health_lines(
            [
                SelectorHealth("choose-name button", True, True, 1),
                SelectorHealth("battle queue button", False, True, 0),
                SelectorHealth("move buttons", False, False, 0, "expected during a turn"),
            ]
        )

        self.assertIn("[ok] choose-name button (1 visible)", lines[0])
        self.assertIn("[fail] battle queue button (0 visible)", lines[1])
        self.assertIn("[info] move buttons (0 visible) - expected during a turn", lines[2])


if __name__ == "__main__":
    unittest.main()
