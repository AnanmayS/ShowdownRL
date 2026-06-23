from __future__ import annotations

import unittest

from showdownrl.live import infer_result


class LiveResultTests(unittest.TestCase):
    def test_infers_result_without_storing_credentials(self) -> None:
        self.assertEqual(infer_result("aquerro won the battle!", "aquerro"), "win")
        self.assertEqual(infer_result("aquerro forfeited.", "aquerro"), "loss")
        self.assertEqual(infer_result("Opponent forfeited.", "aquerro"), "win")
        self.assertEqual(infer_result("The battle ended.", "aquerro"), "unknown")


if __name__ == "__main__":
    unittest.main()
