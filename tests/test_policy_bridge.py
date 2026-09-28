from __future__ import annotations

import os
import sys
import tempfile
import time
import tomllib
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from gymnasium import spaces

from showdownrl import policy_bridge
from showdownrl.battle_features import N_ACTIONS, OBS_SIZE
from showdownrl.policy_bridge import (
    MASKABLE_MODEL_FILENAME,
    RICH_MODEL_FILENAME,
    LivePolicy,
    PolicyLoadError,
    check_model_compatible,
    model_search_paths,
)


class _Model:
    def __init__(self, obs_size: int, n_actions: int):
        self.observation_space = spaces.Box(-1.0, 1.0, (obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(n_actions)


class PolicyBridgeTests(unittest.TestCase):
    def test_default_model_is_declared_as_install_data(self) -> None:
        pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        model_files = data["tool"]["setuptools"]["data-files"]["models"]

        self.assertIn(f"models/{RICH_MODEL_FILENAME}", model_files)
        self.assertIn(f"models/{MASKABLE_MODEL_FILENAME}", model_files)

    def test_model_search_paths_include_installed_data_dir(self) -> None:
        paths = model_search_paths(RICH_MODEL_FILENAME)

        self.assertIn(Path(sys.prefix) / "models" / RICH_MODEL_FILENAME, paths)

    def test_compatible_model_accepted(self) -> None:
        check_model_compatible(_Model(OBS_SIZE, N_ACTIONS))

    def test_legacy_observation_sizes_are_rejected_not_silently_used(self) -> None:
        for obs_size in (14, 46, 106):
            with self.assertRaisesRegex(PolicyLoadError, "OBS_SIZE"):
                check_model_compatible(_Model(obs_size, 4))

    def test_wrong_action_space_rejected(self) -> None:
        with self.assertRaisesRegex(PolicyLoadError, "actions"):
            check_model_compatible(_Model(OBS_SIZE, 4))

    def test_missing_model_raises_load_error(self) -> None:
        with self.assertRaises(PolicyLoadError):
            LivePolicy(Path("/nonexistent/model.zip"))

    def test_no_real_model_raises_load_error(self) -> None:
        with mock.patch.object(policy_bridge, "default_live_model_path", return_value=None):
            with self.assertRaisesRegex(PolicyLoadError, "models/real"):
                LivePolicy()

    def test_default_live_model_is_newest_real_zip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "models" / "real"
            real.mkdir(parents=True)
            old, new = real / "a.zip", real / "b.zip"
            old.write_bytes(b"")
            new.write_bytes(b"")
            now = time.time()
            os.utime(old, (now - 100, now - 100))
            os.utime(new, (now, now))
            with mock.patch.object(policy_bridge, "model_search_paths",
                                   return_value=[Path(tmp) / "models" / "real"]):
                self.assertEqual(policy_bridge.default_live_model_path(), new)


if __name__ == "__main__":
    unittest.main()
