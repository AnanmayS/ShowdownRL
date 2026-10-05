from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

try:
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    from scripts.train_ppo import (
        LeaguePool,
        WinRateEvalCallback,
        build_ppo_kwargs,
        evaluate_win_rate,
        parse_net_arch,
        resolve_algorithm,
    )
    from showdownrl.significance import two_proportion_z_test, wilson_interval
    from showdownrl.simple_env import SimplePokemonMoveEnv
except ImportError as exc:  # pragma: no cover - depends on optional rl extras
    build_ppo_kwargs = None
    parse_net_arch = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(build_ppo_kwargs is None, f"optional RL dependencies are missing: {IMPORT_ERROR}")
class TrainPpoTests(unittest.TestCase):
    def test_parse_net_arch_requires_positive_ints(self) -> None:
        self.assertEqual(parse_net_arch("64,128"), [64, 128])
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_net_arch("64,nope")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_net_arch("0,64")

    def test_build_ppo_kwargs_exposes_tuned_training_controls(self) -> None:
        args = argparse.Namespace(
            learning_rate=2.5e-4,
            n_steps=256,
            batch_size=256,
            n_epochs=8,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.03,
            net_arch=[128, 128],
            ortho_init=False,
        )

        kwargs = build_ppo_kwargs(args)

        self.assertEqual(kwargs["n_steps"], 256)
        self.assertEqual(kwargs["batch_size"], 256)
        self.assertEqual(kwargs["ent_coef"], 0.01)
        self.assertEqual(kwargs["target_kl"], 0.03)
        self.assertEqual(kwargs["policy_kwargs"]["net_arch"], [128, 128])
        self.assertFalse(kwargs["policy_kwargs"]["ortho_init"])

    def test_resolve_algorithm_supports_plain_ppo(self) -> None:
        algorithm_class, uses_action_masks = resolve_algorithm("ppo")

        self.assertEqual(algorithm_class.__name__, "PPO")
        self.assertFalse(uses_action_masks)

    def test_eval_callback_selects_best_checkpoint_by_win_rate(self) -> None:
        class FakeLogger:
            def __init__(self):
                self.records = {}

            def record(self, key, value):
                self.records[key] = value

        class FakeModel:
            def __init__(self):
                self.logger = FakeLogger()
                self.saved_at = []
                self.num_timesteps = 0

            def save(self, path):
                self.saved_at.append(self.num_timesteps)

        def stats(wins: int, mean_reward: float) -> dict:
            low, high = wilson_interval(wins, 200)
            return {
                "episodes": 200,
                "wins": wins,
                "losses": 200 - wins,
                "draws": 0,
                "win_rate": wins / 200,
                "win_rate_ci_low": low,
                "win_rate_ci_high": high,
                "mean_reward": mean_reward,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            model = FakeModel()
            callback = WinRateEvalCallback(
                None,
                eval_freq=1,
                n_eval_episodes=200,
                best_model_save_path=Path(tmpdir),
                use_masks=False,
                verbose=0,
            )
            callback.model = model
            for timesteps, wins, mean_reward in [(1, 100, 0.5), (2, 120, 0.1), (3, 110, 0.9)]:
                model.num_timesteps = callback.num_timesteps = timesteps
                callback.record_evaluation(stats(wins, mean_reward))

            # Step 3 has the best mean reward but a lower win rate than step 2.
            self.assertEqual(model.saved_at, [1, 2])
            self.assertEqual(callback.best_evaluation["timesteps"], 2)
            self.assertAlmostEqual(model.logger.records["eval/win_rate"], 0.55)
            written = json.loads((Path(tmpdir) / "evaluations.json").read_text())
            self.assertEqual(written["best"]["wins"], 120)
            self.assertEqual(len(written["evaluations"]), 3)
            self.assertLess(written["best"]["win_rate_ci_low"], 0.6)
            self.assertGreater(written["best"]["win_rate_ci_high"], 0.6)

    def test_evaluate_win_rate_counts_env_outcomes(self) -> None:
        class FirstLegalActionModel:
            def predict(self, obs, state=None, episode_start=None, deterministic=True, action_masks=None):
                return action_masks.argmax(axis=1), state

        env = DummyVecEnv([lambda: Monitor(SimplePokemonMoveEnv(seed=3, mechanics="rich"))])

        stats = evaluate_win_rate(FirstLegalActionModel(), env, 25, use_masks=True, seed=3)

        self.assertEqual(stats["wins"] + stats["losses"] + stats["draws"], 25)
        self.assertLessEqual(stats["win_rate_ci_low"], stats["win_rate"])
        self.assertGreaterEqual(stats["win_rate_ci_high"], stats["win_rate"])
        # Same seed, same deterministic policy -> identical evaluation.
        self.assertEqual(stats, evaluate_win_rate(FirstLegalActionModel(), env, 25, use_masks=True, seed=3))

    def test_significance_helpers(self) -> None:
        low, high = wilson_interval(50, 100)
        self.assertAlmostEqual(low, 0.4038, places=3)
        self.assertAlmostEqual(high, 0.5962, places=3)
        _, p_small = two_proportion_z_test(35, 100, 30, 100)
        _, p_large = two_proportion_z_test(350, 1000, 300, 1000)
        self.assertGreater(p_small, 0.05)
        self.assertLess(p_large, 0.05)
        self.assertEqual(two_proportion_z_test(0, 10, 0, 10), (0.0, 1.0))

    def test_league_pool_temperature_zero_picks_newest_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = LeaguePool(Path(tmpdir), max_size=5)
            for step in [100, 300, 200]:
                (Path(tmpdir) / f"league_step_{step}.zip").touch()

            self.assertEqual(pool.sample(temperature=0).name, "league_step_300.zip")


if __name__ == "__main__":
    unittest.main()
