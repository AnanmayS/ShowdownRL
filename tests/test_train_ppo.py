from __future__ import annotations

import argparse
import unittest

try:
    from scripts.train_ppo import build_ppo_kwargs, parse_net_arch
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


if __name__ == "__main__":
    unittest.main()
