#!/usr/bin/env python
"""Train a MaskablePPO agent on the real Showdown simulator (gen9randombattle).

Pipeline:
    collect  -> record (obs, mask, action, outcome) from a teacher heuristic
    bc       -> behaviour-clone the teacher into a MaskablePPO policy (+ value head)
    ppo      -> fine-tune with PPO against a mixed opponent pool / self-play league

Examples:
    python scripts/train_real.py collect --battles 20000 --out models/real/bc_data.npz
    python scripts/train_real.py bc --data models/real/bc_data.npz --out models/real/bc.zip
    python scripts/train_real.py ppo --init models/real/bc.zip --steps 3000000 --out models/real/ppo
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_env.environment import SinglesEnv  # noqa: E402
from poke_env.player import Player  # noqa: E402
from poke_env.ps_client import AccountConfiguration  # noqa: E402

from showdownrl.battle_features import N_ACTIONS, OBS_SIZE, embed_battle  # noqa: E402
from showdownrl.real_env import (  # noqa: E402
    BATTLE_FORMAT,
    MaskedShowdownEnv,
    MixedOpponent,
    parse_pool,
    scripted_chooser,
    server_configuration,
)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Data collection


class RecordingPlayer(Player):
    def __init__(self, teacher: str, epsilon: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.teacher = scripted_chooser(teacher)
        self.epsilon = epsilon
        self.records: dict[str, list] = {}

    def choose_move(self, battle):
        order = self.teacher(battle)
        mask = np.array(SinglesEnv.get_action_mask(battle), dtype=np.int8)
        action = int(SinglesEnv.order_to_action(order, battle, strict=False))
        if 0 <= action < N_ACTIONS and mask[action]:
            self.records.setdefault(battle.battle_tag, []).append(
                (embed_battle(battle), mask, action))
        if self.epsilon and random.random() < self.epsilon:
            return self.choose_random_singles_move(battle)
        return order


async def _collect_worker_async(teacher: str, opponents: list[str], battles: int, epsilon: float,
                                port: int, worker: int) -> dict:
    obs, masks, actions, outcomes, steps_left = [], [], [], [], []
    per_opp = max(1, battles // len(opponents))
    wins = total = 0
    for i, opp_name in enumerate(opponents):
        tag = f"{worker}x{i}x{random.randrange(10**5)}"
        rec = RecordingPlayer(
            teacher, epsilon=epsilon, battle_format=BATTLE_FORMAT,
            server_configuration=server_configuration(port), max_concurrent_battles=16,
            account_configuration=AccountConfiguration(f"rec{tag}", None), log_level=40)
        opp = MixedOpponent(
            parse_pool(f"{opp_name}:1"), battle_format=BATTLE_FORMAT,
            server_configuration=server_configuration(port), max_concurrent_battles=16,
            account_configuration=AccountConfiguration(f"opp{tag}", None), log_level=40)
        await rec.battle_against(opp, n_battles=per_opp)
        wins += rec.n_won_battles
        total += rec.n_finished_battles
        for btag, battle in rec.battles.items():
            traj = rec.records.get(btag, [])
            outcome = 1.0 if battle.won else -1.0
            for t, (o, m, a) in enumerate(traj):
                obs.append(o)
                masks.append(m)
                actions.append(a)
                outcomes.append(outcome)
                steps_left.append(len(traj) - t - 1)
        for player in (rec, opp):
            try:
                await player.ps_client.stop_listening()
            except Exception:
                pass
    return dict(obs=np.stack(obs), masks=np.stack(masks), actions=np.array(actions, dtype=np.int64),
                outcomes=np.array(outcomes, dtype=np.float32),
                steps_left=np.array(steps_left, dtype=np.int64), wins=wins, total=total)


def _collect_worker(job: tuple) -> dict:
    return asyncio.run(_collect_worker_async(*job))


def _collect(args) -> None:
    import multiprocessing as mp

    ports = [int(p) for p in args.ports.split(",")]
    opponents = args.opponents.split(",")
    per_worker = args.battles // args.workers
    jobs = [(args.teacher, opponents, per_worker, args.epsilon, ports[w % len(ports)], w)
            for w in range(args.workers)]
    start = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        parts = pool.map(_collect_worker, jobs)
    wins = sum(p["wins"] for p in parts)
    total = sum(p["total"] for p in parts)
    merged = {k: np.concatenate([p[k] for p in parts]) for k in
              ("obs", "masks", "actions", "outcomes", "steps_left")}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **merged)
    print(f"teacher won {wins}/{total}; saved {len(merged['actions'])} samples to {out} "
          f"in {time.time() - start:.0f}s")


# ---------------------------------------------------------------------------
# Model construction


def policy_kwargs(arch: str):
    import torch.nn as nn

    sizes = [int(x) for x in arch.split(",")]
    return dict(net_arch=dict(pi=sizes, vf=sizes), activation_fn=nn.ReLU)


class _SpaceEnv:
    """Minimal env stub so MaskablePPO can be built without a live server."""

    def __init__(self):
        import gymnasium as gym
        from gymnasium import spaces

        class Stub(gym.Env):
            observation_space = spaces.Box(-4.0, 4.0, (OBS_SIZE,), dtype=np.float32)
            action_space = spaces.Discrete(N_ACTIONS)

            def reset(self, *, seed=None, options=None):
                return np.zeros(OBS_SIZE, dtype=np.float32), {}

            def step(self, action):
                return np.zeros(OBS_SIZE, dtype=np.float32), 0.0, True, False, {}

            def action_masks(self):
                return np.ones(N_ACTIONS, dtype=bool)

        self.env = Stub()


def build_model(env, args, **overrides):
    from sb3_contrib import MaskablePPO

    kwargs = dict(
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs(args.arch),
        verbose=0,
        device="cpu",
        seed=args.seed,
    )
    kwargs.update(overrides)
    return MaskablePPO("MlpPolicy", env, **kwargs)


# ---------------------------------------------------------------------------
# Behaviour cloning


def _bc(args) -> None:
    import torch as th

    parts = [np.load(path) for path in args.data.split(",")]
    data = {k: np.concatenate([p[k] for p in parts]) for k in
            ("obs", "masks", "actions", "outcomes", "steps_left")}
    obs = th.tensor(data["obs"], dtype=th.float32)
    masks = th.tensor(data["masks"], dtype=th.bool)
    actions = th.tensor(data["actions"], dtype=th.long)
    returns = th.tensor(data["outcomes"] * (args.gamma ** data["steps_left"]), dtype=th.float32)
    n = len(actions)
    perm = th.randperm(n, generator=th.Generator().manual_seed(args.seed))
    n_val = max(1000, n // 20)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    args.n_steps, args.batch_size, args.n_epochs = 2048, 1024, 4
    model = build_model(_SpaceEnv().env, args)
    policy = model.policy
    if args.init:
        from sb3_contrib import MaskablePPO

        policy.load_state_dict(MaskablePPO.load(args.init, device="cpu").policy.state_dict())
    optim = th.optim.AdamW(policy.parameters(), lr=args.bc_lr, weight_decay=args.weight_decay)

    def run(idx, train: bool):
        policy.train(train)
        total_loss = correct = 0.0
        for start in range(0, len(idx), args.bc_batch):
            b = idx[start:start + args.bc_batch]
            with th.set_grad_enabled(train):
                dist = policy.get_distribution(obs[b], action_masks=masks[b])
                logp = dist.log_prob(actions[b])
                values = policy.predict_values(obs[b]).squeeze(-1)
                loss = -logp.mean() + args.bc_value_coef * ((values - returns[b]) ** 2).mean()
                if train:
                    optim.zero_grad()
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optim.step()
                pred = dist.distribution.probs.argmax(-1)
                correct += (pred == actions[b]).float().sum().item()
                total_loss += loss.item() * len(b)
        return total_loss / len(idx), correct / len(idx)

    import copy

    best_state, best_val, best_acc = None, float("inf"), 0.0
    for epoch in range(args.bc_epochs):
        shuffled = train_idx[th.randperm(len(train_idx))]
        tl, ta = run(shuffled, True)
        vl, va = run(val_idx, False)
        print(f"epoch {epoch + 1}: train loss {tl:.3f} acc {ta:.3f} | val loss {vl:.3f} acc {va:.3f}",
              flush=True)
        if vl < best_val:
            best_val, best_acc = vl, va
            best_state = copy.deepcopy(policy.state_dict())
    policy.load_state_dict(best_state)
    va = best_acc
    print(f"best val loss {best_val:.3f} acc {best_acc:.3f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    _write_meta(out, args, extra={"stage": "bc", "val_acc": va, "samples": n})
    print(f"saved {out}")


def _write_meta(path: Path, args, extra: dict) -> None:
    meta = {"git_sha": git_sha(), "format": BATTLE_FORMAT, "obs_size": OBS_SIZE,
            "n_actions": N_ACTIONS, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "args": {k: v for k, v in vars(args).items() if k != "func"}}
    meta.update(extra)
    Path(str(path).removesuffix(".zip") + ".meta.json").write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# PPO


def _make_env(pool: str, port: int, hp_value: float, fainted_value: float):
    def _init():
        return MaskedShowdownEnv(opponent_pool=pool, port=port, hp_value=hp_value,
                                 fainted_value=fainted_value)
    return _init


def _ppo(args) -> None:
    import torch as th
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    snap_dir = out_dir / "league"
    snap_dir.mkdir(exist_ok=True)
    ports = [int(p) for p in args.ports.split(",")]
    env = SubprocVecEnv([_make_env(args.pool, ports[i % len(ports)], args.hp_value,
                                   args.fainted_value) for i in range(args.n_envs)],
                        start_method="spawn")
    env = VecMonitor(env)

    model = build_model(env, args)
    anchor = None
    if args.init:
        init = MaskablePPO.load(args.init, device="cpu")
        model.policy.load_state_dict(init.policy.state_dict())
        if args.kl_coef > 0:
            anchor = init.policy
            anchor.set_training_mode(False)
            for p in anchor.parameters():
                p.requires_grad_(False)
    if anchor is not None:
        _install_kl_anchor(model, anchor, args.kl_coef, args.kl_decay_steps)

    class LeagueCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.results: list[tuple[str, bool]] = []
            self.snapshots: list[Path] = []
            self.last_snap = 0
            self.last_log = 0
            self.t0 = time.time()

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                if "won" in info:
                    self.results.append((info.get("opponent") or "?", info["won"]))
            if self.num_timesteps - self.last_log >= args.log_every:
                self.last_log = self.num_timesteps
                recent = self.results[-2000:]
                by_opp: dict[str, list[bool]] = {}
                for name, won in recent:
                    by_opp.setdefault(name, []).append(won)
                summary = " ".join(f"{k}={np.mean(v):.2f}({len(v)})" for k, v in sorted(by_opp.items()))
                fps = self.num_timesteps / max(time.time() - self.t0, 1)
                print(f"[{self.num_timesteps}] {fps:.0f} steps/s games={len(self.results)} {summary}",
                      flush=True)
            if args.selfplay_weight > 0 and self.num_timesteps - self.last_snap >= args.snapshot_every:
                self.last_snap = self.num_timesteps
                path = snap_dir / f"snap_{self.num_timesteps}.zip"
                self.model.save(str(path))
                self.snapshots.append(path)
                recent = self.snapshots[-args.league_size:]
                w = args.selfplay_weight / len(recent)
                pool = args.pool + "," + ",".join(f"snap={p}:{w:.4f}" for p in recent)
                self.training_env.env_method("set_opponent_pool", pool)
            if self.num_timesteps and self.num_timesteps % args.save_every < args.n_envs:
                self.model.save(str(out_dir / f"ckpt_{self.num_timesteps}.zip"))
            return True

    cb = LeagueCallback()
    try:
        model.learn(total_timesteps=args.steps, callback=cb, progress_bar=False)
    finally:
        model.save(str(out_dir / "final.zip"))
        _write_meta(out_dir / "final.zip", args, extra={"stage": "ppo", "timesteps": model.num_timesteps})
        env.close()
    print(f"saved {out_dir / 'final.zip'}")


def _install_kl_anchor(model, anchor, coef: float, decay_steps: int) -> None:
    """Add KL(pi || pi_anchor) to the PPO loss by wrapping policy.evaluate_actions.

    MaskablePPO.train() computes loss from evaluate_actions' entropy term, so we
    fold the KL penalty into the returned entropy: loss = ... - ent_coef * entropy.
    Using entropy' = entropy - (coef/ent_coef) * KL gives loss += coef * KL.
    """
    import torch as th

    policy = model.policy
    original = policy.evaluate_actions
    ent_coef = model.ent_coef if model.ent_coef > 0 else 1e-8

    def evaluate_actions(obs, actions, action_masks=None):
        values, log_prob, entropy = original(obs, actions, action_masks=action_masks)
        frac = max(0.0, 1.0 - model.num_timesteps / max(decay_steps, 1))
        if frac <= 0:
            return values, log_prob, entropy
        dist = policy.get_distribution(obs, action_masks=action_masks)
        with th.no_grad():
            anchor_dist = anchor.get_distribution(obs, action_masks=action_masks)
        p = dist.distribution.probs
        logp = th.log(p.clamp_min(1e-8))
        logq = th.log(anchor_dist.distribution.probs.clamp_min(1e-8))
        kl = (p * (logp - logq)).sum(-1)
        if entropy is None:
            entropy = -log_prob
        return values, log_prob, entropy - (coef * frac / ent_coef) * kl

    policy.evaluate_actions = evaluate_actions


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--arch", default="512,256")
        p.add_argument("--lr", type=float, default=1e-4)
        p.add_argument("--gamma", type=float, default=0.99)
        p.add_argument("--gae-lambda", type=float, default=0.95)
        p.add_argument("--clip", type=float, default=0.2)
        p.add_argument("--ent-coef", type=float, default=0.005)
        p.add_argument("--n-steps", type=int, default=1024)
        p.add_argument("--batch-size", type=int, default=1024)
        p.add_argument("--n-epochs", type=int, default=4)

    c = sub.add_parser("collect")
    c.add_argument("--teacher", default="smart")
    c.add_argument("--opponents", default="smart,heuristic,max_power,random")
    c.add_argument("--battles", type=int, default=20000)
    c.add_argument("--epsilon", type=float, default=0.0)
    c.add_argument("--ports", default="8000,8001")
    c.add_argument("--workers", type=int, default=6)
    c.add_argument("--out", required=True)

    b = sub.add_parser("bc")
    common(b)
    b.add_argument("--data", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--bc-epochs", type=int, default=15)
    b.add_argument("--bc-lr", type=float, default=1e-3)
    b.add_argument("--bc-batch", type=int, default=512)
    b.add_argument("--bc-value-coef", type=float, default=0.5)
    b.add_argument("--init", default=None, help="warm-start from this .zip")
    b.add_argument("--weight-decay", type=float, default=1e-4)

    p = sub.add_parser("ppo")
    common(p)
    p.add_argument("--init", default=None, help="BC/PPO .zip to start from")
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--ports", default="8000")
    p.add_argument("--pool", default="smart:0.4,heuristic:0.3,max_power:0.15,random:0.05")
    p.add_argument("--selfplay-weight", type=float, default=0.0)
    p.add_argument("--league-size", type=int, default=5)
    p.add_argument("--snapshot-every", type=int, default=200_000)
    p.add_argument("--save-every", type=int, default=250_000)
    p.add_argument("--log-every", type=int, default=50_000)
    p.add_argument("--hp-value", type=float, default=0.05)
    p.add_argument("--fainted-value", type=float, default=0.05)
    p.add_argument("--kl-coef", type=float, default=0.0)
    p.add_argument("--kl-decay-steps", type=int, default=2_000_000)
    p.add_argument("--out", required=True)

    args = parser.parse_args()
    if args.cmd == "collect":
        _collect(args)
    elif args.cmd == "bc":
        _bc(args)
    else:
        _ppo(args)


if __name__ == "__main__":
    main()
