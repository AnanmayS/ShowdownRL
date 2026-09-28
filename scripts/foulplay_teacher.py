#!/usr/bin/env python
"""Record Foul Play's decisions as behaviour-cloning data in our feature space.

Launches K Foul Play instances (``accept_challenge`` mode, local server only) through
the logging wrapper ``scripts/foulplay_logged_run.py``, which writes every battle-room
websocket frame Foul Play receives and every choice it sends to
``<log-dir>/<battle-tag>.jsonl``. poke-env opponents challenge the instances
round-robin. Finished logs are replayed offline into poke-env ``Battle`` objects from
Foul Play's seat (``showdownrl.foulplay_replay``) and each decision becomes one sample:
``embed_battle`` obs, ``SinglesEnv`` action mask and Foul Play's choice as our action
index. Output is the same .npz layout as ``scripts/train_real.py collect``
(obs, masks, actions, outcomes, steps_left), usable directly with ``train_real.py bc``.

Examples::

    # 30-battle check (1 instance)
    python scripts/foulplay_teacher.py collect --battles 30 --instances 1 \\
        --out results/foulplay_teacher/check.npz

    # long run, incremental saves every 250 battles
    nohup python scripts/foulplay_teacher.py collect --battles 3000 --instances 3 \\
        --search-time-ms 75 --save-every 250 \\
        --out /Users/ananmaysingh/ShowdownRL/models/real/foulplay_teacher.npz > fpt.log 2>&1 &

    # (re)convert existing log dirs, merging into an existing npz
    python scripts/foulplay_teacher.py convert results/foulplay_teacher/logs_ab12 \\
        --out data.npz --append

Opponents (``--opponents``): smart, heuristic, max_power, random, and ``policy`` (needs
``--policy PATH.zip``; played by showdownrl.real_env.PolicyPlayer, stochastic).

See scripts/eval_foulplay.py for the one-time Foul Play setup and the required
local-login patch. Nothing here talks to anything but ``ws://localhost:<port>``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import threading
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from showdownrl.foulplay_replay import BattleSamples, merge_samples, replay_log  # noqa: E402

KEYS = ("obs", "masks", "actions", "outcomes", "steps_left")
OPP_CODES = {"smart": "sm", "heuristic": "he", "max_power": "mx", "random": "rn", "policy": "pp"}


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    import math

    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


# ---------------------------------------------------------------------------
# Conversion / saving


class Converter:
    """Converts finished battle logs (``*.done`` marker present) once each and keeps
    the per-battle results so incremental saves only process new logs."""

    def __init__(self, log_dirs: list[Path], base: Optional[dict] = None,
                 players: Optional[dict[str, str]] = None):
        self.log_dirs = log_dirs
        self.base = base
        self.players = dict(players or {})
        self.results: dict[Path, BattleSamples] = {}

    def _load_players(self) -> None:
        for d in self.log_dirs:
            f = d / "players.json"
            if f.exists():
                try:
                    self.players.update(json.loads(f.read_text()))
                except (OSError, json.JSONDecodeError):
                    pass

    def update(self, finished_only: bool = True) -> int:
        self._load_players()
        new = 0
        for d in self.log_dirs:
            for path in sorted(d.glob("*.jsonl")):
                if path in self.results:
                    continue
                if finished_only and not path.with_suffix(".done").exists():
                    continue
                self.results[path] = replay_log(path)
                new += 1
        return new

    def finished(self) -> list[BattleSamples]:
        return [r for r in self.results.values() if r.finished]

    def arrays(self) -> dict[str, np.ndarray]:
        merged = merge_samples(self.finished())
        if self.base is not None and len(self.base["actions"]):
            merged = {k: np.concatenate([self.base[k], merged[k]]) for k in KEYS}
        return merged

    def opponent_type(self, username: str) -> str:
        from poke_env.data.normalize import to_id_str

        return self.players.get(to_id_str(username), self.players.get(username, "unknown"))

    def summary(self) -> dict:
        per_opp: dict[str, dict] = defaultdict(lambda: dict(battles=0, fp_wins=0, ties=0))
        decisions = labeled = 0
        reasons: dict[str, int] = defaultdict(int)
        replay_errors = 0
        for r in self.finished():
            s = per_opp[self.opponent_type(r.opponent)]
            s["battles"] += 1
            s["fp_wins"] += int(bool(r.won))
            s["ties"] += int(r.won is None)
            decisions += r.decisions
            labeled += len(r.actions)
            replay_errors += len(r.errors)
            for _, why in r.mismatches:
                reasons[why.split(" (")[0][:60]] += 1
        total = dict(battles=sum(s["battles"] for s in per_opp.values()),
                     fp_wins=sum(s["fp_wins"] for s in per_opp.values()))
        for s in list(per_opp.values()) + [total]:
            s["fp_win_rate"] = round(s["fp_wins"] / max(s["battles"], 1), 4)
            s["ci95"] = [round(x, 4) for x in wilson(s["fp_wins"], s["battles"])]
        return dict(
            battles=total["battles"],
            fp_win_rate=total["fp_win_rate"],
            per_opponent=dict(per_opp),
            decisions=decisions,
            samples=labeled,
            mismatches=decisions - labeled,
            mismatch_rate=round((decisions - labeled) / max(decisions, 1), 5),
            mismatch_reasons=dict(sorted(reasons.items(), key=lambda kv: -kv[1])[:15]),
            replay_errors=replay_errors,
            unfinished_logs=sum(1 for r in self.results.values() if not r.finished),
            base_samples=0 if self.base is None else int(len(self.base["actions"])),
        )


def load_npz(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with np.load(path) as data:
        return {k: data[k] for k in KEYS}


def save_npz(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def save(conv: Converter, out: Path, extra: dict) -> dict:
    arrays = conv.arrays()
    save_npz(out, arrays)
    summary = dict(conv.summary(), **extra, out=str(out), total_samples_in_file=int(len(arrays["actions"])))
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Collection


def make_opponent(kind: str, username: str, port: int, policy: Optional[str]):
    from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
    from poke_env.ps_client import AccountConfiguration

    from showdownrl.real_env import BATTLE_FORMAT, PolicyPlayer, server_configuration
    from showdownrl.smart_heuristic import SmartHeuristicsPlayer

    kwargs = dict(battle_format=BATTLE_FORMAT, server_configuration=server_configuration(port),
                  max_concurrent_battles=1, log_level=40,
                  account_configuration=AccountConfiguration(username, None))
    scripted = {"smart": SmartHeuristicsPlayer, "heuristic": SimpleHeuristicsPlayer,
                "max_power": MaxBasePowerPlayer, "random": RandomPlayer}
    if kind in scripted:
        return scripted[kind](**kwargs)
    if kind == "policy":
        if not policy:
            raise SystemExit("--opponents includes 'policy' but --policy was not given")
        import torch
        from sb3_contrib import MaskablePPO

        torch.set_num_threads(1)  # stay inside the CPU budget
        model = MaskablePPO.load(policy, device="cpu")
        return PolicyPlayer(model, deterministic=False, **kwargs)
    raise SystemExit(f"unknown opponent {kind!r}")


class Instance(threading.Thread):
    """One Foul Play process plus its opponents; plays battles until the shared budget
    is exhausted, restarting Foul Play (and the opponents) after a failure."""

    def __init__(self, idx: int, args, run_id: str, log_dir: Path, budget: "Budget",
                 players: dict[str, str], players_lock: threading.Lock):
        super().__init__(daemon=True, name=f"fp{idx}")
        self.idx, self.args, self.run_id, self.log_dir = idx, args, run_id, log_dir
        self.budget = budget
        self.players, self.players_lock = players, players_lock
        self.restarts = 0
        self.played = 0
        self.failures = 0
        self.stop_event = threading.Event()

    def _start(self):
        from eval_foulplay import FoulPlayProcess

        fp_dir = Path(self.args.foulplay_dir).expanduser().resolve()
        python = fp_dir / ".venv" / "bin" / "python"
        name = f"fpt{self.run_id}{self.idx}r{self.restarts}"
        fp = FoulPlayProcess(
            fp_dir, python, name, self.args.port, self.args.search_time_ms,
            self.args.search_parallelism, 10 ** 6,
            self.log_dir / "stdout" / f"{name}.log",
            entry=ROOT / "scripts" / "foulplay_logged_run.py",
            extra_env={"FP_BATTLE_LOG_DIR": str(self.log_dir),
                       "FP_THREAD_SEARCH": "1" if self.args.search_backend == "thread" else "0"})
        opps = {}
        for kind in dict.fromkeys(self.args.opponents):  # repeats in the list = weights
            uname = f"o{OPP_CODES[kind]}{self.run_id}{self.idx}r{self.restarts}"
            with self.players_lock:
                self.players[uname] = kind
                (self.log_dir / "players.json").write_text(json.dumps(self.players, indent=1))
            opps[kind] = make_opponent(kind, uname, self.args.port, self.args.policy)
        deadline = time.time() + 60
        for p in opps.values():
            while not p.ps_client.logged_in.is_set():
                if time.time() > deadline:
                    raise TimeoutError("opponent failed to log in")
                time.sleep(0.05)
        return fp, opps

    @staticmethod
    def _shutdown(fp, opps) -> None:
        from eval_foulplay import on_poke_loop

        if fp is not None:
            fp.stop()
        for p in (opps or {}).values():
            try:
                on_poke_loop(p.ps_client.stop_listening(), timeout=10)
            except Exception:
                pass

    def run(self) -> None:
        from eval_foulplay import play_one

        fp = opps = None
        local = 0
        k = self.idx
        while not self.stop_event.is_set() and self.budget.take():
            try:
                if fp is None:
                    fp, opps = self._start()
                    local = 0
                kind = self.args.opponents[k % len(self.args.opponents)]
                k += 1
                opp = opps[kind]
                play_one(opp, fp, local, self.args.accept_timeout, self.args.battle_timeout)
                local += 1
                self.played += 1
                try:
                    opp.reset_battles()  # keep memory flat over long runs
                except EnvironmentError:
                    pass
            except Exception:
                self.budget.give_back()
                self.failures += 1
                print(f"[fp{self.idx}] failure #{self.failures}, restarting Foul Play:\n"
                      f"{traceback.format_exc()[-1500:]}", flush=True)
                self._shutdown(fp, opps)
                fp = opps = None
                self.restarts += 1
                if self.failures > self.args.max_failures:
                    print(f"[fp{self.idx}] too many failures; giving up", flush=True)
                    break
                time.sleep(5)
        self._shutdown(fp, opps)


class Budget:
    def __init__(self, n: int):
        self.left = n
        self.lock = threading.Lock()

    def take(self) -> bool:
        with self.lock:
            if self.left <= 0:
                return False
            self.left -= 1
            return True

    def give_back(self) -> None:
        with self.lock:
            self.left += 1


def collect(args) -> None:
    if args.nice:
        os.nice(args.nice)  # Foul Play children inherit it; training keeps priority
    run_id = args.run_id or f"{random.randrange(16 ** 4):04x}"
    log_dir = Path(args.log_dir or ROOT / "results" / "foulplay_teacher" / f"logs_{run_id}").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out).expanduser().resolve()
    base = load_npz(out) if args.append else None
    if args.append and base is not None:
        print(f"appending to {out} ({len(base['actions'])} existing samples)", flush=True)
    print(f"run {run_id}: logs -> {log_dir}, output -> {out}", flush=True)

    players: dict[str, str] = {}
    conv = Converter([log_dir], base=base, players=players)
    budget = Budget(args.battles)
    lock = threading.Lock()
    instances = [Instance(i, args, run_id, log_dir, budget, players, lock)
                 for i in range(args.instances)]

    stop = threading.Event()

    def _sig(signum, frame):
        print(f"signal {signum}: stopping after current battles and saving", flush=True)
        stop.set()
        with budget.lock:
            budget.left = 0

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    start = time.time()
    for inst in instances:
        inst.start()
        time.sleep(2)
    last_saved = 0

    def extra():
        elapsed = time.time() - start
        n = sum(1 for r in conv.results.values() if r.finished)
        samples = sum(len(r.actions) for r in conv.finished())
        return dict(run_id=run_id, log_dir=str(log_dir), elapsed_s=round(elapsed),
                    battles_per_hour=round(n / max(elapsed, 1) * 3600, 1),
                    samples_per_hour=round(samples / max(elapsed, 1) * 3600),
                    search_time_ms=args.search_time_ms, instances=args.instances,
                    restarts=sum(i.restarts for i in instances))

    while any(inst.is_alive() for inst in instances):
        time.sleep(5)
        conv.update()
        n = len(conv.finished())
        if n - last_saved >= args.save_every:
            s = save(conv, out, extra())
            last_saved = n
            print(f"[{time.strftime('%H:%M:%S')}] saved {s['total_samples_in_file']} samples "
                  f"({n} battles, FP win {s['fp_win_rate']:.3f}, mismatch "
                  f"{s['mismatch_rate']:.4f}, {s['battles_per_hour']} battles/h)", flush=True)
    for inst in instances:
        inst.join(timeout=30)
    conv.update()
    s = save(conv, out, extra())
    print(json.dumps(s, indent=2), flush=True)


def convert(args) -> None:
    out = Path(args.out).expanduser().resolve()
    base = load_npz(out) if args.append else None
    conv = Converter([Path(d).resolve() for d in args.log_dirs], base=base)
    conv.update(finished_only=not args.include_unmarked)
    s = save(conv, out, {"log_dirs": args.log_dirs})
    print(json.dumps(s, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="play Foul Play vs opponents, log, convert, save")
    c.add_argument("--battles", type=int, default=30)
    c.add_argument("--instances", type=int, default=1, help="concurrent Foul Play processes")
    c.add_argument("--port", type=int, default=8000)
    c.add_argument("--opponents", default="smart,heuristic,max_power,random",
                   help=f"comma list from {sorted(OPP_CODES)}, played round-robin; "
                        "repeat a name to weight it (e.g. smart,smart,heuristic,random)")
    c.add_argument("--policy", default=None, help="MaskablePPO .zip for the 'policy' opponent")
    c.add_argument("--foulplay-dir", default=os.environ.get("FOULPLAY_DIR", "~/foul-play"))
    c.add_argument("--search-time-ms", type=int, default=75)
    c.add_argument("--search-parallelism", type=int, default=1)
    c.add_argument("--search-backend", choices=("thread", "process"), default="thread",
                   help="thread: run Foul Play's MCTS in a worker thread instead of spawning a "
                        "process pool per decision (same search, less CPU; only with "
                        "--search-parallelism 1); process: Foul Play's default")
    c.add_argument("--accept-timeout", type=float, default=20.0)
    c.add_argument("--battle-timeout", type=float, default=900.0)
    c.add_argument("--max-failures", type=int, default=20, help="per instance")
    c.add_argument("--save-every", type=int, default=250, help="battles between saves")
    c.add_argument("--nice", type=int, default=10, help="niceness increment (0 = off)")
    c.add_argument("--run-id", default=None)
    c.add_argument("--log-dir", default=None)
    c.add_argument("--out", required=True)
    c.add_argument("--append", action="store_true", help="merge into an existing --out npz")

    v = sub.add_parser("convert", help="convert existing log dirs to an npz")
    v.add_argument("log_dirs", nargs="+")
    v.add_argument("--out", required=True)
    v.add_argument("--append", action="store_true")
    v.add_argument("--include-unmarked", action="store_true",
                   help="also read logs without a .done marker (finished ones are still kept only)")

    args = parser.parse_args()
    if args.cmd == "collect":
        args.opponents = [o.strip() for o in args.opponents.split(",") if o.strip()]
        for o in args.opponents:
            if o not in OPP_CODES:
                raise SystemExit(f"unknown opponent {o!r}")
        if args.search_parallelism != 1:
            args.search_backend = "process"
        collect(args)
    else:
        convert(args)


if __name__ == "__main__":
    main()
