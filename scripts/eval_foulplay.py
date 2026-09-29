#!/usr/bin/env python
"""Evaluate a ShowdownRL agent against Foul Play (MCTS bot on poke-engine) on a LOCAL server.

Foul Play (https://github.com/pmariglia/foul-play) is launched as a subprocess in
``accept_challenge`` mode and the agent challenges it ``--n`` times, one battle at a
time (Foul Play only plays one battle at once). Format: gen9randombattle.

Examples:
    python scripts/eval_foulplay.py --agent heuristic --n 50
    python scripts/eval_foulplay.py --agent smart --n 50 --search-time-ms 100 --json reports/fp_smart.json
    python scripts/eval_foulplay.py --agent models/real/ppo.zip --n 100

Agents: ``random``, ``max_power``, ``heuristic`` (poke-env SimpleHeuristicsPlayer),
``smart`` (showdownrl.smart_heuristic.SmartHeuristicsPlayer) or a MaskablePPO ``.zip``
path (played via showdownrl.real_env.PolicyPlayer, deterministic unless --stochastic).

One-time Foul Play setup (outside this repo; needs Rust/cargo for poke-engine,
which only ships as an sdist on PyPI and defaults to gen4 unless built with the
gen9 ``terastallization`` feature):

    git clone https://github.com/pmariglia/foul-play.git ~/foul-play
    cd ~/foul-play
    uv venv --seed --python 3.13 .venv          # any Python >= 3.11 venv works
    .venv/bin/pip install -r requirements.txt
    # make sure the engine is the gen9 build (requirements.txt asks for it, this forces it):
    .venv/bin/pip install -v --force-reinstall --no-deps --no-cache-dir poke-engine==0.0.48 \\
        --config-settings="build-args=--features poke-engine/terastallization --no-default-features"

Local patch required in ~/foul-play/fp/websocket_client.py (``PSWebsocketClient.login``):
upstream always POSTs to play.pokemonshowdown.com for a guest assertion. For a
localhost websocket with no password, send ``/trn <name>,0,`` directly instead
(a no-security local server accepts it) and return the username. Without that
patch Foul Play contacts the official login server.

Search budget: ``--search-time-ms`` (default 100) is passed straight to Foul Play.
With ``--search-parallelism 1`` Foul Play samples 2-4 determinized worlds per
decision and searches each for search_time_ms (search_time_ms/2 early in the
battle), sequentially, so a decision costs roughly 2x search_time_ms of one core
plus process-spawn overhead. Foul Play also downloads random-battle set data
from pkmn.github.io on first use (not a Showdown server).
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_env.concurrency import POKE_LOOP  # noqa: E402
from poke_env.player import (  # noqa: E402
    MaxBasePowerPlayer,
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.ps_client import AccountConfiguration  # noqa: E402

from showdownrl.real_env import BATTLE_FORMAT, PolicyPlayer, server_configuration  # noqa: E402
from showdownrl.smart_heuristic import SmartHeuristicsPlayer  # noqa: E402

SCRIPTED = {
    "random": RandomPlayer,
    "max_power": MaxBasePowerPlayer,
    "heuristic": SimpleHeuristicsPlayer,
    "smart": SmartHeuristicsPlayer,
}
READY_MARKER = f"Waiting for a {BATTLE_FORMAT} challenge"


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


# ---------------------------------------------------------------------------
# Foul Play subprocess


class FoulPlayProcess:
    def __init__(self, fp_dir: Path, python: Path, username: str, port: int,
                 search_time_ms: int, parallelism: int, run_count: int, log_path: Path,
                 entry: Optional[Path] = None, extra_env: Optional[dict] = None):
        """``entry``: script to run instead of Foul Play's ``run.py`` (e.g. the logging
        wrapper scripts/foulplay_logged_run.py); ``extra_env``: added to its environment."""
        self.username = username
        self.log_path = log_path
        cmd = [
            str(python), str(entry) if entry else "run.py",
            "--websocket-uri", f"ws://localhost:{port}/showdown/websocket",
            "--ps-username", username,
            "--bot-mode", "accept_challenge",
            "--pokemon-format", BATTLE_FORMAT,
            "--search-time-ms", str(search_time_ms),
            "--search-parallelism", str(parallelism),
            "--run-count", str(run_count),
            "--log-level", "INFO",
        ]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(log_path, "w")
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
                   FOULPLAY_DIR=str(fp_dir), **(extra_env or {}))
        # Own process group so the search worker processes die with it.
        self.proc = subprocess.Popen(cmd, cwd=fp_dir, stdout=self._log, stderr=subprocess.STDOUT,
                                     env=env, start_new_session=True)
        atexit.register(self.stop)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def log_text(self) -> str:
        try:
            return self.log_path.read_text(errors="replace")
        except OSError:
            return ""

    def ready_count(self) -> int:
        return self.log_text().count(READY_MARKER)

    def wait_ready(self, count: int, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.ready_count() >= count:
                return
            if not self.alive():
                raise RuntimeError(f"Foul Play exited (code {self.proc.returncode}); "
                                   f"see {self.log_path}:\n{self.log_text()[-2000:]}")
            time.sleep(0.1)
        raise TimeoutError(f"Foul Play not ready after {timeout}s; see {self.log_path}")

    def stop(self) -> None:
        if self.proc.poll() is None:
            for sig, wait in ((signal.SIGTERM, 5), (signal.SIGKILL, 5)):
                try:
                    os.killpg(self.proc.pid, sig)
                except (ProcessLookupError, PermissionError):
                    break
                try:
                    self.proc.wait(timeout=wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
        if not self._log.closed:
            self._log.close()


# ---------------------------------------------------------------------------
# Agent


def make_agent(spec: str, port: int, deterministic: bool, search_workers: int = 2) -> Player:
    kwargs = dict(
        battle_format=BATTLE_FORMAT,
        server_configuration=server_configuration(port),
        max_concurrent_battles=1,
        account_configuration=AccountConfiguration(
            f"fpeval{os.getpid() % 100000}{random.randint(0, 999)}", None),
        log_level=40,
    )
    if spec in SCRIPTED:
        return SCRIPTED[spec](**kwargs)
    if spec.startswith("search"):
        from showdownrl.search import SearchPlayer, parse_search_spec

        return SearchPlayer(**{"workers": search_workers, **parse_search_spec(spec)}, **kwargs)
    from sb3_contrib import MaskablePPO

    from showdownrl.battle_features import N_ACTIONS, OBS_SIZE

    model = MaskablePPO.load(spec, device="cpu")
    if model.observation_space.shape != (OBS_SIZE,) or model.action_space.n != N_ACTIONS:
        raise SystemExit(f"{spec}: obs {model.observation_space.shape} / {model.action_space} does "
                         f"not match the real-env features ({OBS_SIZE},) / Discrete({N_ACTIONS})")
    return PolicyPlayer(model, deterministic=deterministic, **kwargs)


def on_poke_loop(coro, timeout: Optional[float] = 30):
    return asyncio.run_coroutine_threadsafe(coro, POKE_LOOP).result(timeout)


def play_one(agent: Player, fp: FoulPlayProcess, index: int, accept_timeout: float,
             battle_timeout: float) -> None:
    """Challenge Foul Play once and block until that battle is finished."""
    fp.wait_ready(index + 1, timeout=120)
    started_before = len(agent.battles)
    finished_before = agent.n_finished_battles
    for attempt in range(3):
        on_poke_loop(agent.ps_client.challenge(fp.username, BATTLE_FORMAT, None))
        deadline = time.time() + accept_timeout
        while time.time() < deadline and len(agent.battles) == started_before:
            if not fp.alive():
                raise RuntimeError(f"Foul Play exited; see {fp.log_path}")
            time.sleep(0.05)
        if len(agent.battles) > started_before:
            break
        # Challenge was not picked up; withdraw it and retry.
        on_poke_loop(agent.ps_client.send_message(f"/cancelchallenge {fp.username}"))
        time.sleep(1.0)
    else:
        raise RuntimeError(f"Foul Play did not accept challenge #{index + 1}; see {fp.log_path}")

    deadline = time.time() + battle_timeout
    while agent.n_finished_battles == finished_before:
        if time.time() > deadline:
            raise TimeoutError(f"battle #{index + 1} exceeded {battle_timeout}s")
        if not fp.alive():
            raise RuntimeError(f"Foul Play exited mid-battle; see {fp.log_path}")
        time.sleep(0.1)


def evaluate(args: argparse.Namespace) -> dict:
    fp_dir = Path(args.foulplay_dir).expanduser().resolve()
    python = Path(args.foulplay_python).expanduser() if args.foulplay_python \
        else fp_dir / ".venv" / "bin" / "python"
    if not (fp_dir / "run.py").exists() or not python.exists():
        raise SystemExit(f"Foul Play not found at {fp_dir} (python {python}); see module docstring")
    fp_name = f"foulplay{os.getpid() % 100000}"
    log_path = Path(args.fp_log) if args.fp_log else \
        Path("results") / "foulplay_logs" / f"{fp_name}.log"

    entry, extra_env = None, None
    if args.search_backend == "thread":
        if args.search_parallelism != 1:
            raise SystemExit("--search-backend thread needs --search-parallelism 1")
        # Same search, run in a worker thread instead of a per-decision process pool
        # (the pool occasionally hangs on a loaded machine); see foulplay_logged_run.py.
        entry = Path(__file__).resolve().parent / "foulplay_logged_run.py"
        extra_env = {"FP_THREAD_SEARCH": "1"}
    fp = FoulPlayProcess(fp_dir, python, fp_name, args.port, args.search_time_ms,
                         args.search_parallelism, args.n, log_path,
                         entry=entry, extra_env=extra_env)

    def _sig(signum, frame):  # make SIGTERM/SIGINT run the cleanup path
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _sig)
    agent = None
    durations: list[float] = []
    void_tags: set[str] = set()
    restarts = 0
    try:
        agent = make_agent(args.agent, args.port, not args.stochastic, args.agent_search_workers)
        deadline = time.time() + 30
        while not agent.ps_client.logged_in.is_set():
            if time.time() > deadline:
                raise TimeoutError("agent failed to log in to local server")
            time.sleep(0.05)

        start = time.time()
        offset = 0  # battles played by earlier (restarted) Foul Play processes
        for i in range(args.n):
            t0 = time.time()
            try:
                play_one(agent, fp, i - offset, args.accept_timeout, args.battle_timeout)
            except (TimeoutError, RuntimeError) as exc:
                # Foul Play occasionally hangs inside its search. Forfeit the stuck
                # battle, exclude it from the stats and restart Foul Play.
                print(f"[{i + 1}/{args.n}] voided: {exc}; restarting Foul Play", flush=True)
                for tag, battle in list(agent.battles.items()):
                    if not battle.finished:
                        void_tags.add(tag)
                        try:
                            on_poke_loop(agent.ps_client.send_message("/forfeit", tag), timeout=10)
                        except Exception:
                            pass
                fp.stop()
                restarts += 1
                fp_name = f"foulplay{os.getpid() % 100000}r{restarts}"
                fp = FoulPlayProcess(fp_dir, python, fp_name, args.port, args.search_time_ms,
                                     args.search_parallelism, args.n - i, log_path.with_name(
                                         f"{fp_name}.log"), entry=entry, extra_env=extra_env)
                offset = i + 1
                continue
            durations.append(time.time() - t0)
            if args.verbose or (i + 1) % 10 == 0 or i + 1 == args.n:
                w, f = agent.n_won_battles, agent.n_finished_battles
                print(f"[{i + 1}/{args.n}] {args.agent} vs FoulPlay: {w}/{f} "
                      f"({w / max(f, 1):.3f}), {durations[-1]:.1f}s this battle", flush=True)
    except KeyboardInterrupt:
        print("interrupted; reporting partial results", file=sys.stderr)
    finally:
        fp.stop()
        if agent is not None:
            try:
                on_poke_loop(agent.ps_client.stop_listening(), timeout=10)
            except Exception:
                pass

    done = [b for tag, b in (agent.battles.items() if agent else []) if b.finished
            and tag not in void_tags]
    wins = sum(1 for b in done if b.won)
    finished = len(done)
    ties = sum(1 for b in done if not b.won and not b.lost)
    lo, hi = wilson(wins, finished)
    elapsed = sum(durations)
    # KO counts give signal even when the win rate is ~0.
    opp_ko = sum(sum(m.fainted for m in b.opponent_team.values()) for b in done)
    own_ko = sum(sum(m.fainted for m in b.team.values()) for b in done)
    turns = sum(b.turn for b in done)
    result = {
        "agent": args.agent,
        "opponent": "foulplay",
        "format": BATTLE_FORMAT,
        "search_time_ms": args.search_time_ms,
        "search_parallelism": args.search_parallelism,
        "deterministic": not args.stochastic,
        "wins": wins,
        "ties": ties,
        "voided": len(void_tags),
        "foulplay_restarts": restarts,
        "battles": finished,
        "win_rate": wins / max(finished, 1),
        "ci95": [lo, hi],
        "avg_foulplay_mons_fainted": round(opp_ko / max(len(done), 1), 2),
        "avg_agent_mons_fainted": round(own_ko / max(len(done), 1), 2),
        "avg_turns": round(turns / max(len(done), 1), 1),
        "seconds": round(elapsed, 1),
        "seconds_per_battle": round(elapsed / max(len(durations), 1), 2),
        "foulplay_log": str(log_path),
    }
    if agent is not None and hasattr(agent, "stats") and hasattr(agent, "decision_seconds"):
        secs = agent.decision_seconds
        result["search"] = {**agent.stats.as_dict(),
                            "seconds_per_decision": round(sum(secs) / max(len(secs), 1), 4)}
        agent.close_executor()
    print(f"{args.agent} vs FoulPlay({args.search_time_ms}ms): {wins}/{finished} = "
          f"{result['win_rate']:.3f} (95% CI {lo:.3f}-{hi:.3f}), "
          f"KOs dealt/taken {result['avg_foulplay_mons_fainted']}/"
          f"{result['avg_agent_mons_fainted']} per battle, "
          f"{result['seconds_per_battle']}s/battle", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", required=True,
                        help="scripted name, MaskablePPO .zip path or search spec "
                             "(search[:MODEL.zip][,samples=N][,time_ms=T][,prior=L]; see eval_real.py)")
    parser.add_argument("--agent-search-workers", type=int, default=2,
                        help="engine worker processes for a search agent (0 = inline)")
    parser.add_argument("--n", type=int, default=50, help="number of battles")
    parser.add_argument("--port", type=int, default=8000, help="local Showdown server port")
    parser.add_argument("--foulplay-dir", default=os.environ.get("FOULPLAY_DIR", "~/foul-play"))
    parser.add_argument("--foulplay-python", default=None,
                        help="python for Foul Play (default: <foulplay-dir>/.venv/bin/python)")
    parser.add_argument("--search-time-ms", type=int, default=100)
    parser.add_argument("--search-parallelism", type=int, default=1,
                        help="Foul Play worker processes per decision (CPU cores used)")
    parser.add_argument("--search-backend", choices=("process", "thread"), default="process",
                        help="Foul Play's MCTS in its per-decision process pool (default) or in "
                             "a worker thread (only with --search-parallelism 1)")
    parser.add_argument("--accept-timeout", type=float, default=20.0,
                        help="seconds to wait for Foul Play to accept a challenge before retrying")
    parser.add_argument("--battle-timeout", type=float, default=900.0)
    parser.add_argument("--stochastic", action="store_true", help="sample policy actions")
    parser.add_argument("--fp-log", default=None, help="Foul Play stdout log path")
    parser.add_argument("--json", default=None, help="write results to this JSON file")
    parser.add_argument("--verbose", action="store_true", help="print after every battle")
    args = parser.parse_args()

    if args.port not in range(1, 65536):
        raise SystemExit("bad port")
    result = evaluate(args)
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
