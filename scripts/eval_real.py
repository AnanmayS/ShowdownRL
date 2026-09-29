#!/usr/bin/env python
"""Evaluate an agent on the real Showdown simulator (local server, gen9randombattle).

Examples:
    python scripts/eval_real.py --agent smart --opponents heuristic,max_power --n 1000
    python scripts/eval_real.py --agent models/real/ppo.zip --n 1000 --json results/eval.json
    python scripts/eval_real.py --agent search:models/real/bc_fp_r2.zip,samples=4,time_ms=100,prior=0.3 \
        --opponents heuristic,smart --n 300 --port 8002 --search-workers 2

Search agents (showdownrl.search.SearchPlayer, needs the ``search`` extra):
``search[:MODEL.zip][,samples=N][,time_ms=T][,prior=LAMBDA][,unrevealed=sample|fainted]
[,iterations=K][,workers=W]``. Without a model the fallback is the smart heuristic
and ``prior`` must be 0.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer  # noqa: E402
from poke_env.ps_client import AccountConfiguration  # noqa: E402

from showdownrl.real_env import BATTLE_FORMAT, PolicyPlayer, server_configuration  # noqa: E402
from showdownrl.smart_heuristic import SmartHeuristicsPlayer  # noqa: E402

SCRIPTED = {
    "random": RandomPlayer,
    "max_power": MaxBasePowerPlayer,
    "heuristic": SimpleHeuristicsPlayer,
    "smart": SmartHeuristicsPlayer,
}


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


_counter = 0


def make_player(spec: str, port: int, concurrency: int, deterministic: bool,
                search_workers: int = 2) -> Player:
    global _counter
    _counter += 1
    kwargs = dict(
        battle_format=BATTLE_FORMAT,
        server_configuration=server_configuration(port),
        max_concurrent_battles=concurrency,
        account_configuration=AccountConfiguration(f"ev{_counter}{int(time.time()) % 100000}", None),
        log_level=40,
    )
    if spec in SCRIPTED:
        return SCRIPTED[spec](**kwargs)
    if spec.startswith("search"):
        from showdownrl.search import SearchPlayer, parse_search_spec

        options = {"workers": search_workers, **parse_search_spec(spec)}
        return SearchPlayer(**options, **kwargs)
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(spec, device="cpu")
    return PolicyPlayer(model, deterministic=deterministic, **kwargs)


async def evaluate(agent_spec: str, opponents: list[str], n: int, port: int,
                   concurrency: int, deterministic: bool, search_workers: int = 2) -> dict:
    results = {}
    for opp_spec in opponents:
        agent = make_player(agent_spec, port, concurrency, deterministic, search_workers)
        opponent = make_player(opp_spec, port, concurrency, deterministic)
        start = time.time()
        await agent.battle_against(opponent, n_battles=n)
        wins = agent.n_won_battles
        finished = agent.n_finished_battles
        lo, hi = wilson(wins, finished)
        results[opp_spec] = {
            "wins": wins,
            "battles": finished,
            "win_rate": wins / max(finished, 1),
            "ci95": [lo, hi],
            "seconds": round(time.time() - start, 1),
        }
        extra = ""
        if hasattr(agent, "stats") and hasattr(agent, "decision_seconds"):
            secs = agent.decision_seconds
            results[opp_spec]["search"] = {
                **agent.stats.as_dict(),
                "seconds_per_decision": round(sum(secs) / max(len(secs), 1), 4),
                "decisions_per_battle": round(len(secs) / max(finished, 1), 1),
            }
            s = results[opp_spec]["search"]
            extra = (f" [{s['seconds_per_decision']}s/decision, fallbacks {s['fallbacks']}, "
                     f"engine errors {s['engine_errors']}, {s['iterations_per_sample']} it/sample]")
            agent.close_executor()
        print(f"{agent_spec} vs {opp_spec}: {wins}/{finished} = {wins / max(finished, 1):.3f} "
              f"(95% CI {lo:.3f}-{hi:.3f}) in {time.time() - start:.0f}s{extra}", flush=True)
        for player in (agent, opponent):
            try:
                await player.ps_client.stop_listening()
            except Exception:
                pass
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True, help="scripted name or MaskablePPO .zip path")
    parser.add_argument("--opponents", default="random,max_power,heuristic,smart")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--stochastic", action="store_true", help="sample policy actions")
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--search-workers", type=int, default=2,
                        help="engine worker processes for search agents (0 = inline)")
    args = parser.parse_args()

    results = asyncio.run(evaluate(args.agent, args.opponents.split(","), args.n, args.port,
                                   args.concurrency, not args.stochastic, args.search_workers))
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"agent": args.agent, "format": BATTLE_FORMAT, "n": args.n,
                   "deterministic": not args.stochastic, "results": results}
        path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
