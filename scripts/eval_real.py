#!/usr/bin/env python
"""Evaluate an agent on the real Showdown simulator (local server, gen9randombattle).

Examples:
    python scripts/eval_real.py --agent smart --opponents heuristic,max_power --n 1000
    python scripts/eval_real.py --agent models/real/ppo.zip --n 1000 --json results/eval.json
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


def make_player(spec: str, port: int, concurrency: int, deterministic: bool) -> Player:
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
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(spec, device="cpu")
    return PolicyPlayer(model, deterministic=deterministic, **kwargs)


async def evaluate(agent_spec: str, opponents: list[str], n: int, port: int,
                   concurrency: int, deterministic: bool) -> dict:
    results = {}
    for opp_spec in opponents:
        agent = make_player(agent_spec, port, concurrency, deterministic)
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
        print(f"{agent_spec} vs {opp_spec}: {wins}/{finished} = {wins / max(finished, 1):.3f} "
              f"(95% CI {lo:.3f}-{hi:.3f}) in {time.time() - start:.0f}s", flush=True)
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
    args = parser.parse_args()

    results = asyncio.run(evaluate(args.agent, args.opponents.split(","), args.n, args.port,
                                   args.concurrency, not args.stochastic))
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"agent": args.agent, "format": BATTLE_FORMAT, "n": args.n,
                   "deterministic": not args.stochastic, "results": results}
        path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
