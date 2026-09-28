#!/usr/bin/env python
"""Record raw Showdown protocol from local battles as offline test fixtures.

Plays a few Gen 9 Random Battles on a *local* Showdown server between two
poke-env players and saves, for one player, every raw websocket message of the
battle room plus a summary of the Battle state poke-env saw at each decision.
The fixtures let ``tests/test_protocol_battle.py`` check that
``showdownrl.protocol_battle`` rebuilds the same state live play needs.

Usage::

    python scripts/record_protocol_fixtures.py --battles 2 --out tests/fixtures
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_env.player import Player, SimpleHeuristicsPlayer  # noqa: E402
from poke_env.ps_client import AccountConfiguration  # noqa: E402

from showdownrl.protocol_battle import battle_summary  # noqa: E402
from showdownrl.real_env import BATTLE_FORMAT, server_configuration  # noqa: E402
from showdownrl.smart_heuristic import choose_smart_move  # noqa: E402


class RecordingPlayer(Player):
    """Smart-heuristic player that records raw battle-room messages."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.raw: dict[str, list[str]] = {}
        self.decisions: dict[str, list[dict[str, Any]]] = {}
        original = self.ps_client._handle_message

        async def recording_handle(message: str) -> None:
            if message.startswith(">battle"):
                tag = message.split("\n", 1)[0][1:]
                self.raw.setdefault(tag, []).append(message)
            await original(message)

        self.ps_client._handle_message = recording_handle  # type: ignore[method-assign]

    def choose_move(self, battle):  # noqa: ANN001
        order = choose_smart_move(battle)
        self.decisions.setdefault(battle.battle_tag, []).append(
            {
                "after_message": len(self.raw.get(battle.battle_tag, [])) - 1,
                "summary": battle_summary(battle),
                "order": order.message,
            }
        )
        return order


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battles", type=int, default=2)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--out", type=Path, default=ROOT / "tests" / "fixtures")
    args = parser.parse_args()

    tag = f"{random.randrange(16**4):04x}"
    recorder = RecordingPlayer(
        account_configuration=AccountConfiguration(f"srl rec {tag}", None),
        battle_format=BATTLE_FORMAT,
        server_configuration=server_configuration(args.port),
        log_level=40,
    )
    opponent = SimpleHeuristicsPlayer(
        account_configuration=AccountConfiguration(f"srl opp {tag}", None),
        battle_format=BATTLE_FORMAT,
        server_configuration=server_configuration(args.port),
        log_level=40,
    )
    # Alternate who sends the challenge so fixtures cover both p1 and p2.
    for index in range(args.battles):
        if index % 2 == 0:
            await recorder.battle_against(opponent, n_battles=1)
        else:
            await opponent.battle_against(recorder, n_battles=1)

    args.out.mkdir(parents=True, exist_ok=True)
    for index, (battle_tag, battle) in enumerate(recorder.battles.items(), start=1):
        payload = {
            "battle_tag": battle_tag,
            "username": recorder.username,
            "role": battle.player_role,
            "won": battle.won,
            "messages": recorder.raw.get(battle_tag, []),
            "decisions": recorder.decisions.get(battle_tag, []),
        }
        path = args.out / f"protocol_battle_{index}.json"
        path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        print(f"{path}: role={battle.player_role} messages={len(payload['messages'])} "
              f"decisions={len(payload['decisions'])} won={battle.won}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
