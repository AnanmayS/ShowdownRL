"""Turn Foul Play battle logs into behaviour-cloning samples in our feature space.

A log (written by ``scripts/foulplay_logged_run.py``) is a JSONL file holding, in
order, every websocket frame Foul Play received for one battle room (``"in"``) and
every message it sent to that room (``"out"``). Replaying the ``"in"`` frames into a
fresh poke-env ``Battle`` exactly the way ``poke_env.player.Player`` does
(``_handle_battle_message``: line by line, ``parse_request`` + decide at each
``|request|`` line) reproduces the state a poke-env agent would have seen had it been
sitting in Foul Play's seat. At each decision point we compute
``embed_battle`` + ``SinglesEnv.get_action_mask`` and convert Foul Play's answer
(the next ``/choose move <slot|id> [terastallize]`` or ``/switch <n>`` it sent) into
our 26-way action index with ``SinglesEnv.order_to_action``.

``order_to_action(strict=False)`` silently substitutes a *random* legal move when the
order is not valid, which would poison the labels, so conversion uses ``strict=True``
and anything that fails (or lands outside the mask) is counted as a mismatch and
dropped.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import orjson
from poke_env.battle import Battle, Pokemon
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str
from poke_env.environment import SinglesEnv
from poke_env.player import Player
from poke_env.player.battle_order import SingleBattleOrder

from showdownrl.battle_features import N_ACTIONS, OBS_SIZE, embed_battle

# Same set poke-env's Player skips before dispatching to Battle.parse_message.
MESSAGES_TO_IGNORE = Player.MESSAGES_TO_IGNORE
TERA_OFFSET = 16  # 6-9 plain move -> 22-25 move + terastallize

_QUIET = logging.getLogger("showdownrl.foulplay_replay.battle")
_QUIET.setLevel(logging.CRITICAL)


class ChoiceError(ValueError):
    """A ``/choose`` message that cannot be mapped to a legal action."""


@dataclass
class BattleSamples:
    tag: str
    username: str
    opponent: str = ""
    won: Optional[bool] = None  # None: unfinished or tie
    finished: bool = False
    turns: int = 0
    obs: list = field(default_factory=list)
    masks: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    decisions: int = 0  # decision points seen (incl. ones we could not label)
    mismatches: list = field(default_factory=list)  # (choice, reason)
    errors: list = field(default_factory=list)  # replay exceptions

    @property
    def outcome(self) -> float:
        return 1.0 if self.won else (-1.0 if self.won is False else 0.0)

    def arrays(self) -> dict[str, np.ndarray]:
        n = len(self.actions)
        return dict(
            obs=np.asarray(self.obs, dtype=np.float32).reshape(n, OBS_SIZE),
            masks=np.asarray(self.masks, dtype=np.int8).reshape(n, N_ACTIONS),
            actions=np.asarray(self.actions, dtype=np.int64),
            outcomes=np.full(n, self.outcome, dtype=np.float32),
            steps_left=np.arange(n - 1, -1, -1, dtype=np.int64),
        )


# ---------------------------------------------------------------------------
# /choose -> action


def parse_choice(message: str) -> Optional[tuple[str, str, bool]]:
    """``'battle-x|/choose move 2 terastallize|7'`` -> ``('move', '2', True)``.

    Returns None for non-decision messages (chat, ``/timer on``, ...)."""
    parts = message.split("|")
    body = parts[1].strip() if len(parts) > 1 else message.strip()
    if body.startswith("/choose "):
        body = body[len("/choose "):].strip()
    elif body.startswith("/switch ") or body.startswith("/move "):
        body = body[1:]
    else:
        return None
    if body.startswith("switch "):
        return "switch", body[len("switch "):].strip(), False
    if body.startswith("move "):
        tokens = body[len("move "):].split()
        if not tokens:
            return None
        gimmicks = {"terastallize", "mega", "zmove", "dynamax", "max", "ultra"}
        tera = "terastallize" in tokens
        move = " ".join(t for t in tokens if t not in gimmicks)
        return "move", move, tera
    return None


def choice_to_order(choice: tuple[str, str, bool], battle: Battle,
                    request: dict[str, Any]) -> SingleBattleOrder:
    kind, arg, tera = choice
    if kind == "switch":
        side = request.get("side", {}).get("pokemon", [])
        if arg.isdigit():
            idx = int(arg) - 1
            if not 0 <= idx < len(side):
                raise ChoiceError(f"switch slot {arg} out of range")
            entry = side[idx]
            mon = battle.team.get(entry["ident"])
            if mon is None:
                species = to_id_str(entry["details"].split(",")[0])
                mon = next((m for m in battle.team.values() if m.species == species
                            or m.base_species == species), None)
        else:
            target = to_id_str(arg)
            mon = next((m for m in battle.team.values()
                        if to_id_str(m.name) == target or m.species == target), None)
        if mon is None:
            raise ChoiceError(f"switch target {arg!r} not in team")
        return SingleBattleOrder(mon)

    if arg.isdigit():
        moves = (request.get("active") or [{}])[0].get("moves", [])
        idx = int(arg) - 1
        if not 0 <= idx < len(moves):
            raise ChoiceError(f"move slot {arg} out of range")
        move_id = to_id_str(moves[idx].get("id") or moves[idx]["move"])
    else:
        move_id = to_id_str(arg)
    move = next((m for m in battle.available_moves if m.id == move_id), None)
    if move is None and battle.active_pokemon is not None:
        move = battle.active_pokemon.moves.get(move_id)
    if move is None:
        raise ChoiceError(f"move {move_id!r} not available")
    return SingleBattleOrder(move, terastallize=tera)


def order_to_action(order: SingleBattleOrder, battle: Battle) -> int:
    """Our action index for ``order``; raises ChoiceError instead of randomising."""
    try:
        action = int(SinglesEnv.order_to_action(order, battle, strict=True))
    except (ValueError, AssertionError) as exc:
        raise ChoiceError(str(exc)) from exc
    if not 0 <= action < N_ACTIONS:
        raise ChoiceError(f"action {action} out of range")
    return action


# ---------------------------------------------------------------------------
# Log replay


def read_log(path: Path | str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                break  # truncated last line of a live file
    return records


def _next_choice(records: list[dict], start: int) -> Optional[str]:
    """First decision message Foul Play sent after ``records[start]`` and before the
    next ``|request|`` frame (the last one if it sent several, e.g. after an error)."""
    found = None
    for rec in records[start + 1:]:
        if rec["dir"] == "in" and "\n|request|" in rec["msg"]:
            break
        if rec["dir"] == "out" and parse_choice(rec["msg"]) is not None:
            found = rec["msg"]
    return found


def replay_records(records: list[dict], battle_format: str = "gen9randombattle",
                   tag: str = "") -> BattleSamples:
    username = next((r.get("username", "") for r in records if r["dir"] == "meta"), "")
    out = BattleSamples(tag=tag, username=username)
    gen = GenData.from_format(battle_format).gen
    battle: Optional[Battle] = None

    for i, rec in enumerate(records):
        if rec["dir"] != "in":
            continue
        split_messages = [m.split("|") for m in rec["msg"].split("\n")]
        if not split_messages[0][0].startswith(">battle"):
            continue
        if battle is None:
            if not (len(split_messages) > 1 and len(split_messages[1]) > 1
                    and split_messages[1][1] == "init"):
                continue  # frames before |init| (should not happen)
            battle_tag = split_messages[0][0][1:]
            out.tag = out.tag or battle_tag
            battle = Battle(battle_tag=battle_tag, username=username, logger=_QUIET, gen=gen)
        try:
            _apply_frame(battle, split_messages, records, i, out)
        except Exception as exc:  # keep going; record for the report
            out.errors.append(f"{type(exc).__name__}: {exc}")
    if battle is not None:
        out.turns = battle.turn
        out.finished = battle.finished
        out.won = battle.won
        out.opponent = battle.opponent_username or ""
    return out


def _apply_frame(battle: Battle, split_messages: list[list[str]], records: list[dict],
                 index: int, out: BattleSamples) -> None:
    """Mirror of ``poke_env.player.Player._handle_battle_message`` (singles, no OTS)."""
    for split_message in split_messages[1:]:
        if not split_message or len(split_message) == 1:
            continue
        kind = split_message[1]
        if kind == "":
            battle.parse_message(split_message)
        elif kind in MESSAGES_TO_IGNORE:
            pass
        elif kind == "request":
            if split_message[2]:
                request = orjson.loads(split_message[2])
                battle.parse_request(request, False)
                if not battle._wait and not battle.teampreview:
                    _decision(battle, request, records, index, out)
        elif kind in ("win", "tie"):
            if kind == "win":
                battle.won_by(split_message[2])
            else:
                battle.tied()
        elif kind in ("error", "bigerror", "showteam"):
            pass
        else:
            battle.parse_message(split_message)


def _decision(battle: Battle, request: dict, records: list[dict], index: int,
              out: BattleSamples) -> None:
    out.decisions += 1
    choice_msg = _next_choice(records, index)
    if choice_msg is None:
        out.mismatches.append(("<none>", "no choice sent"))
        return
    choice = parse_choice(choice_msg)
    obs = embed_battle(battle)
    mask = np.asarray(SinglesEnv.get_action_mask(battle), dtype=np.int8)
    try:
        action = order_to_action(choice_to_order(choice, battle, request), battle)
        if not mask[action]:
            raise ChoiceError(f"action {action} masked out (mask={np.flatnonzero(mask).tolist()})")
    except ChoiceError as exc:
        out.mismatches.append((choice_msg.split("|", 1)[-1], str(exc)))
        return
    out.obs.append(obs)
    out.masks.append(mask)
    out.actions.append(action)


def replay_log(path: Path | str, battle_format: str = "gen9randombattle") -> BattleSamples:
    path = Path(path)
    return replay_records(read_log(path), battle_format, tag=path.stem)


def merge_samples(parts: Iterable[BattleSamples]) -> dict[str, np.ndarray]:
    arrays = [p.arrays() for p in parts if p.actions]
    keys = ("obs", "masks", "actions", "outcomes", "steps_left")
    if not arrays:
        return dict(obs=np.zeros((0, OBS_SIZE), np.float32), masks=np.zeros((0, N_ACTIONS), np.int8),
                    actions=np.zeros(0, np.int64), outcomes=np.zeros(0, np.float32),
                    steps_left=np.zeros(0, np.int64))
    return {k: np.concatenate([a[k] for a in arrays]) for k in keys}
