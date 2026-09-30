"""Rebuild poke-env ``Battle`` objects from raw Pokemon Showdown protocol.

Live play reads the protocol the browser client receives (the same websocket
messages poke-env gets during training) and replays it through poke-env's own
``Battle.parse_message`` / ``Battle.parse_request``. The resulting ``Battle``
is embedded with ``showdownrl.battle_features.embed_battle`` exactly like the
training environment, so the model sees the same state representation live.

This module is pure Python (no browser, no network) so it can be unit-tested
against recorded protocol logs.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from poke_env.battle import Battle, Move, Pokemon
from poke_env.data.normalize import to_id_str
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder, SingleBattleOrder

LOGGER = logging.getLogger("showdownrl.protocol")

# Messages poke-env's Player drops before they reach the Battle object.
PLAYER_IGNORED = set(Player.MESSAGES_TO_IGNORE) | {"showteam", "bigerror"}


def parse_room_message(message: str) -> tuple[str, list[str]]:
    """Split one raw server message into ``(roomid, lines)``.

    Showdown prefixes room messages with ``>roomid``; lobby/global messages
    have no prefix and return an empty room id.
    """
    message = message[:-1] if message.endswith("\n") else message
    if message.startswith(">"):
        header, _, body = message.partition("\n")
        return header[1:].strip(), body.split("\n") if body else []
    return "", message.split("\n")


def initial_side_order(lines: Iterable[str], request: dict[str, Any], role: str) -> list[dict[str, Any]]:
    """Recover ``request.side.pokemon`` as it was at battle start.

    Showdown keeps the active Pokemon in slot 0 and swaps it with the incoming
    Pokemon on every switch/drag, so undoing our logged switch-ins (newest
    first) restores the initial order.
    """
    order = list(((request or {}).get("side") or {}).get("pokemon") or [])
    names = [to_id_str(_ident_name(entry.get("ident", ""))) for entry in order]
    switch_ins: list[str] = []
    for line in lines:
        parts = line.split("|")
        if len(parts) > 2 and parts[1] in ("switch", "drag") and parts[2].startswith(role):
            switch_ins.append(to_id_str(_ident_name(parts[2])))
    for index in range(len(switch_ins) - 1, 0, -1):
        incoming, previous = switch_ins[index], switch_ins[index - 1]
        if not names or names[0] != incoming or previous not in names:
            break  # log and request disagree (e.g. Illusion); keep what we have
        j = names.index(previous)
        names[0], names[j] = names[j], names[0]
        order[0], order[j] = order[j], order[0]
    return order


def gen_from_tag(battle_tag: str, default: int = 9) -> int:
    match = re.search(r"gen(\d+)", battle_tag or "")
    return int(match.group(1)) if match else default


@dataclass
class ProtocolBattle:
    """Incrementally mirror one battle room as a poke-env ``Battle``.

    Lines are processed in arrival order with the same dispatch rules as
    ``poke_env.player.Player._handle_battle_message``. Until our side is known
    (from a ``|request|`` or a ``|player|`` line matching ``username``) lines
    are buffered, then replayed into a freshly created ``Battle`` whose
    username is set to the exact name the server uses for our side, so
    poke-env's ``|player|`` handling assigns the correct role.
    """

    battle_tag: str = ""
    username: str = ""
    role: Optional[str] = None
    gen: Optional[int] = None
    battle: Optional[Battle] = None
    request: Optional[dict[str, Any]] = None
    answered_rqid: Any = None
    last_error: str = ""
    parse_errors: list[str] = field(default_factory=list)
    player_names: dict[str, str] = field(default_factory=dict)
    _pending: list[list[str]] = field(default_factory=list)
    _seed_entries: list[dict[str, Any]] = field(default_factory=list)
    _request_seq: int = 0
    _answered_seq: int = -1
    _error_seq: int = -1

    # ------------------------------------------------------------------ input

    def feed(self, message: str) -> None:
        """Feed one raw server message (``>room\\n|line\\n|line...``)."""
        roomid, lines = parse_room_message(message)
        if roomid and not self.battle_tag:
            self.battle_tag = roomid
        self.feed_lines(lines)

    def feed_lines(self, lines: Iterable[str]) -> None:
        for line in lines:
            if not line or not line.startswith("|"):
                continue
            self._feed_split(line.split("|"))

    def feed_request(self, request: dict[str, Any] | str) -> None:
        payload = request if isinstance(request, str) else json.dumps(request)
        self._feed_split(["", "request", payload])

    @classmethod
    def from_messages(cls, messages: Iterable[str], username: str = "", role: Optional[str] = None) -> "ProtocolBattle":
        state = cls(username=username, role=role)
        for message in messages:
            state.feed(message)
        return state

    @classmethod
    def from_snapshot(
        cls,
        lines: Iterable[str],
        request: dict[str, Any] | None,
        *,
        battle_tag: str = "",
        username: str = "",
        role: Optional[str] = None,
    ) -> "ProtocolBattle":
        """Rebuild from a client-side battle log plus the latest request.

        Fallback for when the raw message stream is unavailable. Only the
        latest request is known, so our team is pre-registered in its
        battle-start order (recovered by undoing the logged switches) right
        after our lead enters. That keeps ``battle.team`` - and therefore the
        switch action indices and team features - in the same order training
        produces from the live message stream.
        """
        lines = [line for line in lines if line and not line.startswith("|request|")]
        state = cls(battle_tag=battle_tag, username=username, role=role)
        if request:
            side = request.get("side") or {}
            if state.role is None and side.get("id") in ("p1", "p2"):
                state.role = side["id"]
            if state.role:
                state._seed_entries = initial_side_order(lines, request, state.role)
        state.feed_lines(lines)
        if request:
            state.feed_request(request)
        return state

    # --------------------------------------------------------------- dispatch

    def _feed_split(self, split: list[str]) -> None:
        if len(split) < 2:
            return
        kind = split[1]
        if kind == "player" and len(split) >= 4 and split[2] in ("p1", "p2") and split[3]:
            self.player_names[split[2]] = split[3]
            if self.role is None and self.username and to_id_str(split[3]) == to_id_str(self.username):
                self.role = split[2]
        elif kind == "request" and len(split) > 2 and split[2]:
            payload = "|".join(split[2:])
            try:
                request = json.loads(payload)
            except json.JSONDecodeError as exc:
                self.parse_errors.append(f"bad request json: {exc}")
                return
            split = ["", "request", payload]
            side = request.get("side") or {}
            if self.role is None and side.get("id") in ("p1", "p2"):
                self.role = side["id"]
            if side.get("id") and side.get("name"):
                self.player_names.setdefault(side["id"], side["name"])
        elif kind == "error" and len(split) > 2:
            self.last_error = "|".join(split[2:])
            self._error_seq = self._request_seq

        if self.battle is None:
            self._pending.append(split)
            if self.role is not None:
                self._create_battle()
            return
        self._apply(split)

    def _create_battle(self) -> None:
        assert self.role is not None
        name = self.player_names.get(self.role) or self.username or self.role
        tag = self.battle_tag or "battle-unknown"
        self.battle = Battle(
            battle_tag=tag,
            username=name,
            logger=LOGGER,
            gen=self.gen or gen_from_tag(tag),
            save_replays=False,
        )
        self.battle._player_role = self.role  # noqa: SLF001 - poke-env internals
        pending, self._pending = self._pending, []
        for split in pending:
            self._apply(split)

    def _apply(self, split: list[str]) -> None:
        battle = self.battle
        assert battle is not None
        kind = split[1]
        try:
            if kind == "":
                battle.parse_message(split)
            elif kind in PLAYER_IGNORED or kind == "error":
                return
            elif kind == "request":
                request = json.loads(split[2])
                battle.parse_request(request)
                self.request = request
                self._request_seq += 1
            elif kind == "win":
                if not battle.finished:
                    battle.won_by(split[2])
            elif kind == "tie":
                if not battle.finished:
                    battle.tied()
            elif kind == "player":
                battle.parse_message(split)
                # poke-env infers the role by comparing names; our role is
                # already known for certain, so keep it pinned.
                if len(split) > 3 and split[2] == self.role and split[3]:
                    battle._player_username = split[3]  # noqa: SLF001
                battle._player_role = self.role  # noqa: SLF001
            else:
                battle.parse_message(split)
                if self._seed_entries and kind in ("switch", "drag") and split[2].startswith(str(self.role)):
                    # Stand-in for the battle-start request: register the team
                    # in its initial order, lead active (see from_snapshot).
                    entries, self._seed_entries = self._seed_entries, []
                    battle._update_team_from_request(  # noqa: SLF001
                        {"pokemon": [{**entry, "active": i == 0} for i, entry in enumerate(entries)]}
                    )
        except Exception as exc:  # keep going: one odd line must not kill live play
            self.parse_errors.append(f"{'|'.join(split)[:120]}: {type(exc).__name__}: {exc}")
            LOGGER.debug("protocol parse error", exc_info=True)

    # ------------------------------------------------------------------ state

    @property
    def ready(self) -> bool:
        return self.battle is not None and self.request is not None

    @property
    def finished(self) -> bool:
        return bool(self.battle is not None and self.battle.finished)

    @property
    def request_seq(self) -> int:
        """Number of requests applied so far (identifies the current request)."""
        return self._request_seq

    @property
    def rqid(self) -> Any:
        return (self.request or {}).get("rqid")

    @property
    def needs_decision(self) -> bool:
        """True when the latest request asks us for a move/switch not yet sent."""
        if not self.ready or self.finished:
            return False
        request = self.request or {}
        if request.get("wait") or request.get("teamPreview"):
            return False
        if self._error_seq == self._request_seq and self._answered_seq == self._request_seq:
            return True  # our last choice for this request was rejected
        return self._answered_seq != self._request_seq

    def mark_answered(self) -> None:
        self._answered_seq = self._request_seq
        self.answered_rqid = self.rqid
        self._error_seq = -1


class BattleRoomTracker:
    """Route a stream of raw client messages to one ``ProtocolBattle`` per room.

    Also supports the snapshot fallback, where each poll rebuilds the battle
    from the client's stored log and latest request; answered request ids are
    remembered so a rebuilt battle does not ask for the same decision twice.
    """

    def __init__(self, username: str = ""):
        self.username = username
        self.rooms: dict[str, ProtocolBattle] = {}
        self.last_room: str = ""
        self._snapshot_answered: dict[str, Any] = {}

    def ingest(self, messages: Iterable[str]) -> None:
        for message in messages:
            roomid, _ = parse_room_message(message)
            if not roomid.startswith("battle-"):
                continue
            state = self.rooms.get(roomid)
            if state is not None and "\n|init|battle" in message:
                # Rejoining a room (e.g. after a reconnect) replays its whole log:
                # start from a fresh battle instead of applying it twice.
                state = None
            if state is None:
                state = self.rooms[roomid] = ProtocolBattle(battle_tag=roomid, username=self.username)
            state.feed(message)
            if state.battle is not None:
                self.last_room = roomid

    def ingest_snapshot(self, roomid: str, lines: Iterable[str], request: dict[str, Any] | None) -> ProtocolBattle:
        state = ProtocolBattle.from_snapshot(lines, request, battle_tag=roomid, username=self.username)
        if state.rqid is not None and self._snapshot_answered.get(roomid) == state.rqid:
            state.mark_answered()
        self.rooms[roomid] = state
        if state.battle is not None:
            self.last_room = roomid
        return state

    def mark_answered(self, state: ProtocolBattle) -> None:
        state.mark_answered()
        if state.battle_tag:
            self._snapshot_answered[state.battle_tag] = state.rqid

    def current(self) -> Optional[ProtocolBattle]:
        """The battle we play in that most recently received a message."""
        state = self.rooms.get(self.last_room)
        return state if state is not None and state.battle is not None else None

    def forget(self, roomid: str) -> None:
        self.rooms.pop(roomid, None)
        self._snapshot_answered.pop(roomid, None)
        if self.last_room == roomid:
            self.last_room = ""


# ---------------------------------------------------------------------------
# Order -> browser click


@dataclass(frozen=True)
class ClickPlan:
    """What to click in the Showdown web client for one poke-env order."""

    kind: str  # "move", "switch" or "none"
    slot: int = 0  # 1-based move slot / 1-based request.side.pokemon slot
    label: str = ""
    move_id: str = ""
    species: str = ""
    terastallize: bool = False
    selectors: tuple[str, ...] = ()
    tera_selectors: tuple[str, ...] = ()
    choose_command: str = ""


TERA_SELECTORS = (
    "input[name='terastallize']",  # classic client (client-battle.js)
    "input[name='tera']",  # preact client (panel-battle.tsx)
)


def _request_moves(request: dict[str, Any] | None) -> list[dict[str, Any]]:
    active = (request or {}).get("active") or []
    return list((active[0] if active else {}).get("moves") or [])


def _request_side(request: dict[str, Any] | None) -> list[dict[str, Any]]:
    return list(((request or {}).get("side") or {}).get("pokemon") or [])


def _ident_name(ident: str) -> str:
    return ident.split(": ", 1)[1] if ": " in ident else ident


def move_slot(move: Move, request: dict[str, Any] | None) -> int:
    """1-based slot of ``move`` in the request's move list (0 if absent)."""
    moves = _request_moves(request)
    if move.id == "recharge":
        return 1
    for index, entry in enumerate(moves, start=1):
        entry_id = entry.get("id") or to_id_str(entry.get("move", ""))
        if entry_id == move.id:
            return index
    if move.id.startswith("hiddenpower"):
        for index, entry in enumerate(moves, start=1):
            if str(entry.get("id", "")).startswith("hiddenpower"):
                return index
    return 0


def switch_slot(pokemon: Pokemon, request: dict[str, Any] | None) -> int:
    """1-based slot of ``pokemon`` in ``request.side.pokemon`` (0 if absent)."""
    target = to_id_str(pokemon.name)
    for index, entry in enumerate(_request_side(request), start=1):
        if to_id_str(_ident_name(entry.get("ident", ""))) == target:
            return index
    return 0


def order_to_click(order: BattleOrder, request: dict[str, Any] | None) -> ClickPlan:
    """Translate a poke-env order into web-client button selectors.

    Selectors cover both Showdown clients: the classic Backbone client served
    at play.pokemonshowdown.com (``button[name=chooseMove][value=N]``,
    ``button[name=chooseSwitch][value=i]`` with 0-based ``i``) and the preact
    client (``button[data-cmd="/move N"]``, ``button[data-cmd="/switch N"]``).
    """
    if not isinstance(order, SingleBattleOrder):
        return ClickPlan(kind="none", label=str(order))
    target = order.order
    if isinstance(target, Move):
        slot = move_slot(target, request)
        moves = _request_moves(request)
        name = moves[slot - 1].get("move", target.id) if 0 < slot <= len(moves) else target.id
        selectors: list[str] = []
        if slot:
            selectors += [
                f".movemenu button[name='chooseMove'][value='{slot}']",
                f".movemenu button[data-cmd='/move {slot}']",
                f".movemenu button[data-cmd^='/move {slot} ']",
            ]
        selectors.append(f".movemenu button[data-move={json.dumps(name)}]")
        tera = bool(order.terastallize)
        return ClickPlan(
            kind="move",
            slot=slot,
            label=f"{name}{' + Tera' if tera else ''}",
            move_id=target.id,
            terastallize=tera,
            selectors=tuple(selectors),
            tera_selectors=TERA_SELECTORS if tera else (),
            choose_command=f"move {slot}{' terastallize' if tera else ''}" if slot else "",
        )
    if isinstance(target, Pokemon):
        slot = switch_slot(target, request)
        selectors = []
        if slot:
            selectors += [
                f".switchmenu button[name='chooseSwitch'][value='{slot - 1}']",
                f".switchmenu button[data-cmd='/switch {slot}']",
            ]
        return ClickPlan(
            kind="switch",
            slot=slot,
            label=f"Switch: {target.name}",
            species=target.species,
            selectors=tuple(selectors),
            choose_command=f"switch {slot}" if slot else "",
        )
    return ClickPlan(kind="none", label=str(order.message))


# ---------------------------------------------------------------------------
# Debug / test helpers


def battle_summary(battle: Battle) -> dict[str, Any]:
    """Small JSON-able summary of the decision-relevant battle state."""
    me = battle.active_pokemon
    opp = battle.opponent_active_pokemon
    return {
        "role": battle.player_role,
        "turn": battle.turn,
        "active": me.species if me else None,
        "active_hp": [me.current_hp, me.max_hp] if me else None,
        "opponent": opp.species if opp else None,
        "opponent_hp_fraction": round(opp.current_hp_fraction, 4) if opp else None,
        "moves": [m.id for m in battle.available_moves],
        "switches": [m.species for m in battle.available_switches],
        "force_switch": bool(battle.force_switch),
        "can_tera": bool(battle.can_tera),
        "trapped": bool(battle.trapped),
    }
