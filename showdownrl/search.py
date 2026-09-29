"""Decision-time search with poke-engine over sampled opponent sets.

``poke-engine`` (https://github.com/pmariglia/poke-engine, MIT) is a Rust
battle simulator with Python bindings and a Monte Carlo tree search. It needs
fully specified states, while a Random Battle is a game of imperfect
information: we see our own team exactly but only what the opponent revealed.

This module

1. converts a poke-env ``Battle`` (our perspective) into several poke-engine
   ``State`` "determinizations": our side is copied from the request, each
   opponent Pokemon gets a Random Battle set (role, moves, item, ability, tera
   type) sampled from ``data/gen9randombattle.json`` consistent with what has
   been revealed, and unrevealed opponent slots are filled with sampled species
   (or left as fainted placeholders);
2. runs ``poke_engine.monte_carlo_tree_search`` on each determinization and
   averages side one's visit shares across samples;
3. maps each engine move-choice string back to our 26-action poke-env index and
   optionally blends it with a policy network's action probabilities:
   ``score = (1 - prior_weight) * visit_share + prior_weight * policy_prob``,
   taking the argmax over legal actions.

poke-engine is an optional dependency (``pip install .[search]``, see
pyproject.toml: it must be built with the gen9 ``terastallization`` feature).
Everything here fails soft: any conversion or engine error falls back to the
policy (or the smart heuristic) and is counted.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from concurrent.futures import Executor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional, Sequence

import numpy as np
from poke_env.battle import (
    AbstractBattle,
    Battle,
    Effect,
    Field,
    Pokemon,
    PokemonType,
    SideCondition,
    Status,
    Weather,
)
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str
from poke_env.environment import SinglesEnv
from poke_env.player import BattleOrder, Player

from showdownrl.battle_features import (
    DEFAULT_RANDBATS_EV,
    DEFAULT_RANDBATS_IV,
    N_ACTIONS,
    _randbats,
    _species_entry,
    embed_battle,
)

try:  # optional dependency
    import poke_engine as pe
except ImportError:  # pragma: no cover - exercised when the extra is missing
    pe = None

LOGGER = logging.getLogger(__name__)
ENGINE_AVAILABLE = pe is not None
_POKEDEX = GenData.from_gen(9).pokedex
STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")

# ---------------------------------------------------------------------------
# poke-env -> poke-engine vocabularies

STATUS_MAP = {
    Status.BRN: "burn",
    Status.FRZ: "freeze",
    Status.PAR: "paralyze",
    Status.PSN: "poison",
    Status.TOX: "toxic",
    Status.SLP: "sleep",
}
WEATHER_MAP = {
    Weather.SUNNYDAY: "sun",
    Weather.RAINDANCE: "rain",
    Weather.SANDSTORM: "sand",
    Weather.HAIL: "hail",
    Weather.SNOWSCAPE: "snow",
    Weather.DESOLATELAND: "harshsun",
    Weather.PRIMORDIALSEA: "heavyrain",
}
TERRAIN_MAP = {
    Field.ELECTRIC_TERRAIN: "electricterrain",
    Field.GRASSY_TERRAIN: "grassyterrain",
    Field.MISTY_TERRAIN: "mistyterrain",
    Field.PSYCHIC_TERRAIN: "psychicterrain",
}
# Volatile statuses poke-env tracks as Effects and poke-engine models.
EFFECT_VOLATILES = {
    Effect.AQUA_RING: "aquaring",
    Effect.ATTRACT: "attract",
    Effect.CHARGE: "charge",
    Effect.CONFUSION: "confusion",
    Effect.CURSE: "curse",
    Effect.DEFENSE_CURL: "defensecurl",
    Effect.DESTINY_BOND: "destinybond",
    Effect.ENCORE: "encore",
    Effect.FLASH_FIRE: "flashfire",
    Effect.FOCUS_ENERGY: "focusenergy",
    Effect.GLAIVE_RUSH: "glaiverush",
    Effect.HEAL_BLOCK: "healblock",
    Effect.INGRAIN: "ingrain",
    Effect.LASER_FOCUS: "laserfocus",
    Effect.LEECH_SEED: "leechseed",
    Effect.LOCKED_MOVE: "lockedmove",
    Effect.MAGNET_RISE: "magnetrise",
    Effect.MINIMIZE: "minimize",
    Effect.NO_RETREAT: "noretreat",
    Effect.OCTOLOCK: "octolock",
    Effect.PERISH1: "perish1",
    Effect.PERISH2: "perish2",
    Effect.PERISH3: "perish3",
    Effect.POWER_TRICK: "powertrick",
    Effect.PROTOSYNTHESISATK: "protosynthesisatk",
    Effect.PROTOSYNTHESISDEF: "protosynthesisdef",
    Effect.PROTOSYNTHESISSPA: "protosynthesisspa",
    Effect.PROTOSYNTHESISSPD: "protosynthesisspd",
    Effect.PROTOSYNTHESISSPE: "protosynthesisspe",
    Effect.QUARKDRIVEATK: "quarkdriveatk",
    Effect.QUARKDRIVEDEF: "quarkdrivedef",
    Effect.QUARKDRIVESPA: "quarkdrivespa",
    Effect.QUARKDRIVESPD: "quarkdrivespd",
    Effect.QUARKDRIVESPE: "quarkdrivespe",
    Effect.ROOST: "roost",
    Effect.SALT_CURE: "saltcure",
    Effect.SLOW_START: "slowstart",
    Effect.SMACK_DOWN: "smackdown",
    Effect.SUBSTITUTE: "substitute",
    Effect.SYRUP_BOMB: "syrupbomb",
    Effect.TAR_SHOT: "tarshot",
    Effect.TAUNT: "taunt",
    Effect.THROAT_CHOP: "throatchop",
    Effect.TORMENT: "torment",
    Effect.TYPECHANGE: "typechange",
    Effect.UPROAR: "uproar",
    Effect.YAWN: "yawn",
}
TRAPPING_EFFECTS = {
    Effect.PARTIALLY_TRAPPED, Effect.BIND, Effect.WRAP, Effect.FIRE_SPIN, Effect.WHIRLPOOL,
    Effect.SAND_TOMB, Effect.MAGMA_STORM, Effect.INFESTATION, Effect.SNAP_TRAP,
    Effect.THUNDER_CAGE, Effect.CLAMP,
}
# Two-turn moves poke-engine tracks as a volatile named after the move.
CHARGE_VOLATILES = {
    "bounce", "dig", "dive", "electroshot", "fly", "freezeshock", "geomancy", "iceburn",
    "meteorbeam", "phantomforce", "razorwind", "shadowforce", "skullbash", "skyattack",
    "skydrop", "solarbeam", "solarblade",
}
TIMED_SIDE_CONDITIONS = {  # poke-env stores the start turn; engine wants turns left
    SideCondition.REFLECT: ("reflect", 5),
    SideCondition.LIGHT_SCREEN: ("light_screen", 5),
    SideCondition.AURORA_VEIL: ("aurora_veil", 5),
    SideCondition.TAILWIND: ("tailwind", 4),
    SideCondition.SAFEGUARD: ("safeguard", 5),
    SideCondition.MIST: ("mist", 5),
    SideCondition.LUCKY_CHANT: ("lucky_chant", 5),
}
BOOST_FIELDS = {
    "atk": "attack_boost", "def": "defense_boost", "spa": "special_attack_boost",
    "spd": "special_defense_boost", "spe": "speed_boost", "accuracy": "accuracy_boost",
    "evasion": "evasion_boost",
}
CHOICE_ITEMS = {"choiceband", "choicespecs", "choicescarf"}


def _type_name(t: Optional[PokemonType]) -> str:
    if t is None:
        return "typeless"
    name = t.name.lower()
    return name if name not in {"three_question_marks"} else "typeless"


def _type_pair(types: Sequence[Optional[PokemonType]]) -> tuple[str, str]:
    names = [_type_name(t) for t in types if t is not None][:2]
    while len(names) < 2:
        names.append("typeless")
    return names[0], names[1]


def _type_from_str(name: str) -> str:
    return to_id_str(name) or "typeless"


@lru_cache(maxsize=1)
def engine_is_gen9() -> bool:
    """True if poke-engine was built with the gen9 ``terastallization`` feature.

    The sdist's default feature is gen4; that build never offers ``-tera`` choices.
    """
    if pe is None:
        return False
    mon = pe.Pokemon(id="pikachu", moves=[pe.Move(id="tackle")], tera_type="normal")
    state = pe.State(side_one=pe.Side(pokemon=[mon]), side_two=pe.Side(pokemon=[mon]))
    result = pe.monte_carlo_tree_search(state, duration_ms=0, iterations=10)
    return any(r.move_choice.endswith("-tera") for r in result.side_one)


# ---------------------------------------------------------------------------
# Engine id validation (unknown ids silently map to NONE inside the engine)


def _roundtrip_mon(**kwargs: Any):
    state = pe.State(side_one=pe.Side(pokemon=[pe.Pokemon(**kwargs)]))
    return pe.State.from_string(state.to_string()).side_one.pokemon[0]


@lru_cache(maxsize=None)
def engine_species(name: str) -> Optional[str]:
    """``name`` if poke-engine knows the species, else None."""
    if not name or name == "none":
        return None
    try:
        back = _roundtrip_mon(id=name).id.lower()
    except BaseException as exc:  # pyo3 panics are BaseException
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return None
    return name if back == name else None


@lru_cache(maxsize=None)
def engine_move(move_id: str) -> Optional[str]:
    if not move_id or move_id == "none":
        return None
    try:
        back = _roundtrip_mon(id="pikachu", moves=[pe.Move(id=move_id)]).moves[0].id.lower()
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return None
    return move_id if back == move_id else None


@lru_cache(maxsize=None)
def _fallback_species_pool() -> tuple[str, ...]:
    names = [n for n in ("pikachu", "eevee", "ditto", "snorlax", "mew", "rattata", "pidgey")
             if engine_species(n)]
    return tuple(names)


# ---------------------------------------------------------------------------
# Stats


def calc_stats(base: dict, level: int, evs: Optional[dict] = None,
               ivs: Optional[dict] = None) -> dict[str, int]:
    """Neutral-nature stats (Random Battles use neutral natures)."""
    out = {}
    for stat in STAT_KEYS:
        b = base.get(stat, 80)
        ev = (evs or {}).get(stat, DEFAULT_RANDBATS_EV)
        iv = (ivs or {}).get(stat, DEFAULT_RANDBATS_IV)
        core = math.floor((2 * b + iv + ev // 4) * level / 100)
        out[stat] = core + level + 10 if stat == "hp" else core + 5
    if base.get("hp") == 1:  # Shedinja
        out["hp"] = 1
    return out


# ---------------------------------------------------------------------------
# Opponent set sampling


@dataclass
class MonSpec:
    """Everything needed to build one poke-engine Pokemon."""

    species: str
    level: int
    types: tuple[str, str]
    base_types: tuple[str, str]
    hp: int
    maxhp: int
    stats: dict[str, int]
    ability: str
    item: str
    moves: list[tuple[str, int, bool]]  # (engine id, pp, disabled)
    tera_type: str = "typeless"
    terastallized: bool = False
    status: str = "none"
    sleep_turns: int = 0
    rest_turns: int = 0
    weight_kg: float = 50.0
    evs: tuple[int, ...] = (85, 85, 85, 85, 85, 85)

    def to_engine(self):
        return pe.Pokemon(
            id=self.species, level=int(self.level), types=self.types, base_types=self.base_types,
            hp=int(self.hp), maxhp=int(self.maxhp), ability=self.ability or "none",
            item=self.item or "none", nature="serious", evs=tuple(int(e) for e in self.evs),
            attack=int(self.stats["atk"]), defense=int(self.stats["def"]),
            special_attack=int(self.stats["spa"]), special_defense=int(self.stats["spd"]),
            speed=int(self.stats["spe"]), status=self.status, rest_turns=int(self.rest_turns),
            sleep_turns=int(self.sleep_turns), weight_kg=float(self.weight_kg),
            moves=[pe.Move(id=m, pp=int(pp), disabled=bool(d)) for m, pp, d in self.moves],
            terastallized=bool(self.terastallized), tera_type=self.tera_type,
        )


def _role_violations(role: dict, mon: Pokemon) -> int:
    role_moves = {to_id_str(m) for m in role.get("moves", [])}
    bad = sum(1 for m in mon.moves if m not in role_moves)
    item = mon.item
    if item and item != "unknown_item":
        if item not in {to_id_str(i) for i in role.get("items", [])}:
            bad += 1
    if mon.ability:
        if to_id_str(mon.ability) not in {to_id_str(a) for a in role.get("abilities", [])}:
            bad += 1
    if mon.is_terastallized and mon.tera_type is not None:
        if _type_name(mon.tera_type) not in {_type_from_str(t) for t in role.get("teraTypes", [])}:
            bad += 1
    return bad


def sample_role(entry: Optional[dict], mon: Optional[Pokemon], rng: random.Random) -> dict:
    """Random Battle role consistent with what ``mon`` revealed (fewest violations)."""
    if not entry or not entry.get("roles"):
        return {}
    roles = list(entry["roles"].values())
    if mon is None:
        return rng.choice(roles)
    scored = [(_role_violations(r, mon), r) for r in roles]
    best = min(s for s, _ in scored)
    return rng.choice([r for s, r in scored if s == best])


def _species_types(dex: dict) -> tuple[str, str]:
    return _type_pair([PokemonType.from_name(t) for t in dex.get("types", [])])


def _sample_moves(revealed: list[str], pool: list[str], rng: random.Random) -> list[str]:
    moves = [m for m in revealed if engine_move(m)][:4]
    rest = [m for m in pool if m not in moves and engine_move(m)]
    rng.shuffle(rest)
    return moves + rest[: max(0, 4 - len(moves))]


def _opponent_mon_spec(mon: Pokemon, rng: random.Random) -> MonSpec:
    entry = _species_entry(mon)
    role = sample_role(entry, mon, rng)
    level = mon.level or (entry or {}).get("level", 80)
    evs = dict((entry or {}).get("evs", {}))
    evs.update(role.get("evs", {}))
    ivs = dict((entry or {}).get("ivs", {}))
    ivs.update(role.get("ivs", {}))
    stats = calc_stats(mon.base_stats, level, evs, ivs)
    maxhp = stats["hp"]
    hp = 0 if mon.fainted else max(1, int(round(mon.current_hp_fraction * maxhp)))

    role_moves = [to_id_str(m) for m in role.get("moves", [])]
    move_ids = _sample_moves(list(mon.moves.keys()), role_moves, rng)
    last = mon.last_move.id if mon.last_move is not None else None

    item = mon.item
    if item == "unknown_item" or item is None:
        items = [to_id_str(i) for i in role.get("items", [])] or ["leftovers"]
        item = rng.choice(items)
    # Choice lock: an active choiced mon that already used a move is locked into it.
    locked = mon.active and item in CHOICE_ITEMS and last in move_ids

    moves = []
    for mid in move_ids:
        known = mon.moves.get(mid)
        pp = known.current_pp if known is not None else 16
        moves.append((mid, max(1, min(int(pp), 64)), bool(locked and mid != last)))

    ability = to_id_str(mon.ability) if mon.ability else ""
    if not ability:
        options = [to_id_str(a) for a in role.get("abilities", [])]
        possible = {to_id_str(a) for a in (mon.possible_abilities or [])}
        both = [a for a in options if not possible or a in possible]
        ability = rng.choice(both or options or sorted(possible) or ["none"])

    if mon.tera_type is not None and (mon.is_terastallized or not role.get("teraTypes")):
        tera = _type_name(mon.tera_type)
    else:
        tera = _type_from_str(rng.choice(role.get("teraTypes") or ["Normal"]))

    evs_t = tuple(int(evs.get(s, DEFAULT_RANDBATS_EV)) for s in STAT_KEYS)
    species = _engine_name_for(mon)
    return MonSpec(
        species=species, level=level, types=_type_pair(mon.base_types),
        base_types=_type_pair([mon._type_1, mon._type_2]), hp=hp, maxhp=maxhp, stats=stats,
        ability=ability, item=item or "none", moves=moves, tera_type=tera,
        terastallized=mon.is_terastallized, weight_kg=mon.weight or 50.0, evs=evs_t,
        **_status_fields(mon),
    )


def _unrevealed_spec(species_id: str, rng: random.Random) -> Optional[MonSpec]:
    entry = _randbats().get(species_id)
    dex = _POKEDEX.get(species_id)
    name = engine_species(species_id)
    if not entry or not dex or not name:
        return None
    role = sample_role(entry, None, rng)
    level = entry.get("level", 80)
    evs = dict(entry.get("evs", {}))
    evs.update(role.get("evs", {}))
    ivs = dict(entry.get("ivs", {}))
    ivs.update(role.get("ivs", {}))
    stats = calc_stats(dex["baseStats"], level, evs, ivs)
    moves = _sample_moves([], [to_id_str(m) for m in role.get("moves", [])], rng)
    types = _species_types(dex)
    return MonSpec(
        species=name, level=level, types=types, base_types=types, hp=stats["hp"],
        maxhp=stats["hp"], stats=stats,
        ability=to_id_str(rng.choice(role.get("abilities") or ["none"])),
        item=to_id_str(rng.choice(role.get("items") or ["leftovers"])),
        moves=[(m, 16, False) for m in moves],
        tera_type=_type_from_str(rng.choice(role.get("teraTypes") or ["Normal"])),
        weight_kg=float(dex.get("weightkg", 50.0)),
        evs=tuple(int(evs.get(s, DEFAULT_RANDBATS_EV)) for s in STAT_KEYS),
    )


def _engine_name_for(mon: Pokemon) -> str:
    for name in (mon.species, to_id_str(mon.base_species)):
        valid = engine_species(name)
        if valid:
            return valid
    return "none"


def _status_fields(mon: Pokemon) -> dict:
    status = STATUS_MAP.get(mon.status, "none") if mon.status is not None else "none"
    out = {"status": status, "sleep_turns": 0, "rest_turns": 0}
    if mon.status == Status.SLP:
        counter = int(mon.status_counter or 0)
        last = mon.last_move.id if mon.last_move is not None else None
        if last == "rest":
            out["rest_turns"] = max(1, min(3, 3 - counter))
        else:
            out["sleep_turns"] = max(0, min(3, counter))
    return out


# ---------------------------------------------------------------------------
# Our side


def _request_tera_types(battle: Battle) -> dict[str, str]:
    """ident -> tera type from the last request (poke-env does not store ours)."""
    out = {}
    try:
        for entry in (battle.last_request or {}).get("side", {}).get("pokemon", []):
            if entry.get("teraType"):
                out[entry["ident"]] = _type_from_str(entry["teraType"])
    except (AttributeError, TypeError):
        pass
    return out


def _our_mon_spec(mon: Pokemon, battle: Battle, name: str, active: bool,
                  tera_types: Optional[dict] = None, ident: str = "") -> MonSpec:
    stats = dict(mon.stats or {})
    level = mon.level or 80
    if not all(stats.get(s) for s in ("atk", "def", "spa", "spd", "spe")):
        stats = calc_stats(mon.base_stats, level)
    maxhp = int(mon.max_hp or stats.get("hp") or 100)
    stats["hp"] = maxhp
    hp = 0 if mon.fainted else max(1, int(mon.current_hp or 0))

    tera = _type_name(mon.tera_type) if mon.tera_type is not None else \
        (tera_types or {}).get(ident, "typeless")
    known = list(mon.moves.values())[:4]
    avail = {m.id for m in battle.available_moves} if active else None
    restrict = active and not battle.force_switch and bool(battle.available_moves)
    moves = []
    for move in known:
        mid = engine_move(move.id)
        pp = max(0, min(int(move.current_pp), 64))
        if mid is None:
            moves.append(("none", 0, True))
            continue
        disabled = bool(restrict and move.id not in avail)
        if restrict and not disabled:
            pp = max(1, pp)  # the request says it is usable (PP tracking can drift)
        moves.append((mid, pp, disabled))
    return MonSpec(
        species=name, level=level, types=_type_pair(mon.base_types),
        base_types=_type_pair([mon._type_1, mon._type_2]), hp=hp, maxhp=maxhp,
        stats={k: int(v) for k, v in stats.items() if v is not None},
        ability=to_id_str(mon.ability or "") or "none", item=to_id_str(mon.item or "") or "none",
        moves=moves, tera_type=tera, terastallized=mon.is_terastallized,
        weight_kg=mon.weight or 50.0, **_status_fields(mon),
    )


# ---------------------------------------------------------------------------
# Side / field conversion


def _side_conditions(battle: Battle, conditions: dict, active: Optional[Pokemon]):
    kwargs: dict[str, int] = {}
    if SideCondition.STEALTH_ROCK in conditions:
        kwargs["stealth_rock"] = 1
    if SideCondition.STICKY_WEB in conditions:
        kwargs["sticky_web"] = 1
    kwargs["spikes"] = min(3, int(conditions.get(SideCondition.SPIKES, 0)))
    kwargs["toxic_spikes"] = min(2, int(conditions.get(SideCondition.TOXIC_SPIKES, 0)))
    for cond, (name, duration) in TIMED_SIDE_CONDITIONS.items():
        if cond in conditions:
            elapsed = max(0, battle.turn - int(conditions[cond]))
            kwargs[name] = max(1, duration - elapsed)
    if active is not None and not active.fainted:
        if active.status == Status.TOX:
            kwargs["toxic_count"] = max(0, min(15, int(active.status_counter or 0)))
        kwargs["protect"] = max(0, min(5, int(getattr(active, "protect_counter", 0) or 0)))
    return pe.SideConditions(**kwargs)


def _side_volatiles(active: Optional[Pokemon]) -> tuple[set[str], dict]:
    vols: set[str] = set()
    extra: dict[str, Any] = {}
    if active is None or active.fainted:
        return vols, extra
    for effect in active.effects:
        if effect in EFFECT_VOLATILES:
            vols.add(EFFECT_VOLATILES[effect])
        elif effect in TRAPPING_EFFECTS:
            vols.add("partiallytrapped")
    if active.must_recharge:
        vols.add("mustrecharge")
    preparing = active.preparing_move
    if preparing is not None and preparing.id in CHARGE_VOLATILES:
        vols.add(preparing.id)
    if "substitute" in vols:
        extra["substitute_health"] = max(1, int(active.max_hp or 100) // 4)
    for stat, fname in BOOST_FIELDS.items():
        extra[fname] = int(max(-6, min(6, active.boosts.get(stat, 0))))
    return vols, extra


def _last_used_move(mon: Optional[Pokemon], spec: Optional[MonSpec], volatiles: set[str]) -> str:
    """poke-engine ``last_used_move`` ("move:<slot>") for the active Pokemon.

    The engine panics if Encore is active without a last-used move, so Encore is
    dropped from ``volatiles`` when the encored move cannot be located.
    """
    last = mon.last_move.id if mon is not None and mon.last_move is not None else None
    ids = [m for m, _, _ in spec.moves] if spec is not None else []
    if last in ids:
        return f"move:{ids.index(last)}"
    volatiles.discard("encore")
    return "move:none"


def _build_side(specs: list[MonSpec], active_index: int, conditions, volatiles: set[str],
                extra: dict, force_switch: bool = False, force_trapped: bool = False,
                active_mon: Optional[Pokemon] = None):
    spec = specs[active_index] if active_index < len(specs) else None
    extra = {**extra, "last_used_move": _last_used_move(active_mon, spec, volatiles)}
    mons = [s.to_engine() for s in specs]
    while len(mons) < 6:
        mons.append(pe.Pokemon.create_fainted())
    return pe.Side(
        pokemon=mons[:6], active_index=str(active_index), side_conditions=conditions,
        volatile_statuses=volatiles, force_switch=force_switch, force_trapped=force_trapped,
        **extra,
    )


def _field_kwargs(battle: Battle) -> dict:
    kwargs: dict[str, Any] = {}
    for weather, start in battle.weather.items():
        if weather in WEATHER_MAP:
            kwargs["weather"] = WEATHER_MAP[weather]
            if weather in (Weather.DESOLATELAND, Weather.PRIMORDIALSEA):
                kwargs["weather_turns_remaining"] = -1
            else:
                kwargs["weather_turns_remaining"] = max(1, 5 - max(0, battle.turn - int(start)))
    for fld, start in battle.fields.items():
        if fld in TERRAIN_MAP:
            kwargs["terrain"] = TERRAIN_MAP[fld]
            kwargs["terrain_turns_remaining"] = max(1, 5 - max(0, battle.turn - int(start)))
        elif fld == Field.TRICK_ROOM:
            kwargs["trick_room"] = True
            kwargs["trick_room_turns_remaining"] = max(1, 5 - max(0, battle.turn - int(start)))
    return kwargs


# ---------------------------------------------------------------------------
# Battle -> states


@dataclass
class ChoiceMap:
    """How engine move-choice strings map to our poke-env action indices."""

    moves: dict[str, int] = field(default_factory=dict)  # engine move id -> slot 0..3
    switches: dict[str, int] = field(default_factory=dict)  # engine species -> team index

    def action(self, choice: str) -> Optional[int]:
        choice = choice.strip().lower()
        if choice.startswith("switch "):
            idx = self.switches.get(choice[len("switch "):])
            return idx
        tera = choice.endswith("-tera")
        name = choice[:-5] if tera else choice
        slot = self.moves.get(name)
        if slot is None:
            return None
        return 22 + slot if tera else 6 + slot


def _our_side(battle: Battle) -> tuple[list[MonSpec], int, ChoiceMap]:
    team = list(battle.team.values())[:6]
    active_mon = battle.active_pokemon
    active_idx = next((i for i, m in enumerate(team) if m is active_mon), None)
    if active_idx is None:
        active_idx = next((i for i, m in enumerate(team) if m.active), 0)
    names: list[str] = []
    spare = [n for n in _fallback_species_pool()]
    for mon in team:
        name = _engine_name_for(mon)
        if name == "none" or name in names:
            # Unknown / duplicate species: use a unique stand-in name so switch
            # strings stay unambiguous (stats are explicit, so the name barely matters).
            spare = [n for n in spare if n not in names and n not in
                     {_engine_name_for(m) for m in team}]
            name = spare.pop(0) if spare else "none"
        names.append(name)
    tera_types = _request_tera_types(battle)
    idents = list(battle.team.keys())[:6]
    specs = [_our_mon_spec(m, battle, names[i], i == active_idx, tera_types, idents[i])
             for i, m in enumerate(team)]
    cmap = ChoiceMap()
    for i, name in enumerate(names):
        if name != "none":
            cmap.switches[name] = i
    for slot, (mid, _, _) in enumerate(specs[active_idx].moves if specs else []):
        if mid != "none" and mid not in cmap.moves:
            cmap.moves[mid] = slot
    # poke-engine's can_use_tera() is "no Pokemon on this side is terastallized".
    if not battle.can_tera and not any(s.terastallized for s in specs):
        used = getattr(battle, "_used_tera", False)
        fainted = [s for s in specs if s.hp == 0]
        if used and fainted:
            fainted[0].terastallized = True
    return specs, active_idx, cmap


def _opponent_side(battle: Battle, rng: random.Random, unrevealed: str
                   ) -> tuple[list[MonSpec], int]:
    revealed = list(battle.opponent_team.values())[:6]
    specs = [_opponent_mon_spec(m, rng) for m in revealed]
    opp_active = battle.opponent_active_pokemon
    active_idx = next((i for i, m in enumerate(revealed) if m is opp_active), None)
    if active_idx is None:
        active_idx = next((i for i, s in enumerate(specs) if s.hp > 0), 0)
    if unrevealed == "sample" and len(specs) < 6:
        taken = {to_id_str(m.base_species) for m in revealed} | {m.species for m in revealed}
        pool = [k for k in _randbats() if k not in taken]
        rng.shuffle(pool)
        for species_id in pool:
            if len(specs) >= 6:
                break
            base = to_id_str(_POKEDEX.get(species_id, {}).get("baseSpecies", species_id))
            if base in taken:
                continue
            spec = _unrevealed_spec(species_id, rng)
            if spec is not None:
                specs.append(spec)
                taken.add(base)
    # Tera already spent by the opponent but the user is gone/unknown: flag a fainted slot.
    if getattr(battle, "_opponent_used_tera", False) and not any(s.terastallized for s in specs):
        fainted = [s for s in specs if s.hp == 0]
        if fainted:
            fainted[0].terastallized = True
    return specs, active_idx


def battle_to_states(battle: Battle, n_samples: int = 4, rng: Optional[random.Random] = None,
                     unrevealed: str = "sample") -> tuple[list, ChoiceMap]:
    """Sample ``n_samples`` poke-engine States for ``battle`` (side one = us).

    ``unrevealed``: ``"sample"`` fills unseen opponent slots with random
    Random Battle species, ``"fainted"`` leaves them as fainted placeholders.
    Returns the states and the ChoiceMap for translating side-one choices.
    """
    if pe is None:
        raise RuntimeError("poke_engine is not installed (pip install .[search])")
    rng = rng or random.Random()
    our_specs, our_active, cmap = _our_side(battle)
    our_active_mon = list(battle.team.values())[our_active] if battle.team else None
    our_vols, our_extra = _side_volatiles(our_active_mon)
    our_conditions = _side_conditions(battle, battle.side_conditions, our_active_mon)
    force_switch = bool(battle.force_switch)
    force_trapped = bool(battle.trapped) and not force_switch
    opp_active = battle.opponent_active_pokemon
    opp_vols, opp_extra = _side_volatiles(opp_active)
    fkw = _field_kwargs(battle)

    states = []
    for _ in range(max(1, n_samples)):
        side_one = _build_side(our_specs, our_active, our_conditions, set(our_vols),
                               dict(our_extra), force_switch=force_switch,
                               force_trapped=force_trapped, active_mon=our_active_mon)
        opp_specs, opp_idx = _opponent_side(battle, rng, unrevealed)
        opp_conditions = _side_conditions(battle, battle.opponent_side_conditions, opp_active)
        extra = dict(opp_extra)
        if "substitute_health" in extra and opp_idx < len(opp_specs):
            extra["substitute_health"] = max(1, opp_specs[opp_idx].maxhp // 4)
        side_two = _build_side(opp_specs, opp_idx, opp_conditions, set(opp_vols), extra,
                               active_mon=opp_active)
        states.append(pe.State(side_one=side_one, side_two=side_two, **fkw))
    return states, cmap


# ---------------------------------------------------------------------------
# Search


def mcts_worker(state_str: str, time_ms: int, iterations: int = 0) -> tuple:
    """Run MCTS on a serialized state (picklable; used in worker processes).

    Returns ``(side_one [(choice, visits, total_score)], total_visits, error, seconds)``.
    """
    t0 = time.perf_counter()
    try:
        import poke_engine as engine

        state = engine.State.from_string(state_str)
        result = engine.monte_carlo_tree_search(state, duration_ms=int(time_ms),
                                                iterations=int(iterations), threads=1)
        side_one = [(r.move_choice, int(r.visits), float(r.total_score)) for r in result.side_one]
        return side_one, int(result.total_visits), None, time.perf_counter() - t0
    except BaseException as exc:  # Rust panics surface as pyo3 PanicException
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return [], 0, f"{type(exc).__name__}: {exc}", time.perf_counter() - t0


def aggregate_visits(results: Sequence[tuple], cmap: ChoiceMap,
                     unmapped: Optional[list] = None) -> tuple[np.ndarray, list[str]]:
    """Average per-sample side-one visit shares, mapped to action indices.

    Choices that do not map to an action (e.g. ``No Move``) are appended to
    ``unmapped`` when given.
    """
    shares = np.zeros(N_ACTIONS, dtype=np.float64)
    errors: list[str] = []
    used = 0
    for side_one, _total, err, *_ in results:
        if err:
            errors.append(err)
            continue
        total = sum(v for _, v, _ in side_one)
        if total <= 0:
            continue
        used += 1
        for choice, visits, _ in side_one:
            action = cmap.action(choice)
            if action is not None and 0 <= action < N_ACTIONS:
                shares[action] += visits / total
            elif unmapped is not None:
                unmapped.append(choice)
    if used:
        shares /= used
    return shares, errors


def policy_probs(model: Any, battle: Battle, mask: np.ndarray) -> np.ndarray:
    """MaskablePPO action probabilities (zeros on illegal actions)."""
    import torch

    obs = embed_battle(battle)
    obs_t, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_t, action_masks=mask[None, :])
        probs = dist.distribution.probs.cpu().numpy().reshape(-1)
    probs = np.where(mask, probs, 0.0)
    s = probs.sum()
    return probs / s if s > 0 else probs


def combine_scores(shares: np.ndarray, probs: Optional[np.ndarray], mask: np.ndarray,
                   prior_weight: float) -> Optional[int]:
    """argmax over legal actions of (1-λ)·visit_share + λ·policy_prob (None if no signal)."""
    shares = np.where(mask, shares, 0.0)
    if shares.sum() <= 0:
        return None
    total = shares / shares.sum()
    if probs is not None and prior_weight > 0:
        total = (1.0 - prior_weight) * total + prior_weight * probs
    total = np.where(mask, total, -np.inf)
    return int(np.argmax(total))


def legal_mask(battle: Battle) -> np.ndarray:
    return np.array(SinglesEnv.get_action_mask(battle), dtype=bool)


def fallback_order(battle: Battle, model: Any = None, mask: Optional[np.ndarray] = None,
                   probs: Optional[np.ndarray] = None) -> BattleOrder:
    """Policy argmax if a model is given, else the smart heuristic."""
    try:
        if model is not None:
            mask = legal_mask(battle) if mask is None else mask
            if mask.any():
                if probs is None:
                    probs = policy_probs(model, battle, mask)
                return action_order(int(np.argmax(np.where(mask, probs, -1.0))), battle)
        from showdownrl.smart_heuristic import choose_smart_move

        return choose_smart_move(battle)
    except Exception:
        LOGGER.exception("fallback failed; choosing randomly")
        return Player.choose_random_singles_move(battle)


def action_order(action: int, battle: Battle) -> BattleOrder:
    return SinglesEnv.action_to_order(np.int64(action), battle, strict=False)


@dataclass
class SearchStats:
    decisions: int = 0
    searched: int = 0
    trivial: int = 0
    fallbacks: int = 0
    engine_errors: int = 0
    unmapped_choices: int = 0
    illegal_choices: int = 0
    engine_seconds: float = 0.0
    search_seconds: float = 0.0
    iterations: int = 0
    samples: int = 0
    last_error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "decisions": self.decisions,
            "searched": self.searched,
            "trivial": self.trivial,
            "fallbacks": self.fallbacks,
            "engine_errors": self.engine_errors,
            "unmapped_choices": self.unmapped_choices,
            "illegal_choices": self.illegal_choices,
            "seconds_per_search": round(self.search_seconds / max(self.searched, 1), 4),
            "engine_cpu_seconds_per_search": round(self.engine_seconds / max(self.searched, 1), 4),
            "iterations_per_sample": round(self.iterations / max(self.samples, 1), 1),
            "last_error": self.last_error,
        }


@dataclass
class SearchConfig:
    n_samples: int = 4
    time_ms: int = 100
    prior_weight: float = 0.0
    unrevealed: str = "sample"
    iterations: int = 0  # >0: fixed iteration budget per sample instead of time


def _prepare(battle: Battle, cfg: SearchConfig, rng: random.Random):
    states, cmap = battle_to_states(battle, cfg.n_samples, rng, cfg.unrevealed)
    return [s.to_string() for s in states], cmap


def _decide(battle: Battle, model: Any, cfg: SearchConfig, mask: np.ndarray, results,
            cmap: ChoiceMap, stats: SearchStats) -> BattleOrder:
    unmapped: list[str] = []
    shares, errors = aggregate_visits(results, cmap, unmapped)
    stats.engine_errors += len(errors)
    stats.unmapped_choices += sum(1 for c in unmapped if c != "No Move")
    # Engine options our request says are illegal (conversion mismatch diagnostics).
    stats.illegal_choices += int(np.count_nonzero((shares > 0) & ~mask))
    if errors:
        stats.last_error = errors[-1]
    for _, total, err, secs in results:
        stats.engine_seconds += secs
        if not err:
            stats.iterations += total
            stats.samples += 1
    probs = policy_probs(model, battle, mask) if model is not None and cfg.prior_weight > 0 \
        else None
    action = combine_scores(shares, probs, mask, cfg.prior_weight)
    if action is None:
        stats.fallbacks += 1
        seen = sorted({c for r in results for c, _, _ in r[0]})
        stats.last_error = f"no legal search signal; engine options {seen[:8]}"
        return fallback_order(battle, model, mask, probs)
    return action_order(action, battle)


def search_policy(battle: Battle, model: Any = None, n_samples: int = 4, time_ms: int = 100,
                  prior_weight: float = 0.0, unrevealed: str = "sample",
                  rng: Optional[random.Random] = None, stats: Optional[SearchStats] = None,
                  iterations: int = 0) -> BattleOrder:
    """Synchronous search decision (engine runs inline in this thread)."""
    cfg = SearchConfig(n_samples, time_ms, prior_weight, unrevealed, iterations)
    stats = stats if stats is not None else SearchStats()
    rng = rng or random.Random()
    stats.decisions += 1
    mask = legal_mask(battle)
    if mask.sum() <= 1:
        stats.trivial += 1
        if mask.sum() == 1:
            return action_order(int(np.flatnonzero(mask)[0]), battle)
        return Player.choose_random_singles_move(battle)
    try:
        t0 = time.perf_counter()
        state_strs, cmap = _prepare(battle, cfg, rng)
        results = [mcts_worker(s, cfg.time_ms, cfg.iterations) for s in state_strs]
        stats.search_seconds += time.perf_counter() - t0
        stats.searched += 1
        return _decide(battle, model, cfg, mask, results, cmap, stats)
    except Exception as exc:
        LOGGER.debug("search failed", exc_info=True)
        stats.fallbacks += 1
        stats.last_error = f"{type(exc).__name__}: {exc}"
        return fallback_order(battle, model, mask)


class SearchPlayer(Player):
    """poke-env Player: poke-engine MCTS over sampled determinizations + policy prior.

    ``workers`` > 0 runs the engine in a process pool (the Rust search holds the
    GIL, so threads would stall poke-env's event loop); ``workers=0`` runs it
    inline (tests / single battles).
    """

    def __init__(self, model_path: Optional[str] = None, n_samples: int = 4,
                 time_ms: int = 100, prior_weight: float = 0.0, unrevealed: str = "sample",
                 workers: int = 1, iterations: int = 0, seed: Optional[int] = None,
                 model: Any = None, **kwargs: Any):
        if pe is None:
            raise RuntimeError("poke_engine is not installed (see pyproject.toml [search])")
        if not engine_is_gen9():
            raise RuntimeError("poke_engine was not built with the gen9 terastallization "
                               "feature (see pyproject.toml [search])")
        super().__init__(**kwargs)
        if model is None and model_path:
            import torch
            from sb3_contrib import MaskablePPO

            torch.set_num_threads(1)
            model = MaskablePPO.load(model_path, device="cpu")
        self.model = model
        self.config = SearchConfig(n_samples, time_ms, prior_weight, unrevealed, iterations)
        self.stats = SearchStats()
        self.rng = random.Random(seed)
        self.workers = workers
        self._executor: Optional[Executor] = None
        self.decision_seconds: list[float] = []

    def _get_executor(self) -> Optional[Executor]:
        if self.workers <= 0:
            return None
        if self._executor is None:
            import multiprocessing as mp

            self._executor = ProcessPoolExecutor(max_workers=self.workers,
                                                 mp_context=mp.get_context("spawn"))
        return self._executor

    def close_executor(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def choose_move(self, battle: AbstractBattle):
        assert isinstance(battle, Battle)
        if self.workers <= 0:
            t0 = time.perf_counter()
            order = search_policy(battle, self.model, self.config.n_samples, self.config.time_ms,
                                  self.config.prior_weight, self.config.unrevealed, self.rng,
                                  self.stats, self.config.iterations)
            self.decision_seconds.append(time.perf_counter() - t0)
            return order
        return self._choose_async(battle)

    async def _choose_async(self, battle: Battle) -> BattleOrder:
        t0 = time.perf_counter()
        stats, cfg = self.stats, self.config
        stats.decisions += 1
        mask = legal_mask(battle)
        if mask.sum() <= 1:
            stats.trivial += 1
            if mask.sum() == 1:
                return action_order(int(np.flatnonzero(mask)[0]), battle)
            return Player.choose_random_singles_move(battle)
        try:
            state_strs, cmap = _prepare(battle, cfg, self.rng)
            loop = asyncio.get_running_loop()
            executor = self._get_executor()
            futures = [loop.run_in_executor(executor, mcts_worker, s, cfg.time_ms, cfg.iterations)
                       for s in state_strs]
            results = await asyncio.gather(*futures)
            stats.search_seconds += time.perf_counter() - t0
            stats.searched += 1
            order = _decide(battle, self.model, cfg, mask, results, cmap, stats)
        except BrokenProcessPool as exc:
            self._executor = None  # recreate on the next decision
            stats.fallbacks += 1
            stats.last_error = f"BrokenProcessPool: {exc}"
            order = fallback_order(battle, self.model, mask)
        except Exception as exc:
            LOGGER.debug("search failed", exc_info=True)
            stats.fallbacks += 1
            stats.last_error = f"{type(exc).__name__}: {exc}"
            order = fallback_order(battle, self.model, mask)
        self.decision_seconds.append(time.perf_counter() - t0)
        return order


def parse_search_spec(spec: str) -> dict:
    """``search[:MODEL.zip][,samples=4][,time_ms=100][,prior=0.3][,unrevealed=sample]...``"""
    if not spec.startswith("search"):
        raise ValueError(f"not a search spec: {spec!r}")
    head, *opts = spec.split(",")
    model_path = head.split(":", 1)[1] if ":" in head else None
    out: dict[str, Any] = {"model_path": model_path or None}
    aliases = {"samples": "n_samples", "n": "n_samples", "time": "time_ms", "ms": "time_ms",
               "prior": "prior_weight", "lambda": "prior_weight", "lam": "prior_weight",
               "iters": "iterations"}
    casts = {"n_samples": int, "time_ms": int, "prior_weight": float, "unrevealed": str,
             "workers": int, "iterations": int, "seed": int}
    for opt in opts:
        key, _, value = opt.partition("=")
        key = aliases.get(key.strip(), key.strip())
        if key not in casts:
            raise ValueError(f"unknown search option {key!r}")
        out[key] = casts[key](value.strip())
    return out
