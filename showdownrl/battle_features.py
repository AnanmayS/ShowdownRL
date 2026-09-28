"""Battle-state feature extraction shared by training (poke-env) and live play.

Everything here works on poke-env ``Battle`` objects, so the exact same code
path embeds observations on the local training server and on a live battle
reconstructed from the Showdown protocol.

Opponent hidden information is estimated from Gen 9 Random Battle set data:
levels are shown in battle, EVs/IVs are fixed by the generator, and the
candidate moves for each species come from ``data/gen9randombattle.json``.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from poke_env.battle import (
    Battle,
    Effect,
    Field,
    Move,
    MoveCategory,
    Pokemon,
    PokemonType,
    SideCondition,
    Status,
    Weather,
)
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

GEN = 9
RANDBATS_PATH = Path(__file__).resolve().parent / "data" / "gen9randombattle.json"
N_ACTIONS = 26  # poke-env SinglesEnv gen 9 action space

BOOST_STATS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
STATUSES = (Status.BRN, Status.FRZ, Status.PAR, Status.PSN, Status.TOX, Status.SLP)
WEATHERS = (Weather.SUNNYDAY, Weather.DESOLATELAND, Weather.RAINDANCE, Weather.PRIMORDIALSEA,
            Weather.SANDSTORM, Weather.SNOWSCAPE, Weather.HAIL)
TERRAINS = (Field.ELECTRIC_TERRAIN, Field.GRASSY_TERRAIN, Field.MISTY_TERRAIN, Field.PSYCHIC_TERRAIN)
HAZARDS = (SideCondition.STEALTH_ROCK, SideCondition.SPIKES, SideCondition.TOXIC_SPIKES,
           SideCondition.STICKY_WEB)
SCREENS = (SideCondition.REFLECT, SideCondition.LIGHT_SCREEN, SideCondition.AURORA_VEIL,
           SideCondition.TAILWIND)

HAZARD_MOVES = {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
HAZARD_REMOVAL = {"rapidspin", "defog", "mortalspin", "tidyup", "courtchange"}
PIVOT_MOVES = {"uturn", "voltswitch", "flipturn", "partingshot", "teleport", "chillyreception",
               "shedtail", "batonpass"}
FIXED_DAMAGE_LEVEL = {"seismictoss", "nightshade"}
HALF_HP_MOVES = {"superfang", "ruination", "naturesmadness"}
IMMUNITY_ABILITIES = {
    "levitate": PokemonType.GROUND,
    "flashfire": PokemonType.FIRE,
    "wellbakedbody": PokemonType.FIRE,
    "waterabsorb": PokemonType.WATER,
    "stormdrain": PokemonType.WATER,
    "dryskin": PokemonType.WATER,
    "voltabsorb": PokemonType.ELECTRIC,
    "lightningrod": PokemonType.ELECTRIC,
    "motordrive": PokemonType.ELECTRIC,
    "sapsipper": PokemonType.GRASS,
    "eartheater": PokemonType.GROUND,
}
DEFAULT_RANDBATS_EV = 85
DEFAULT_RANDBATS_IV = 31

_TYPE_CHART = GenData.from_gen(GEN).type_chart
_MOVE_CACHE: dict[str, Move] = {}


# ---------------------------------------------------------------------------
# Random battle set data


@lru_cache(maxsize=1)
def _randbats() -> dict[str, dict]:
    try:
        raw = json.loads(RANDBATS_PATH.read_text())
    except (OSError, ValueError):
        return {}
    return {to_id_str(name): entry for name, entry in raw.items()}


def _species_entry(mon: Pokemon) -> Optional[dict]:
    data = _randbats()
    for key in (mon.species, to_id_str(mon.base_species)):
        if key in data:
            return data[key]
    return None


def _get_move(move_id: str) -> Move:
    move = _MOVE_CACHE.get(move_id)
    if move is None:
        move = Move(move_id, gen=GEN)
        _MOVE_CACHE[move_id] = move
    return move


def candidate_moves(mon: Pokemon) -> list[Move]:
    """Revealed moves plus every move the species can carry in random battles."""
    moves = {m.id: m for m in mon.moves.values()}
    entry = _species_entry(mon)
    if entry and len(moves) < 4:
        for role in entry.get("roles", {}).values():
            for name in role.get("moves", []):
                move_id = to_id_str(name)
                if move_id not in moves:
                    try:
                        moves[move_id] = _get_move(move_id)
                    except Exception:  # unknown move id in local dex
                        continue
    return list(moves.values())


def candidate_tera_types(mon: Pokemon) -> list[PokemonType]:
    entry = _species_entry(mon)
    types: set[PokemonType] = set()
    if entry:
        for role in entry.get("roles", {}).values():
            for name in role.get("teraTypes", []):
                try:
                    types.add(PokemonType.from_name(name))
                except Exception:
                    continue
    return sorted(types, key=lambda t: t.value)


# ---------------------------------------------------------------------------
# Stat estimation


def _boost_multiplier(stage: int) -> float:
    stage = max(-6, min(6, stage))
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


def estimate_stat(mon: Pokemon, stat: str) -> float:
    """Unboosted stat. Exact for our own Pokemon, randbats-estimated for the opponent."""
    stats = getattr(mon, "stats", None) or {}
    value = stats.get(stat)
    if value:
        return float(value)
    base = mon.base_stats.get(stat, 80)
    level = mon.level or 80
    ev = DEFAULT_RANDBATS_EV
    entry = _species_entry(mon)
    if entry and stat in entry.get("evs", {}):
        ev = entry["evs"][stat]
    core = math.floor((2 * base + DEFAULT_RANDBATS_IV + ev // 4) * level / 100)
    if stat == "hp":
        return float(core + level + 10)
    return float(core + 5)


def estimate_max_hp(mon: Pokemon) -> float:
    if mon.max_hp and mon.max_hp > 100:
        return float(mon.max_hp)
    return estimate_stat(mon, "hp")


def effective_speed(mon: Pokemon, battle: Battle, ours: bool) -> float:
    speed = estimate_stat(mon, "spe") * _boost_multiplier(mon.boosts.get("spe", 0))
    if mon.status == Status.PAR:
        speed *= 0.5
    item = mon.item or ""
    if item == "choicescarf":
        speed *= 1.5
    side = battle.side_conditions if ours else battle.opponent_side_conditions
    if SideCondition.TAILWIND in side:
        speed *= 2
    return speed


def speed_advantage(battle: Battle) -> float:
    """1 if our active moves first, 0 if the opponent does, 0.5 for ties/unknown."""
    me, opp = battle.active_pokemon, battle.opponent_active_pokemon
    if me is None or opp is None:
        return 0.5
    ours, theirs = effective_speed(me, battle, True), effective_speed(opp, battle, False)
    if Field.TRICK_ROOM in battle.fields:
        ours, theirs = theirs, ours
    if ours > theirs:
        return 1.0
    if ours < theirs:
        return 0.0
    return 0.5


# ---------------------------------------------------------------------------
# Damage estimation


def _type_multiplier(move_type: PokemonType, defender: Pokemon) -> float:
    try:
        return defender.damage_multiplier(move_type)
    except Exception:
        return 1.0


def _ability_immune(move_type: PokemonType, defender: Pokemon) -> float:
    """Multiplier from immunity abilities (1 = none, 0 = certain immunity)."""
    if defender.ability:
        immune = IMMUNITY_ABILITIES.get(defender.ability)
        return 0.0 if immune == move_type else 1.0
    possible = [to_id_str(a) for a in (defender.possible_abilities or [])]
    if not possible:
        return 1.0
    hits = sum(1 for a in possible if IMMUNITY_ABILITIES.get(a) == move_type)
    return 1.0 - hits / len(possible)


def estimate_damage(*args, **kwargs) -> float:
    """Expected damage as a fraction of the defender's *max* HP (accuracy included).

    Returns 0 for moves poke-env has incomplete data for (e.g. ``recharge``).
    """
    try:
        return _estimate_damage(*args, **kwargs)
    except (KeyError, AttributeError, TypeError, ValueError):
        return 0.0


def _estimate_damage(
    move: Move,
    attacker: Pokemon,
    defender: Pokemon,
    battle: Battle,
    attacker_is_us: bool,
    tera: bool = False,
) -> float:
    if move.category == MoveCategory.STATUS:
        return 0.0
    move_id = move.id
    move_type = move.type
    if tera and attacker.tera_type is not None and move_id == "terablast":
        move_type = attacker.tera_type

    eff = _type_multiplier(move_type, defender) * _ability_immune(move_type, defender)
    if eff == 0:
        return 0.0
    accuracy = move.accuracy if isinstance(move.accuracy, float) else 1.0
    defender_hp = estimate_max_hp(defender)

    if move_id in FIXED_DAMAGE_LEVEL:
        return accuracy * (attacker.level or 80) / defender_hp
    if move_id in HALF_HP_MOVES:
        return accuracy * 0.5 * defender.current_hp_fraction

    base_power = move.base_power or 60
    physical = move.category == MoveCategory.PHYSICAL
    atk_stat = "atk" if physical else "spa"
    def_stat = "def" if (physical or move_id in {"psyshock", "psystrike", "secretsword"}) else "spd"
    atk_source = defender if move_id == "foulplay" else attacker
    if move_id == "bodypress":
        atk_stat = "def"
    attack = estimate_stat(atk_source, atk_stat) * _boost_multiplier(atk_source.boosts.get(atk_stat, 0))
    defense = estimate_stat(defender, def_stat) * _boost_multiplier(defender.boosts.get(def_stat, 0))
    level = attacker.level or 80
    base = math.floor(math.floor(2 * level / 5 + 2) * base_power * attack / max(defense, 1) / 50) + 2

    mods = eff * 0.925 * (move.expected_hits or 1)
    attacker_types = [t for t in attacker.original_types if t is not None] if hasattr(
        attacker, "original_types") else [t for t in attacker.types if t is not None]
    if tera and attacker.tera_type is not None and not attacker.is_terastallized:
        if move_type == attacker.tera_type:
            mods *= 2.0 if move_type in attacker_types else 1.5
        elif move_type in attacker_types:
            mods *= 1.5
    elif move_type in [t for t in attacker.types if t is not None] or move_type in attacker_types:
        stab = 1.5
        if attacker.is_terastallized and attacker.tera_type == move_type and move_type in attacker_types:
            stab = 2.0
        mods *= stab
    if physical and attacker.status == Status.BRN and attacker.ability != "guts":
        mods *= 0.5
    weather = set(battle.weather)
    if weather & {Weather.SUNNYDAY, Weather.DESOLATELAND}:
        mods *= 1.5 if move_type == PokemonType.FIRE else 0.5 if move_type == PokemonType.WATER else 1
    elif weather & {Weather.RAINDANCE, Weather.PRIMORDIALSEA}:
        mods *= 1.5 if move_type == PokemonType.WATER else 0.5 if move_type == PokemonType.FIRE else 1
    item = attacker.item or ""
    if item == "choiceband" and physical or item == "choicespecs" and not physical:
        mods *= 1.5
    elif item == "lifeorb":
        mods *= 1.3
    elif item == "expertbelt" and eff > 1:
        mods *= 1.2
    defender_side = battle.opponent_side_conditions if attacker_is_us else battle.side_conditions
    if SideCondition.AURORA_VEIL in defender_side or (
        physical and SideCondition.REFLECT in defender_side) or (
        not physical and SideCondition.LIGHT_SCREEN in defender_side):
        mods *= 0.5
    return accuracy * base * mods / defender_hp


def best_damage(moves: Iterable[Move], attacker: Pokemon, defender: Pokemon, battle: Battle,
                attacker_is_us: bool) -> float:
    best = 0.0
    for move in moves:
        best = max(best, estimate_damage(move, attacker, defender, battle, attacker_is_us))
    return best


def opponent_threat(opp: Pokemon, target: Pokemon, battle: Battle) -> float:
    """Estimated best damage fraction the opponent mon can deal to ``target``."""
    return best_damage(candidate_moves(opp), opp, target, battle, attacker_is_us=False)


def hazard_entry_damage(mon: Pokemon, side: dict) -> float:
    damage = 0.0
    if SideCondition.STEALTH_ROCK in side:
        damage += 0.125 * _type_multiplier(PokemonType.ROCK, mon)
    layers = side.get(SideCondition.SPIKES, 0)
    grounded = PokemonType.FLYING not in mon.types and mon.ability != "levitate"
    if layers and grounded:
        damage += {1: 1 / 8, 2: 1 / 6}.get(layers, 1 / 4)
    if (mon.item or "") == "heavydutyboots":
        return 0.0
    return damage


# ---------------------------------------------------------------------------
# Observation


def _status_onehot(mon: Optional[Pokemon]) -> list[float]:
    status = mon.status if mon is not None else None
    return [1.0 if status == s else 0.0 for s in STATUSES]


def _boosts(mon: Optional[Pokemon]) -> list[float]:
    if mon is None:
        return [0.0] * len(BOOST_STATS)
    return [mon.boosts.get(s, 0) / 6.0 for s in BOOST_STATS]


def _log_eff(multiplier: float) -> float:
    if multiplier <= 0:
        return -1.5
    return max(-1.5, min(1.5, math.log2(multiplier) / 2))


def _move_slots(battle: Battle) -> list[Optional[Move]]:
    """Moves in the same order poke-env uses for actions 6-9."""
    active = battle.active_pokemon
    if active is None:
        return [None] * 4
    avail_ids = [m.id for m in battle.available_moves]
    known = list(active.moves.values())[:4]
    known_ids = [m.id for m in known]
    if len(avail_ids) == 1 and avail_ids[0] not in known_ids:
        slots = list(battle.available_moves)
    else:
        slots = known
    return (slots + [None] * 4)[:4]


MOVE_FEATURES = 20
TEAM_FEATURES = 16
OPP_TEAM_FEATURES = 8


def _move_features(move: Optional[Move], battle: Battle, available: set[str],
                   opp_hp: float) -> list[float]:
    try:
        return _move_features_unsafe(move, battle, available, opp_hp)
    except (KeyError, AttributeError, TypeError, ValueError):
        flag = 1.0 if move is not None and move.id in available else 0.0
        return [flag] + [0.0] * (MOVE_FEATURES - 1)


def _move_features_unsafe(move: Optional[Move], battle: Battle, available: set[str],
                          opp_hp: float) -> list[float]:
    active, opp = battle.active_pokemon, battle.opponent_active_pokemon
    if move is None or active is None:
        return [0.0] * MOVE_FEATURES
    dmg = estimate_damage(move, active, opp, battle, True) if opp is not None else 0.0
    tera_dmg = (estimate_damage(move, active, opp, battle, True, tera=True)
                if opp is not None and battle.can_tera else dmg)
    eff = _type_multiplier(move.type, opp) if opp is not None else 1.0
    self_boost = move.self_boost or (move.boosts if move.target == "self" else None) or {}
    target_drop = move.boosts if move.boosts and move.target != "self" else {}
    accuracy = move.accuracy if isinstance(move.accuracy, float) else 1.0
    return [
        1.0 if move.id in available else 0.0,
        min(dmg, 2.0),
        min(tera_dmg, 2.0),
        1.0 if opp is not None and dmg * 0.9 >= opp_hp else 0.0,
        _log_eff(eff),
        min(move.base_power, 250) / 150.0,
        accuracy,
        move.priority / 3.0,
        1.0 if move.category == MoveCategory.STATUS else 0.0,
        1.0 if move.category == MoveCategory.PHYSICAL else 0.0,
        sum(max(v, 0) for v in self_boost.values()) / 3.0,
        -sum(min(v, 0) for v in target_drop.values()) / 3.0,
        move.heal + move.drain,
        1.0 if move.id in HAZARD_MOVES else 0.0,
        1.0 if move.id in HAZARD_REMOVAL else 0.0,
        1.0 if move.id in PIVOT_MOVES or move.self_switch else 0.0,
        1.0 if move.status is not None or move.volatile_status is not None else 0.0,
        1.0 if move.is_protect_move else 0.0,
        move.recoil,
        (move.current_pp / move.max_pp) if move.max_pp else 1.0,
    ]


def _team_features(mon: Pokemon, battle: Battle, available_switches: set[str]) -> list[float]:
    opp = battle.opponent_active_pokemon
    if mon.fainted:
        return [0.0, 1.0] + [0.0] * (TEAM_FEATURES - 2)
    offense = best_damage(mon.moves.values(), mon, opp, battle, True) if opp is not None else 0.0
    threat = opponent_threat(opp, mon, battle) if opp is not None else 0.0
    faster = 0.5
    if opp is not None:
        ours, theirs = effective_speed(mon, battle, True), effective_speed(opp, battle, False)
        faster = 1.0 if ours > theirs else 0.0 if ours < theirs else 0.5
    best_eff = max((_type_multiplier(t, opp) for t in mon.types if t is not None), default=1.0) \
        if opp is not None else 1.0
    worst_eff = max((_type_multiplier(t, mon) for t in opp.types if t is not None), default=1.0) \
        if opp is not None else 1.0
    return [
        mon.current_hp_fraction,
        0.0,
        1.0 if mon.active else 0.0,
        1.0 if mon.base_species in available_switches else 0.0,
        1.0 if mon.status is not None else 0.0,
        min(offense, 2.0),
        min(threat, 2.0),
        1.0 if opp is not None and offense * 0.9 >= opp.current_hp_fraction else 0.0,
        1.0 if threat * 0.9 >= mon.current_hp_fraction else 0.0,
        faster,
        _log_eff(best_eff),
        _log_eff(worst_eff),
        min(hazard_entry_damage(mon, battle.side_conditions), 1.0),
        estimate_stat(mon, "spe") / 400.0,
        (mon.level or 80) / 100.0,
        1.0 if (mon.item or "").startswith("choice") else 0.0,
    ]


def _opp_team_features(battle: Battle) -> list[float]:
    opp_mons = list(battle.opponent_team.values())
    revealed = len(opp_mons)
    fainted = sum(1 for m in opp_mons if m.fainted)
    hp_known = sum(m.current_hp_fraction for m in opp_mons if not m.fainted)
    unrevealed = max(0, 6 - revealed)
    statused = sum(1 for m in opp_mons if m.status is not None and not m.fainted)
    me = battle.active_pokemon
    worst_threat = 0.0
    if me is not None:
        for mon in opp_mons:
            if not mon.fainted and not mon.active:
                worst_threat = max(worst_threat, opponent_threat(mon, me, battle))
    return [
        revealed / 6.0,
        fainted / 6.0,
        (hp_known + unrevealed) / 6.0,
        unrevealed / 6.0,
        statused / 6.0,
        min(worst_threat, 2.0),
        1.0 if battle.opponent_used_tera else 0.0 if hasattr(battle, "opponent_used_tera") else 0.0,
        min(hazard_entry_damage(battle.opponent_active_pokemon, battle.opponent_side_conditions), 1.0)
        if battle.opponent_active_pokemon is not None else 0.0,
    ]


def _active_features(mon: Optional[Pokemon], battle: Battle, ours: bool) -> list[float]:
    if mon is None:
        return [0.0] * 24
    effects = mon.effects
    return [
        mon.current_hp_fraction,
        *_boosts(mon),
        *_status_onehot(mon),
        1.0 if mon.is_terastallized else 0.0,
        1.0 if mon.first_turn else 0.0,
        1.0 if mon.must_recharge else 0.0,
        1.0 if Effect.SUBSTITUTE in effects else 0.0,
        1.0 if Effect.LEECH_SEED in effects else 0.0,
        1.0 if Effect.CONFUSION in effects else 0.0,
        estimate_stat(mon, "spe") / 400.0,
        (mon.level or 80) / 100.0,
        1.0 if (mon.item or "").startswith("choice") else 0.0,
        1.0 if ours and battle.trapped else 0.0,
    ]


def _side_features(side: dict) -> list[float]:
    return [
        1.0 if SideCondition.STEALTH_ROCK in side else 0.0,
        side.get(SideCondition.SPIKES, 0) / 3.0,
        side.get(SideCondition.TOXIC_SPIKES, 0) / 2.0,
        1.0 if SideCondition.STICKY_WEB in side else 0.0,
        *[1.0 if s in side else 0.0 for s in SCREENS],
    ]


def embed_battle(battle: Battle) -> np.ndarray:
    active, opp = battle.active_pokemon, battle.opponent_active_pokemon
    our_alive = sum(1 for m in battle.team.values() if not m.fainted)
    opp_fainted = sum(1 for m in battle.opponent_team.values() if m.fainted)
    weather = set(battle.weather)
    fields = set(battle.fields)
    opp_hp = opp.current_hp_fraction if opp is not None else 1.0

    threat = opponent_threat(opp, active, battle) if active is not None and opp is not None else 0.0
    our_best = best_damage(battle.available_moves, active, opp, battle, True) \
        if active is not None and opp is not None else 0.0
    opp_priority_threat = 0.0
    if active is not None and opp is not None:
        for move in candidate_moves(opp):
            try:
                is_priority = move.priority > 0 and move.category != MoveCategory.STATUS
            except KeyError:
                continue
            if is_priority:
                opp_priority_threat = max(
                    opp_priority_threat, estimate_damage(move, opp, active, battle, False))

    global_feats = [
        min(battle.turn, 60) / 30.0,
        our_alive / 6.0,
        (6 - opp_fainted) / 6.0,
        (our_alive - (6 - opp_fainted)) / 6.0,
        *[1.0 if w in weather else 0.0 for w in WEATHERS],
        *[1.0 if t in fields else 0.0 for t in TERRAINS],
        1.0 if Field.TRICK_ROOM in fields else 0.0,
        1.0 if battle.can_tera else 0.0,
        speed_advantage(battle),
        min(threat, 2.0),
        1.0 if active is not None and threat * 0.9 >= active.current_hp_fraction else 0.0,
        min(our_best, 2.0),
        1.0 if our_best * 0.9 >= opp_hp else 0.0,
        min(opp_priority_threat, 2.0),
        1.0 if battle.force_switch else 0.0,
        1.0 if not battle.available_moves else 0.0,
    ]

    available_moves = {m.id for m in battle.available_moves}
    move_feats: list[float] = []
    for move in _move_slots(battle):
        move_feats.extend(_move_features(move, battle, available_moves, opp_hp))

    available_switches = {m.base_species for m in battle.available_switches}
    team = list(battle.team.values())[:6]
    team_feats: list[float] = []
    for mon in team:
        team_feats.extend(_team_features(mon, battle, available_switches))
    team_feats.extend([0.0] * (TEAM_FEATURES * (6 - len(team))))

    feats = (
        global_feats
        + _active_features(active, battle, True)
        + _active_features(opp, battle, False)
        + _side_features(battle.side_conditions)
        + _side_features(battle.opponent_side_conditions)
        + move_feats
        + team_feats
        + _opp_team_features(battle)
    )
    return np.asarray(feats, dtype=np.float32)


OBS_SIZE = 25 + 24 + 24 + 8 + 8 + 4 * MOVE_FEATURES + 6 * TEAM_FEATURES + OPP_TEAM_FEATURES
