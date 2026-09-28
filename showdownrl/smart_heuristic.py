"""Damage-calc heuristic for Gen 9 Random Battles.

Stronger than poke-env's SimpleHeuristicsPlayer because it reasons about
speed order, KO ranges, estimated opponent threat (random battle set data)
and hazard damage on switch-in. Used as a live fallback, a behaviour-cloning
teacher and an evaluation opponent.
"""

from __future__ import annotations

from poke_env.battle import AbstractBattle, Battle, MoveCategory, Pokemon
from poke_env.player import Player

from showdownrl.battle_features import (
    HAZARD_MOVES,
    HAZARD_REMOVAL,
    effective_speed,
    estimate_damage,
    hazard_entry_damage,
    opponent_threat,
    speed_advantage,
)

_HAZARD_CONDITION = {
    "stealthrock": "STEALTH_ROCK",
    "spikes": "SPIKES",
    "toxicspikes": "TOXIC_SPIKES",
    "stickyweb": "STICKY_WEB",
}


def _switch_score(mon: Pokemon, battle: Battle) -> float:
    opp = battle.opponent_active_pokemon
    if opp is None:
        return mon.current_hp_fraction
    offense = max((estimate_damage(m, mon, opp, battle, True) for m in mon.moves.values()),
                  default=0.0)
    threat = opponent_threat(opp, mon, battle)
    entry = hazard_entry_damage(mon, battle.side_conditions)
    faster = effective_speed(mon, battle, True) > effective_speed(opp, battle, False)
    hp_after = mon.current_hp_fraction - entry
    score = min(offense, 1.2) - 1.2 * min(threat, 1.2) - entry + 0.3 * hp_after
    if faster and offense * 0.9 >= opp.current_hp_fraction:
        score += 0.6
    if threat * 0.9 >= hp_after:
        score -= 0.5
    return score


def best_switch(battle: Battle):
    switches = battle.available_switches
    if not switches:
        return None
    return max(switches, key=lambda m: _switch_score(m, battle))


def choose_smart_move(battle: AbstractBattle):
    assert isinstance(battle, Battle)
    try:
        return _choose_smart_move(battle)
    except (KeyError, AttributeError, TypeError, ValueError):
        # poke-env lacks dex data for a few pseudo-moves (e.g. "recharge").
        return Player.choose_random_singles_move(battle)


def _choose_smart_move(battle: Battle):
    me, opp = battle.active_pokemon, battle.opponent_active_pokemon
    if me is None or opp is None or not battle.available_moves:
        target = best_switch(battle)
        if target is not None:
            return Player.create_order(target)
        return Player.choose_random_singles_move(battle)

    faster = speed_advantage(battle) >= 1.0
    threat = opponent_threat(opp, me, battle)
    we_die = threat * 0.9 >= me.current_hp_fraction
    opp_hp = opp.current_hp_fraction

    scored = []
    for move in battle.available_moves:
        dmg = estimate_damage(move, me, opp, battle, True)
        tera_dmg = estimate_damage(move, me, opp, battle, True, tera=True) if battle.can_tera else dmg
        scored.append((move, dmg, tera_dmg))

    # 1. Take a KO when we can land it before being KO'd.
    kos = [(m, d) for m, d, _ in scored if d * 0.9 >= opp_hp and (m.accuracy or 1) >= 0.85]
    if kos:
        first = [m for m, _ in kos if faster or m.priority > 0]
        if first or not we_die:
            pick = max(first or [m for m, _ in kos], key=lambda m: (m.priority, m.accuracy))
            return Player.create_order(pick)

    # 2. Get out of losing matchups.
    best_attack = max(scored, key=lambda x: x[1])
    if battle.available_switches and not battle.trapped:
        danger = we_die and not faster
        outclassed = threat >= 0.55 and best_attack[1] < 0.35
        if danger or outclassed:
            target = best_switch(battle)
            if target is not None and _switch_score(target, battle) > -0.2:
                return Player.create_order(target)

    opp_alive = 6 - sum(1 for m in battle.opponent_team.values() if m.fainted)
    my_alive = sum(1 for m in battle.team.values() if not m.fainted)
    safe = threat < 0.35 or (faster and threat < 0.6)

    for move, _, _ in scored:
        # 3. Hazards early, removal when useful.
        if move.id in HAZARD_MOVES and opp_alive >= 3 and safe:
            condition_name = _HAZARD_CONDITION[move.id]
            if not any(c.name == condition_name for c in battle.opponent_side_conditions):
                return Player.create_order(move)
        if move.id in HAZARD_REMOVAL and battle.side_conditions and my_alive >= 3 and safe \
                and move.category == MoveCategory.STATUS:
            return Player.create_order(move)

    # 4. Recover when healthy enough to matter.
    if me.current_hp_fraction < 0.5 and not we_die:
        for move, _, _ in scored:
            if move.heal >= 0.5:
                return Player.create_order(move)

    # 5. Setup when the opponent can't punish it.
    if me.current_hp_fraction >= 0.7 and threat < 0.3:
        for move, _, _ in scored:
            boosts = move.self_boost or (move.boosts if move.target == "self" else None)
            if move.category == MoveCategory.STATUS and boosts and sum(boosts.values()) >= 2 \
                    and max(me.boosts.values()) < 2:
                return Player.create_order(move)

    # 6. Strongest attack, terastallizing when it clearly helps.
    move, dmg, tera_dmg = max(scored, key=lambda x: max(x[1], x[2]))
    if dmg <= 0.02:
        status_moves = [m for m, _, _ in scored if m.category == MoveCategory.STATUS]
        if status_moves and opp.status is None:
            for m in status_moves:
                if m.status is not None:
                    return Player.create_order(m)
    use_tera = battle.can_tera and (tera_dmg > dmg * 1.25 or my_alive == 1)
    return Player.create_order(move, terastallize=use_tera)


class SmartHeuristicsPlayer(Player):
    def choose_move(self, battle: AbstractBattle):
        return choose_smart_move(battle)
