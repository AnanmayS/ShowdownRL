import logging

import numpy as np
import pytest

pytest.importorskip("poke_env")

from poke_env.battle import Battle  # noqa: E402
from poke_env.environment import SinglesEnv  # noqa: E402

from showdownrl.battle_features import (  # noqa: E402
    OBS_SIZE,
    candidate_moves,
    embed_battle,
    estimate_damage,
    estimate_stat,
    speed_advantage,
)
from showdownrl.smart_heuristic import choose_smart_move  # noqa: E402


def _mon(ident, details, condition, active, stats, moves, item, ability, tera):
    return {
        "ident": ident, "details": details, "condition": condition, "active": active,
        "stats": stats, "moves": moves, "baseAbility": ability, "item": item,
        "pokeball": "pokeball", "ability": ability, "commanding": False, "reviving": False,
        "teraType": tera, "terastallized": "",
    }


REQUEST = {
    "active": [{
        "moves": [
            {"move": "Earthquake", "id": "earthquake", "pp": 16, "maxpp": 16, "target": "allAdjacent", "disabled": False},
            {"move": "Stealth Rock", "id": "stealthrock", "pp": 32, "maxpp": 32, "target": "foeSide", "disabled": False},
            {"move": "Stone Edge", "id": "stoneedge", "pp": 8, "maxpp": 8, "target": "normal", "disabled": False},
            {"move": "Megahorn", "id": "megahorn", "pp": 16, "maxpp": 16, "target": "normal", "disabled": False},
        ],
        "canTerastallize": "Water",
    }],
    "side": {
        "name": "tester", "id": "p1",
        "pokemon": [
            _mon("p1: Rhydon", "Rhydon, L85, M", "300/300", True,
                 {"atk": 270, "def": 253, "spa": 125, "spd": 125, "spe": 117},
                 ["earthquake", "stealthrock", "stoneedge", "megahorn"], "eviolite", "lightningrod", "Water"),
            _mon("p1: Lanturn", "Lanturn, L89, M", "320/320", False,
                 {"atk": 108, "def": 154, "spa": 186, "spd": 186, "spe": 170},
                 ["thunderbolt", "scald", "icebeam", "thunderwave"], "leftovers", "voltabsorb", "Flying"),
        ],
    },
    "rqid": 3,
}


@pytest.fixture
def battle():
    b = Battle("battle-gen9randombattle-1", "tester", logging.getLogger("test"), gen=9)
    b.player_role = "p1"
    b.parse_request(REQUEST)
    b.parse_message(["", "switch", "p1a: Rhydon", "Rhydon, L85, M", "300/300"])
    b.parse_message(["", "switch", "p2a: Charizard", "Charizard, L84, M", "100/100"])
    return b


def test_embedding_shape_and_bounds(battle):
    obs = embed_battle(battle)
    assert obs.shape == (OBS_SIZE,)
    assert np.all(np.isfinite(obs))
    assert np.abs(obs).max() <= 4.0


def test_our_and_opponent_actives_are_not_swapped(battle):
    assert battle.active_pokemon.species == "rhydon"
    assert battle.opponent_active_pokemon.species == "charizard"


def test_opponent_stats_estimated_from_randbats_level(battle):
    opp = battle.opponent_active_pokemon
    # Charizard base 100 speed, level 84, 85 EVs, 31 IVs -> 216 (Mew L82 check: 211)
    assert estimate_stat(opp, "spe") == 216
    assert speed_advantage(battle) == 0.0  # Rhydon (117) is slower


def test_damage_estimate_respects_type_effectiveness(battle):
    me, opp = battle.active_pokemon, battle.opponent_active_pokemon
    moves = me.moves
    rock = estimate_damage(moves["stoneedge"], me, opp, battle, True)
    ground = estimate_damage(moves["earthquake"], me, opp, battle, True)
    assert ground == 0.0  # Charizard is Flying
    assert rock > 0.8  # 4x STAB Stone Edge from 270 Atk is a likely KO


def test_candidate_moves_include_randbats_set(battle):
    ids = {m.id for m in candidate_moves(battle.opponent_active_pokemon)}
    assert ids  # unrevealed opponent still has a candidate move pool
    assert len(ids) >= 4


def test_smart_heuristic_picks_legal_super_effective_move(battle):
    order = choose_smart_move(battle)
    assert str(order) in {str(o) for o in battle.valid_orders}
    assert order.order.id == "stoneedge"
    action = SinglesEnv.order_to_action(order, battle)
    assert SinglesEnv.get_action_mask(battle)[int(action)] == 1
