"""Offline tests for showdownrl.search (poke-engine decision-time search)."""

import json
import logging
import random
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("poke_env")
pe = pytest.importorskip("poke_engine")

from poke_env.battle import Battle  # noqa: E402
from poke_env.environment import SinglesEnv  # noqa: E402
from poke_env.ps_client import AccountConfiguration  # noqa: E402

from showdownrl import search as S  # noqa: E402
from showdownrl.protocol_battle import ProtocolBattle  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


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


def _replay_decisions(name):
    """Yield the Battle at every recorded decision point of a protocol fixture."""
    fixture = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    points = {d["after_message"] for d in fixture["decisions"]}
    state = ProtocolBattle(username=fixture["username"])
    for index, message in enumerate(fixture["messages"]):
        state.feed(message)
        if index in points:
            yield state.battle
            state.mark_answered()


# ---------------------------------------------------------------------------
# Conversion


def test_engine_is_gen9_build():
    assert S.engine_is_gen9(), "rebuild poke-engine with --features poke-engine/terastallization"


def test_states_copy_our_side_exactly(battle):
    states, cmap = S.battle_to_states(battle, 3, random.Random(0))
    assert len(states) == 3
    ours = states[0].side_one
    assert ours.active_index == "0"
    rhydon, lanturn = ours.pokemon[0], ours.pokemon[1]
    assert (rhydon.id, rhydon.level, rhydon.hp, rhydon.maxhp) == ("rhydon", 85, 300, 300)
    assert (rhydon.attack, rhydon.defense, rhydon.speed) == (270, 253, 117)
    assert rhydon.item == "eviolite" and rhydon.ability == "lightningrod"
    assert rhydon.tera_type == "water" and not rhydon.terastallized
    assert [m.id for m in rhydon.moves] == ["earthquake", "stealthrock", "stoneedge", "megahorn"]
    assert rhydon.moves[2].pp == 8
    assert lanturn.id == "lanturn" and lanturn.tera_type == "flying"
    assert ours.pokemon[2].hp == 0  # padded with fainted placeholders
    # Round-trips through the engine's own serialisation.
    back = pe.State.from_string(states[0].to_string())
    assert back.side_one.pokemon[0].id.lower() == "rhydon"
    assert cmap.moves == {"earthquake": 0, "stealthrock": 1, "stoneedge": 2, "megahorn": 3}
    assert cmap.switches == {"rhydon": 0, "lanturn": 1}


def test_opponent_sets_sampled_from_randbats(battle):
    states, _ = S.battle_to_states(battle, 4, random.Random(1), unrevealed="fainted")
    entry = S._randbats()["charizard"]
    role_moves = {m.lower().replace(" ", "").replace("-", "")
                  for r in entry["roles"].values() for m in r["moves"]}
    for state in states:
        zard = state.side_two.pokemon[0]
        assert zard.id == "charizard" and zard.level == 84
        assert zard.hp == zard.maxhp
        assert zard.speed == 216  # same randbats estimate as battle_features.estimate_stat
        ids = [m.id for m in zard.moves if m.id != "none"]
        assert len(ids) == 4 and set(ids) <= role_moves
        assert zard.item != "unknown_item"
        # unrevealed slots are fainted placeholders in this mode
        assert all(p.hp == 0 for p in state.side_two.pokemon[1:])


def test_revealed_information_is_kept(battle):
    battle.parse_message(["", "move", "p2a: Charizard", "Hurricane", "p1a: Rhydon"])
    battle.parse_message(["", "-enditem", "p2a: Charizard", "Heavy-Duty Boots"])
    battle.parse_message(["", "-damage", "p2a: Charizard", "40/100"])
    battle.parse_message(["", "-status", "p2a: Charizard", "brn"])
    battle.parse_message(["", "-boost", "p2a: Charizard", "spa", "1"])
    battle.opponent_active_pokemon.item = "heavydutyboots"
    states, _ = S.battle_to_states(battle, 3, random.Random(2))
    for state in states:
        zard = state.side_two.pokemon[0]
        assert "hurricane" in [m.id for m in zard.moves]
        assert zard.item == "heavydutyboots"
        assert zard.status == "burn"
        assert abs(zard.hp / zard.maxhp - 0.4) < 0.01
        assert state.side_two.special_attack_boost == 1


def test_unrevealed_slots_sampled_without_duplicates(battle):
    states, _ = S.battle_to_states(battle, 5, random.Random(3), unrevealed="sample")
    for state in states:
        ids = [p.id for p in state.side_two.pokemon]
        assert len(set(ids)) == 6 and all(p.hp > 0 for p in state.side_two.pokemon)
        assert ids[0] == "charizard"


def test_opponent_choice_lock(battle):
    zard = battle.opponent_active_pokemon
    battle.parse_message(["", "move", "p2a: Charizard", "Flamethrower", "p1a: Rhydon"])
    zard.item = "choicespecs"
    states, _ = S.battle_to_states(battle, 2, random.Random(4))
    for state in states:
        moves = state.side_two.pokemon[0].moves
        assert [m.id for m in moves if not m.disabled] == ["flamethrower"]


def test_encore_needs_last_used_move(battle):
    battle.parse_message(["", "move", "p2a: Charizard", "Flamethrower", "p1a: Rhydon"])
    battle.parse_message(["", "-start", "p2a: Charizard", "Encore"])
    states, _ = S.battle_to_states(battle, 2, random.Random(6))
    for state in states:
        side = state.side_two
        slot = [m.id for m in side.pokemon[0].moves].index("flamethrower")
        assert side.last_used_move == f"move:{slot}" and "encore" in side.volatile_statuses
        # searching this state must not hit the engine's encore panic
        assert S.mcts_worker(state.to_string(), 0, 200)[2] is None
    # Encore without a known last move is dropped instead of crashing the engine.
    battle.parse_message(["", "-start", "p1a: Rhydon", "Encore"])
    state = S.battle_to_states(battle, 1, random.Random(6))[0][0]
    assert "encore" not in state.side_one.volatile_statuses
    assert state.side_one.last_used_move == "move:none"


def test_field_and_side_conditions(battle):
    battle.parse_message(["", "-weather", "RainDance"])
    battle.parse_message(["", "-sidestart", "p2: opp", "move: Stealth Rock"])
    battle.parse_message(["", "-sidestart", "p1: tester", "Spikes"])
    battle.parse_message(["", "-sidestart", "p1: tester", "Spikes"])
    battle.parse_message(["", "-sidestart", "p1: tester", "move: Reflect"])
    battle.parse_message(["", "-fieldstart", "move: Trick Room"])
    state = S.battle_to_states(battle, 1, random.Random(5))[0][0]
    state = pe.State.from_string(state.to_string())  # what the engine actually parsed
    assert state.weather.lower() == "rain" and state.weather_turns_remaining >= 1
    assert state.trick_room
    assert state.side_two.side_conditions.stealth_rock == 1
    assert state.side_one.side_conditions.spikes == 2
    assert state.side_one.side_conditions.reflect == 5


# ---------------------------------------------------------------------------
# Choice strings <-> actions


def test_choice_map_round_trip(battle):
    _, cmap = S.battle_to_states(battle, 1, random.Random(0))
    assert cmap.action("earthquake") == 6
    assert cmap.action("megahorn") == 9
    assert cmap.action("stoneedge-tera") == 24
    assert cmap.action("switch lanturn") == 1
    assert cmap.action("No Move") is None
    assert cmap.action("switch pikachu") is None
    for action in (6, 9, 24, 1):
        order = S.action_order(action, battle)
        assert str(order) in {str(o) for o in battle.valid_orders}


def test_engine_choices_map_to_legal_actions(battle):
    states, cmap = S.battle_to_states(battle, 1, random.Random(0))
    result = S.mcts_worker(states[0].to_string(), 0, 300)
    side_one, total, err, _ = result
    assert err is None and total >= 300
    mask = S.legal_mask(battle)
    actions = {cmap.action(c) for c, _, _ in side_one}
    assert None not in actions
    assert all(mask[a] for a in actions)
    # every legal action is an engine option (4 moves, 4 tera moves, 1 switch)
    assert actions == set(np.flatnonzero(mask))


def test_aggregate_and_blend():
    cmap = S.ChoiceMap(moves={"tackle": 0, "growl": 1}, switches={"eevee": 2})
    results = [
        ([("tackle", 60, 30.0), ("growl", 40, 10.0)], 100, None, 0.1),
        ([("tackle", 20, 5.0), ("switch eevee", 80, 50.0)], 100, None, 0.1),
        ([], 0, "PanicException: boom", 0.0),
    ]
    shares, errors = S.aggregate_visits(results, cmap)
    assert errors == ["PanicException: boom"]
    assert shares[6] == pytest.approx(0.4) and shares[7] == pytest.approx(0.2)
    assert shares[2] == pytest.approx(0.4)
    mask = np.zeros(S.N_ACTIONS, dtype=bool)
    mask[[2, 6, 7]] = True
    assert S.combine_scores(shares, None, mask, 0.0) in (2, 6)
    probs = np.zeros(S.N_ACTIONS)
    probs[7] = 1.0
    assert S.combine_scores(shares, probs, mask, 0.8) == 7
    assert S.combine_scores(np.zeros(S.N_ACTIONS), probs, mask, 0.5) is None
    mask[7] = False  # illegal actions never win, however strong the prior
    assert S.combine_scores(shares, probs, mask, 0.99) != 7


def test_parse_search_spec():
    spec = S.parse_search_spec("search:models/x.zip,samples=6,time_ms=80,prior=0.3,unrevealed=fainted")
    assert spec == {"model_path": "models/x.zip", "n_samples": 6, "time_ms": 80,
                    "prior_weight": 0.3, "unrevealed": "fainted"}
    assert S.parse_search_spec("search") == {"model_path": None}
    with pytest.raises(ValueError):
        S.parse_search_spec("search,bogus=1")


# ---------------------------------------------------------------------------
# Legality over recorded battles + fallbacks


@pytest.mark.parametrize("fixture", ["protocol_p1.json", "protocol_p2.json"])
def test_search_orders_are_legal_on_recorded_battles(fixture):
    stats = S.SearchStats()
    rng = random.Random(7)
    n = 0
    for battle in _replay_decisions(fixture):
        order = S.search_policy(battle, None, n_samples=2, time_ms=0, iterations=200,
                                rng=rng, stats=stats)
        assert str(order) in {str(o) for o in battle.valid_orders}
        action = SinglesEnv.order_to_action(order, battle)
        assert S.legal_mask(battle)[int(action)]
        n += 1
    assert n > 10
    assert stats.fallbacks == 0 and stats.engine_errors == 0
    assert stats.illegal_choices == 0 and stats.unmapped_choices == 0


def test_engine_error_falls_back(battle, monkeypatch):
    monkeypatch.setattr(S, "mcts_worker", lambda *a, **k: ([], 0, "PanicException: x", 0.0))
    stats = S.SearchStats()
    order = S.search_policy(battle, None, n_samples=2, time_ms=1, stats=stats)
    assert str(order) in {str(o) for o in battle.valid_orders}
    assert stats.fallbacks == 1 and stats.engine_errors == 2


def test_conversion_error_falls_back_to_policy(battle, monkeypatch):
    def boom(*_a, **_k):
        raise KeyError("unknown species")

    class FakePolicyModel:
        pass

    monkeypatch.setattr(S, "battle_to_states", boom)
    monkeypatch.setattr(S, "policy_probs",
                        lambda model, b, mask: np.where(mask, np.eye(S.N_ACTIONS)[8], 0.0))
    stats = S.SearchStats()
    order = S.search_policy(battle, FakePolicyModel(), stats=stats)
    assert stats.fallbacks == 1 and "unknown species" in stats.last_error
    assert order.order.id == "stoneedge"  # the fake policy's argmax (action 8)


def test_search_player_inline(battle):
    player = S.SearchPlayer(n_samples=2, time_ms=0, iterations=100, workers=0, seed=0,
                            account_configuration=AccountConfiguration("searchtest", None),
                            start_listening=False, log_level=50)
    order = player.choose_move(battle)
    assert str(order) in {str(o) for o in battle.valid_orders}
    assert player.stats.searched == 1 and player.stats.fallbacks == 0


def test_run_searches_raises_timeout_when_engine_hangs(monkeypatch):
    import time
    from concurrent.futures import ThreadPoolExecutor

    import showdownrl.search as search

    monkeypatch.setattr(search, "mcts_worker", lambda *a, **k: time.sleep(2))
    monkeypatch.setattr(search, "search_timeout_s", lambda cfg: 0.1)
    cfg = search.SearchConfig(1, 10, 0.0, "sample", 0)
    with ThreadPoolExecutor(1) as pool:
        with pytest.raises(TimeoutError):
            search.run_searches(pool, ["state"], cfg)
