"""Offline tests: rebuild poke-env Battles from recorded Showdown protocol.

Fixtures in ``tests/fixtures/protocol_p*.json`` were recorded on a local
Showdown server with ``scripts/record_protocol_fixtures.py``: every raw
websocket message of one player's battle room plus a summary of the Battle
state poke-env's own ``Player`` saw at each decision.
"""

from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from poke_env.concurrency import POKE_LOOP
from poke_env.environment import SinglesEnv
from poke_env.player import Player
from poke_env.ps_client import AccountConfiguration

from showdownrl.battle_features import N_ACTIONS, OBS_SIZE, embed_battle
from showdownrl.policy_bridge import LivePolicy, PolicyLoadError, choose_order, order_is_valid
from showdownrl.protocol_battle import (
    ProtocolBattle,
    battle_summary,
    initial_side_order,
    order_to_click,
    parse_room_message,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class _ReplayPlayer(Player):
    """poke-env's own Player message path, run offline over recorded messages."""

    def __init__(self, username: str):
        super().__init__(
            account_configuration=AccountConfiguration(username, None),
            start_listening=False,
            log_level=50,
        )
        self.snapshots: list[tuple[np.ndarray, list[int], dict[str, Any]]] = []

        async def no_send(*_args: Any, **_kwargs: Any) -> None:
            return None

        self.ps_client.send_message = no_send  # type: ignore[method-assign]

    def choose_move(self, battle):  # noqa: ANN001
        self.snapshots.append((embed_battle(battle), SinglesEnv.get_action_mask(battle), battle_summary(battle)))
        return self.choose_random_singles_move(battle)


def golden_snapshots(fixture: dict[str, Any]) -> list[tuple[np.ndarray, list[int], dict[str, Any]]]:
    player = _ReplayPlayer(fixture["username"])

    async def run() -> None:
        for message in fixture["messages"]:
            await player._handle_battle_message([line.split("|") for line in message.split("\n")])

    asyncio.run_coroutine_threadsafe(run(), POKE_LOOP).result(timeout=60)
    return player.snapshots


def decision_points(fixture: dict[str, Any]) -> list[int]:
    return [d["after_message"] for d in fixture["decisions"]]


def request_at(fixture: dict[str, Any], index: int) -> dict[str, Any]:
    _, lines = parse_room_message(fixture["messages"][index])
    for line in lines:
        if line.startswith("|request|"):
            return json.loads(line[len("|request|"):])
    raise AssertionError(f"message {index} has no request")


class FakeModel:
    """Stands in for a MaskablePPO model with the live feature space."""

    def __init__(self, choose=None, obs_size: int = OBS_SIZE):  # noqa: ANN001
        self.observation_space = spaces.Box(-4.0, 4.0, (obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.choose = choose or (lambda mask: int(np.flatnonzero(mask)[0]))
        self.seen: list[tuple[np.ndarray, np.ndarray]] = []

    def predict(self, obs, action_masks=None, deterministic=True):  # noqa: ANN001
        self.seen.append((obs, action_masks))
        return np.int64(self.choose(action_masks)), None


class ProtocolReplayTests(unittest.TestCase):
    """The builder must see exactly what poke-env saw during training."""

    maxDiff = None

    def check_fixture(self, name: str, expected_role: str) -> None:
        fixture = load_fixture(name)
        golden = golden_snapshots(fixture)
        points = decision_points(fixture)
        self.assertEqual(len(golden), len(points))

        state = ProtocolBattle(username=fixture["username"])
        decisions = iter(zip(points, fixture["decisions"], golden))
        point, recorded, (obs, mask, summary) = next(decisions)
        for index, message in enumerate(fixture["messages"]):
            state.feed(message)
            if index != point:
                continue
            self.assertTrue(state.needs_decision, f"no decision pending after message {index}")
            battle = state.battle
            assert battle is not None
            self.assertEqual(battle.player_role, expected_role)
            self.assertEqual(battle_summary(battle), recorded["summary"])
            self.assertEqual(battle_summary(battle), summary)
            np.testing.assert_allclose(embed_battle(battle), obs, rtol=0, atol=1e-6)
            self.assertEqual(SinglesEnv.get_action_mask(battle), mask)
            state.mark_answered()
            self.assertFalse(state.needs_decision)
            point, recorded, (obs, mask, summary) = next(decisions, (None, None, (None, None, None)))
        self.assertIsNone(point)
        self.assertTrue(state.finished)
        self.assertEqual(state.battle.won, fixture["won"])
        self.assertEqual(state.parse_errors, [])

    def test_replay_matches_poke_env_as_p1(self) -> None:
        self.check_fixture("protocol_p1.json", "p1")

    def test_replay_matches_poke_env_as_p2(self) -> None:
        self.check_fixture("protocol_p2.json", "p2")

    def test_role_from_request_without_username(self) -> None:
        fixture = load_fixture("protocol_p2.json")
        state = ProtocolBattle.from_messages(fixture["messages"][: decision_points(fixture)[0] + 1])
        self.assertEqual(state.role, "p2")
        self.assertEqual(state.battle.player_role, "p2")
        self.assertEqual(battle_summary(state.battle), fixture["decisions"][0]["summary"])

    def test_role_from_username_case_insensitive(self) -> None:
        fixture = load_fixture("protocol_p1.json")
        first_request = decision_points(fixture)[0]
        state = ProtocolBattle(username=fixture["username"].upper().replace(" ", ""))
        for message in fixture["messages"][:first_request]:
            state.feed(message)
        # Role is known from |player| before any request arrives.
        self.assertEqual(state.role, "p1")
        self.assertIsNotNone(state.battle)
        self.assertFalse(state.needs_decision)

    def test_active_vs_opponent_and_hp_come_from_the_right_side(self) -> None:
        fixture = load_fixture("protocol_p2.json")
        point = decision_points(fixture)[0]
        state = ProtocolBattle.from_messages(fixture["messages"][: point + 1], username=fixture["username"])
        request = request_at(fixture, point)
        active_entry = next(p for p in request["side"]["pokemon"] if p["active"])
        battle = state.battle
        self.assertEqual(battle.active_pokemon.name, active_entry["ident"].split(": ", 1)[1])
        hp, max_hp = (int(v) for v in active_entry["condition"].split()[0].split("/"))
        self.assertEqual((battle.active_pokemon.current_hp, battle.active_pokemon.max_hp), (hp, max_hp))
        # The opponent's active Pokemon is the p1 mon from the log, never ours.
        opp = battle.opponent_active_pokemon
        self.assertIsNotNone(opp)
        self.assertNotIn(opp, battle.team.values())
        self.assertIn(opp, battle.opponent_team.values())
        self.assertEqual({m.id for m in battle.available_moves},
                         {m["id"] for m in request["active"][0]["moves"] if not m.get("disabled")})
        self.assertEqual({m.name for m in battle.available_switches},
                         {p["ident"].split(": ", 1)[1] for p in request["side"]["pokemon"]
                          if not p["active"] and not p["condition"].endswith("fnt")})

    def test_snapshot_rebuild_matches_stream(self) -> None:
        for name in ("protocol_p1.json", "protocol_p2.json"):
            fixture = load_fixture(name)
            golden = golden_snapshots(fixture)
            for point, (obs, mask, summary) in list(zip(decision_points(fixture), golden)):
                lines: list[str] = []
                for message in fixture["messages"][: point + 1]:
                    lines.extend(parse_room_message(message)[1])
                state = ProtocolBattle.from_snapshot(lines, request_at(fixture, point),
                                                     battle_tag=fixture["battle_tag"])
                self.assertEqual(battle_summary(state.battle), summary, f"{name} @ {point}")
                np.testing.assert_allclose(embed_battle(state.battle), obs, atol=1e-6)
                self.assertEqual(SinglesEnv.get_action_mask(state.battle), mask)

    def test_initial_side_order_undoes_switches(self) -> None:
        request = {"side": {"pokemon": [{"ident": "p1: C"}, {"ident": "p1: A"}, {"ident": "p1: B"}]}}
        lines = ["|switch|p1a: A|A, L80|100/100", "|switch|p1a: B|B|100/100", "|switch|p1a: C|C|100/100"]
        # start [A, B, C] -> B in: [B, A, C] -> C in: [C, A, B]
        order = [e["ident"] for e in initial_side_order(lines, request, "p1")]
        self.assertEqual(order, ["p1: A", "p1: B", "p1: C"])

    def test_error_reopens_the_decision(self) -> None:
        fixture = load_fixture("protocol_p1.json")
        point = decision_points(fixture)[0]
        state = ProtocolBattle.from_messages(fixture["messages"][: point + 1])
        state.mark_answered()
        self.assertFalse(state.needs_decision)
        state.feed(f">{fixture['battle_tag']}\n|error|[Invalid choice] Can't move: Invalid move")
        self.assertTrue(state.needs_decision)
        self.assertIn("Invalid choice", state.last_error)

    def test_wait_request_needs_no_decision(self) -> None:
        fixture = load_fixture("protocol_p1.json")
        index = next(i for i, m in enumerate(fixture["messages"]) if '|request|{"wait":true' in m)
        state = ProtocolBattle.from_messages(fixture["messages"][: index + 1])
        self.assertTrue(state.ready)
        self.assertFalse(state.needs_decision)


class OrderToClickTests(unittest.TestCase):
    def state_at(self, name: str, predicate) -> tuple[ProtocolBattle, dict[str, Any]]:  # noqa: ANN001
        fixture = load_fixture(name)
        for decision in fixture["decisions"]:
            if predicate(decision):
                point = decision["after_message"]
                state = ProtocolBattle.from_messages(fixture["messages"][: point + 1], username=fixture["username"])
                return state, request_at(fixture, point)
        raise AssertionError("no matching decision in fixture")

    def test_move_order_maps_to_request_slot(self) -> None:
        state, request = self.state_at("protocol_p1.json", lambda d: d["summary"]["moves"])
        battle = state.battle
        moves = request["active"][0]["moves"]
        for slot, entry in enumerate(moves, start=1):
            if entry.get("disabled"):
                continue
            move = next(m for m in battle.available_moves if m.id == entry["id"])
            plan = order_to_click(Player.create_order(move), request)
            self.assertEqual((plan.kind, plan.slot, plan.move_id), ("move", slot, entry["id"]))
            self.assertIn(f".movemenu button[name='chooseMove'][value='{slot}']", plan.selectors)
            self.assertIn(f".movemenu button[data-cmd='/move {slot}']", plan.selectors)
            self.assertFalse(plan.terastallize)
            self.assertEqual(plan.tera_selectors, ())

    def test_tera_order_checks_terastallize_box(self) -> None:
        state, request = self.state_at("protocol_p2.json", lambda d: d["summary"]["can_tera"])
        move = state.battle.available_moves[0]
        plan = order_to_click(Player.create_order(move, terastallize=True), request)
        self.assertTrue(plan.terastallize)
        self.assertIn("input[name='terastallize']", plan.tera_selectors)
        self.assertIn("input[name='tera']", plan.tera_selectors)
        self.assertTrue(plan.choose_command.endswith("terastallize"))

    def test_switch_order_maps_to_request_slot(self) -> None:
        state, request = self.state_at("protocol_p2.json", lambda d: d["summary"]["switches"])
        side = request["side"]["pokemon"]
        for mon in state.battle.available_switches:
            plan = order_to_click(Player.create_order(mon), request)
            self.assertEqual(plan.kind, "switch")
            self.assertEqual(side[plan.slot - 1]["ident"].split(": ", 1)[1], mon.name)
            self.assertIn(f".switchmenu button[name='chooseSwitch'][value='{plan.slot - 1}']", plan.selectors)
            self.assertIn(f".switchmenu button[data-cmd='/switch {plan.slot}']", plan.selectors)
            self.assertFalse(side[plan.slot - 1]["active"])


class LiveDecisionTests(unittest.TestCase):
    def voluntary_switch_state(self, need_tera: bool = False) -> tuple[ProtocolBattle, dict[str, Any]]:
        fixture = load_fixture("protocol_p2.json")
        for decision in fixture["decisions"]:
            summary = decision["summary"]
            if need_tera and not summary["can_tera"]:
                continue
            if summary["moves"] and summary["switches"] and not summary["force_switch"]:
                point = decision["after_message"]
                state = ProtocolBattle.from_messages(fixture["messages"][: point + 1])
                return state, request_at(fixture, point)
        raise AssertionError("fixture has no voluntary-switch decision")

    def test_model_can_switch_voluntarily(self) -> None:
        state, request = self.voluntary_switch_state()
        mask = np.array(SinglesEnv.get_action_mask(state.battle), dtype=bool)
        self.assertTrue(mask[:6].any(), "switch actions must be unmasked on a normal turn")
        self.assertTrue(mask[6:10].any())
        model = FakeModel(choose=lambda m: int(np.flatnonzero(m[:6])[0]))
        decision = choose_order(state.battle, LivePolicy(model=model))
        self.assertEqual(decision.source, "ppo")
        self.assertEqual(decision.fallback_reason, "")
        plan = order_to_click(decision.order, request)
        self.assertEqual(plan.kind, "switch")
        self.assertGreater(plan.slot, 1)
        obs, _ = model.seen[0]
        np.testing.assert_allclose(obs, embed_battle(state.battle))

    def test_model_can_terastallize(self) -> None:
        state, request = self.voluntary_switch_state(need_tera=True)
        self.assertTrue(any(SinglesEnv.get_action_mask(state.battle)[22:26]))
        model = FakeModel(choose=lambda m: int(np.flatnonzero(m[22:26])[0] + 22))
        decision = choose_order(state.battle, LivePolicy(model=model))
        self.assertTrue(decision.order.terastallize)
        self.assertIn(decision.action, range(22, 26))
        self.assertTrue(order_to_click(decision.order, request).terastallize)

    def test_without_model_uses_smart_heuristic(self) -> None:
        state, _ = self.voluntary_switch_state()
        decision = choose_order(state.battle, None)
        self.assertEqual(decision.source, "smart")
        self.assertTrue(order_is_valid(decision.order, state.battle))

    def test_incompatible_model_is_rejected(self) -> None:
        with self.assertRaises(PolicyLoadError):
            LivePolicy(model=FakeModel(obs_size=106))

    def test_failing_model_falls_back_to_heuristic(self) -> None:
        state, _ = self.voluntary_switch_state()

        def boom(_mask):  # noqa: ANN001
            raise RuntimeError("boom")

        decision = choose_order(state.battle, LivePolicy(model=FakeModel(choose=boom)))
        self.assertEqual(decision.source, "smart")
        self.assertIn("boom", decision.fallback_reason)

    def test_forced_switch_excludes_active(self) -> None:
        fixture = load_fixture("protocol_p1.json")
        decision = next(d for d in fixture["decisions"] if d["summary"]["force_switch"])
        point = decision["after_message"]
        state = ProtocolBattle.from_messages(fixture["messages"][: point + 1])
        request = request_at(fixture, point)
        mask = SinglesEnv.get_action_mask(state.battle)
        self.assertFalse(any(mask[6:]))
        order = choose_order(state.battle, None).order
        plan = order_to_click(order, request)
        self.assertEqual(plan.kind, "switch")
        entry = request["side"]["pokemon"][plan.slot - 1]
        self.assertFalse(entry["active"])
        self.assertFalse(entry["condition"].endswith("fnt"))


if __name__ == "__main__":
    unittest.main()


def test_rejoin_replay_resets_room_instead_of_double_applying():
    from showdownrl.protocol_battle import BattleRoomTracker

    tracker = BattleRoomTracker("tester")
    room = "battle-gen9randombattle-42"
    first = f">{room}\n|init|battle\n|title|x vs y\n"
    tracker.ingest([first])
    old_state = tracker.rooms[room]
    tracker.ingest([f">{room}\n|t:|1\n"])
    assert tracker.rooms[room] is old_state
    tracker.ingest([first])  # reconnect: server replays the room from |init|
    assert tracker.rooms[room] is not old_state
