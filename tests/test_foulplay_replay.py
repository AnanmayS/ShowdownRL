"""Foul Play log -> BC sample conversion (showdownrl.foulplay_replay).

The fixture is a real Foul Play battle log recorded by scripts/foulplay_logged_run.py
on a local server (Foul Play as p2 vs poke-env MaxBasePowerPlayer): 12 decisions,
including a choice-locked turn, a forced switch after a faint, a voluntary switch
and a terastallized move.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from showdownrl.battle_features import N_ACTIONS, OBS_SIZE
from showdownrl.foulplay_replay import (
    merge_samples,
    parse_choice,
    read_log,
    replay_records,
)

FIXTURE = Path(__file__).parent / "fixtures" / "foulplay_battle.jsonl"
EXPECTED_ACTIONS = [8, 8, 8, 2, 8, 4, 8, 9, 9, 22, 9, 9]


@pytest.mark.parametrize("msg,expected", [
    ("battle-x|/choose move 2 terastallize|7", ("move", "2", True)),
    ("battle-x|/choose move icebeam|3", ("move", "icebeam", False)),
    ("battle-x|/choose move ironhead terastallize|21", ("move", "ironhead", True)),
    ("battle-x|/switch 3|9", ("switch", "3", False)),
    ("battle-x|/choose switch 5|9", ("switch", "5", False)),
    ("battle-x|/choose switch great tusk", ("switch", "great tusk", False)),
    ("battle-x|/timer on", None),
    ("battle-x|hf", None),
    ("battle-x|gg", None),
])
def test_parse_choice(msg, expected):
    assert parse_choice(msg) == expected


def test_replay_fixture():
    s = replay_records(read_log(FIXTURE), tag="fixture")
    assert s.username == "fptc30a1r0"
    assert s.opponent == "omxc30a1r0"
    assert s.finished and s.won is True
    assert s.errors == []
    assert s.mismatches == []
    assert s.decisions == len(s.actions) == len(EXPECTED_ACTIONS)
    assert s.actions == EXPECTED_ACTIONS

    arrays = s.arrays()
    n = len(EXPECTED_ACTIONS)
    assert arrays["obs"].shape == (n, OBS_SIZE) and arrays["obs"].dtype == np.float32
    assert arrays["masks"].shape == (n, N_ACTIONS) and arrays["masks"].dtype == np.int8
    assert np.isfinite(arrays["obs"]).all()
    assert all(arrays["masks"][i, a] == 1 for i, a in enumerate(arrays["actions"]))
    assert arrays["outcomes"].tolist() == [1.0] * n
    assert arrays["steps_left"].tolist() == list(range(n - 1, -1, -1))
    # the forced switch after the faint only allows switches
    forced = 3
    assert arrays["actions"][forced] < 6 and not arrays["masks"][forced, 6:].any()
    # terastallized move lands in 22-25 with tera legal at that point
    tera = EXPECTED_ACTIONS.index(22)
    assert arrays["masks"][tera, 22]

    merged = merge_samples([s, s])
    assert len(merged["actions"]) == 2 * n


def _rewrite_moves_as_slots(records):
    """Turn '/choose move <id>' into '/choose move <slot>' using the preceding request."""
    out, request = [], None
    for rec in records:
        if rec["dir"] == "in" and "\n|request|" in rec["msg"]:
            body = rec["msg"].split("\n|request|", 1)[1].split("\n", 1)[0]
            if body:
                request = json.loads(body)
        if rec["dir"] == "out" and "/choose move " in rec["msg"]:
            room, cmd, *rest = rec["msg"].split("|")
            tokens = cmd[len("/choose move "):].split()
            ids = [m["id"] for m in request["active"][0]["moves"]]
            tokens[0] = str(ids.index(tokens[0]) + 1)
            rec = dict(rec, msg="|".join([room, "/choose move " + " ".join(tokens), *rest]))
        out.append(rec)
    return out


def test_slot_numbers_map_to_same_actions():
    s = replay_records(_rewrite_moves_as_slots(read_log(FIXTURE)))
    assert s.mismatches == [] and s.actions == EXPECTED_ACTIONS


def test_illegal_choice_is_counted_not_labelled():
    records = read_log(FIXTURE)
    first = next(i for i, r in enumerate(records) if r["dir"] == "out" and "/choose move" in r["msg"])
    room, _, *rest = records[first]["msg"].split("|")
    records[first] = dict(records[first], msg="|".join([room, "/choose move splash", *rest]))
    s = replay_records(records)
    assert len(s.mismatches) == 1 and "splash" in s.mismatches[0][0]
    assert s.actions == EXPECTED_ACTIONS[1:]
    assert s.decisions == len(EXPECTED_ACTIONS)


def test_unfinished_log_has_no_outcome():
    records = read_log(FIXTURE)
    cut = next(i for i, r in enumerate(records) if r["dir"] == "in" and "\n|win|" in r["msg"])
    s = replay_records(records[:cut])
    assert not s.finished and s.won is None
