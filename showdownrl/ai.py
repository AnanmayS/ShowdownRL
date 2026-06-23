"""Move scoring policy used by the live browser player."""

from __future__ import annotations

import re
from typing import Any

TYPE_SCORE = {
    "normal": 1.0,
    "fire": 1.2,
    "water": 1.2,
    "electric": 1.2,
    "grass": 1.1,
    "ice": 1.3,
    "fighting": 1.3,
    "poison": 0.9,
    "ground": 1.3,
    "flying": 1.2,
    "psychic": 1.1,
    "bug": 0.9,
    "rock": 1.2,
    "ghost": 1.2,
    "dragon": 1.2,
    "dark": 1.2,
    "steel": 1.0,
    "fairy": 1.2,
}

NON_ATTACKS = {
    "agility",
    "amnesia",
    "aromatherapy",
    "bulkup",
    "calmmind",
    "defog",
    "disable",
    "dragondance",
    "encore",
    "healbell",
    "irondefense",
    "leechseed",
    "lightscreen",
    "nastyplot",
    "protect",
    "recover",
    "reflect",
    "rest",
    "roar",
    "rockpolish",
    "roost",
    "softboiled",
    "spikes",
    "stealthrock",
    "stickyweb",
    "substitute",
    "swordsdance",
    "tailwind",
    "taunt",
    "teleport",
    "thunderwave",
    "toxic",
    "toxicspikes",
    "trickroom",
    "whirlwind",
    "willowisp",
    "wish",
}


def normalize_move_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def score_move(move: dict[str, Any]) -> float:
    """Return a simple score for a Pokemon Showdown move button payload."""
    score = TYPE_SCORE.get(move.get("type", "").lower(), 1.0)
    if normalize_move_name(move.get("name", "")) in NON_ATTACKS:
        score *= 0.2
    return score
