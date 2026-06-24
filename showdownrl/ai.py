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

RECOVERY_MOVES = {
    "healorder",
    "milkdrink",
    "moonlight",
    "morningsun",
    "recover",
    "rest",
    "roost",
    "shoreup",
    "slackoff",
    "softboiled",
    "synthesis",
    "wish",
}

STATUS_MOVES = {
    "glare",
    "hypnosis",
    "sleeppowder",
    "spore",
    "stunspore",
    "thunderwave",
    "toxic",
    "willowisp",
    "yawn",
}

SETUP_MOVES = {
    "agility",
    "bulkup",
    "calmmind",
    "dragondance",
    "irondefense",
    "nastyplot",
    "quiverdance",
    "rockpolish",
    "swordsdance",
    "tailglow",
    "workup",
}


def normalize_move_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def parse_move_power(move: dict[str, Any]) -> int:
    raw_power = move.get("power") or move.get("base_power") or move.get("basePower")
    if raw_power:
        try:
            return int(raw_power)
        except (TypeError, ValueError):
            pass

    text = str(move.get("text", ""))
    power_match = re.search(r"(?:power|base power)\D{0,8}(\d{1,3})", text, re.I)
    if power_match:
        return int(power_match.group(1))

    numbers = [int(value) for value in re.findall(r"\b(\d{2,3})\b", text)]
    plausible = [value for value in numbers if 20 <= value <= 250]
    return plausible[0] if plausible else 0


def move_category(move: dict[str, Any]) -> str:
    return str(move.get("category") or "").strip().lower()


def active_types(move: dict[str, Any]) -> set[str]:
    raw = move.get("active_types") or move.get("user_types") or []
    if isinstance(raw, str):
        raw = [raw]
    return {str(value).strip().lower() for value in raw if str(value).strip()}


def hp_percent(pokemon: dict[str, Any] | None) -> int | None:
    if not pokemon:
        return None
    value = pokemon.get("hp_percent")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def has_status(pokemon: dict[str, Any] | None) -> bool:
    if not pokemon:
        return False
    return bool(str(pokemon.get("status") or "").strip())


def context_types(turn_state: dict[str, Any] | None) -> set[str]:
    if not turn_state:
        return set()
    active = turn_state.get("active") or {}
    raw = active.get("types") or []
    if isinstance(raw, str):
        raw = [raw]
    return {str(value).strip().lower() for value in raw if str(value).strip()}


def score_move(move: dict[str, Any], turn_state: dict[str, Any] | None = None) -> float:
    """Return a simple score for a Pokemon Showdown move button payload."""
    text = str(move.get("text") or "")
    text_lower = text.lower()
    move_type = str(move.get("type") or "").lower()
    name = normalize_move_name(str(move.get("name") or ""))
    category = move_category(move)
    power = parse_move_power(move)
    own_hp = hp_percent((turn_state or {}).get("active"))
    opponent_hp = hp_percent((turn_state or {}).get("opponent"))
    opponent_has_status = has_status((turn_state or {}).get("opponent"))

    if "doesn't affect" in text_lower or "does not affect" in text_lower or "immune" in text_lower:
        return -100.0
    if "disabled" in text_lower or "no pp" in text_lower:
        return -50.0

    score = TYPE_SCORE.get(move_type, 1.0)
    if power:
        score += min(power, 180) / 100

    user_types = active_types(move) or context_types(turn_state)
    if move_type and move_type in user_types:
        score *= 1.25

    if "super effective" in text_lower:
        score *= 1.35
    if "not very effective" in text_lower:
        score *= 0.65

    if opponent_hp is not None and power:
        if opponent_hp <= 25:
            score += 0.75
        elif opponent_hp <= 45:
            score += 0.3
    if any(word in text_lower for word in ("ko", "faint", "finish")):
        score += 0.5

    recovery_when_low = False
    if name in RECOVERY_MOVES:
        if own_hp is not None and own_hp <= 35:
            score += 1.0
            recovery_when_low = True
        elif own_hp is not None and own_hp >= 70:
            score *= 0.4
    if name in STATUS_MOVES and opponent_has_status:
        score *= 0.35
    if name in SETUP_MOVES and own_hp is not None and own_hp <= 35:
        score *= 0.45

    if category == "status" or name in NON_ATTACKS or (not power and category != "physical" and category != "special"):
        score *= 0.75 if recovery_when_low else 0.3
    return score


def ranked_moves(
    moves: list[dict[str, Any]],
    turn_state: dict[str, Any] | None = None,
) -> list[tuple[dict[str, Any], float]]:
    scored = [(move, score_move(move, turn_state)) for move in moves]
    return sorted(scored, key=lambda item: item[1], reverse=True)
