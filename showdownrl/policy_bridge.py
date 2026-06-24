"""Bridge live Pokemon Showdown state into trained move-selection policies."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from showdownrl.ai import parse_move_power, ranked_moves

MODEL_FILENAME = "ppo_move_selection_v2_typed.zip"
LEGACY_MODEL_FILENAME = "ppo_move_selection_v1.zip"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / MODEL_FILENAME


def default_model_path() -> Path:
    candidates = [
        DEFAULT_MODEL_PATH,
        Path.cwd() / "models" / MODEL_FILENAME,
        Path(__file__).resolve().parent / "models" / MODEL_FILENAME,
        Path(__file__).resolve().parent.parent / "models" / LEGACY_MODEL_FILENAME,
        Path.cwd() / "models" / LEGACY_MODEL_FILENAME,
        Path(__file__).resolve().parent / "models" / LEGACY_MODEL_FILENAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


TYPE_CHART: dict[str, dict[str, float]] = {
    "normal": {"rock": 0.5, "ghost": 0.0, "steel": 0.5},
    "fire": {"fire": 0.5, "water": 0.5, "grass": 2.0, "ice": 2.0, "bug": 2.0, "rock": 0.5, "dragon": 0.5, "steel": 2.0},
    "water": {"fire": 2.0, "water": 0.5, "grass": 0.5, "ground": 2.0, "rock": 2.0, "dragon": 0.5},
    "electric": {"water": 2.0, "electric": 0.5, "grass": 0.5, "ground": 0.0, "flying": 2.0, "dragon": 0.5},
    "grass": {"fire": 0.5, "water": 2.0, "grass": 0.5, "poison": 0.5, "ground": 2.0, "flying": 0.5, "bug": 0.5, "rock": 2.0, "dragon": 0.5, "steel": 0.5},
    "ice": {"fire": 0.5, "water": 0.5, "grass": 2.0, "ice": 0.5, "ground": 2.0, "flying": 2.0, "dragon": 2.0, "steel": 0.5},
    "fighting": {"normal": 2.0, "ice": 2.0, "poison": 0.5, "flying": 0.5, "psychic": 0.5, "bug": 0.5, "rock": 2.0, "ghost": 0.0, "dark": 2.0, "steel": 2.0, "fairy": 0.5},
    "poison": {"grass": 2.0, "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5, "steel": 0.0, "fairy": 2.0},
    "ground": {"fire": 2.0, "electric": 2.0, "grass": 0.5, "poison": 2.0, "flying": 0.0, "bug": 0.5, "rock": 2.0, "steel": 2.0},
    "flying": {"electric": 0.5, "grass": 2.0, "fighting": 2.0, "bug": 2.0, "rock": 0.5, "steel": 0.5},
    "psychic": {"fighting": 2.0, "poison": 2.0, "psychic": 0.5, "dark": 0.0, "steel": 0.5},
    "bug": {"fire": 0.5, "grass": 2.0, "fighting": 0.5, "poison": 0.5, "flying": 0.5, "psychic": 2.0, "ghost": 0.5, "dark": 2.0, "steel": 0.5, "fairy": 0.5},
    "rock": {"fire": 2.0, "ice": 2.0, "fighting": 0.5, "ground": 0.5, "flying": 2.0, "bug": 2.0, "steel": 0.5},
    "ghost": {"normal": 0.0, "psychic": 2.0, "ghost": 2.0, "dark": 0.5},
    "dragon": {"dragon": 2.0, "steel": 0.5, "fairy": 0.0},
    "dark": {"fighting": 0.5, "psychic": 2.0, "ghost": 2.0, "dark": 0.5, "fairy": 0.5},
    "steel": {"fire": 0.5, "water": 0.5, "electric": 0.5, "ice": 2.0, "rock": 2.0, "steel": 0.5, "fairy": 2.0},
    "fairy": {"fire": 0.5, "fighting": 2.0, "poison": 0.5, "dragon": 2.0, "dark": 2.0, "steel": 0.5},
}


class PolicyLoadError(RuntimeError):
    """Raised when an optional model-backed policy cannot be loaded."""


@dataclass
class PolicyChoice:
    ranked: list[tuple[dict[str, Any], float]]
    source: str
    fallback_reason: str = ""


def _hp_fraction(pokemon: dict[str, Any] | None) -> float:
    if not pokemon:
        return 1.0
    try:
        return max(0.0, min(1.0, float(pokemon.get("hp_percent")) / 100.0))
    except (TypeError, ValueError):
        return 1.0


def _accuracy_fraction(move: dict[str, Any]) -> float:
    raw = move.get("accuracy") or move.get("acc")
    if raw:
        try:
            value = float(str(raw).strip().removesuffix("%"))
            return max(0.0, min(1.0, value / 100.0 if value > 1 else value))
        except ValueError:
            pass

    text = str(move.get("text") or "")
    match = re.search(r"(?:accuracy|acc)\D{0,8}(\d{1,3})\s*%?", text, re.I)
    if match:
        return max(0.0, min(1.0, int(match.group(1)) / 100.0))
    return 1.0


def _normalized_types(raw: Any) -> list[str]:
    if isinstance(raw, str):
        raw = [raw]
    if not raw:
        return []
    return [str(value).strip().lower() for value in raw if str(value).strip()]


def _effectiveness_from_text(text: str) -> float | None:
    lower = text.lower()
    if "doesn't affect" in lower or "does not affect" in lower or "immune" in lower:
        return 0.0
    if "super effective" in lower:
        return 2.0
    if "not very effective" in lower:
        return 0.5
    return None


def type_effectiveness(move: dict[str, Any], opponent: dict[str, Any] | None) -> float:
    text_value = _effectiveness_from_text(str(move.get("text") or ""))
    if text_value is not None:
        return text_value

    move_type = str(move.get("type") or "").strip().lower()
    opponent_types = _normalized_types((opponent or {}).get("types"))
    if not move_type or not opponent_types:
        return 1.0

    chart = TYPE_CHART.get(move_type, {})
    multiplier = 1.0
    for pokemon_type in opponent_types:
        multiplier *= chart.get(pokemon_type, 1.0)
    return multiplier


def damage_multiplier(move: dict[str, Any], turn_state: dict[str, Any] | None) -> float:
    state = turn_state or {}
    move_type = str(move.get("type") or "").strip().lower()
    active_types = _normalized_types((state.get("active") or {}).get("types"))
    stab = 1.5 if move_type and move_type in active_types else 1.0
    return stab * type_effectiveness(move, state.get("opponent"))


def turn_state_to_observation(
    turn_state: dict[str, Any] | None,
    moves: list[dict[str, Any]],
) -> list[float]:
    """Project a live turn-state payload into the 14-float PPO observation."""
    state = turn_state or {}
    observation = [0.0] * 14
    observation[0] = _hp_fraction(state.get("active"))
    observation[1] = _hp_fraction(state.get("opponent"))

    for index in range(4):
        base = 2 + index * 3
        if index >= len(moves):
            continue
        move = moves[index]
        observation[base] = max(0.0, min(1.0, parse_move_power(move) / 100.0))
        observation[base + 1] = _accuracy_fraction(move)
        observation[base + 2] = damage_multiplier(move, state)
    return observation


def _rank_with_selected_first(
    moves: list[dict[str, Any]],
    turn_state: dict[str, Any] | None,
    selected_index: int,
) -> list[tuple[dict[str, Any], float]]:
    selected = moves[selected_index]
    heuristic_tail = [
        (move, score)
        for move, score in ranked_moves(moves, turn_state)
        if move is not selected
    ]
    return [(selected, 1.0), *heuristic_tail]


class PPOMovePolicy:
    """Small adapter around the trained stable-baselines PPO model."""

    def __init__(self, model_path: Path | None = None, model: Any | None = None):
        self.model_path = model_path or default_model_path()
        if model is not None:
            self.model = model
            return

        if not self.model_path.exists():
            raise PolicyLoadError(f"PPO model not found: {self.model_path}")
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise PolicyLoadError("Install RL dependencies with `pip install -e '.[rl]'` to use --policy ppo.") from exc

        self.model = PPO.load(str(self.model_path))

    def choose(self, moves: list[dict[str, Any]], turn_state: dict[str, Any] | None) -> PolicyChoice:
        fallback = ranked_moves(moves, turn_state)
        if not moves:
            return PolicyChoice(fallback, "heuristic", "no available moves")

        observation = turn_state_to_observation(turn_state, moves)
        try:
            try:
                import numpy as np

                model_observation: Any = np.array(observation, dtype=np.float32)
            except ImportError:
                model_observation = observation
            action, _ = self.model.predict(model_observation, deterministic=True)
            if isinstance(action, (list, tuple)):
                action = action[0]
            elif hasattr(action, "item"):
                action = action.item()
            action_index = int(action)
        except Exception as exc:
            return PolicyChoice(fallback, "heuristic", f"PPO predict failed: {exc}")

        if action_index < 0 or action_index >= len(moves):
            return PolicyChoice(fallback, "heuristic", f"PPO chose unavailable action {action_index}")
        return PolicyChoice(_rank_with_selected_first(moves, turn_state, action_index), "ppo")
