"""Bridge live Pokemon Showdown state into trained move-selection policies."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from showdownrl.ai import (
    RECOVERY_MOVES,
    SETUP_MOVES,
    STATUS_MOVES,
    normalize_move_name,
    parse_move_power,
    ranked_moves,
)

MODEL_FILENAME = "ppo_move_selection_v2_typed.zip"
RICH_MODEL_FILENAME = "ppo_move_selection_v3_rich.zip"
FINETUNED_MODEL_FILENAME = "ppo_move_selection_v5_rich_finetuned.zip"
MASKABLE_MODEL_FILENAME = "maskable_ppo_v11_conservative_3M.zip"
LEGACY_MASKABLE_MODEL_FILENAME = "maskable_ppo_move_selection_v6_rich.zip"
LEGACY_MODEL_FILENAME = "ppo_move_selection_v1.zip"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / MODEL_FILENAME
SIMPLE_OBS_SIZE = 14
RICH_FEATURES_PER_MOVE = 8
RICH_OBS_SIZE = SIMPLE_OBS_SIZE + 4 * RICH_FEATURES_PER_MOVE
DEFAULT_MAX_BENCH_SIZE = 3
BENCH_FEATURES_PER_POKEMON = 20
TEAM_RICH_OBS_SIZE = RICH_OBS_SIZE + DEFAULT_MAX_BENCH_SIZE * BENCH_FEATURES_PER_POKEMON


def model_search_paths(filename: str = RICH_MODEL_FILENAME) -> list[Path]:
    """Return model locations used by editable, source, and wheel installs."""
    root = Path(__file__).resolve().parent.parent
    package_dir = Path(__file__).resolve().parent
    return [
        root / "models" / filename,
        Path.cwd() / "models" / filename,
        package_dir / "models" / filename,
        Path(sys.prefix) / "models" / filename,
    ]


def default_model_path() -> Path:
    candidates = [
        *model_search_paths(MASKABLE_MODEL_FILENAME),
        *model_search_paths(LEGACY_MASKABLE_MODEL_FILENAME),
        *model_search_paths(FINETUNED_MODEL_FILENAME),
        *model_search_paths(RICH_MODEL_FILENAME),
        *model_search_paths(MODEL_FILENAME),
        *model_search_paths(LEGACY_MODEL_FILENAME),
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


def _status_penalty(value: Any) -> float:
    return 0.2 if str(value or "").strip() else 0.0


def defensive_type_score(defender_types: list[str], attacker_types: list[str]) -> float:
    if not defender_types or not attacker_types:
        return 0.0
    incoming = 0.0
    for attacker_type in attacker_types:
        chart = TYPE_CHART.get(attacker_type, {})
        multiplier = 1.0
        for defender_type in defender_types:
            multiplier *= chart.get(defender_type, 1.0)
        incoming += multiplier
    average_incoming = incoming / len(attacker_types)
    return max(-1.0, min(1.0, 1.0 - average_incoming))


def score_switch_option(option: dict[str, Any], turn_state: dict[str, Any] | None = None) -> float:
    text = str(option.get("text") or option.get("name") or "").lower()
    if "fainted" in text or "disabled" in text or "0%" in text:
        return -100.0

    score = _hp_fraction(option) * 2.0
    score -= _status_penalty(option.get("status"))
    switch_types = _normalized_types(option.get("types"))
    opponent_types = _normalized_types(((turn_state or {}).get("opponent") or {}).get("types"))
    score += defensive_type_score(switch_types, opponent_types)
    return score


def ranked_switches(
    switch_options: list[dict[str, Any]],
    turn_state: dict[str, Any] | None = None,
) -> list[tuple[dict[str, Any], float]]:
    scored = [(option, score_switch_option(option, turn_state)) for option in switch_options]
    return sorted(scored, key=lambda item: item[1], reverse=True)


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


def turn_state_to_rich_observation(
    turn_state: dict[str, Any] | None,
    moves: list[dict[str, Any]],
) -> list[float]:
    """Project a live turn-state payload into the rich v3 PPO observation."""
    state = turn_state or {}
    observation = turn_state_to_observation(state, moves) + [0.0] * (RICH_OBS_SIZE - SIMPLE_OBS_SIZE)
    opponent_hp = _hp_fraction(state.get("opponent"))

    for index in range(4):
        if index >= len(moves):
            continue
        move = moves[index]
        rich_base = SIMPLE_OBS_SIZE + index * RICH_FEATURES_PER_MOVE
        power = max(0.0, min(1.0, parse_move_power(move) / 100.0))
        accuracy = _accuracy_fraction(move)
        multiplier = damage_multiplier(move, state)
        expected_damage = power * accuracy * multiplier * 0.25
        move_type = str(move.get("type") or "").strip().lower()
        active_types = _normalized_types((state.get("active") or {}).get("types"))
        effectiveness = type_effectiveness(move, state.get("opponent"))
        name = normalize_move_name(str(move.get("name") or ""))
        category = str(move.get("category") or "").strip().lower()

        observation[rich_base] = min(4.0, power * accuracy * multiplier)
        observation[rich_base + 1] = 1.0 if move_type and move_type in active_types else 0.0
        observation[rich_base + 2] = 1.0 if effectiveness > 1.0 else 0.0
        observation[rich_base + 3] = 1.0 if effectiveness < 1.0 else 0.0
        observation[rich_base + 4] = 1.0 if expected_damage >= opponent_hp else 0.0
        observation[rich_base + 5] = 1.0 if name in RECOVERY_MOVES else 0.0
        observation[rich_base + 6] = 1.0 if name in SETUP_MOVES else 0.0
        observation[rich_base + 7] = 1.0 if category == "status" or name in STATUS_MOVES else 0.0
    return observation


def turn_state_to_team_observation(
    turn_state: dict[str, Any] | None,
    moves: list[dict[str, Any]],
) -> list[float]:
    """Project live state into the 106-float bench-aware simulator observation."""
    state = turn_state or {}
    observation = turn_state_to_rich_observation(state, moves)
    switch_options = list((state.get("switch_options") or [])[:DEFAULT_MAX_BENCH_SIZE])
    observation.extend([0.0] * (TEAM_RICH_OBS_SIZE - len(observation)))
    bench_base = RICH_OBS_SIZE
    for index, option in enumerate(switch_options):
        slot_base = bench_base + index * BENCH_FEATURES_PER_POKEMON
        observation[slot_base] = _hp_fraction(option)
        switch_types = _normalized_types(option.get("types"))
        for type_index, pokemon_type in enumerate(TYPE_CHART):
            observation[slot_base + 1 + type_index] = 1.0 if pokemon_type in switch_types else 0.0
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
            model_shape = getattr(getattr(model, "observation_space", None), "shape", ())
            self.observation_size = int(model_shape[0]) if model_shape else RICH_OBS_SIZE
            action_space = getattr(model, "action_space", None)
            self.action_size = int(getattr(action_space, "n", 4) or 4)
            return

        if not self.model_path.exists():
            raise PolicyLoadError(f"PPO model not found: {self.model_path}")
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise PolicyLoadError("Install RL dependencies with `pip install -e '.[rl]'` to use --policy ppo.") from exc

        if "maskable" in self.model_path.stem:
            try:
                from sb3_contrib import MaskablePPO
            except ImportError as exc:
                raise PolicyLoadError(
                    "Install RL dependencies with `pip install -e '.[rl]'` to use a MaskablePPO model."
                ) from exc
            self.model = MaskablePPO.load(str(self.model_path))
        else:
            try:
                self.model = PPO.load(str(self.model_path))
            except Exception as ppo_error:
                try:
                    from sb3_contrib import MaskablePPO
                except ImportError as exc:
                    raise PolicyLoadError(
                        "Install RL dependencies with `pip install -e '.[rl]'` to use a MaskablePPO model."
                    ) from exc
                try:
                    self.model = MaskablePPO.load(str(self.model_path))
                except Exception as maskable_error:
                    raise PolicyLoadError(
                        f"Could not load PPO model: {ppo_error}; MaskablePPO fallback also failed: {maskable_error}"
                    ) from maskable_error
        self.observation_size = int(getattr(self.model.observation_space, "shape", (SIMPLE_OBS_SIZE,))[0])
        self.action_size = int(getattr(getattr(self.model, "action_space", None), "n", 4) or 4)

    def choose(self, moves: list[dict[str, Any]], turn_state: dict[str, Any] | None) -> PolicyChoice:
        fallback = ranked_moves(moves, turn_state)
        if not moves:
            return PolicyChoice(fallback, "heuristic", "no available moves")

        if getattr(self, "observation_size", SIMPLE_OBS_SIZE) > RICH_OBS_SIZE:
            observation = turn_state_to_team_observation(turn_state, moves)
        elif getattr(self, "observation_size", SIMPLE_OBS_SIZE) > SIMPLE_OBS_SIZE:
            observation = turn_state_to_rich_observation(turn_state, moves)
        else:
            observation = turn_state_to_observation(turn_state, moves)
        try:
            try:
                import numpy as np

                model_observation: Any = np.array(observation, dtype=np.float32)
            except ImportError:
                model_observation = observation
            predict_kwargs = {"deterministic": True}
            if "sb3_contrib" in type(self.model).__module__:
                import numpy as np

                action_masks = np.zeros(getattr(self, "action_size", 4), dtype=bool)
                action_masks[: min(len(moves), 4)] = True
                predict_kwargs["action_masks"] = action_masks
            action, _ = self.model.predict(model_observation, **predict_kwargs)
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
