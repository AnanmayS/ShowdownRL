"""Choose live orders from a poke-env ``Battle`` rebuilt from the protocol.

Live play reconstructs the battle with ``showdownrl.protocol_battle`` and asks
this module for a poke-env ``BattleOrder``. A MaskablePPO model trained on
``showdownrl.battle_features`` (``OBS_SIZE`` observations, ``N_ACTIONS``
poke-env ``SinglesEnv`` actions) is used through ``real_env.policy_order`` -
the exact function the training/evaluation code uses - so moves, switches and
Terastallization are all available to the model. Without a compatible model
the damage-calc heuristic ``smart_heuristic.choose_smart_move`` decides.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

MODEL_FILENAME = "ppo_move_selection_v2_typed.zip"
RICH_MODEL_FILENAME = "ppo_move_selection_v3_rich.zip"
FINETUNED_MODEL_FILENAME = "ppo_move_selection_v5_rich_finetuned.zip"
MASKABLE_MODEL_FILENAME = "maskable_ppo_v11_conservative_3M.zip"
LEGACY_MASKABLE_MODEL_FILENAME = "maskable_ppo_move_selection_v6_rich.zip"
LEGACY_MODEL_FILENAME = "ppo_move_selection_v1.zip"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / MODEL_FILENAME
REAL_MODEL_DIRNAME = "real"

# Type chart used by the lightweight simulator in ``simple_env``.
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
    """Default model for the legacy simple-simulator scripts."""
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


def default_live_model_path() -> Optional[Path]:
    """Newest real-simulator model (``models/real/*.zip``), if any exists."""
    seen: list[Path] = []
    for directory in {path.parent for path in model_search_paths(REAL_MODEL_DIRNAME)}:
        real_dir = directory / REAL_MODEL_DIRNAME
        if real_dir.is_dir():
            seen.extend(real_dir.glob("*.zip"))
    if not seen:
        return None
    return max(seen, key=lambda path: path.stat().st_mtime)


class PolicyLoadError(RuntimeError):
    """Raised when a model-backed live policy cannot be loaded."""


@dataclass
class Decision:
    """One live decision: the poke-env order plus where it came from."""

    order: Any  # poke_env BattleOrder
    source: str  # "search", "ppo", "smart" or "random"
    fallback_reason: str = ""
    action: Optional[int] = None


def check_model_compatible(model: Any) -> None:
    """Raise ``PolicyLoadError`` unless ``model`` uses the live feature space."""
    from showdownrl.battle_features import N_ACTIONS, OBS_SIZE

    shape = tuple(getattr(getattr(model, "observation_space", None), "shape", None) or ())
    n_actions = getattr(getattr(model, "action_space", None), "n", None)
    if shape != (OBS_SIZE,):
        raise PolicyLoadError(
            f"model observation shape {shape} does not match battle_features OBS_SIZE ({OBS_SIZE},); "
            "it was trained on a different feature set"
        )
    if n_actions is not None and int(n_actions) != N_ACTIONS:
        raise PolicyLoadError(f"model has {n_actions} actions, expected {N_ACTIONS} (poke-env SinglesEnv)")


class LivePolicy:
    """MaskablePPO policy over ``battle_features`` observations."""

    def __init__(self, model_path: Path | None = None, model: Any | None = None, deterministic: bool = True):
        self.deterministic = deterministic
        if model is not None:
            self.model = model
            self.model_path = Path(model_path) if model_path else Path("<in-memory>")
        else:
            path = Path(model_path) if model_path else default_live_model_path()
            if path is None:
                raise PolicyLoadError("no real-simulator model found in models/real/")
            if not path.exists():
                raise PolicyLoadError(f"model not found: {path}")
            try:
                from sb3_contrib import MaskablePPO
            except ImportError as exc:
                raise PolicyLoadError("Install RL dependencies with `pip install -e '.[rl]'`.") from exc
            try:
                self.model = MaskablePPO.load(str(path), device="cpu")
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                raise PolicyLoadError(f"could not load MaskablePPO model {path}: {exc}") from exc
            self.model_path = path
        check_model_compatible(self.model)

    def choose(self, battle: Any) -> Decision:
        from poke_env.environment import SinglesEnv

        from showdownrl.real_env import policy_order

        order = policy_order(self.model, battle, self.deterministic)
        action = None
        try:
            action = int(SinglesEnv.order_to_action(order, battle, strict=False))
        except Exception:  # noqa: BLE001 - action index is informational only
            pass
        return Decision(order=order, source="ppo", action=action)


class SearchLivePolicy(LivePolicy):
    """poke-engine MCTS over sampled opponent sets, with the PPO policy as prior/fallback."""

    def __init__(self, model_path: Path | None = None, n_samples: int = 4, time_ms: int = 100,
                 prior_weight: float = 0.0):
        from showdownrl.search import ENGINE_AVAILABLE, SearchStats, engine_is_gen9

        if not ENGINE_AVAILABLE or not engine_is_gen9():
            raise PolicyLoadError(
                "search needs poke-engine built for gen9: pip install -e '.[search]' (see pyproject.toml)")
        super().__init__(model_path)
        self.n_samples = n_samples
        self.time_ms = time_ms
        self.prior_weight = prior_weight
        self.stats = SearchStats()

    def choose(self, battle: Any) -> Decision:
        from poke_env.environment import SinglesEnv

        from showdownrl.search import search_policy

        order = search_policy(battle, self.model, n_samples=self.n_samples, time_ms=self.time_ms,
                              prior_weight=self.prior_weight, stats=self.stats)
        action = None
        try:
            action = int(SinglesEnv.order_to_action(order, battle, strict=False))
        except Exception:  # noqa: BLE001 - action index is informational only
            pass
        return Decision(order=order, source="search", action=action)


def order_is_valid(order: Any, battle: Any) -> bool:
    """True when ``order`` is a move/switch the current request allows."""
    from poke_env.battle import Move, Pokemon
    from poke_env.player.battle_order import SingleBattleOrder

    if not isinstance(order, SingleBattleOrder):
        return False
    target = order.order
    if isinstance(target, Move):
        if order.terastallize and not battle.can_tera:
            return False
        return any(move.id == target.id for move in battle.available_moves)
    if isinstance(target, Pokemon):
        return any(mon is target or mon.name == target.name for mon in battle.available_switches)
    return False


def heuristic_decision(battle: Any, reason: str = "") -> Decision:
    from poke_env.player import Player

    from showdownrl.smart_heuristic import choose_smart_move

    try:
        order = choose_smart_move(battle)
        if order_is_valid(order, battle):
            return Decision(order=order, source="smart", fallback_reason=reason)
        reason = (reason + "; " if reason else "") + f"smart heuristic chose invalid order {order}"
    except Exception as exc:  # noqa: BLE001 - never crash live play
        reason = (reason + "; " if reason else "") + f"smart heuristic failed: {exc}"
    return Decision(order=Player.choose_random_singles_move(battle), source="random", fallback_reason=reason)


def choose_order(battle: Any, policy: Optional[LivePolicy] = None) -> Decision:
    """Pick an order for ``battle`` with the model, falling back to the heuristic."""
    if policy is None:
        return heuristic_decision(battle)
    try:
        decision = policy.choose(battle)
    except Exception as exc:  # noqa: BLE001
        return heuristic_decision(battle, f"PPO predict failed: {exc}")
    if not order_is_valid(decision.order, battle):
        return heuristic_decision(battle, f"PPO chose unavailable order {decision.order}")
    return decision
