"""Gymnasium environment that trains against the real Pokemon Showdown simulator.

Wraps poke-env's ``SinglesEnv`` (local server, Gen 9 Random Battles) with the
shared feature extractor from ``battle_features`` and a configurable pool of
opponents (scripted baselines and frozen policy snapshots).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from poke_env.battle import AbstractBattle, Battle
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player import (
    MaxBasePowerPlayer,
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from showdownrl.battle_features import N_ACTIONS, OBS_SIZE, embed_battle

BATTLE_FORMAT = "gen9randombattle"


def server_configuration(port: int = 8000) -> ServerConfiguration:
    return ServerConfiguration(
        f"ws://localhost:{port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )


class ShowdownFeatureEnv(SinglesEnv):
    """SinglesEnv with ShowdownRL features and potential-based reward shaping."""

    def __init__(self, *, hp_value: float = 0.0, fainted_value: float = 0.0,
                 victory_value: float = 1.0, **kwargs: Any):
        super().__init__(**kwargs)
        self._hp_value = hp_value
        self._fainted_value = fainted_value
        self._victory_value = victory_value
        low = np.full(OBS_SIZE, -4.0, dtype=np.float32)
        high = np.full(OBS_SIZE, 4.0, dtype=np.float32)
        self.observation_spaces = {
            agent: spaces.Box(low, high, dtype=np.float32) for agent in self.possible_agents
        }

    def calc_reward(self, battle: AbstractBattle) -> float:
        return self.reward_computing_helper(
            battle,
            fainted_value=self._fainted_value,
            hp_value=self._hp_value,
            victory_value=self._victory_value,
        )

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        assert isinstance(battle, Battle)
        if battle.player_username != self.agent1.username:
            # The opponent Player embeds its own view when it needs one.
            return np.zeros(OBS_SIZE, dtype=np.float32)
        return embed_battle(battle)


# ---------------------------------------------------------------------------
# Opponents


class PolicyPlayer(Player):
    """Plays with a frozen MaskablePPO policy (CPU inference)."""

    def __init__(self, model: Any, deterministic: bool = False, **kwargs: Any):
        super().__init__(**kwargs)
        self.model = model
        self.deterministic = deterministic

    def choose_move(self, battle: AbstractBattle):
        assert isinstance(battle, Battle)
        return policy_order(self.model, battle, self.deterministic)


def policy_order(model: Any, battle: Battle, deterministic: bool = True):
    obs = embed_battle(battle)
    mask = np.array(SinglesEnv.get_action_mask(battle), dtype=bool)
    if mask.sum() == 0:
        return Player.choose_random_singles_move(battle)
    action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
    return SinglesEnv.action_to_order(np.int64(action), battle, strict=False)


class MixedOpponent(Player):
    """Picks one strategy per battle from a weighted pool.

    Pool entries are ``(weight, chooser)`` where chooser is a callable
    ``battle -> BattleOrder``. Snapshots can be added/replaced at runtime.
    """

    def __init__(self, pool: Sequence[tuple[float, Any]], **kwargs: Any):
        super().__init__(**kwargs)
        self.pool = list(pool)
        self._assignment: dict[str, Any] = {}
        self.last_choice_name: Optional[str] = None

    def set_pool(self, pool: Sequence[tuple[float, Any]]) -> None:
        self.pool = list(pool)

    def _pick(self) -> Any:
        weights = [w for w, _ in self.pool]
        return random.choices([c for _, c in self.pool], weights=weights, k=1)[0]

    def choose_move(self, battle: AbstractBattle):
        chooser = self._assignment.get(battle.battle_tag)
        if chooser is None:
            if len(self._assignment) > 64:
                self._assignment.clear()
            chooser = self._pick()
            self._assignment[battle.battle_tag] = chooser
            self.last_choice_name = getattr(chooser, "name", None)
        return chooser(battle)


class NamedChooser:
    def __init__(self, name: str, fn: Any):
        self.name = name
        self.fn = fn

    def __call__(self, battle: AbstractBattle):
        return self.fn(battle)


def scripted_chooser(name: str) -> NamedChooser:
    if name == "random":
        return NamedChooser(name, Player.choose_random_singles_move)
    if name == "max_power":
        return NamedChooser(name, MaxBasePowerPlayer.choose_singles_move)
    if name == "heuristic":
        return NamedChooser(name, lambda b: SimpleHeuristicsPlayer.choose_singles_move(b)[0])
    if name == "smart":
        from showdownrl.smart_heuristic import choose_smart_move

        return NamedChooser(name, choose_smart_move)
    raise ValueError(f"unknown scripted opponent {name!r}")


def snapshot_chooser(path: str | Path, deterministic: bool = False) -> NamedChooser:
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(str(path), device="cpu")
    return NamedChooser(f"snapshot:{Path(path).stem}",
                        lambda b: policy_order(model, b, deterministic))


def parse_pool(spec: str) -> list[tuple[float, NamedChooser]]:
    """Parse ``"heuristic:0.6,max_power:0.2,random:0.1,snap=path.zip:0.1"``."""
    pool = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, weight = part.rpartition(":")
        if not name:
            name, weight = weight, "1"
        if name.startswith("snap="):
            pool.append((float(weight), snapshot_chooser(name[5:])))
        else:
            pool.append((float(weight), scripted_chooser(name)))
    return pool


# ---------------------------------------------------------------------------
# Gym wrapper for SB3


class MaskedShowdownEnv(gym.Env):
    """Flat-observation env exposing ``action_masks()`` for MaskablePPO."""

    metadata = {"render_modes": []}

    def __init__(self, opponent_pool: str = "heuristic:1", port: int = 8000,
                 hp_value: float = 0.0, fainted_value: float = 0.0, seed: Optional[int] = None):
        super().__init__()
        tag = f"{random.randrange(16**6):06x}"
        self.poke_env = ShowdownFeatureEnv(
            battle_format=BATTLE_FORMAT,
            server_configuration=server_configuration(port),
            account_configuration1=AccountConfiguration(f"srla{tag}", None),
            account_configuration2=AccountConfiguration(f"srlb{tag}", None),
            hp_value=hp_value,
            fainted_value=fainted_value,
            strict=False,
            log_level=40,
        )
        self.opponent = MixedOpponent(
            parse_pool(opponent_pool),
            battle_format=BATTLE_FORMAT,
            start_listening=False,
            log_level=40,
        )
        self.env = SingleAgentWrapper(self.poke_env, self.opponent)
        self.observation_space = spaces.Box(-4.0, 4.0, (OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._mask = np.ones(N_ACTIONS, dtype=bool)
        self._episode_opponent: Optional[str] = None

    def set_opponent_pool(self, spec: str) -> None:
        self.opponent.set_pool(parse_pool(spec))

    def _unpack(self, obs: dict) -> np.ndarray:
        self._mask = np.asarray(obs["action_mask"], dtype=bool)
        if not self._mask.any():
            self._mask[0] = True
        return np.asarray(obs["observation"], dtype=np.float32)

    def action_masks(self) -> np.ndarray:
        return self._mask

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._episode_opponent = None
        return self._unpack(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(np.int64(action))
        if self._episode_opponent is None:
            self._episode_opponent = self.opponent.last_choice_name
        info = dict(info)
        if terminated or truncated:
            battle = self.poke_env.battle1
            info["won"] = bool(battle is not None and battle.won)
            info["opponent"] = self._episode_opponent
        return self._unpack(obs), float(reward), bool(terminated), bool(truncated), info

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
