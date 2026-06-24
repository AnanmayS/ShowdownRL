"""
Simple Pokemon Move Selection RL Environment.

A simplified Gymnasium environment where an agent selects from four moves.
The opponent can pick randomly or use a stronger baseline policy.

State (14 floats):
    own_hp, opponent_hp, then 4 moves x (base_power, accuracy, damage_multiplier)

Actions: Discrete(4) — choose move index 0–3.

Reward:
    + (opponent HP decrease) for hitting the opponent
    - (own HP decrease) for taking damage
    +1 for winning
    -1 for losing

Episode ends when either Pokemon faints or max_turns is reached.
"""

import gymnasium
import numpy as np
from gymnasium import spaces

from showdownrl.policy_bridge import TYPE_CHART

POKEMON_TYPES = tuple(TYPE_CHART.keys())


class SimplePokemonMoveEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_turns: int = 50,
        seed: int = None,
        opponent_policy: str = "random",
        mechanics: str = "toy",
    ):
        super().__init__()

        self.max_turns = max_turns
        self.opponent_policy = opponent_policy
        self.mechanics = mechanics

        # Observation: own_hp, opponent_hp, 4 moves x (bp, acc, damage multiplier)
        self.observation_space = spaces.Box(
            low=0.0, high=4.0, shape=(14,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(4)

        self.rng = np.random.default_rng(seed)

        # Internal state
        self.own_hp = 1.0
        self.opponent_hp = 1.0
        self.moves = None
        self.own_types = []
        self.opponent_types = []
        self.turn = 0

    def _sample_types(self) -> list[str]:
        count = 1 if self.rng.random() < 0.7 else 2
        return list(self.rng.choice(POKEMON_TYPES, size=count, replace=False))

    def _type_effectiveness(self, move_type: str, defender_types: list[str]) -> float:
        multiplier = 1.0
        chart = TYPE_CHART.get(move_type, {})
        for defender_type in defender_types:
            multiplier *= chart.get(defender_type, 1.0)
        return multiplier

    def _damage_multiplier(self, move_type: str, attacker_types: list[str], defender_types: list[str]) -> float:
        stab = 1.5 if move_type in attacker_types else 1.0
        return stab * self._type_effectiveness(move_type, defender_types)

    def _generate_moves(self):
        """Generate four moves with base_power, accuracy, and a damage multiplier."""
        moves = []
        if self.mechanics == "typed":
            self.own_types = self._sample_types()
            self.opponent_types = self._sample_types()
            move_types = list(self.rng.choice(POKEMON_TYPES, size=4, replace=True))
            if self.rng.random() < 0.75:
                move_types[0] = self.own_types[0]
            for move_type in move_types:
                bp = self.rng.uniform(0.3, 1.0)
                acc = self.rng.uniform(0.7, 1.0)
                multiplier = self._damage_multiplier(move_type, self.own_types, self.opponent_types)
                moves.append((bp, acc, multiplier))
            return moves

        self.own_types = []
        self.opponent_types = []
        for _ in range(4):
            bp = self.rng.uniform(0.3, 1.0)   # base_power in [0.3, 1.0]
            acc = self.rng.uniform(0.7, 1.0)   # accuracy in [0.7, 1.0]
            multiplier = self.rng.uniform(0.5, 2.0)
            moves.append((bp, acc, multiplier))
        return moves

    def _get_obs(self):
        """Build the 14-element observation vector."""
        obs = np.zeros(14, dtype=np.float32)
        obs[0] = self.own_hp
        obs[1] = self.opponent_hp
        for i, (bp, acc, te) in enumerate(self.moves):
            base = 2 + i * 3
            obs[base] = bp
            obs[base + 1] = acc
            obs[base + 2] = te
        return obs

    def _get_info(self):
        return {
            "turn": self.turn,
            "own_hp": self.own_hp,
            "opponent_hp": self.opponent_hp,
            "own_types": self.own_types,
            "opponent_types": self.opponent_types,
            "mechanics": self.mechanics,
        }

    def _opponent_action(self) -> int:
        if self.opponent_policy == "max_damage":
            return int(np.argmax([move[0] for move in self.moves]))
        if self.opponent_policy == "type_aware":
            return int(np.argmax([bp * acc * te for bp, acc, te in self.moves]))
        return int(self.rng.integers(0, 4))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.own_hp = 1.0
        self.opponent_hp = 1.0
        self.moves = self._generate_moves()
        self.turn = 0

        return self._get_obs(), self._get_info()

    def step(self, action):
        # Clamp action to valid range
        action = min(max(int(action), 0), 3)

        prev_own = self.own_hp
        prev_opp = self.opponent_hp

        # --- Our attack ---
        bp, acc, te = self.moves[action]
        # Damage = base_power * accuracy * type_effectiveness, scaled down
        damage_dealt = bp * acc * te * 0.25
        # Accuracy check
        if self.rng.random() > acc:
            damage_dealt = 0.0
        self.opponent_hp = max(0.0, self.opponent_hp - damage_dealt)

        # --- Opponent attack ---
        opp_action = self._opponent_action()
        opp_bp, opp_acc, opp_te = self.moves[opp_action]
        damage_taken = opp_bp * opp_acc * opp_te * 0.25
        if self.rng.random() > opp_acc:
            damage_taken = 0.0
        self.own_hp = max(0.0, self.own_hp - damage_taken)

        # --- Reward ---
        opp_delta = prev_opp - self.opponent_hp
        own_delta = prev_own - self.own_hp
        reward = opp_delta - own_delta

        # --- Terminal conditions ---
        terminated = False
        if self.opponent_hp <= 0:
            reward += 1.0
            terminated = True
        elif self.own_hp <= 0:
            reward -= 1.0
            terminated = True

        self.turn += 1
        truncated = self.turn >= self.max_turns

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        print(f"Turn {self.turn}: own={self.own_hp:.2f}, opp={self.opponent_hp:.2f}")

    def close(self):
        pass
