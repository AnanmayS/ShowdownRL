"""
Simple Pokemon Move Selection RL Environment.

A simplified Gymnasium environment where an agent selects from four moves.
The opponent can pick randomly or use a stronger baseline policy.

State:
    simple observation: own_hp, opponent_hp, then 4 moves x
        (base_power, accuracy, damage_multiplier), then bench info
    rich observation: simple observation plus 4 moves x
        (expected_damage, STAB, super-effective, resisted/immune,
        finish_flag, recovery_flag, setup_flag, status_flag)

Actions: moves 0–3, switches 4+.

Reward:
    + (opponent HP decrease) for hitting the opponent
    - (own HP decrease) for taking damage
    +1 for winning
    -1 for losing

Episode ends when either team has no living Pokemon or max_turns is reached.
"""

import gymnasium
import numpy as np
from gymnasium import spaces

from showdownrl.policy_bridge import TYPE_CHART

POKEMON_TYPES = tuple(TYPE_CHART.keys())
MOVE_ACTIONS = 4
SWITCH_ACTION = 4
DEFAULT_MAX_BENCH_SIZE = 3
BENCH_FEATURES_PER_POKEMON = 20
BASE_SIMPLE_OBS_SIZE = 14
SIMPLE_OBS_SIZE = BASE_SIMPLE_OBS_SIZE + DEFAULT_MAX_BENCH_SIZE * BENCH_FEATURES_PER_POKEMON
RICH_FEATURES_PER_MOVE = 8
BASE_RICH_OBS_SIZE = BASE_SIMPLE_OBS_SIZE + MOVE_ACTIONS * RICH_FEATURES_PER_MOVE
RICH_OBS_SIZE = BASE_RICH_OBS_SIZE + DEFAULT_MAX_BENCH_SIZE * BENCH_FEATURES_PER_POKEMON
ROLE_ATTACK = "attack"
ROLE_RECOVER = "recover"
ROLE_SETUP = "setup"
ROLE_STATUS = "status"


class SimplePokemonMoveEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_turns: int = 50,
        seed: int = None,
        opponent_policy: str = "random",
        mechanics: str = "toy",
        observation_mode: str = "simple",
        max_bench_size: int = DEFAULT_MAX_BENCH_SIZE,
    ):
        super().__init__()

        self.max_turns = max_turns
        self.opponent_policy = opponent_policy
        self.mechanics = mechanics
        self.observation_mode = observation_mode
        self.max_bench_size = max(0, int(max_bench_size))

        base_obs_size = BASE_RICH_OBS_SIZE if observation_mode == "rich" else BASE_SIMPLE_OBS_SIZE
        obs_size = base_obs_size + self.max_bench_size * BENCH_FEATURES_PER_POKEMON
        self.observation_space = spaces.Box(
            low=0.0, high=4.0, shape=(obs_size,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(MOVE_ACTIONS + self.max_bench_size)

        self.rng = np.random.default_rng(seed)

        # Internal state
        self.own_hp = 1.0
        self.opponent_hp = 1.0
        self.moves = None
        self.opponent_moves = None
        self.own_types = []
        self.opponent_types = []
        self.own_attack_boost = 1.0
        self.opponent_attack_boost = 1.0
        self.bench = []
        self.opponent_bench = []
        self.active_pokemon = None
        self.opponent_active = None
        self.current_opponent_policy = opponent_policy
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

    def _move_values(self, move):
        if len(move) == 3:
            bp, acc, multiplier = move
            return bp, acc, multiplier, 1.0, multiplier, ROLE_ATTACK
        return move

    def _sample_role(self, attack_slots_remaining: int) -> str:
        if self.mechanics != "rich":
            return ROLE_ATTACK
        if attack_slots_remaining > 0:
            return ROLE_ATTACK
        return str(
            self.rng.choice(
                [ROLE_ATTACK, ROLE_RECOVER, ROLE_SETUP, ROLE_STATUS],
                p=[0.58, 0.16, 0.13, 0.13],
            )
        )

    def _generate_toy_moves(self):
        moves = []
        for _ in range(4):
            bp = self.rng.uniform(0.3, 1.0)
            acc = self.rng.uniform(0.7, 1.0)
            multiplier = self.rng.uniform(0.5, 2.0)
            moves.append((bp, acc, multiplier))
        return moves

    def _generate_typed_moves(self, attacker_types: list[str], defender_types: list[str]):
        moves = []
        move_types = list(self.rng.choice(POKEMON_TYPES, size=4, replace=True))
        if self.rng.random() < 0.75:
            move_types[0] = attacker_types[0]
        for index, move_type in enumerate(move_types):
            attack_slots_remaining = max(0, 2 - index)
            role = self._sample_role(attack_slots_remaining)
            bp = self.rng.uniform(0.3, 1.0)
            acc = self.rng.uniform(0.7, 1.0)
            effectiveness = self._type_effectiveness(move_type, defender_types)
            stab = 1.5 if move_type in attacker_types else 1.0
            multiplier = stab * effectiveness
            if role != ROLE_ATTACK:
                bp = 0.0
                multiplier = 0.0
            moves.append((bp, acc, multiplier, stab, effectiveness, role))
        return moves

    def _generate_moves(self):
        """Generate four moves for the agent-facing observation."""
        if self.mechanics in {"typed", "rich"}:
            self.own_types = self._sample_types()
            self.opponent_types = self._sample_types()
            return self._generate_typed_moves(self.own_types, self.opponent_types)

        self.own_types = []
        self.opponent_types = []
        return self._generate_toy_moves()

    def _generate_opponent_moves(self):
        """Generate hidden opponent moves from the opponent's attacking perspective."""
        if self.mechanics in {"typed", "rich"}:
            return self._generate_typed_moves(self.opponent_types, self.own_types)
        return self._generate_toy_moves()

    def _make_pokemon(
        self, types: list[str], defender_types: list[str], hp: float = 1.0
    ) -> dict:
        return {
            "hp": hp,
            "types": types,
            "moves": self._generate_typed_moves(types, defender_types),
            "attack_boost": 1.0,
        }

    def _sync_active_aliases(self) -> None:
        if self.active_pokemon is not None:
            self.own_hp = self.active_pokemon["hp"]
            self.moves = self.active_pokemon["moves"]
            self.own_types = self.active_pokemon["types"]
            self.own_attack_boost = self.active_pokemon["attack_boost"]
        if self.opponent_active is not None:
            self.opponent_hp = self.opponent_active["hp"]
            self.opponent_moves = self.opponent_active["moves"]
            self.opponent_types = self.opponent_active["types"]
            self.opponent_attack_boost = self.opponent_active["attack_boost"]

    def _sync_active_pokemon(self) -> None:
        if self.active_pokemon is not None:
            self.active_pokemon["hp"] = self.own_hp
            self.active_pokemon["moves"] = self.moves
            self.active_pokemon["types"] = self.own_types
            self.active_pokemon["attack_boost"] = self.own_attack_boost
        if self.opponent_active is not None:
            self.opponent_active["hp"] = self.opponent_hp
            self.opponent_active["moves"] = self.opponent_moves
            self.opponent_active["types"] = self.opponent_types
            self.opponent_active["attack_boost"] = self.opponent_attack_boost

    def _first_living_bench(self, bench: list[dict]) -> dict | None:
        for pokemon in bench:
            if pokemon["hp"] > 0:
                return pokemon
        return None

    def _team_has_living_pokemon(self, active: dict | None, bench: list[dict]) -> bool:
        return (
            active is not None and active["hp"] > 0
        ) or self._first_living_bench(bench) is not None

    def _auto_switch_fainted(self) -> None:
        if self.active_pokemon is not None and self.active_pokemon["hp"] <= 0:
            replacement = self._first_living_bench(self.bench)
            if replacement is not None:
                self.active_pokemon = replacement
        if self.opponent_active is not None and self.opponent_active["hp"] <= 0:
            replacement = self._first_living_bench(self.opponent_bench)
            if replacement is not None:
                self.opponent_active = replacement
        self._sync_active_aliases()

    def _get_obs(self):
        """Build the observation vector."""
        base_size = (
            BASE_RICH_OBS_SIZE if self.observation_mode == "rich" else BASE_SIMPLE_OBS_SIZE
        )
        size = base_size + self.max_bench_size * BENCH_FEATURES_PER_POKEMON
        obs = np.zeros(size, dtype=np.float32)
        obs[0] = self.own_hp
        obs[1] = self.opponent_hp
        for i, move in enumerate(self.moves):
            bp, acc, multiplier, stab, effectiveness, role = self._move_values(move)
            base = 2 + i * 3
            obs[base] = bp
            obs[base + 1] = acc
            obs[base + 2] = multiplier
            if self.observation_mode == "rich":
                rich_base = BASE_SIMPLE_OBS_SIZE + i * RICH_FEATURES_PER_MOVE
                expected_damage = bp * acc * multiplier * 0.25
                obs[rich_base] = min(4.0, bp * acc * multiplier)
                obs[rich_base + 1] = 1.0 if stab > 1.0 else 0.0
                obs[rich_base + 2] = 1.0 if effectiveness > 1.0 else 0.0
                obs[rich_base + 3] = 1.0 if effectiveness < 1.0 else 0.0
                obs[rich_base + 4] = 1.0 if expected_damage >= self.opponent_hp else 0.0
                obs[rich_base + 5] = 1.0 if role == ROLE_RECOVER else 0.0
                obs[rich_base + 6] = 1.0 if role == ROLE_SETUP else 0.0
                obs[rich_base + 7] = 1.0 if role == ROLE_STATUS else 0.0
        bench_base = base_size
        for index, pokemon in enumerate(self.bench[: self.max_bench_size]):
            slot_base = bench_base + index * BENCH_FEATURES_PER_POKEMON
            obs[slot_base] = pokemon["hp"]
            for type_index, pokemon_type in enumerate(POKEMON_TYPES):
                obs[slot_base + 1 + type_index] = 1.0 if pokemon_type in pokemon["types"] else 0.0
            obs[slot_base + 19] = (
                1.0
                if any(self._move_values(move)[5] == ROLE_RECOVER for move in pokemon["moves"])
                else 0.0
            )
        return obs

    def _get_info(self):
        return {
            "turn": self.turn,
            "own_hp": self.own_hp,
            "opponent_hp": self.opponent_hp,
            "own_types": self.own_types,
            "opponent_types": self.opponent_types,
            "mechanics": self.mechanics,
            "observation_mode": self.observation_mode,
            "bench_size": len(self.bench),
            "opponent_bench_size": len(self.opponent_bench),
        }

    def action_masks(self) -> np.ndarray:
        """Return a state-dependent valid-action mask for MaskablePPO."""
        masks = np.ones(MOVE_ACTIONS + self.max_bench_size, dtype=bool)
        if not self.moves:
            return masks

        for index, move in enumerate(self.moves):
            bp, acc, multiplier, _, _, role = self._move_values(move)
            expected_damage = bp * acc * multiplier * self.own_attack_boost * 0.25
            if role == ROLE_ATTACK:
                masks[index] = expected_damage > 0.0
            elif role == ROLE_RECOVER:
                masks[index] = self.own_hp <= 0.85
            elif role == ROLE_SETUP:
                masks[index] = self.own_attack_boost < 1.8 and self.own_hp >= 0.35
            elif role == ROLE_STATUS:
                masks[index] = self.opponent_attack_boost > 0.45 and self.opponent_hp >= 0.25

        for index in range(self.max_bench_size):
            masks[SWITCH_ACTION + index] = (
                index < len(self.bench) and self.bench[index]["hp"] > 0
            )

        if not masks.any():
            return np.ones(MOVE_ACTIONS + self.max_bench_size, dtype=bool)
        return masks

    def _opponent_action(self) -> int:
        living_bench_indices = [
            index for index, pokemon in enumerate(self.opponent_bench) if pokemon["hp"] > 0
        ]
        if living_bench_indices and self.rng.random() < 0.10:
            return SWITCH_ACTION + int(self.rng.choice(living_bench_indices))

        policy = self.current_opponent_policy
        opponent_moves = self.opponent_moves or self.moves
        if policy == "max_damage":
            return int(np.argmax([self._move_values(move)[0] for move in opponent_moves]))
        if policy == "type_aware":
            scores = []
            for move in opponent_moves:
                bp, acc, multiplier, _, _, role = self._move_values(move)
                if role == ROLE_RECOVER:
                    scores.append(1.2 if self.opponent_hp <= 0.35 else 0.05)
                elif role in {ROLE_SETUP, ROLE_STATUS}:
                    scores.append(0.15)
                else:
                    scores.append(bp * acc * multiplier)
            return int(np.argmax(scores))
        return int(self.rng.integers(0, MOVE_ACTIONS))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.own_hp = 1.0
        self.opponent_hp = 1.0
        self.own_attack_boost = 1.0
        self.opponent_attack_boost = 1.0
        if self.opponent_policy == "mixed":
            self.current_opponent_policy = str(
                self.rng.choice(
                    ["random", "max_damage", "type_aware"],
                    p=[0.15, 0.25, 0.60],
                )
            )
        else:
            self.current_opponent_policy = self.opponent_policy

        if self.max_bench_size == 0:
            self.active_pokemon = None
            self.opponent_active = None
            self.bench = []
            self.opponent_bench = []
            self.moves = self._generate_moves()
            self.opponent_moves = self._generate_opponent_moves()
        else:
            own_type_sets = [self._sample_types() for _ in range(1 + self.max_bench_size)]
            opponent_type_sets = [self._sample_types() for _ in range(1 + self.max_bench_size)]
            self.active_pokemon = self._make_pokemon(own_type_sets[0], opponent_type_sets[0])
            self.opponent_active = self._make_pokemon(opponent_type_sets[0], own_type_sets[0])
            self.bench = [
                self._make_pokemon(types, opponent_type_sets[0])
                for types in own_type_sets[1:]
            ]
            self.opponent_bench = [
                self._make_pokemon(types, own_type_sets[0])
                for types in opponent_type_sets[1:]
            ]
            self._sync_active_aliases()
        self.turn = 0

        return self._get_obs(), self._get_info()

    def _apply_action(self, action: int, *, is_own: bool) -> None:
        if action >= SWITCH_ACTION:
            bench = self.bench if is_own else self.opponent_bench
            bench_index = action - SWITCH_ACTION
            if bench_index < len(bench) and bench[bench_index]["hp"] > 0:
                if is_own:
                    self.active_pokemon = bench[bench_index]
                else:
                    self.opponent_active = bench[bench_index]
                self._sync_active_aliases()
            return

        moves = self.moves if is_own else (self.opponent_moves or self.moves)
        bp, acc, multiplier, _, _, role = self._move_values(moves[action])
        if self.rng.random() > acc:
            return

        if is_own:
            if role == ROLE_RECOVER:
                self.own_hp = min(1.0, self.own_hp + 0.35)
            elif role == ROLE_SETUP:
                self.own_attack_boost = min(2.2, self.own_attack_boost + 0.6)
            elif role == ROLE_STATUS:
                self.opponent_attack_boost = max(0.4, self.opponent_attack_boost - 0.45)
            else:
                damage_dealt = bp * acc * multiplier * self.own_attack_boost * 0.25
                self.opponent_hp = max(0.0, self.opponent_hp - damage_dealt)
            self._sync_active_pokemon()
            return

        if role == ROLE_RECOVER:
            self.opponent_hp = min(1.0, self.opponent_hp + 0.35)
        elif role == ROLE_SETUP:
            self.opponent_attack_boost = min(2.2, self.opponent_attack_boost + 0.6)
        elif role == ROLE_STATUS:
            self.own_attack_boost = max(0.4, self.own_attack_boost - 0.45)
        else:
            damage_taken = bp * acc * multiplier * self.opponent_attack_boost * 0.25
            self.own_hp = max(0.0, self.own_hp - damage_taken)

        self._sync_active_pokemon()

    def step(self, action):
        # Clamp action to valid range
        action = min(max(int(action), 0), self.action_space.n - 1)

        prev_opp = self.opponent_hp
        prev_own_boost = self.own_attack_boost
        prev_opp_boost = self.opponent_attack_boost
        is_switch = action >= SWITCH_ACTION
        role = None
        expected_damage = 0.0

        self._apply_action(action, is_own=True)
        prev_own = self.own_hp

        if not is_switch:
            bp, acc, multiplier, _, _, role = self._move_values(self.moves[action])
            expected_damage = bp * acc * multiplier * prev_own_boost * 0.25

        # --- Opponent attack ---
        if self.opponent_hp > 0:
            opp_action = self._opponent_action()
            self._apply_action(opp_action, is_own=False)

        post_own_hp = self.own_hp
        post_opp_hp = self.opponent_hp
        self._auto_switch_fainted()

        # --- Reward ---
        opp_delta = max(0.0, prev_opp - post_opp_hp)
        own_delta = max(0.0, prev_own - post_own_hp)
        reward = opp_delta - own_delta - 0.01
        if is_switch:
            reward -= 0.01
        elif role == ROLE_ATTACK and expected_damage >= prev_opp:
            reward += 0.18
        elif role == ROLE_RECOVER:
            if prev_own <= 0.55:
                reward += min(1.0 - prev_own, 0.35) * 0.18
            else:
                reward -= 0.12
        elif role == ROLE_SETUP:
            if prev_own >= 0.45 and prev_opp >= 0.35 and prev_own_boost < 1.8:
                reward += 0.03
            else:
                reward -= 0.10
        elif role == ROLE_STATUS:
            if prev_opp >= 0.35 and prev_opp_boost > 0.45:
                reward += 0.02
            else:
                reward -= 0.08

        # --- Terminal conditions ---
        terminated = False
        opponent_lost = (
            self.opponent_hp <= 0
            if self.max_bench_size == 0
            else not self._team_has_living_pokemon(
                self.opponent_active, self.opponent_bench
            )
        )
        own_lost = (
            self.own_hp <= 0
            if self.max_bench_size == 0
            else not self._team_has_living_pokemon(self.active_pokemon, self.bench)
        )
        if opponent_lost:
            reward += 1.0
            terminated = True
        elif own_lost:
            reward -= 1.0
            terminated = True

        self.turn += 1
        truncated = self.turn >= self.max_turns
        if truncated and not terminated:
            reward -= 0.75

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        print(f"Turn {self.turn}: own={self.own_hp:.2f}, opp={self.opponent_hp:.2f}")

    def close(self):
        pass
