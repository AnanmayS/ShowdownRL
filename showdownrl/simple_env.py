"""
Simple Pokemon Move Selection RL Environment.

A simplified Gymnasium environment where an agent selects from four moves.
The opponent can pick randomly or use a stronger baseline policy.

Mechanics levels:
  - toy:   No types, no roles, just raw (bp, acc, multiplier) tuples
  - typed: Types exist, roles exist, but env is otherwise simplified
  - rich:  Full observation with rich per-move features
  - advanced:  Everything from rich + status conditions + held items

State:
    simple observation: own_hp, opponent_hp, then 4 moves x
        (base_power, accuracy, damage_multiplier), then bench info
    rich observation: simple observation plus 4 moves x
        (expected_damage, STAB, super-effective, resisted/immune,
        finish_flag, recovery_flag, setup_flag, status_flag)
    advanced observation: rich observation plus own/opponent status+item
        (the opponent's item/ability stay hidden until revealed in battle)
    rich_v2 observation: rich observation plus own/opponent types, attack
        boosts and speeds (new size; older checkpoints use simple/rich)

Actions: moves 0-3, switches 4+.

Turn order: both sides choose simultaneously; switches resolve first, then
the faster active Pokemon moves (random tie-break). Move effectiveness is
recomputed whenever either active Pokemon changes.

Reward:
    + (opponent team HP decrease) for hitting the opponent
    - (own team HP decrease) for taking damage
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
RICH_V2_OBS_EXTRA = 2 * len(POKEMON_TYPES) + 5
OBSERVATION_MODES = ("simple", "rich", "rich_v2")
RICH_OBSERVATION_MODES = ("rich", "rich_v2")
# Bump whenever battle dynamics or observation layouts change so results and
# checkpoints can be traced to the environment they came from.
ENV_VERSION = 2
ROLE_ATTACK = "attack"
ROLE_RECOVER = "recover"
ROLE_SETUP = "setup"
ROLE_STATUS = "status"

# Status condition constants
STATUS_NONE = 0
STATUS_BURN = 1      # halved attack, 1/16 HP per turn
STATUS_POISON = 2    # 1/8 HP per turn
STATUS_PARALYSIS = 3 # 25% chance of not moving
STATUS_SLEEP = 4     # can't move for 1-3 turns (wears off)
STATUS_FREEZE = 5    # can't move, 20% thaw chance
STATUS_LABELS = ["none", "brn", "psn", "par", "slp", "frz"]

# Held item constants
ITEM_NONE = 0
ITEM_LEFTOVERS = 1       # 1/16 HP per turn
ITEM_CHOICE_BAND = 2     # 1.5x attack, locks into first move used
ITEM_LIFE_ORB = 3        # 1.3x damage, 1/10 HP recoil
ITEM_FOCUS_SASH = 4      # survive at 1 HP from full
ITEM_ASSAULT_VEST = 5    # 1.5x SpD, no status moves
ITEM_LABELS = ["none", "leftovers", "choice_band", "life_orb", "focus_sash", "assault_vest"]

# Extra observation features for advanced mode: status(4) + hazards(6) + abilities(2) = 12
ADVANCED_OBS_EXTRA = 12

# Hazard constants
HAZARD_TYPE_NONE = 0
HAZARD_STEALTH_ROCK = 1
HAZARD_SPIKES = 2
HAZARD_TOXIC_SPIKES = 3

# Ability constants
ABILITY_NONE = 0
ABILITY_INTIMIDATE = 1   # Lowers opponent Attack on switch-in
ABILITY_REGENERATOR = 2  # Heals 1/3 HP on switch-out
ABILITY_NATURAL_CURE = 3 # Cures status on switch-out


def _base_obs_size(mechanics: str, observation_mode: str) -> int:
    """Return the observation size before the bench block."""
    size = BASE_RICH_OBS_SIZE if observation_mode in RICH_OBSERVATION_MODES else BASE_SIMPLE_OBS_SIZE
    if mechanics == "advanced":
        size += ADVANCED_OBS_EXTRA
    if observation_mode == "rich_v2":
        size += RICH_V2_OBS_EXTRA
    return size


def _obs_size_for_mechanics(mechanics: str, observation_mode: str, max_bench_size: int) -> int:
    """Return the total observation size for a given mechanics/observation_mode combo."""
    return _base_obs_size(mechanics, observation_mode) + max_bench_size * BENCH_FEATURES_PER_POKEMON


def _parse_advanced_move(move: tuple) -> tuple:
    """Parse a move tuple, handling both advanced (7-tuple) and standard (6-tuple) formats."""
    if len(move) == 3:
        bp, acc, multiplier = move
        return bp, acc, multiplier, 1.0, multiplier, ROLE_ATTACK, STATUS_NONE
    if len(move) >= 7:
        bp, acc, multiplier, stab, effectiveness, role, status_type = move[:7]
        return bp, acc, multiplier, stab, effectiveness, role, status_type
    bp, acc, multiplier, stab, effectiveness, role = move[:6]
    return bp, acc, multiplier, stab, effectiveness, role, STATUS_NONE


def _move_type(move: tuple) -> str | None:
    """Return the move's type when the tuple carries it (8-tuple format)."""
    return move[7] if len(move) >= 8 else None


def _last_fields(move: tuple) -> tuple:
    """Extract (role, status_type) from any move format."""
    if len(move) == 3:
        return ROLE_ATTACK, STATUS_NONE
    if len(move) >= 7:
        return move[5], move[6]
    # 6-element tuple (rich/typed mode)
    return move[5], STATUS_NONE


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
        if observation_mode not in OBSERVATION_MODES:
            raise ValueError(f"observation_mode must be one of {OBSERVATION_MODES}")

        self.max_turns = max_turns
        self.opponent_policy = opponent_policy
        self.mechanics = mechanics
        self.observation_mode = observation_mode
        self.max_bench_size = max(0, int(max_bench_size))

        obs_size = _obs_size_for_mechanics(mechanics, observation_mode, self.max_bench_size)
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
        self.own_speed = 0.5
        self.opponent_speed = 0.5
        self.bench = []
        self.opponent_bench = []
        self.active_pokemon = None
        self.opponent_active = None
        self.current_opponent_policy = opponent_policy
        self.turn = 0

        # Advanced mode state
        self.own_status = STATUS_NONE
        self.opponent_status = STATUS_NONE
        self.own_item = ITEM_NONE
        self.opponent_item = ITEM_NONE
        self.own_choice_locked = False
        self.own_choice_action = -1
        self.opponent_choice_locked = False
        self.opponent_choice_action = -1
        # Track focus sash usage (once per battle per pokemon)
        self.own_sash_used = False
        self.opponent_sash_used = False

        # Hazard state
        self.own_stealth_rock = 0
        self.opp_stealth_rock = 0
        self.own_spikes = 0
        self.opp_spikes = 0
        self.own_toxic_spikes = 0
        self.opp_toxic_spikes = 0

        # Ability state (synced from pokemon dicts)
        self.own_ability = ABILITY_NONE
        self.opponent_ability = ABILITY_NONE

    # ------------------------------------------------------------------ #
    #  Type / damage helpers
    # ------------------------------------------------------------------ #

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

    def _sample_status_type(self) -> int:
        """Sample a status condition or hazard for a status move."""
        if self.mechanics == "advanced":
            return int(self.rng.choice(
                [STATUS_BURN, STATUS_POISON, STATUS_PARALYSIS, STATUS_SLEEP,
                 HAZARD_STEALTH_ROCK, HAZARD_SPIKES, HAZARD_TOXIC_SPIKES],
                p=[0.15, 0.15, 0.15, 0.15, 0.2, 0.1, 0.1],
            ))
        return int(self.rng.choice([STATUS_BURN, STATUS_POISON, STATUS_PARALYSIS], p=[0.3, 0.3, 0.4]))

    def _sample_item(self) -> int:
        """Sample a held item."""
        if self.mechanics != "advanced":
            return ITEM_NONE
        return int(self.rng.choice(
            [ITEM_LEFTOVERS, ITEM_CHOICE_BAND, ITEM_LIFE_ORB, ITEM_FOCUS_SASH, ITEM_ASSAULT_VEST],
            p=[0.3, 0.2, 0.2, 0.15, 0.15],
        ))

    def _sample_ability(self) -> int:
        """Sample an ability for a Pokemon (advanced mode only)."""
        if self.mechanics != "advanced":
            return ABILITY_NONE
        return int(self.rng.choice(
            [ABILITY_NONE, ABILITY_INTIMIDATE, ABILITY_REGENERATOR, ABILITY_NATURAL_CURE],
            p=[0.4, 0.25, 0.2, 0.15],
        ))

    # ------------------------------------------------------------------ #
    #  Move generation
    # ------------------------------------------------------------------ #

    def _move_values(self, move):
        """Return (bp, acc, multiplier, stab, effectiveness, role, status_type)."""
        return _parse_advanced_move(move)

    def _sample_role(self, attack_slots_remaining: int) -> str:
        if self.mechanics not in ("rich", "advanced"):
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
            status_type = STATUS_NONE
            if role == ROLE_STATUS and self.mechanics == "advanced":
                status_type = self._sample_status_type()
            if role != ROLE_ATTACK:
                bp = 0.0
                multiplier = 0.0
            moves.append((bp, acc, multiplier, stab, effectiveness, role, status_type, move_type))
        return moves

    def _retarget_move(self, move: tuple, defender_types: list[str]) -> tuple:
        """Recompute a typed move's effectiveness against a new defender."""
        move_type = _move_type(move)
        if move_type is None:
            return move
        bp, acc, _, stab, _, role, status_type = _parse_advanced_move(move)
        effectiveness = self._type_effectiveness(move_type, defender_types)
        multiplier = stab * effectiveness if role == ROLE_ATTACK else 0.0
        return (bp, acc, multiplier, stab, effectiveness, role, status_type, move_type)

    def _generate_moves(self):
        """Generate four moves for the agent-facing observation."""
        if self.mechanics in {"typed", "rich", "advanced"}:
            self.own_types = self._sample_types()
            self.opponent_types = self._sample_types()
            return self._generate_typed_moves(self.own_types, self.opponent_types)

        self.own_types = []
        self.opponent_types = []
        return self._generate_toy_moves()

    def _generate_opponent_moves(self):
        """Generate hidden opponent moves from the opponent's attacking perspective."""
        if self.mechanics in {"typed", "rich", "advanced"}:
            return self._generate_typed_moves(self.opponent_types, self.own_types)
        return self._generate_toy_moves()

    # ------------------------------------------------------------------ #
    #  Pokemon / team management
    # ------------------------------------------------------------------ #

    def _make_pokemon(
        self, types: list[str], defender_types: list[str], hp: float = 1.0
    ) -> dict:
        return {
            "hp": hp,
            "types": types,
            "moves": self._generate_typed_moves(types, defender_types),
            "attack_boost": 1.0,
            "speed": float(self.rng.uniform(0.2, 1.0)),
            "status": STATUS_NONE,
            "item": self._sample_item() if self.mechanics == "advanced" else ITEM_NONE,
            "sleep_turns": 0,
            "choice_locked": False,
            "choice_action": -1,
            "sash_used": False,
            "ability": self._sample_ability() if self.mechanics == "advanced" else ABILITY_NONE,
            "item_revealed": False,
            "ability_revealed": False,
        }

    def _sync_active_aliases(self) -> None:
        if self.active_pokemon is not None:
            self.own_hp = self.active_pokemon["hp"]
            self.moves = self.active_pokemon["moves"]
            self.own_types = self.active_pokemon["types"]
            self.own_attack_boost = self.active_pokemon["attack_boost"]
            self.own_speed = self.active_pokemon.get("speed", 0.5)
            self.own_status = self.active_pokemon.get("status", STATUS_NONE)
            self.own_item = self.active_pokemon.get("item", ITEM_NONE)
            self.own_choice_locked = self.active_pokemon.get("choice_locked", False)
            self.own_choice_action = self.active_pokemon.get("choice_action", -1)
            self.own_sash_used = self.active_pokemon.get("sash_used", False)
            self.own_ability = self.active_pokemon.get("ability", ABILITY_NONE)
        if self.opponent_active is not None:
            self.opponent_hp = self.opponent_active["hp"]
            self.opponent_moves = self.opponent_active["moves"]
            self.opponent_types = self.opponent_active["types"]
            self.opponent_attack_boost = self.opponent_active["attack_boost"]
            self.opponent_speed = self.opponent_active.get("speed", 0.5)
            self.opponent_status = self.opponent_active.get("status", STATUS_NONE)
            self.opponent_item = self.opponent_active.get("item", ITEM_NONE)
            self.opponent_choice_locked = self.opponent_active.get("choice_locked", False)
            self.opponent_choice_action = self.opponent_active.get("choice_action", -1)
            self.opponent_sash_used = self.opponent_active.get("sash_used", False)
            self.opponent_ability = self.opponent_active.get("ability", ABILITY_NONE)

    def _sync_active_pokemon(self) -> None:
        if self.active_pokemon is not None:
            self.active_pokemon["hp"] = self.own_hp
            self.active_pokemon["moves"] = self.moves
            self.active_pokemon["types"] = self.own_types
            self.active_pokemon["attack_boost"] = self.own_attack_boost
            self.active_pokemon["speed"] = self.own_speed
            self.active_pokemon["status"] = self.own_status
            self.active_pokemon["item"] = self.own_item
            self.active_pokemon["choice_locked"] = self.own_choice_locked
            self.active_pokemon["choice_action"] = self.own_choice_action
            self.active_pokemon["sash_used"] = self.own_sash_used
            self.active_pokemon["ability"] = self.own_ability
        if self.opponent_active is not None:
            self.opponent_active["hp"] = self.opponent_hp
            self.opponent_active["moves"] = self.opponent_moves
            self.opponent_active["types"] = self.opponent_types
            self.opponent_active["attack_boost"] = self.opponent_attack_boost
            self.opponent_active["speed"] = self.opponent_speed
            self.opponent_active["status"] = self.opponent_status
            self.opponent_active["item"] = self.opponent_item
            self.opponent_active["choice_locked"] = self.opponent_choice_locked
            self.opponent_active["choice_action"] = self.opponent_choice_action
            self.opponent_active["sash_used"] = self.opponent_sash_used
            self.opponent_active["ability"] = self.opponent_ability

    def _first_living_bench(self, bench: list[dict]) -> dict | None:
        for pokemon in bench:
            if pokemon["hp"] > 0:
                return pokemon
        return None

    def _team_has_living_pokemon(self, active: dict | None, bench: list[dict]) -> bool:
        return (
            active is not None and active["hp"] > 0
        ) or self._first_living_bench(bench) is not None

    def _team_hp(self, *, is_own: bool) -> float:
        """Total HP across one side's team (active + bench)."""
        hp = self.own_hp if is_own else self.opponent_hp
        bench = self.bench if is_own else self.opponent_bench
        return hp + sum(pokemon["hp"] for pokemon in bench)

    def _side_alive(self, *, is_own: bool) -> bool:
        if self.max_bench_size == 0:
            return (self.own_hp if is_own else self.opponent_hp) > 0
        if is_own:
            return self._team_has_living_pokemon(self.active_pokemon, self.bench)
        return self._team_has_living_pokemon(self.opponent_active, self.opponent_bench)

    def battle_result(self) -> str | None:
        """Return "win"/"loss" from the agent's side once a team is out, else None.

        If both teams run out on the same turn the agent is credited with the
        win, matching the terminal reward.
        """
        if not self._side_alive(is_own=False):
            return "win"
        if not self._side_alive(is_own=True):
            return "loss"
        return None

    def _refresh_matchups(self) -> None:
        """Recompute every move's effectiveness against the current opposing active Pokemon."""
        if self.active_pokemon is None or self.opponent_active is None:
            return
        for pokemon in [self.active_pokemon, *self.bench]:
            pokemon["moves"] = [
                self._retarget_move(move, self.opponent_active["types"]) for move in pokemon["moves"]
            ]
        for pokemon in [self.opponent_active, *self.opponent_bench]:
            pokemon["moves"] = [
                self._retarget_move(move, self.active_pokemon["types"]) for move in pokemon["moves"]
            ]
        self._sync_active_aliases()

    def _on_switch_out(self, outgoing: dict) -> None:
        # Boosts and Choice lock never survive leaving the field.
        outgoing["attack_boost"] = 1.0
        outgoing["choice_locked"] = False
        outgoing["choice_action"] = -1
        if self.mechanics != "advanced" or outgoing["hp"] <= 0:
            return
        # On switch-out: Regenerator heals, Natural Cure cures
        if outgoing.get("ability") == ABILITY_REGENERATOR:
            outgoing["hp"] = min(1.0, outgoing["hp"] + (1.0 / 3.0))
            self._reveal(outgoing, "ability")
        if outgoing.get("ability") == ABILITY_NATURAL_CURE and outgoing.get("status", STATUS_NONE):
            outgoing["status"] = STATUS_NONE
            self._reveal(outgoing, "ability")

    def _on_switch_in(self, *, is_own: bool) -> None:
        """Apply entry hazards and Intimidate to the Pokemon that just came in."""
        if self.mechanics != "advanced":
            return
        incoming = self.active_pokemon if is_own else self.opponent_active
        incoming_types = self.own_types if is_own else self.opponent_types
        damage = 0.0

        # Stealth Rock: 1/8 max HP scaled by Rock effectiveness
        if self.opp_stealth_rock if is_own else self.own_stealth_rock:
            damage += 0.125 * self._type_effectiveness("rock", incoming_types)

        # Spikes: fixed damage per layer
        spikes = self.opp_spikes if is_own else self.own_spikes
        if spikes:
            damage += {1: 0.125, 2: 0.167, 3: 0.25}[spikes]

        if is_own:
            self.own_hp = max(0.0, self.own_hp - damage)
        else:
            self.opponent_hp = max(0.0, self.opponent_hp - damage)

        # Toxic Spikes: poison on switch
        if self.opp_toxic_spikes if is_own else self.own_toxic_spikes:
            self._inflict_status(STATUS_POISON, target_is_own=is_own)

        # Intimidate: lowers opponent's attack on switch-in
        incoming_ability = self.own_ability if is_own else self.opponent_ability
        if incoming_ability == ABILITY_INTIMIDATE:
            self._reveal(incoming, "ability")
            if is_own:
                self.opponent_attack_boost = max(0.4, self.opponent_attack_boost - 0.3)
            else:
                self.own_attack_boost = max(0.4, self.own_attack_boost - 0.3)

    def _switch_to(self, bench: list[dict], bench_index: int, *, is_own: bool) -> None:
        """Swap one side's active Pokemon with bench[bench_index] and resolve switch effects."""
        self._sync_active_pokemon()
        replacement = bench[bench_index]
        outgoing = self.active_pokemon if is_own else self.opponent_active
        if outgoing is not None:
            self._on_switch_out(outgoing)
            bench[bench_index] = outgoing
        if is_own:
            self.active_pokemon = replacement
        else:
            self.opponent_active = replacement
        self._sync_active_aliases()
        self._refresh_matchups()
        self._on_switch_in(is_own=is_own)
        self._sync_active_pokemon()

    def _auto_switch_fainted(self) -> None:
        self._sync_active_pokemon()
        for is_own in (True, False):
            bench = self.bench if is_own else self.opponent_bench
            # Loop: a replacement can itself faint to entry hazards.
            while True:
                active = self.active_pokemon if is_own else self.opponent_active
                if active is None or active["hp"] > 0:
                    break
                replacement_index = next(
                    (index for index, pokemon in enumerate(bench) if pokemon["hp"] > 0), None
                )
                if replacement_index is None:
                    break
                self._switch_to(bench, replacement_index, is_own=is_own)
        self._sync_active_aliases()

    def _effective_speed(self, *, is_own: bool) -> float:
        speed = self.own_speed if is_own else self.opponent_speed
        status = self.own_status if is_own else self.opponent_status
        if self.mechanics == "advanced" and status == STATUS_PARALYSIS:
            speed *= 0.5
        return speed

    def _turn_order(self, own_action: int, opponent_action: int) -> tuple[bool, bool]:
        """Return the sides (is_own flags) in resolution order.

        Switches resolve before moves; otherwise the faster Pokemon acts first,
        with a coin flip on speed ties.
        """
        own_switch = own_action >= SWITCH_ACTION
        opponent_switch = opponent_action >= SWITCH_ACTION
        if own_switch != opponent_switch:
            own_first = own_switch
        else:
            own_speed = self._effective_speed(is_own=True)
            opponent_speed = self._effective_speed(is_own=False)
            if own_speed == opponent_speed:
                own_first = bool(self.rng.random() < 0.5)
            else:
                own_first = own_speed > opponent_speed
        return (True, False) if own_first else (False, True)

    @staticmethod
    def _reveal(pokemon: dict | None, key: str) -> None:
        """Mark a Pokemon's item/ability as seen by the other side."""
        if pokemon is not None:
            pokemon[f"{key}_revealed"] = True

    @staticmethod
    def _public_value(pokemon: dict | None, key: str) -> int:
        """Return an item/ability only once it has been revealed in battle."""
        if pokemon is None or not pokemon.get(f"{key}_revealed", False):
            return 0
        return int(pokemon.get(key, 0))

    # ------------------------------------------------------------------ #
    #  Observation
    # ------------------------------------------------------------------ #

    def _build_observation(
        self,
        *,
        own_hp: float,
        opponent_hp: float,
        moves: list,
        bench: list[dict],
        own_status: int = STATUS_NONE,
        opponent_status: int = STATUS_NONE,
        own_item: int = ITEM_NONE,
        opponent_item: int = ITEM_NONE,
        own_ability: int = ABILITY_NONE,
        opponent_ability: int = ABILITY_NONE,
        hazards: tuple = (0, 0, 0, 0, 0, 0),
        own_types: list[str] = (),
        opponent_types: list[str] = (),
        own_attack_boost: float = 1.0,
        opponent_attack_boost: float = 1.0,
        own_speed: float = 0.0,
        opponent_speed: float = 0.0,
    ):
        """Build an observation vector from one side's perspective.

        Callers must pass only information that side can see: the opposing
        item/ability should be 0 until revealed.
        """
        is_advanced = self.mechanics == "advanced"
        is_rich = self.observation_mode in RICH_OBSERVATION_MODES
        base_size = _base_obs_size(self.mechanics, self.observation_mode)
        size = base_size + self.max_bench_size * BENCH_FEATURES_PER_POKEMON
        obs = np.zeros(size, dtype=np.float32)
        obs[0] = own_hp
        obs[1] = opponent_hp
        for i, move in enumerate(moves or []):
            bp, acc, multiplier, stab, effectiveness, role, status_type = self._move_values(move)
            base = 2 + i * 3
            obs[base] = bp
            obs[base + 1] = acc
            obs[base + 2] = multiplier
            if is_rich:
                rich_base = BASE_SIMPLE_OBS_SIZE + i * RICH_FEATURES_PER_MOVE
                expected_damage = bp * acc * multiplier * 0.25
                obs[rich_base] = min(4.0, bp * acc * multiplier)
                obs[rich_base + 1] = 1.0 if stab > 1.0 else 0.0
                obs[rich_base + 2] = 1.0 if effectiveness > 1.0 else 0.0
                obs[rich_base + 3] = 1.0 if effectiveness < 1.0 else 0.0
                obs[rich_base + 4] = 1.0 if expected_damage >= opponent_hp else 0.0
                obs[rich_base + 5] = 1.0 if role == ROLE_RECOVER else 0.0
                obs[rich_base + 6] = 1.0 if role == ROLE_SETUP else 0.0
                obs[rich_base + 7] = 1.0 if role == ROLE_STATUS else 0.0

        # Advanced: add status + item + hazard + ability info at end of base obs, before bench
        extra_base = BASE_RICH_OBS_SIZE if is_rich else BASE_SIMPLE_OBS_SIZE
        if is_advanced:
            obs[extra_base] = own_status
            obs[extra_base + 1] = opponent_status
            obs[extra_base + 2] = own_item
            obs[extra_base + 3] = opponent_item
            # Hazards: 6 floats (own-set/opponent-set SR, Spikes, Toxic Spikes)
            obs[extra_base + 4 : extra_base + 10] = hazards
            # Abilities: 2 floats
            obs[extra_base + 10] = own_ability
            obs[extra_base + 11] = opponent_ability
            extra_base += ADVANCED_OBS_EXTRA

        # rich_v2: types, attack boosts and speed (who moves first)
        if self.observation_mode == "rich_v2":
            type_count = len(POKEMON_TYPES)
            for type_index, pokemon_type in enumerate(POKEMON_TYPES):
                obs[extra_base + type_index] = 1.0 if pokemon_type in own_types else 0.0
                obs[extra_base + type_count + type_index] = (
                    1.0 if pokemon_type in opponent_types else 0.0
                )
            stats_base = extra_base + 2 * type_count
            obs[stats_base] = own_attack_boost
            obs[stats_base + 1] = opponent_attack_boost
            obs[stats_base + 2] = own_speed
            obs[stats_base + 3] = opponent_speed
            obs[stats_base + 4] = 0.5 if own_speed == opponent_speed else float(own_speed > opponent_speed)

        # Bench info
        bench_base = base_size
        for index, pokemon in enumerate(bench[: self.max_bench_size]):
            slot_base = bench_base + index * BENCH_FEATURES_PER_POKEMON
            obs[slot_base] = pokemon["hp"]
            for type_index, pokemon_type in enumerate(POKEMON_TYPES):
                obs[slot_base + 1 + type_index] = 1.0 if pokemon_type in pokemon["types"] else 0.0
            # Check if any move on bench pokemon is recovery
            poke_moves = pokemon.get("moves", [])
            has_recovery = any(
                _parse_advanced_move(m)[5] == ROLE_RECOVER for m in poke_moves
            )
            obs[slot_base + 19] = 1.0 if has_recovery else 0.0
        return obs

    def _get_obs(self):
        """Build the observation vector."""
        return self._build_observation(
            own_hp=self.own_hp,
            opponent_hp=self.opponent_hp,
            moves=self.moves,
            bench=self.bench,
            own_status=self.own_status,
            opponent_status=self.opponent_status,
            own_item=self.own_item,
            opponent_item=self._public_value(self.opponent_active, "item"),
            own_ability=self.own_ability,
            opponent_ability=self._public_value(self.opponent_active, "ability"),
            hazards=(
                self.own_stealth_rock, self.opp_stealth_rock,
                self.own_spikes, self.opp_spikes,
                self.own_toxic_spikes, self.opp_toxic_spikes,
            ),
            own_types=self.own_types,
            opponent_types=self.opponent_types,
            own_attack_boost=self.own_attack_boost,
            opponent_attack_boost=self.opponent_attack_boost,
            own_speed=self._effective_speed(is_own=True),
            opponent_speed=self._effective_speed(is_own=False),
        )

    def get_opponent_observation(self):
        """Build the observation vector from the opponent's perspective."""
        return self._build_observation(
            own_hp=self.opponent_hp,
            opponent_hp=self.own_hp,
            moves=self.opponent_moves or self.moves,
            bench=self.opponent_bench,
            own_status=self.opponent_status,
            opponent_status=self.own_status,
            own_item=self.opponent_item,
            opponent_item=self._public_value(self.active_pokemon, "item"),
            own_ability=self.opponent_ability,
            opponent_ability=self._public_value(self.active_pokemon, "ability"),
            hazards=(
                self.opp_stealth_rock, self.own_stealth_rock,
                self.opp_spikes, self.own_spikes,
                self.opp_toxic_spikes, self.own_toxic_spikes,
            ),
            own_types=self.opponent_types,
            opponent_types=self.own_types,
            own_attack_boost=self.opponent_attack_boost,
            opponent_attack_boost=self.own_attack_boost,
            own_speed=self._effective_speed(is_own=False),
            opponent_speed=self._effective_speed(is_own=True),
        )

    def _get_info(self):
        return {
            "turn": self.turn,
            "own_hp": self.own_hp,
            "opponent_hp": self.opponent_hp,
            "own_types": self.own_types,
            "opponent_types": self.opponent_types,
            "mechanics": self.mechanics,
            "observation_mode": self.observation_mode,
            "env_version": ENV_VERSION,
            "bench_size": len(self.bench),
            "opponent_bench_size": len(self.opponent_bench),
        }

    # ------------------------------------------------------------------ #
    #  Action masks
    # ------------------------------------------------------------------ #

    def _action_masks_for(
        self,
        *,
        moves: list | None,
        bench: list[dict],
        own_hp: float,
        opponent_hp: float,
        own_attack_boost: float,
        opponent_attack_boost: float,
        own_status: int = STATUS_NONE,
        own_item: int = ITEM_NONE,
        own_choice_locked: bool = False,
        own_choice_action: int = -1,
    ) -> np.ndarray:
        masks = np.ones(MOVE_ACTIONS + self.max_bench_size, dtype=bool)
        if not moves:
            return masks

        is_advanced = self.mechanics == "advanced"

        for index, move in enumerate(moves):
            bp, acc, multiplier, _, _, role, status_type = self._move_values(move)
            expected_damage = bp * acc * multiplier * own_attack_boost * 0.25

            # Choice Band lock: can only use the first move selected
            if is_advanced and own_choice_locked and own_choice_action >= 0:
                masks[index] = (index == own_choice_action)
                continue

            # Assault Vest: can't use status moves
            if is_advanced and own_item == ITEM_ASSAULT_VEST and role == ROLE_STATUS:
                masks[index] = False
                continue

            # Sleep / Freeze: can't act at all — handled by not masking (model learns through observation)
            if is_advanced and own_status in (STATUS_SLEEP, STATUS_FREEZE):
                continue  # All move masks stay True; model sees status in obs

            # Paralysis: 25% can't move — handled stochastically in _apply_action; masks stay True

            if role == ROLE_ATTACK:
                masks[index] = expected_damage > 0.0
            elif role == ROLE_RECOVER:
                masks[index] = own_hp <= 0.85
            elif role == ROLE_SETUP:
                masks[index] = own_attack_boost < 1.8 and own_hp >= 0.35
            elif role == ROLE_STATUS:
                masks[index] = opponent_attack_boost > 0.45 and opponent_hp >= 0.25

        for index in range(self.max_bench_size):
            masks[SWITCH_ACTION + index] = (
                index < len(bench) and bench[index]["hp"] > 0
            )

        if not masks.any():
            return np.ones(MOVE_ACTIONS + self.max_bench_size, dtype=bool)
        return masks

    def action_masks(self) -> np.ndarray:
        """Return a state-dependent valid-action mask for MaskablePPO."""
        return self._action_masks_for(
            moves=self.moves,
            bench=self.bench,
            own_hp=self.own_hp,
            opponent_hp=self.opponent_hp,
            own_attack_boost=self.own_attack_boost,
            opponent_attack_boost=self.opponent_attack_boost,
            own_status=self.own_status,
            own_item=self.own_item,
            own_choice_locked=self.own_choice_locked,
            own_choice_action=self.own_choice_action,
        )

    def opponent_action_masks(self) -> np.ndarray:
        """Return valid-action masks from the opponent's perspective."""
        return self._action_masks_for(
            moves=self.opponent_moves or self.moves,
            bench=self.opponent_bench,
            own_hp=self.opponent_hp,
            opponent_hp=self.own_hp,
            own_attack_boost=self.opponent_attack_boost,
            opponent_attack_boost=self.own_attack_boost,
            own_status=self.opponent_status,
            own_item=self.opponent_item,
            own_choice_locked=self.opponent_choice_locked,
            own_choice_action=self.opponent_choice_action,
        )

    def _defensive_type_score(self, defender_types: list[str], attacker_types: list[str]) -> float:
        if not defender_types or not attacker_types:
            return 0.0
        incoming = 0.0
        for attacker_type in attacker_types:
            incoming += self._type_effectiveness(attacker_type, defender_types)
        average_incoming = incoming / len(attacker_types)
        return max(-1.0, min(1.0, 1.0 - average_incoming))

    # ------------------------------------------------------------------ #
    #  Opponent action
    # ------------------------------------------------------------------ #

    def _opponent_action(self) -> int:
        living_bench_indices = [
            index for index, pokemon in enumerate(self.opponent_bench) if pokemon["hp"] > 0
        ]
        if living_bench_indices and self.rng.random() < 0.10:
            return SWITCH_ACTION + int(self.rng.choice(living_bench_indices))

        policy = self.current_opponent_policy
        opponent_moves = self.opponent_moves or self.moves
        is_advanced = self.mechanics == "advanced"

        if policy == "max_damage":
            return int(np.argmax([self._move_values(move)[0] for move in opponent_moves]))

        if policy == "type_aware":
            scores = []
            for move in opponent_moves:
                bp, acc, multiplier, _, _, role, status_type = self._move_values(move)

                # Status- and item-aware scoring
                base_score = bp * acc * multiplier

                # Don't use status moves if Assault Vest
                if is_advanced and self.opponent_item == ITEM_ASSAULT_VEST and role == ROLE_STATUS:
                    scores.append(-0.1)
                    continue

                if role == ROLE_RECOVER:
                    score = 1.2 if self.opponent_hp <= 0.35 else 0.05
                elif role == ROLE_SETUP:
                    score = 0.15
                elif role == ROLE_STATUS:
                    # More valuable if opponent isn't statused yet
                    if self.own_status != STATUS_NONE:
                        score = 0.1  # already statused
                    else:
                        score = 0.4 if self.own_hp >= 0.3 else 0.1
                else:
                    # Burn halves physical attack — account for it
                    if is_advanced and self.own_status == STATUS_BURN:
                        base_score *= 0.5  # Attack effectively halved
                    score = base_score
                scores.append(score)
            return int(np.argmax(scores))

        return int(self.rng.integers(0, MOVE_ACTIONS))

    # ------------------------------------------------------------------ #
    #  Reset
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.own_hp = 1.0
        self.opponent_hp = 1.0
        self.own_attack_boost = 1.0
        self.opponent_attack_boost = 1.0
        self.own_status = STATUS_NONE
        self.opponent_status = STATUS_NONE
        self.own_item = ITEM_NONE
        self.opponent_item = ITEM_NONE
        self.own_choice_locked = False
        self.own_choice_action = -1
        self.opponent_choice_locked = False
        self.opponent_choice_action = -1
        self.own_sash_used = False
        self.opponent_sash_used = False
        self.own_ability = ABILITY_NONE
        self.opponent_ability = ABILITY_NONE
        # Reset hazards
        self.own_stealth_rock = 0
        self.opp_stealth_rock = 0
        self.own_spikes = 0
        self.opp_spikes = 0
        self.own_toxic_spikes = 0
        self.opp_toxic_spikes = 0

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
            self.own_speed = float(self.rng.uniform(0.2, 1.0))
            self.opponent_speed = float(self.rng.uniform(0.2, 1.0))
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

    # ------------------------------------------------------------------ #
    #  Apply action (the core combat logic)
    # ------------------------------------------------------------------ #

    def _apply_action(self, action: int, *, is_own: bool) -> None:
        is_advanced = self.mechanics == "advanced"

        # --- Switches ---
        if action >= SWITCH_ACTION:
            bench = self.bench if is_own else self.opponent_bench
            bench_index = action - SWITCH_ACTION
            if bench_index < len(bench) and bench[bench_index]["hp"] > 0:
                self._switch_to(bench, bench_index, is_own=is_own)
            return

        # --- Use a move ---
        moves = self.moves if is_own else (self.opponent_moves or self.moves)
        bp, acc, multiplier, stab, effectiveness, role, status_type = self._move_values(moves[action])

        # Handle Choice Band lock: set on first move used this switch-in
        if is_own:
            if is_advanced and self.own_item == ITEM_CHOICE_BAND and not self.own_choice_locked:
                self.own_choice_locked = True
                self.own_choice_action = action
        else:
            if is_advanced and self.opponent_item == ITEM_CHOICE_BAND and not self.opponent_choice_locked:
                self.opponent_choice_locked = True
                self.opponent_choice_action = action

        # Handle Sleep / Freeze — can't act
        if is_advanced:
            actor_status = self.own_status if is_own else self.opponent_status
            poke = self.active_pokemon if is_own else self.opponent_active
            if actor_status == STATUS_SLEEP and poke is not None:
                # Decrement sleep counter, wake if done
                remaining = poke.get("sleep_turns", 0) - 1
                if remaining <= 0:
                    if is_own:
                        self.own_status = STATUS_NONE
                    else:
                        self.opponent_status = STATUS_NONE
                    poke["status"] = STATUS_NONE
                else:
                    poke["sleep_turns"] = remaining
                self._sync_active_pokemon()
                return  # Didn't act this turn

            if actor_status == STATUS_FREEZE and poke is not None:
                # 20% thaw chance each turn
                if self.rng.random() < 0.2:
                    if is_own:
                        self.own_status = STATUS_NONE
                    else:
                        self.opponent_status = STATUS_NONE
                    poke["status"] = STATUS_NONE
                    self._sync_active_pokemon()
                else:
                    self._sync_active_pokemon()
                    return  # Still frozen, can't act

            if actor_status == STATUS_PARALYSIS:
                # 25% chance to be fully paralyzed
                if self.rng.random() < 0.25:
                    self._sync_active_pokemon()
                    return  # Paralysis prevented action

        # Accuracy check
        if self.rng.random() > acc:
            self._sync_active_pokemon()
            return

        # --- Apply move effects ---
        # Accuracy only gates the hit above; damage is not scaled by it again.
        if is_own:
            if role == ROLE_RECOVER:
                heal = 0.35
                if is_advanced and self.own_item == ITEM_LEFTOVERS:
                    heal += 0.06  # Extra healing from leftovers synergy
                self.own_hp = min(1.0, self.own_hp + heal)
                # Burn doesn't affect recovery
            elif role == ROLE_SETUP:
                boost = 0.6
                self.own_attack_boost = min(2.2, self.own_attack_boost + boost)
            elif role == ROLE_STATUS:
                if is_advanced and status_type != STATUS_NONE:
                    self._apply_status_move(status_type, is_own=True)
                else:
                    self.opponent_attack_boost = max(0.4, self.opponent_attack_boost - 0.45)
            else:
                # Attack move
                attack_boost = self.own_attack_boost

                # Burn halves attack (physical)
                if is_advanced and self.own_status == STATUS_BURN:
                    attack_boost *= 0.5

                # Item modifiers
                if is_advanced:
                    if self.own_item == ITEM_LIFE_ORB:
                        attack_boost *= 1.3
                    elif self.own_item == ITEM_CHOICE_BAND:
                        attack_boost *= 1.5

                damage_dealt = bp * multiplier * attack_boost * 0.25

                # Focus Sash: survive at 1 HP from full
                if is_advanced and self.opponent_item == ITEM_FOCUS_SASH and not self.opponent_sash_used:
                    if self.opponent_hp >= 0.99 and damage_dealt >= self.opponent_hp:
                        damage_dealt = self.opponent_hp - (1.0 / 100.0)  # Leave a sliver
                        self.opponent_sash_used = True
                        self._reveal(self.opponent_active, "item")

                self.opponent_hp = max(0.0, self.opponent_hp - damage_dealt)

                # Life Orb recoil: 1/10 of the attacker's max HP
                if is_advanced and self.own_item == ITEM_LIFE_ORB and damage_dealt > 0:
                    self.own_hp = max(0.0, self.own_hp - 0.1)
                    self._reveal(self.active_pokemon, "item")

                self._maybe_freeze(moves[action], target_is_own=False)

            self._sync_active_pokemon()
            return

        # --- Opponent's action ---
        if role == ROLE_RECOVER:
            heal = 0.35
            if is_advanced and self.opponent_item == ITEM_LEFTOVERS:
                heal += 0.06
            self.opponent_hp = min(1.0, self.opponent_hp + heal)
        elif role == ROLE_SETUP:
            self.opponent_attack_boost = min(2.2, self.opponent_attack_boost + 0.6)
        elif role == ROLE_STATUS:
            if is_advanced and status_type != STATUS_NONE:
                self._apply_status_move(status_type, is_own=False)
            else:
                self.own_attack_boost = max(0.4, self.own_attack_boost - 0.45)
        else:
            attack_boost = self.opponent_attack_boost
            if is_advanced and self.opponent_status == STATUS_BURN:
                attack_boost *= 0.5
            if is_advanced:
                if self.opponent_item == ITEM_LIFE_ORB:
                    attack_boost *= 1.3
                elif self.opponent_item == ITEM_CHOICE_BAND:
                    attack_boost *= 1.5

            damage_taken = bp * multiplier * attack_boost * 0.25

            # Focus Sash for the player's side
            if is_advanced and self.own_item == ITEM_FOCUS_SASH and not self.own_sash_used:
                if self.own_hp >= 0.99 and damage_taken >= self.own_hp:
                    damage_taken = self.own_hp - (1.0 / 100.0)
                    self.own_sash_used = True
                    self._reveal(self.active_pokemon, "item")

            self.own_hp = max(0.0, self.own_hp - damage_taken)

            # Life Orb recoil hits the attacker (the opponent)
            if is_advanced and self.opponent_item == ITEM_LIFE_ORB and damage_taken > 0:
                self.opponent_hp = max(0.0, self.opponent_hp - 0.1)
                self._reveal(self.opponent_active, "item")

            self._maybe_freeze(moves[action], target_is_own=True)

        self._sync_active_pokemon()

    def _apply_status_move(self, status_type: int, *, is_own: bool) -> None:
        """Resolve an advanced-mode status move: set a hazard or inflict a status."""
        if status_type == HAZARD_STEALTH_ROCK:
            if is_own:
                self.own_stealth_rock = 1
            else:
                self.opp_stealth_rock = 1
        elif status_type == HAZARD_SPIKES:
            if is_own and self.own_spikes < 3:
                self.own_spikes += 1
            elif not is_own and self.opp_spikes < 3:
                self.opp_spikes += 1
        elif status_type == HAZARD_TOXIC_SPIKES:
            if is_own and self.own_toxic_spikes < 2:
                self.own_toxic_spikes += 1
            elif not is_own and self.opp_toxic_spikes < 2:
                self.opp_toxic_spikes += 1
        else:
            self._inflict_status(status_type, target_is_own=not is_own)

    def _inflict_status(self, status_type: int, *, target_is_own: bool) -> None:
        """Give the target a major status if it doesn't already have one."""
        target_status = self.own_status if target_is_own else self.opponent_status
        target_active = self.active_pokemon if target_is_own else self.opponent_active
        if target_status != STATUS_NONE or target_active is None:
            return
        if target_is_own:
            self.own_status = status_type
        else:
            self.opponent_status = status_type
        target_active["status"] = status_type
        if status_type == STATUS_SLEEP:
            target_active["sleep_turns"] = int(self.rng.integers(1, 4))

    def _maybe_freeze(self, move: tuple, *, target_is_own: bool) -> None:
        """Ice-type attacks have a 10% chance to freeze a surviving target (advanced only)."""
        if self.mechanics != "advanced" or _move_type(move) != "ice":
            return
        target_hp = self.own_hp if target_is_own else self.opponent_hp
        if target_hp > 0 and self.rng.random() < 0.1:
            self._inflict_status(STATUS_FREEZE, target_is_own=target_is_own)

    # ------------------------------------------------------------------ #
    #  Status ticks (burn / poison damage, item healing)
    # ------------------------------------------------------------------ #

    def _apply_status_ticks(self) -> None:
        """Apply end-of-turn status damage and item healing."""
        if self.mechanics != "advanced":
            return

        # Burn: 1/16 max HP damage
        if self.own_status == STATUS_BURN:
            self.own_hp = max(0.0, self.own_hp - (1.0 / 16.0))
        if self.opponent_status == STATUS_BURN:
            self.opponent_hp = max(0.0, self.opponent_hp - (1.0 / 16.0))

        # Poison: 1/8 max HP damage
        if self.own_status == STATUS_POISON:
            self.own_hp = max(0.0, self.own_hp - (1.0 / 8.0))
        if self.opponent_status == STATUS_POISON:
            self.opponent_hp = max(0.0, self.opponent_hp - (1.0 / 8.0))

        # Leftovers: 1/16 max HP heal
        if self.own_item == ITEM_LEFTOVERS and self.own_hp > 0 and self.own_hp < 1.0:
            self.own_hp = min(1.0, self.own_hp + (1.0 / 16.0))
            self._reveal(self.active_pokemon, "item")
        if self.opponent_item == ITEM_LEFTOVERS and self.opponent_hp > 0 and self.opponent_hp < 1.0:
            self.opponent_hp = min(1.0, self.opponent_hp + (1.0 / 16.0))
            self._reveal(self.opponent_active, "item")

        self._sync_active_pokemon()

    # ------------------------------------------------------------------ #
    #  Step
    # ------------------------------------------------------------------ #

    def step(self, action):
        # Clamp action to valid range
        action = min(max(int(action), 0), self.action_space.n - 1)

        prev_opp = self._team_hp(is_own=False)
        prev_own = self._team_hp(is_own=True)

        # Both sides commit to an action before either resolves.
        opp_action = self._opponent_action()
        actions = {True: action, False: opp_action}
        for is_own in self._turn_order(action, opp_action):
            # A Pokemon knocked out earlier in the turn doesn't get to act.
            if (self.own_hp if is_own else self.opponent_hp) <= 0:
                continue
            self._apply_action(actions[is_own], is_own=is_own)

        # --- Status ticks (burn/poison damage, leftovers heal) ---
        if self.mechanics == "advanced":
            self._apply_status_ticks()

        self._auto_switch_fainted()

        # --- Reward (team hp-delta, so switches don't hide damage; no role shaping) ---
        opp_delta = max(0.0, prev_opp - self._team_hp(is_own=False))
        own_delta = max(0.0, prev_own - self._team_hp(is_own=True))
        reward = opp_delta - own_delta - 0.01

        # --- Terminal conditions ---
        terminated = False
        result = self.battle_result()
        if result == "win":
            reward += 1.0
            terminated = True
        elif result == "loss":
            reward -= 1.0
            terminated = True

        self.turn += 1
        truncated = self.turn >= self.max_turns
        if truncated and not terminated:
            reward -= 0.75
            result = "draw"

        info = self._get_info()
        if result is not None:
            info["result"] = result
            info["is_success"] = result == "win"
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        status_str = ""
        if self.mechanics == "advanced":
            s_own = STATUS_LABELS[self.own_status]
            s_opp = STATUS_LABELS[self.opponent_status]
            i_own = ITEM_LABELS[self.own_item]
            i_opp = ITEM_LABELS[self.opponent_item]
            status_str = f" | status: {s_own}/{s_opp} | items: {i_own}/{i_opp}"
        print(f"Turn {self.turn}: own={self.own_hp:.2f}, opp={self.opponent_hp:.2f}{status_str}")

    def close(self):
        pass
