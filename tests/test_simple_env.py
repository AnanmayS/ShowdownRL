from __future__ import annotations

import unittest

try:
    from showdownrl.simple_env import (
        ABILITY_INTIMIDATE,
        ABILITY_REGENERATOR,
        BASE_RICH_OBS_SIZE,
        BASE_SIMPLE_OBS_SIZE,
        ITEM_LEFTOVERS,
        ITEM_LIFE_ORB,
        ITEM_NONE,
        POKEMON_TYPES,
        RICH_OBS_SIZE,
        RICH_V2_OBS_EXTRA,
        STATUS_FREEZE,
        STATUS_NONE,
        STATUS_SLEEP,
        SimplePokemonMoveEnv,
    )
except ImportError as exc:  # pragma: no cover - depends on optional rl extras
    SimplePokemonMoveEnv = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(SimplePokemonMoveEnv is None, f"optional RL dependencies are missing: {IMPORT_ERROR}")
class SimpleEnvTests(unittest.TestCase):
    def test_type_aware_opponent_policy_selects_best_expected_damage(self) -> None:
        env = SimplePokemonMoveEnv(opponent_policy="type_aware", seed=1)
        env.moves = [
            (0.4, 1.0, 1.0),
            (0.5, 1.0, 2.0),
            (1.0, 0.5, 1.0),
            (0.3, 1.0, 1.0),
        ]

        self.assertEqual(env._opponent_action(), 1)

    def test_typed_mechanics_samples_types_and_damage_multipliers(self) -> None:
        env = SimplePokemonMoveEnv(mechanics="typed", seed=2)
        obs, info = env.reset()

        self.assertEqual(info["mechanics"], "typed")
        self.assertTrue(info["own_types"])
        self.assertTrue(info["opponent_types"])
        self.assertEqual(len(env.moves), 4)
        self.assertEqual(len(env.opponent_moves), 4)
        self.assertTrue(any(obs[2 + index * 3 + 2] != 1.0 for index in range(4)))

    def test_opponent_policy_uses_hidden_opponent_moves(self) -> None:
        env = SimplePokemonMoveEnv(opponent_policy="max_damage", seed=5)
        env.moves = [
            (1.0, 1.0, 1.0),
            (0.9, 1.0, 1.0),
            (0.8, 1.0, 1.0),
            (0.7, 1.0, 1.0),
        ]
        env.opponent_moves = [
            (0.1, 1.0, 1.0),
            (0.2, 1.0, 1.0),
            (0.3, 1.0, 1.0),
            (0.4, 1.0, 1.0),
        ]

        self.assertEqual(env._opponent_action(), 3)

    def test_opponent_does_not_retaliate_after_fainting(self) -> None:
        env = SimplePokemonMoveEnv(opponent_policy="max_damage", max_bench_size=0, seed=6)
        env.reset()
        env.own_hp = 0.2
        env.opponent_hp = 0.1
        env.moves = [
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ]
        env.opponent_moves = [
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ]
        env.own_speed = 0.9
        env.opponent_speed = 0.3

        _, _, terminated, _, info = env.step(0)

        self.assertTrue(terminated)
        self.assertEqual(info["own_hp"], 0.2)
        self.assertEqual(info["opponent_hp"], 0.0)
        self.assertEqual(info["result"], "win")

    def test_faster_opponent_moves_first(self) -> None:
        env = SimplePokemonMoveEnv(opponent_policy="max_damage", max_bench_size=0, seed=6)
        env.reset()
        env.own_hp = 0.2
        env.opponent_hp = 0.1
        env.moves = [(1.0, 1.0, 1.0)] + [(0.0, 1.0, 0.0)] * 3
        env.opponent_moves = [(1.0, 1.0, 1.0)] + [(0.0, 1.0, 0.0)] * 3
        env.own_speed = 0.3
        env.opponent_speed = 0.9

        _, _, terminated, _, info = env.step(0)

        self.assertTrue(terminated)
        self.assertEqual(info["own_hp"], 0.0)
        self.assertEqual(info["opponent_hp"], 0.1)
        self.assertEqual(info["result"], "loss")

    def test_speed_ties_are_broken_randomly(self) -> None:
        env = SimplePokemonMoveEnv(max_bench_size=0, seed=11)
        env.reset()
        env.own_speed = env.opponent_speed = 0.5

        orders = {env._turn_order(0, 0) for _ in range(50)}

        self.assertEqual(orders, {(True, False), (False, True)})

    def test_switch_resolves_before_faster_attacker(self) -> None:
        env = SimplePokemonMoveEnv(mechanics="typed", seed=12)
        env.reset()
        original_active = env.active_pokemon
        replacement = env.bench[0]
        replacement["types"] = ["water"]
        env.opponent_active["moves"] = [(1.0, 1.0, 1.0, 1.0, 1.0, "attack", 0, "normal")] * 4
        env._sync_active_aliases()
        env.own_speed = 0.1
        env.opponent_speed = 0.9
        env._opponent_action = lambda: 0

        env.step(4)

        self.assertIs(env.active_pokemon, replacement)
        self.assertAlmostEqual(replacement["hp"], 0.75)
        self.assertEqual(original_active["hp"], 1.0)

    def test_move_multipliers_recomputed_after_either_side_switches(self) -> None:
        env = SimplePokemonMoveEnv(mechanics="typed", seed=13)
        env.reset()
        water = (0.8, 1.0, 1.0, 1.0, 1.0, "attack", 0, "water")
        env.active_pokemon["types"] = ["normal"]
        env.active_pokemon["moves"] = [water] * 4
        env.opponent_active["types"] = ["fire"]
        env.opponent_bench[0]["types"] = ["grass"]
        env._sync_active_aliases()
        env._refresh_matchups()
        self.assertEqual(env.moves[0][2], 2.0)

        env._apply_action(4, is_own=False)  # opponent switches fire -> grass

        self.assertEqual(env.opponent_types, ["grass"])
        self.assertEqual(env.moves[0][2], 0.5)
        self.assertAlmostEqual(env._get_obs()[4], 0.5)

        stale_electric = (0.8, 1.0, 9.9, 1.0, 9.9, "attack", 0, "electric")
        env.bench[0]["types"] = ["normal"]
        env.bench[0]["moves"] = [stale_electric] * 4
        env._apply_action(4, is_own=True)  # own switch-in picks up the current matchup

        self.assertEqual(env.moves[0][2], 0.5)
        self.assertEqual(env.moves[0][4], 0.5)

    def test_accuracy_only_gates_the_hit(self) -> None:
        env = SimplePokemonMoveEnv(mechanics="rich", max_bench_size=0, seed=14)
        env.reset()
        env.moves = [(0.8, 0.5, 1.0, 1.0, 1.0, "attack")] * 4
        damages = set()
        for _ in range(60):
            env.opponent_hp = 1.0
            env._apply_action(0, is_own=True)
            damages.add(round(1.0 - env.opponent_hp, 6))

        # A hit deals the full bp * multiplier * 0.25 (no second accuracy discount).
        self.assertEqual(damages, {0.0, 0.2})

    def test_hazards_hit_pokemon_sent_in_after_faint(self) -> None:
        env = SimplePokemonMoveEnv(mechanics="advanced", seed=15)
        env.reset()
        fainted = env.active_pokemon
        fainted["ability"] = ABILITY_REGENERATOR
        replacement = env.bench[0]
        replacement["types"] = ["fire"]  # 2x weak to Rock
        replacement["hp"] = 0.8
        env.opp_stealth_rock = 1
        env.own_hp = 0.0
        env._sync_active_pokemon()

        env._auto_switch_fainted()

        self.assertIs(env.active_pokemon, replacement)
        # Stealth Rock deals 1/8 max HP * 2 regardless of current HP.
        self.assertAlmostEqual(env.own_hp, 0.8 - 0.25)
        # The fainted Pokemon takes the replacement's bench slot and stays fainted.
        self.assertIs(env.bench[0], fainted)
        self.assertEqual(fainted["hp"], 0.0)
        self.assertEqual(len({id(p) for p in [env.active_pokemon, *env.bench]}), 4)

    def test_boosts_and_choice_lock_reset_on_switch(self) -> None:
        env = SimplePokemonMoveEnv(mechanics="advanced", seed=16)
        env.reset()
        original_active = env.active_pokemon
        env.own_attack_boost = 2.2
        env.own_choice_locked = True
        env.own_choice_action = 2
        env._sync_active_pokemon()

        env._apply_action(4, is_own=True)
        env._apply_action(4, is_own=True)

        self.assertIs(env.active_pokemon, original_active)
        self.assertEqual(env.own_attack_boost, 1.0)
        self.assertFalse(env.own_choice_locked)
        self.assertEqual(env.own_choice_action, -1)

    def test_life_orb_recoil_hits_the_attacker(self) -> None:
        env = SimplePokemonMoveEnv(mechanics="advanced", seed=17)
        env.reset()
        env.own_hp = env.opponent_hp = 1.0
        env.own_status = env.opponent_status = STATUS_NONE
        env.own_item = ITEM_NONE
        env.opponent_item = ITEM_LIFE_ORB
        env.opponent_attack_boost = 1.0
        env.opponent_moves = [(0.4, 1.0, 1.0, 1.0, 1.0, "attack", 0)] * 4

        env._apply_action(0, is_own=False)

        self.assertAlmostEqual(env.own_hp, 1.0 - 0.4 * 1.3 * 0.25)
        self.assertAlmostEqual(env.opponent_hp, 0.9)

    def test_status_moves_lower_only_the_target_attack_outside_advanced(self) -> None:
        env = SimplePokemonMoveEnv(mechanics="rich", max_bench_size=0, seed=18)
        env.reset()
        status_move = (0.0, 1.0, 0.0, 1.0, 1.0, "status")
        env.moves = [status_move] * 4
        env.opponent_moves = [status_move] * 4

        env._apply_action(0, is_own=True)
        self.assertAlmostEqual(env.opponent_attack_boost, 0.55)
        self.assertEqual(env.own_attack_boost, 1.0)

        env._apply_action(0, is_own=False)
        self.assertAlmostEqual(env.own_attack_boost, 0.55)
        self.assertAlmostEqual(env.opponent_attack_boost, 0.55)

    def test_opponent_item_and_ability_hidden_until_revealed(self) -> None:
        env = SimplePokemonMoveEnv(mechanics="advanced", seed=19)
        env.reset()
        env.opponent_active["item"] = ITEM_LEFTOVERS
        env.opponent_active["ability"] = ABILITY_INTIMIDATE
        env._sync_active_aliases()
        extra = BASE_SIMPLE_OBS_SIZE

        obs = env._get_obs()
        self.assertEqual(obs[extra + 2], env.own_item)
        self.assertEqual(obs[extra + 3], 0)
        self.assertEqual(obs[extra + 11], 0)

        env.opponent_status = STATUS_NONE
        env.opponent_hp = 0.5
        env._apply_status_ticks()  # Leftovers heals visibly

        self.assertEqual(env._get_obs()[extra + 3], ITEM_LEFTOVERS)

    def test_sleep_and_freeze_can_occur(self) -> None:
        env = SimplePokemonMoveEnv(mechanics="advanced", seed=20)
        env.reset()
        self.assertIn(STATUS_SLEEP, {env._sample_status_type() for _ in range(300)})

        env.own_status = STATUS_NONE
        env.moves = [(0.1, 1.0, 1.0, 1.0, 1.0, "attack", 0, "ice")] * 4
        statuses = set()
        for _ in range(200):
            env.opponent_hp = 1.0
            env.opponent_status = STATUS_NONE
            env.opponent_active["status"] = STATUS_NONE
            env._apply_action(0, is_own=True)
            statuses.add(env.opponent_status)
        self.assertIn(STATUS_FREEZE, statuses)

    def test_rich_v2_observation_adds_types_boosts_and_speed(self) -> None:
        env = SimplePokemonMoveEnv(mechanics="rich", observation_mode="rich_v2", seed=21)
        obs, _ = env.reset()
        self.assertEqual(len(obs), RICH_OBS_SIZE + RICH_V2_OBS_EXTRA)

        env.own_speed, env.opponent_speed = 0.8, 0.4
        env.own_attack_boost = 1.6
        obs = env._get_obs()
        base = BASE_RICH_OBS_SIZE
        for pokemon_type in env.own_types:
            self.assertEqual(obs[base + POKEMON_TYPES.index(pokemon_type)], 1.0)
        for pokemon_type in env.opponent_types:
            self.assertEqual(obs[base + len(POKEMON_TYPES) + POKEMON_TYPES.index(pokemon_type)], 1.0)
        stats = base + 2 * len(POKEMON_TYPES)
        self.assertAlmostEqual(obs[stats], 1.6, places=5)
        self.assertAlmostEqual(obs[stats + 2], 0.8, places=5)
        self.assertAlmostEqual(obs[stats + 3], 0.4, places=5)
        self.assertEqual(obs[stats + 4], 1.0)

    def test_reward_counts_damage_to_a_pokemon_that_switched_in(self) -> None:
        env = SimplePokemonMoveEnv(mechanics="typed", seed=22)
        env.reset()
        env.opponent_bench[0]["types"] = ["water"]
        env.active_pokemon["moves"] = [(1.0, 1.0, 1.0, 1.0, 1.0, "attack", 0, "normal")] * 4
        env._sync_active_aliases()
        env._opponent_action = lambda: 4

        _, reward, _, _, _ = env.step(0)

        self.assertAlmostEqual(env.opponent_hp, 0.75)
        self.assertAlmostEqual(reward, 0.25 - 0.01)

    def test_rich_observation_includes_support_move_flags(self) -> None:
        for seed in range(1, 20):
            env = SimplePokemonMoveEnv(mechanics="rich", observation_mode="rich", seed=seed)
            obs, info = env.reset()
            if any(
                obs[14 + index * 8 + 5] or obs[14 + index * 8 + 6] or obs[14 + index * 8 + 7]
                for index in range(4)
            ):
                self.assertEqual(info["observation_mode"], "rich")
                self.assertEqual(len(obs), RICH_OBS_SIZE)
                return
        self.fail("rich mechanics did not generate a support move in deterministic sample")

    def test_action_masks_filter_state_dependent_no_ops(self) -> None:
        env = SimplePokemonMoveEnv(mechanics="rich", max_bench_size=0, seed=3)
        env.reset()
        env.own_hp = 1.0
        env.own_attack_boost = 2.2
        env.opponent_attack_boost = 0.4
        env.opponent_hp = 0.1
        env.moves = [
            (0.7, 1.0, 1.0, 1.0, 1.0, "attack"),
            (0.0, 1.0, 0.0, 1.0, 1.0, "recover"),
            (0.0, 1.0, 0.0, 1.0, 1.0, "setup"),
            (0.0, 1.0, 0.0, 1.0, 1.0, "status"),
        ]

        masks = env.action_masks().tolist()
        self.assertEqual(masks[:4], [True, False, False, False])
        self.assertTrue(all(not m for m in masks[4:]))

    def test_action_masks_keep_fallback_action_when_all_filtered(self) -> None:
        env = SimplePokemonMoveEnv(mechanics="rich", max_bench_size=0, seed=4)
        env.reset()
        env.own_hp = 1.0
        env.own_attack_boost = 2.2
        env.opponent_attack_boost = 0.4
        env.opponent_hp = 0.1
        env.moves = [
            (0.0, 1.0, 0.0, 1.0, 1.0, "recover"),
            (0.0, 1.0, 0.0, 1.0, 1.0, "recover"),
            (0.0, 1.0, 0.0, 1.0, 1.0, "setup"),
            (0.0, 1.0, 0.0, 1.0, 1.0, "status"),
        ]

        self.assertEqual(env.action_masks().tolist(), [True, True, True, True])

    def test_opponent_observation_does_not_mutate_env_state(self) -> None:
        env = SimplePokemonMoveEnv(mechanics="rich", observation_mode="rich", seed=8)
        env.reset()
        env.own_hp = 0.4
        env.opponent_hp = 0.7
        before = (
            env.own_hp,
            env.opponent_hp,
            env.moves,
            env.opponent_moves,
            env.bench,
            env.opponent_bench,
            env.active_pokemon,
            env.opponent_active,
        )

        obs = env.get_opponent_observation()
        masks = env.opponent_action_masks()

        self.assertAlmostEqual(obs[0], 0.7, places=4)
        self.assertAlmostEqual(obs[1], 0.4, places=4)
        self.assertEqual(len(masks), env.action_space.n)
        self.assertEqual(
            before,
            (
                env.own_hp,
                env.opponent_hp,
                env.moves,
                env.opponent_moves,
                env.bench,
                env.opponent_bench,
                env.active_pokemon,
                env.opponent_active,
            ),
        )

    def test_switch_swaps_active_into_selected_bench_slot(self) -> None:
        env = SimplePokemonMoveEnv(mechanics="typed", seed=9)
        env.reset()
        original_active = env.active_pokemon
        replacement = env.bench[0]

        env._apply_action(4, is_own=True)

        self.assertIs(env.active_pokemon, replacement)
        self.assertIs(env.bench[0], original_active)


if __name__ == "__main__":
    unittest.main()
