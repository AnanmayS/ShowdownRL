# AI Stats

This report summarizes the published benchmark in
`docs/benchmarks/current_evaluation.csv`, generated from `SimplePokemonMoveEnv`.

![ShowdownRL policy benchmark](assets/ai_policy_comparison.png)

## Current Result

The trained PPO v3 policy uses a richer 46-feature observation that includes
per-move expected damage, STAB, type advantage, finish ranges, recovery, setup,
and status flags. On rich mechanics against the type-aware opponent, v3 beat the
typed v2 checkpoint on two 1000-episode seeds:

- Seed 42: v3 won 597 of 1000 episodes (59.7%) versus 57.8% for v2.
- Seed 99: v3 won 585 of 1000 episodes (58.5%) versus 57.0% for v2.

The strongest policy is still the hand-coded **Type aware** baseline, which won
60.8% on seed 42 and 59.6% on seed 99. The next model target is to beat that
baseline consistently.

## Benchmark Table

| Scenario | Policy | Episodes | Record | Win rate | Avg reward | Avg turns |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Rich/type-aware seed 42 | Type aware | 1000 | 608-392 | 60.8% | +0.237 | 6.67 |
| Rich/type-aware seed 42 | Trained PPO v3 | 1000 | 597-403 | 59.7% | +0.205 | 7.76 |
| Rich/type-aware seed 42 | Trained PPO v2 | 1000 | 578-422 | 57.8% | +0.165 | 6.64 |
| Rich/type-aware seed 99 | Type aware | 1000 | 596-404 | 59.6% | +0.213 | 6.70 |
| Rich/type-aware seed 99 | Trained PPO v3 | 1000 | 585-415 | 58.5% | +0.182 | 7.76 |
| Rich/type-aware seed 99 | Trained PPO v2 | 1000 | 570-430 | 57.0% | +0.147 | 6.67 |
| Typed/type-aware | Type aware | 1000 | 747-253 | 74.7% | +0.498 | 5.08 |
| Typed/type-aware | Trained PPO v2 | 1000 | 717-283 | 71.7% | +0.422 | 5.11 |
| Typed/type-aware | Trained PPO v1 | 1000 | 404-596 | 40.4% | -0.387 | 5.28 |
| Toy/random | Type aware | 1000 | 933-67 | 93.3% | +1.132 | 5.29 |
| Toy/random | Trained PPO v2 | 1000 | 923-77 | 92.3% | +1.105 | 5.33 |
| Toy/random | Trained PPO v1 | 1000 | 729-271 | 72.9% | +0.568 | 6.18 |

## What These Stats Mean

- **Win rate** is the clearest headline metric.
- **Average reward** captures how decisively the policy wins or loses in the
  simulator's reward function.
- **Average turns** is useful as an efficiency signal, but lower is only better
  when win rate and reward remain strong.
- **Reward / turn** is a compact tie-breaker for policies with similar win
  rates.

## Reproducing

```bash
pip install -e ".[rl]"
python scripts/train_ppo.py --timesteps 100000 --seed 42 --mechanics typed --opponent-policy type_aware --output models/ppo_move_selection_v2_typed.zip
python scripts/train_ppo.py --timesteps 300000 --seed 45 --mechanics rich --observation-mode rich --opponent-policy type_aware --output models/ppo_move_selection_v3_rich.zip
python scripts/evaluate_model.py --episodes 1000 --seed 42 --mechanics typed --opponent-policy type_aware --model models/ppo_move_selection_v1.zip --model models/ppo_move_selection_v2_typed.zip --output results/evaluation_v1_vs_v2_typed.csv
python scripts/evaluate_model.py --episodes 1000 --seed 42 --mechanics toy --opponent-policy random --model models/ppo_move_selection_v1.zip --model models/ppo_move_selection_v2_typed.zip --output results/evaluation_v1_vs_v2_toy.csv
python scripts/evaluate_model.py --episodes 1000 --seed 42 --mechanics rich --opponent-policy type_aware --model models/ppo_move_selection_v2_typed.zip --model models/ppo_move_selection_v3_rich.zip --output results/evaluation_v2_vs_v3_rich_seed42.csv
python scripts/evaluate_model.py --episodes 1000 --seed 99 --mechanics rich --opponent-policy type_aware --model models/ppo_move_selection_v2_typed.zip --model models/ppo_move_selection_v3_rich.zip --output results/evaluation_v2_vs_v3_rich_seed99.csv
```

Model artifacts: `ppo_move_selection_v2_typed.zip`, `ppo_move_selection_v3_rich.zip`  
Evaluation CSV timestamp: `2026-06-24`

These numbers benchmark the experimental simulator policy. The live browser
player can use the trained model with `showdownrl live --policy ppo`, with a
heuristic fallback if the checkpoint is missing or selects an unavailable move.
Live ladder performance should still be tracked separately because real
Pokemon Showdown battles include mechanics beyond this move-selection simulator.
