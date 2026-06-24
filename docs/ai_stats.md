# AI Stats

This report summarizes the published benchmark in
`docs/benchmarks/current_evaluation.csv`, generated from `SimplePokemonMoveEnv`.

![ShowdownRL policy benchmark](assets/ai_policy_comparison.png)

## Current Result

The trained PPO v2 policy won 717 of 1000 episodes (71.7%) on the typed
mechanics benchmark with an average reward of 0.422. On the same benchmark, the
older PPO v1 checkpoint won 404 of 1000 episodes (40.4%).

The strongest policy in this run was still **Type aware** at 74.7% win rate.
That means the trained model is now much better than the previous checkpoint,
but the next model target is to beat the type-aware baseline consistently.

## Benchmark Table

| Scenario | Policy | Episodes | Record | Win rate | Avg reward | Avg turns |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
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
python scripts/evaluate_model.py --episodes 1000 --seed 42 --mechanics typed --opponent-policy type_aware --model models/ppo_move_selection_v1.zip --model models/ppo_move_selection_v2_typed.zip --output results/evaluation_v1_vs_v2_typed.csv
python scripts/evaluate_model.py --episodes 1000 --seed 42 --mechanics toy --opponent-policy random --model models/ppo_move_selection_v1.zip --model models/ppo_move_selection_v2_typed.zip --output results/evaluation_v1_vs_v2_toy.csv
```

Model artifact: `ppo_move_selection_v2_typed.zip (0.2 MB, local artifact)`  
Evaluation CSV timestamp: `2026-06-24`

These numbers benchmark the experimental simulator policy. The live browser
player can use the trained model with `showdownrl live --policy ppo`, with a
heuristic fallback if the checkpoint is missing or selects an unavailable move.
Live ladder performance should still be tracked separately because real
Pokemon Showdown battles include mechanics beyond this move-selection simulator.
