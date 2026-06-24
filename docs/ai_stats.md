# AI Stats

This report summarizes the published benchmark in
`docs/benchmarks/current_evaluation.csv`, generated from `SimplePokemonMoveEnv`.

## Current Result

The trained PPO v3 policy uses a richer 46-feature observation that includes
per-move expected damage, STAB, type advantage, finish ranges, recovery, setup,
and status flags. On rich mechanics, win rate counts actual simulated KOs and
unfinished episodes are tracked as draws.

The v4 experiment added mixed-opponent training and anti-stall reward shaping,
but it did not beat v3 on either evaluation seed. The default live PPO model
therefore remains `ppo_move_selection_v3_rich.zip`.

## Benchmark Table

| Scenario | Policy | Episodes | Record | Win rate | Avg reward | Avg turns |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Rich/type-aware seed 42 | Trained PPO v3 | 1000 | 246-410-344 | 24.6% | +0.182 | 7.76 |
| Rich/type-aware seed 42 | Type aware | 1000 | 236-398-366 | 23.6% | +0.304 | 6.67 |
| Rich/type-aware seed 42 | Experimental PPO v4 | 1000 | 233-426-341 | 23.3% | +0.193 | 6.92 |
| Rich/type-aware seed 99 | Trained PPO v3 | 1000 | 240-421-339 | 24.0% | +0.159 | 7.76 |
| Rich/type-aware seed 99 | Type aware | 1000 | 228-409-363 | 22.8% | +0.277 | 6.70 |
| Rich/type-aware seed 99 | Experimental PPO v4 | 1000 | 225-436-339 | 22.5% | +0.168 | 6.95 |

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
python scripts/train_ppo.py --timesteps 200000 --seed 47 --mechanics rich --observation-mode rich --opponent-policy mixed --output models/ppo_move_selection_v4_rich.zip
python scripts/evaluate_model.py --episodes 1000 --seed 42 --mechanics rich --opponent-policy type_aware --model models/ppo_move_selection_v2_typed.zip --model models/ppo_move_selection_v3_rich.zip --model models/ppo_move_selection_v4_rich.zip --output results/evaluation_v2_v3_v4_rich_seed42.csv
python scripts/evaluate_model.py --episodes 1000 --seed 99 --mechanics rich --opponent-policy type_aware --model models/ppo_move_selection_v2_typed.zip --model models/ppo_move_selection_v3_rich.zip --model models/ppo_move_selection_v4_rich.zip --output results/evaluation_v2_v3_v4_rich_seed99.csv
```

Model artifacts: `ppo_move_selection_v2_typed.zip`, `ppo_move_selection_v3_rich.zip`, `ppo_move_selection_v4_rich.zip`  
Evaluation CSV timestamp: `2026-06-24`

These numbers benchmark the experimental simulator policy. The live browser
player can use the trained model with `showdownrl live --policy ppo`, with a
heuristic fallback if the checkpoint is missing or selects an unavailable move.
Live ladder performance should still be tracked separately because real
Pokemon Showdown battles include mechanics beyond this move-selection simulator.
