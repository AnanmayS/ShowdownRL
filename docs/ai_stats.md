# AI Stats

This report summarizes the published benchmark in
`docs/benchmarks/current_evaluation.csv`, generated from `SimplePokemonMoveEnv`.

## Current Result

The MaskablePPO v6 policy uses a richer 46-feature observation that includes
per-move expected damage, STAB, type advantage, finish ranges, recovery, setup,
and status flags. It also receives state-dependent action masks that filter
tactical no-op moves during training and evaluation. On rich mechanics, win
rate counts actual simulated KOs and unfinished episodes are tracked as draws.

The v6 checkpoint beats v5 on both fixed evaluation seeds, so the default live
PPO model is now `maskable_ppo_move_selection_v6_rich.zip`.

## Benchmark Table

| Scenario | Policy | Episodes | Record (W-L-D) | Win rate | Avg reward | Avg turns |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Rich/type-aware seed 42 | Maskable PPO v6 | 1000 | 368-327-305 | 36.8% | +0.569 | 8.01 |
| Rich/type-aware seed 42 | Fine-tuned PPO v5 | 1000 | 338-351-311 | 33.8% | +0.441 | 7.81 |
| Rich/type-aware seed 42 | Trained PPO v3 | 1000 | 246-410-344 | 24.6% | +0.182 | 7.76 |
| Rich/type-aware seed 42 | Type aware | 1000 | 236-398-366 | 23.6% | +0.304 | 6.67 |
| Rich/type-aware seed 42 | Experimental PPO v4 | 1000 | 233-426-341 | 23.3% | +0.193 | 6.92 |
| Rich/type-aware seed 99 | Maskable PPO v6 | 1000 | 354-341-305 | 35.4% | +0.526 | 8.06 |
| Rich/type-aware seed 99 | Fine-tuned PPO v5 | 1000 | 324-365-311 | 32.4% | +0.405 | 7.82 |
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
python scripts/regenerate_benchmarks.py
```

To smoke-test the benchmark command generation without running RL evaluation:

```bash
python scripts/regenerate_benchmarks.py --dry-run --episodes 2
```

Model artifacts: `ppo_move_selection_v2_typed.zip`, `ppo_move_selection_v3_rich.zip`, `ppo_move_selection_v4_rich.zip`, `ppo_move_selection_v5_rich_finetuned.zip`, `maskable_ppo_move_selection_v6_rich.zip`

Evaluation CSV timestamp: `2026-06-24`

These numbers benchmark the experimental simulator policy. The live browser
player can use the trained model with `showdownrl live --policy ppo`, with a
heuristic fallback if the checkpoint is missing or selects an unavailable move.
Live ladder performance should still be tracked separately because real
Pokemon Showdown battles include mechanics beyond this move-selection simulator.
