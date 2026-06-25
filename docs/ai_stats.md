# AI Stats

This report summarizes the published benchmark in
`docs/benchmarks/current_evaluation.csv`, generated from `SimplePokemonMoveEnv`.

![ShowdownRL policy benchmark](assets/ai_policy_comparison.png)

## Current Result

The default trained simulator policy is `maskable_ppo_move_selection_v6_rich.zip`.
Across the fixed rich/type-aware evaluation seeds it recorded
1124-859-17 (W-L-D) over 2000 episodes:
56.2% win rate, 57.0% non-loss rate,
and +0.325 average reward.

The strongest policy in the corrected simulator benchmark is
**Maskable PPO v6** at 56.2% win rate.

## Benchmark Table

| Scenario | Policy | Episodes | Record (W-L-D) | Win rate | Non-loss | Avg reward | Avg turns |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rich/type-aware seed 42 | Maskable PPO v6 | 1000 | 565-427-8 | 56.5% | 57.3% | +0.338 | 6.51 |
| Rich/type-aware seed 42 | PPO v5 fine-tuned | 1000 | 527-470-3 | 52.7% | 53.0% | +0.222 | 6.16 |
| Rich/type-aware seed 42 | PPO v3 rich | 1000 | 498-499-3 | 49.8% | 50.1% | +0.116 | 6.23 |
| Rich/type-aware seed 42 | Type aware | 1000 | 494-506-0 | 49.4% | 49.4% | +0.149 | 5.58 |
| Rich/type-aware seed 42 | PPO v4 rich | 1000 | 492-508-0 | 49.2% | 49.2% | +0.122 | 5.71 |
| Rich/type-aware seed 42 | PPO v2 typed | 1000 | 487-513-0 | 48.7% | 48.7% | +0.122 | 5.57 |
| Rich/type-aware seed 42 | Max damage | 1000 | 362-638-0 | 36.2% | 36.2% | -0.327 | 5.88 |
| Rich/type-aware seed 42 | Random | 1000 | 306-686-8 | 30.6% | 31.4% | -0.714 | 7.67 |
| Rich/type-aware seed 99 | Maskable PPO v6 | 1000 | 559-432-9 | 55.9% | 56.8% | +0.312 | 6.63 |
| Rich/type-aware seed 99 | PPO v5 fine-tuned | 1000 | 523-473-4 | 52.3% | 52.7% | +0.206 | 6.17 |
| Rich/type-aware seed 99 | PPO v3 rich | 1000 | 494-502-4 | 49.4% | 49.8% | +0.102 | 6.25 |
| Rich/type-aware seed 99 | Type aware | 1000 | 488-511-1 | 48.8% | 48.9% | +0.126 | 5.63 |
| Rich/type-aware seed 99 | PPO v4 rich | 1000 | 485-514-1 | 48.5% | 48.6% | +0.097 | 5.77 |
| Rich/type-aware seed 99 | PPO v2 typed | 1000 | 482-517-1 | 48.2% | 48.3% | +0.101 | 5.62 |
| Rich/type-aware seed 99 | Max damage | 1000 | 359-641-0 | 35.9% | 35.9% | -0.342 | 5.89 |
| Rich/type-aware seed 99 | Random | 1000 | 285-707-8 | 28.5% | 29.3% | -0.780 | 7.80 |

## What These Stats Mean

- **Win rate** is the clearest headline metric.
- **Non-loss rate** counts wins plus simulator draws.
- **Average reward** captures how decisively the policy wins or loses in the
  simulator's reward function.
- **Average turns** is useful as an efficiency signal, but lower is only better
  when win rate and reward remain strong.

## Reproducing

```bash
pip install -e ".[rl]"
python scripts/regenerate_benchmarks.py
python scripts/generate_ai_stats.py
```

Benchmark Git SHA(s): `3054469`
Evaluation CSV timestamp: `2026-06-24`

These numbers benchmark the experimental simulator policy. The live browser
player can use the trained model with `showdownrl live --policy ppo`, with a
heuristic fallback if the checkpoint is missing or selects an unavailable move.
Live ladder performance should still be tracked separately because real
Pokemon Showdown battles include mechanics beyond this move-selection simulator.
