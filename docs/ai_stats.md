# AI Stats

This report summarizes the published benchmark in
`docs/benchmarks/current_evaluation.csv`, generated from `SimplePokemonMoveEnv`.

![ShowdownRL policy benchmark](assets/ai_policy_comparison.png)

## Current Result

The default trained bench-simulator policy is `maskable_ppo_v11_conservative_3M.zip`.
Across the fixed rich/type-aware evaluation seeds it recorded
1578-341-81 (W-L-D) over 2000 episodes:
78.9% win rate, 83.0% non-loss rate,
and +2.208 average reward.

The strongest policy in the corrected simulator benchmark is
**Maskable PPO v11 (bench simulator)** at 78.9% win rate.

## Benchmark Table

| Scenario | Policy | Episodes | Record (W-L-D) | Win rate | Non-loss | Avg reward | Avg turns |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rich/type-aware seed 42 | Maskable PPO v11 (bench simulator) | 1000 | 790-169-41 | 79.0% | 83.1% | +2.211 | 22.93 |
| Rich/type-aware seed 42 | Type aware | 1000 | 752-222-26 | 75.2% | 77.8% | +1.905 | 22.10 |
| Rich/type-aware seed 42 | Max damage | 1000 | 542-429-29 | 54.2% | 57.1% | +0.569 | 24.37 |
| Rich/type-aware seed 42 | Random | 1000 | 319-544-137 | 31.9% | 45.6% | -1.822 | 31.64 |
| Rich/type-aware seed 99 | Maskable PPO v11 (bench simulator) | 1000 | 788-172-40 | 78.8% | 82.8% | +2.205 | 22.83 |
| Rich/type-aware seed 99 | Type aware | 1000 | 753-222-25 | 75.3% | 77.8% | +1.900 | 22.00 |
| Rich/type-aware seed 99 | Max damage | 1000 | 545-428-27 | 54.5% | 57.2% | +0.577 | 24.25 |
| Rich/type-aware seed 99 | Random | 1000 | 323-522-155 | 32.3% | 47.8% | -1.806 | 31.98 |

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

Benchmark Git SHA(s): `6d57ea6`
Evaluation CSV timestamp: `2026-06-25`

These numbers benchmark the experimental simulator policy. The live browser
player can use the trained model with `showdownrl live --policy ppo`, with a
heuristic fallback if the checkpoint is missing or selects an unavailable move.
Live ladder performance should still be tracked separately because real
Pokemon Showdown battles include mechanics beyond this move-selection simulator.
