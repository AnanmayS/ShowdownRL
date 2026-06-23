# AI Stats

This report summarizes the published benchmark in
`docs/benchmarks/current_evaluation.csv`, generated from
`results/evaluation.csv` and `SimplePokemonMoveEnv`.

![ShowdownRL policy benchmark](assets/ai_policy_comparison.png)

## Current Result

The trained PPO policy won 34 of 50 episodes (68%) with an average reward of 0.484.

The strongest policy in this run was **Type aware** at
92% win rate. That means the trained model is available and
working, but the simple hand-written type-aware baseline is still the bar to
beat in this simulator.

## Benchmark Table

| Policy | Episodes | Record | Win rate | Avg reward | Avg turns | Reward / turn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Type aware | 50 | 46-4 | 92% | +1.103 | 5.16 | +0.2138 |
| Max damage | 50 | 37-13 | 74% | +0.605 | 6.12 | +0.0988 |
| Trained PPO | 50 | 34-16 | 68% | +0.484 | 5.94 | +0.0814 |
| Random | 50 | 32-18 | 64% | +0.331 | 6.96 | +0.0475 |

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
python scripts/evaluate_model.py --episodes 50
python scripts/generate_ai_stats.py
```

Model artifact: `ppo_move_selection_v1.zip (0.2 MB, local artifact)`  
Evaluation CSV timestamp: `2026-06-23`

These numbers benchmark the experimental simulator policy. The live browser
player currently uses a lightweight move-scoring policy for the official
Pokemon Showdown website, so live ladder performance should be tracked
separately once battle logging is added.
