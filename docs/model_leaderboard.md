# Model Leaderboard

Generated from `docs/benchmarks/current_evaluation.csv`.

| Scenario | Policy | Episodes | Record (W-L-D) | Win rate | Non-loss | Avg reward | Avg turns | Git SHA |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| rich_type_aware_seed42 | Maskable PPO v6 | 1000 | 565-427-8 | 56.5% | 57.3% | +0.338 | 6.51 | 3054469 |
| rich_type_aware_seed42 | PPO v5 fine-tuned | 1000 | 527-470-3 | 52.7% | 53.0% | +0.222 | 6.16 | 3054469 |
| rich_type_aware_seed42 | PPO v3 rich | 1000 | 498-499-3 | 49.8% | 50.1% | +0.116 | 6.23 | 3054469 |
| rich_type_aware_seed42 | Type aware | 1000 | 494-506-0 | 49.4% | 49.4% | +0.149 | 5.58 | 3054469 |
| rich_type_aware_seed42 | PPO v4 rich | 1000 | 492-508-0 | 49.2% | 49.2% | +0.122 | 5.71 | 3054469 |
| rich_type_aware_seed42 | PPO v2 typed | 1000 | 487-513-0 | 48.7% | 48.7% | +0.122 | 5.57 | 3054469 |
| rich_type_aware_seed42 | Max damage | 1000 | 362-638-0 | 36.2% | 36.2% | -0.327 | 5.88 | 3054469 |
| rich_type_aware_seed42 | Random | 1000 | 306-686-8 | 30.6% | 31.4% | -0.714 | 7.67 | 3054469 |
| rich_type_aware_seed99 | Maskable PPO v6 | 1000 | 559-432-9 | 55.9% | 56.8% | +0.312 | 6.63 | 3054469 |
| rich_type_aware_seed99 | PPO v5 fine-tuned | 1000 | 523-473-4 | 52.3% | 52.7% | +0.206 | 6.17 | 3054469 |
| rich_type_aware_seed99 | PPO v3 rich | 1000 | 494-502-4 | 49.4% | 49.8% | +0.102 | 6.25 | 3054469 |
| rich_type_aware_seed99 | Type aware | 1000 | 488-511-1 | 48.8% | 48.9% | +0.126 | 5.63 | 3054469 |
| rich_type_aware_seed99 | PPO v4 rich | 1000 | 485-514-1 | 48.5% | 48.6% | +0.097 | 5.77 | 3054469 |
| rich_type_aware_seed99 | PPO v2 typed | 1000 | 482-517-1 | 48.2% | 48.3% | +0.101 | 5.62 | 3054469 |
| rich_type_aware_seed99 | Max damage | 1000 | 359-641-0 | 35.9% | 35.9% | -0.342 | 5.89 | 3054469 |
| rich_type_aware_seed99 | Random | 1000 | 285-707-8 | 28.5% | 29.3% | -0.780 | 7.80 | 3054469 |

Full benchmark refresh:

```bash
pip install -e ".[rl]"
python scripts/regenerate_benchmarks.py
```
