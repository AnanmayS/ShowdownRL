# Model Leaderboard

Generated from `docs/benchmarks/current_evaluation.csv`.

| Scenario | Policy | Episodes | Record (W-L-D) | Win rate | Non-loss | Avg reward | Avg turns | Git SHA |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| rich_type_aware_seed42 | Maskable PPO v11 (bench simulator) | 1000 | 790-169-41 | 79.0% | 83.1% | +2.211 | 22.93 | 6d57ea6 |
| rich_type_aware_seed42 | Type aware | 1000 | 752-222-26 | 75.2% | 77.8% | +1.905 | 22.10 | 6d57ea6 |
| rich_type_aware_seed42 | Max damage | 1000 | 542-429-29 | 54.2% | 57.1% | +0.569 | 24.37 | 6d57ea6 |
| rich_type_aware_seed42 | Random | 1000 | 319-544-137 | 31.9% | 45.6% | -1.822 | 31.64 | 6d57ea6 |
| rich_type_aware_seed99 | Maskable PPO v11 (bench simulator) | 1000 | 788-172-40 | 78.8% | 82.8% | +2.205 | 22.83 | 6d57ea6 |
| rich_type_aware_seed99 | Type aware | 1000 | 753-222-25 | 75.3% | 77.8% | +1.900 | 22.00 | 6d57ea6 |
| rich_type_aware_seed99 | Max damage | 1000 | 545-428-27 | 54.5% | 57.2% | +0.577 | 24.25 | 6d57ea6 |
| rich_type_aware_seed99 | Random | 1000 | 323-522-155 | 32.3% | 47.8% | -1.806 | 31.98 | 6d57ea6 |

Full benchmark refresh:

```bash
pip install -e ".[rl]"
python scripts/regenerate_benchmarks.py
```
