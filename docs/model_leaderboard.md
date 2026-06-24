# Model Leaderboard

This leaderboard tracks trained PPO checkpoints against fixed simulator
benchmarks. On rich mechanics, win rate counts actual simulated KOs; unfinished
episodes are tracked as draws. A model should beat the previous default PPO
checkpoint on at least two evaluation seeds before becoming the live default.

| Model | Mechanics | Train opponent | Train seed | Timesteps | Eval seed | Record (W-L-D) | Win rate | Non-loss | Avg reward | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `maskable_ppo_move_selection_v6_rich` | rich | type_aware | 246 | 200000 | 42 | 368-327-305 | 36.8% | 67.3% | +0.569 | default |
| `maskable_ppo_move_selection_v6_rich` | rich | type_aware | 246 | 200000 | 99 | 354-341-305 | 35.4% | 65.9% | +0.526 | default |
| `ppo_move_selection_v5_rich_finetuned` | rich | type_aware | 145 | 200000 | 42 | 338-351-311 | 33.8% | 64.9% | +0.441 | previous default |
| `ppo_move_selection_v5_rich_finetuned` | rich | type_aware | 145 | 200000 | 99 | 324-365-311 | 32.4% | 63.5% | +0.405 | previous default |
| `ppo_move_selection_v3_rich` | rich | type_aware | 45 | 300000 | 42 | 246-410-344 | 24.6% | 59.0% | +0.182 | previous default |
| `ppo_move_selection_v3_rich` | rich | type_aware | 45 | 300000 | 99 | 240-421-339 | 24.0% | 57.9% | +0.159 | previous default |
| `ppo_move_selection_v4_rich` | rich | mixed | 47 | 200000 | 42 | 233-426-341 | 23.3% | 57.4% | +0.193 | experimental |
| `ppo_move_selection_v4_rich` | rich | mixed | 47 | 200000 | 99 | 225-436-339 | 22.5% | 56.4% | +0.168 | experimental |
| `ppo_move_selection_v2_typed` | typed | type_aware | 42 | 100000 | 42 | 717-283 | 71.7% | n/a | +0.422 | archived baseline |

The v6 checkpoint trains MaskablePPO with state-dependent action masks for
tactical no-op moves. It beats v5's KO win rate, non-loss rate, and average
reward on both fixed rich/type-aware evaluation seeds, so
`maskable_ppo_move_selection_v6_rich.zip` is the default live checkpoint.
