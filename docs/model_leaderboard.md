# Model Leaderboard

This leaderboard tracks trained PPO checkpoints against fixed simulator
benchmarks. On rich mechanics, win rate counts actual simulated KOs; unfinished
episodes are tracked as draws. A model should beat the previous default PPO
checkpoint on at least two evaluation seeds before becoming the live default.

| Model | Mechanics | Train opponent | Train seed | Timesteps | Eval seed | Record | Win rate | Non-loss | Avg reward | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ppo_move_selection_v5_rich_finetuned` | rich | type_aware | 145 | 200000 | 42 | 338-351-311 | 33.8% | 64.9% | +0.441 | default |
| `ppo_move_selection_v5_rich_finetuned` | rich | type_aware | 145 | 200000 | 99 | 324-365-311 | 32.4% | 63.5% | +0.405 | default |
| `ppo_move_selection_v3_rich` | rich | type_aware | 45 | 300000 | 42 | 246-410-344 | 24.6% | 59.0% | +0.182 | previous default |
| `ppo_move_selection_v3_rich` | rich | type_aware | 45 | 300000 | 99 | 240-421-339 | 24.0% | 57.9% | +0.159 | previous default |
| `ppo_move_selection_v4_rich` | rich | mixed | 47 | 200000 | 42 | 233-426-341 | 23.3% | 57.4% | +0.193 | experimental |
| `ppo_move_selection_v4_rich` | rich | mixed | 47 | 200000 | 99 | 225-436-339 | 22.5% | 56.4% | +0.168 | experimental |
| `ppo_move_selection_v2_typed` | typed | type_aware | 42 | 100000 | 42 | 717-283 | 71.7% | n/a | +0.422 | archived baseline |

The v5 checkpoint fine-tunes the previous rich PPO default using vectorized
rollouts and deterministic evaluation checkpoints. It beats v3's KO win rate
and non-loss rate on both fixed rich/type-aware evaluation seeds, so
`ppo_move_selection_v5_rich_finetuned.zip` is the default live checkpoint.
