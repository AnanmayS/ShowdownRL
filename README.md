# ShowdownRL

[![CI](https://github.com/AnanmayS/ShowdownRL/actions/workflows/ci.yml/badge.svg)](https://github.com/AnanmayS/ShowdownRL/actions/workflows/ci.yml)

Watch an AI play Pokemon Showdown in a visible browser.

ShowdownRL opens Pokemon Showdown, signs in with your account or a guest name,
queues a Random Battle, and visibly clicks moves while you watch. It can also
save a WebM recording of the battle.

## Current AI Benchmark

The default trained simulator policy is `ppo_move_selection_v3_rich.zip`. It
extends the original 14-feature PPO input with per-move context for expected
damage, STAB, type advantage, finish ranges, recovery, setup, and status moves.
The v4 experiment added mixed-opponent training and anti-stall reward shaping,
but v3 still has the best KO win rate on the current rich-mechanics benchmark.

| Scenario | Policy | Episodes | Record | Win rate | Avg reward | Avg turns |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Rich/type-aware seed 42 | Trained PPO v3 | 1000 | 246-410-344 | 24.6% | +0.182 | 7.76 |
| Rich/type-aware seed 42 | Type aware | 1000 | 236-398-366 | 23.6% | +0.304 | 6.67 |
| Rich/type-aware seed 42 | Experimental PPO v4 | 1000 | 233-426-341 | 23.3% | +0.193 | 6.92 |
| Rich/type-aware seed 99 | Trained PPO v3 | 1000 | 240-421-339 | 24.0% | +0.159 | 7.76 |
| Rich/type-aware seed 99 | Type aware | 1000 | 228-409-363 | 22.8% | +0.277 | 6.70 |
| Rich/type-aware seed 99 | Experimental PPO v4 | 1000 | 225-436-339 | 22.5% | +0.168 | 6.95 |

Records are shown as wins-losses-draws. See
[docs/benchmarks/current_evaluation.csv](docs/benchmarks/current_evaluation.csv)
and [docs/model_leaderboard.md](docs/model_leaderboard.md) for the side-by-side
benchmark data.

## Install

For the first public version, install directly from GitHub with `pipx`:

```bash
pipx install "git+https://github.com/AnanmayS/ShowdownRL.git"
```

For local development from this folder:

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

## First Run

Run the setup wizard:

```bash
showdownrl setup
```

Setup will:

- install Playwright Chromium
- ask for your Pokemon Showdown username and password
- save credentials locally in `~/Library/Application Support/ShowdownRL/config.env`

Your password is only sent to Pokemon Showdown during login. It is not uploaded
anywhere else by ShowdownRL.

Use guest mode instead of a password:

```bash
showdownrl setup --guest
```

## Check Everything

Before queueing a battle:

```bash
showdownrl check
```

To only verify the public website controls without logging in:

```bash
showdownrl check --skip-login
```

## Watch the AI Play

```bash
showdownrl live
```

Useful options:

```bash
# Stop after login, before queueing
showdownrl live --login-only

# Run without recording
showdownrl live --no-record

# Slow the clicks down
showdownrl live --slow-mo-ms 500 --click-delay 1.25

# Limit the battle loop for a smoke test
showdownrl live --max-turns 3

# Play more than one battle in the same run
showdownrl live --max-battles 3

# Stop a long session after 30 minutes
showdownrl live --max-battles 50 --max-time 30

# Show move scores while the AI is choosing
showdownrl live --debug-policy

# Try the trained PPO move selector, falling back to the heuristic if needed
showdownrl live --policy ppo

# Use a specific PPO checkpoint
showdownrl live --policy ppo --model-path models/ppo_move_selection_v3_rich.zip

# Do not write local battle stats
showdownrl live --no-stats
```

Recordings are saved to `~/Movies/ShowdownRL/` when installed normally. If you
run from this repository folder, recordings are saved to `results/`.

## Live Stats

`showdownrl live` writes local battle stats by default. Stats are stored on your
machine only and are not uploaded by ShowdownRL.

Print a terminal summary:

```bash
showdownrl stats
```

Generate a local HTML report:

```bash
showdownrl stats --html
showdownrl stats --open
showdownrl stats --trend
```

Filter the report:

```bash
showdownrl stats --since 2026-06-23
showdownrl stats --format "Random Battle"
```

Stats are saved under the local app data directory, separate from the config
file that stores credentials. You can override the location for a run:

```bash
showdownrl live --stats-dir ./my-stats
showdownrl stats --stats-dir ./my-stats
```

## Account and Privacy

Delete saved local credentials:

```bash
showdownrl logout
```

Print diagnostics without exposing secrets:

```bash
showdownrl doctor
```

You can also use environment variables for one-off overrides:

```bash
PS_USERNAME=your_name PS_PASSWORD=your_password showdownrl live
```

Battle logs do not store your password. They include local-only battle metadata
such as result, turns, selected moves, forced switches, policy source, rating
when it can be detected from the page, errors, and video path.
When `--debug-policy` is used, ShowdownRL also saves local redacted turn-state
snapshots under the stats directory so you can inspect what the AI saw before
clicking.

## Troubleshooting

- Missing Chromium: run `showdownrl setup`.
- Missing credentials: run `showdownrl setup` or use `showdownrl live --guest --username SomeGuestName`.
- Login failed: run `showdownrl logout && showdownrl setup`.
- Website controls changed: run `showdownrl doctor`, then `showdownrl check --skip-login`.
- Stats look empty: play a full battle with `showdownrl live`, then run `showdownrl stats`.

## Developer Notes

The current public CLI focuses on the live AI player for
`https://play.pokemonshowdown.com/`.

The repository also contains experimental reinforcement-learning scripts under
`scripts/` and helper modules in `showdownrl/`:

```bash
pip install -e ".[rl]"
python scripts/smoke_test.py
python scripts/train_ppo.py --timesteps 2048 --opponent-policy type_aware
python scripts/evaluate_model.py --episodes 100 --opponent-policy type_aware
python scripts/generate_ai_stats.py
PYTHONPATH=. python -m unittest discover -s tests
```

Those training/evaluation workflows are not part of the v1 nontechnical user
flow yet.
