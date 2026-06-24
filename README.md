# ShowdownRL

Watch an AI play Pokemon Showdown in a visible browser.

ShowdownRL opens Pokemon Showdown, signs in with your account or a guest name,
queues a Random Battle, and visibly clicks moves while you watch. It can also
save a WebM recording of the battle.

## Current AI Benchmark

The default trained simulator policy is `ppo_move_selection_v3_rich.zip`. It
extends the original 14-feature PPO input with per-move context for expected
damage, STAB, type advantage, finish ranges, recovery, setup, and status moves.
Across two 1000-episode rich-mechanics benchmarks, v3 beat the typed v2
checkpoint on both seeds while remaining slightly behind the hand-coded
type-aware baseline.

![ShowdownRL policy benchmark](docs/assets/ai_policy_comparison.png)

| Scenario | Policy | Episodes | Record | Win rate | Avg reward | Avg turns |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Rich/type-aware seed 42 | Type aware | 1000 | 608-392 | 60.8% | +0.237 | 6.67 |
| Rich/type-aware seed 42 | Trained PPO v3 | 1000 | 597-403 | 59.7% | +0.205 | 7.76 |
| Rich/type-aware seed 42 | Trained PPO v2 | 1000 | 578-422 | 57.8% | +0.165 | 6.64 |
| Rich/type-aware seed 99 | Type aware | 1000 | 596-404 | 59.6% | +0.213 | 6.70 |
| Rich/type-aware seed 99 | Trained PPO v3 | 1000 | 585-415 | 58.5% | +0.182 | 7.76 |
| Rich/type-aware seed 99 | Trained PPO v2 | 1000 | 570-430 | 57.0% | +0.147 | 6.67 |
| Typed/type-aware | Type aware | 1000 | 747-253 | 74.7% | +0.498 | 5.08 |
| Typed/type-aware | Trained PPO v2 | 1000 | 717-283 | 71.7% | +0.422 | 5.11 |
| Typed/type-aware | Trained PPO v1 | 1000 | 404-596 | 40.4% | -0.387 | 5.28 |
| Toy/random | Type aware | 1000 | 933-67 | 93.3% | +1.132 | 5.29 |
| Toy/random | Trained PPO v2 | 1000 | 923-77 | 92.3% | +1.105 | 5.33 |
| Toy/random | Trained PPO v1 | 1000 | 729-271 | 72.9% | +0.568 | 6.18 |

The type-aware baseline is still slightly stronger on the richer benchmark, so
the next training goal is to beat that baseline consistently. See
[docs/benchmarks/current_evaluation.csv](docs/benchmarks/current_evaluation.csv)
for the side-by-side benchmark data.

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
