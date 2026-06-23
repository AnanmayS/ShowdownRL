# ShowdownRL

Watch an AI play Pokemon Showdown in a visible browser.

ShowdownRL opens Pokemon Showdown, signs in with your account or a guest name,
queues a Random Battle, and visibly clicks moves while you watch. It can also
save a WebM recording of the battle.

## Current AI Benchmark

The current trained simulator policy is `ppo_move_selection_v1.zip`. In the
latest local benchmark, the trained PPO policy won **34 of 50 episodes** for a
**68% win rate**.

![ShowdownRL policy benchmark](docs/assets/ai_policy_comparison.png)

| Policy | Episodes | Record | Win rate | Avg reward | Avg turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| Type aware | 50 | 46-4 | 92% | +1.103 | 5.16 |
| Max damage | 50 | 37-13 | 74% | +0.605 | 6.12 |
| Trained PPO | 50 | 34-16 | 68% | +0.484 | 5.94 |
| Random | 50 | 32-18 | 64% | +0.331 | 6.96 |

The simple type-aware baseline is still the strongest policy in this simulator,
so the next training goal is to beat that baseline. See
[docs/ai_stats.md](docs/ai_stats.md) for the full report and regeneration
commands.

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

# Show move scores while the AI is choosing
showdownrl live --debug-policy

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
such as result, turns, selected moves, forced switches, errors, and video path.

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
python scripts/train_ppo.py --timesteps 2048
python scripts/evaluate_model.py --episodes 100
python scripts/generate_ai_stats.py
PYTHONPATH=. python -m unittest discover -s tests
```

Those training/evaluation workflows are not part of the v1 nontechnical user
flow yet.
