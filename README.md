# ShowdownRL

[![CI](https://github.com/AnanmayS/ShowdownRL/actions/workflows/ci.yml/badge.svg)](https://github.com/AnanmayS/ShowdownRL/actions/workflows/ci.yml)

Watch an AI play Pokemon Showdown in a visible browser.

## About

ShowdownRL is a local command-line tool for watching an automated Pokemon
Showdown player in action. It opens the real Pokemon Showdown website, signs in
with your account or a guest name, queues a Random Battle, and clicks moves in a
visible browser so you can follow every decision.

Decisions are made from the raw battle protocol the web client receives, not
from scraped page text: ShowdownRL records the battle room's messages, rebuilds
the battle with poke-env's own parser, and feeds the same features used in
training (`showdownrl/battle_features.py`) to the policy. By default the policy is
a search agent. It runs poke-engine MCTS over sampled guesses of the opponent's
hidden sets and falls back to a MaskablePPO model trained on the real simulator,
then to a damage-calc heuristic. It can pick any legal action: moves, voluntary
switches, and Terastallization.
It can also save WebM recordings, write local battle stats, and
generate local reports for comparing runs over time. Credentials, battle logs,
debug snapshots, recordings, and stats stay on your machine unless you choose to
share them.

<video src="docs/assets/demo_battle.webm" controls width="100%" poster="docs/assets/ai_policy_comparison.png">
  Your browser doesn't support the video tag. <a href="docs/assets/demo_battle.webm">Download the demo video</a>.
</video>

## Current AI Benchmark

All numbers below come from the **real Pokemon Showdown simulator**: a local
server running Gen 9 Random Battles, driven through poke-env. They are not from a
simplified environment. Each row is 1,000 battles, and ranges are Wilson 95%
confidence intervals. See [docs/real_simulator.md](docs/real_simulator.md) for
the training pipeline.

| Agent | vs poke-env SimpleHeuristics | vs ShowdownRL smart heuristic | vs Foul Play (100 ms search) |
| --- | ---: | ---: | ---: |
| Smart heuristic (`showdownrl/smart_heuristic.py`) | 61.9% (2,000 battles) | 50% (itself) | 2/50 |
| BC from the smart heuristic (`bc_smart`) | 61.8% (58.7-64.8) | 49.8% (46.7-52.9) | - |
| BC distilled from Foul Play (`bc_fp_r2`) | 71.3% (68.4-74.0) | 62.4% (59.4-65.3) | 3/40 (7.5%) |
| **Search** (`--policy search`: poke-engine MCTS, 4 sampled opponent sets, `bc_fp_r2` fallback) | **91.7% (88.0-94.3)**, 300 battles | **85.0% (80.5-88.6)**, 300 battles | **29/58 (50.0%, 37.5-62.5)** |

Search runs MCTS on [poke-engine](https://github.com/pmariglia/poke-engine)
over several sampled guesses of the opponent's hidden sets, drawn from the
Gen 9 Random Battle set data. The benchmark used 50 ms per sample against the
heuristics and 100 ms per sample against Foul Play. At that budget it plays
evenly with Foul Play, the strongest open-source Random Battle bot. Every
plain-network policy we trained loses to Foul Play more than 90% of the time.

The models trained in the older simplified environment (`maskable_ppo_v11` and
later) use a different observation. They cannot play real battles, and their
simplified-env scores do not transfer. See
[docs/model_leaderboard.md](docs/model_leaderboard.md) for that history.

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

# Save a redacted state snapshot for every decision
showdownrl live --debug-policy

# Default: MCTS search (needs the `search` extra: pip install -e ".[search]").
# More samples / time per sample play stronger but slower.
showdownrl live --policy search --search-samples 4 --search-time-ms 100

# Use the newest real-simulator model in models/real/ without search, falling
# back to the damage-calc heuristic if none loads
showdownrl live --policy ppo

# Use a specific MaskablePPO checkpoint trained on battle_features
showdownrl live --policy ppo --model-path models/real/my_model.zip

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
python scripts/train_ppo.py --timesteps 2048 --mechanics rich --observation-mode rich --opponent-policy type_aware --output models/ppo_smoke.zip
python scripts/evaluate_model.py --episodes 100 --mechanics rich --opponent-policy type_aware --model models/ppo_smoke.zip
python scripts/regenerate_benchmarks.py --dry-run --episodes 2
python scripts/run_experiments.py --dry-run --timesteps 2048 --episodes 20
python scripts/run_experiments.py --dry-run --tune-trials 4 --timesteps 2048 --episodes 20
PYTHONPATH=. python -m unittest discover -s tests
```

Those training/evaluation workflows are not part of the v1 nontechnical user
flow yet.
