# ShowdownRL Dashboard

A lightweight React dashboard for battle history and analytics.

## Features

- **Win Rate Over Time** — Rolling win-rate chart from battle logs
- **Matchup Win Rates** — Win rates broken down by opponent team archetype (random, max damage, type aware)
- **Common Misplays** — Worst-performing move selections (moves over-represented in losses)
- **Policy Version Comparison** — A/B view of all trained models from benchmark CSV
- **Battle Replay Viewer** — Play back saved battle logs with turn-by-turn detail

## Getting Started

```bash
cd dashboard
npm install
npm run dev
```

This starts a local dev server. Open http://localhost:5173 in your browser.

## Loading Data

1. **Benchmark data** (built-in): The dashboard loads `docs/benchmarks/current_evaluation.csv` automatically. You can also upload a custom CSV.
2. **Battle logs**: Upload a `battles.jsonl` file from your ShowdownRL stats directory:
   - macOS: `~/Library/Application Support/ShowdownRL/stats/battles.jsonl`
   - Linux: `~/.local/share/showdownrl/stats/battles.jsonl`

## Building for Static Deployment

```bash
npm run build
```

Output goes to `dashboard/dist/`. Serve the `dist/` folder with any static file server:

```bash
npx serve dist
```

## Tech Stack

- React 18
- Vite 5
- Recharts (charting)
- Pure CSS (no CSS frameworks)
