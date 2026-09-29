# Training on the real Showdown simulator

Earlier ShowdownRL models were trained in `showdownrl/simple_env.py`, a
simplified 4v4 duel. That environment was a different game from real Pokemon
Showdown. For example, the agent always moved first, and type matchups were
never recomputed after a switch. Its win rates did not carry over to the site.
The current pipeline trains and evaluates against a **local Pokemon Showdown
server** through poke-env, in Gen 9 Random Battles.

## Pieces

| File | Role |
| --- | --- |
| `showdownrl/battle_features.py` | 273-feature observation built from a poke-env `Battle`. Includes damage estimates, speed order, KO flags, hazards, and opponent sets estimated from `data/gen9randombattle.json`. |
| `showdownrl/real_env.py` | Gymnasium env (`MaskedShowdownEnv`) wrapping poke-env's `SinglesEnv` with the 26-action gen9 mask, plus a mixed opponent pool and self-play snapshots. |
| `showdownrl/smart_heuristic.py` | Damage-calc heuristic. Serves as the live fallback, a behaviour-cloning teacher, and an evaluation opponent. |
| `showdownrl/protocol_battle.py` | Rebuilds the same poke-env `Battle` from the web client's raw protocol, so live play sees the exact features used in training. |
| `scripts/train_real.py` | `collect` (heuristic teacher data), `bc` (behaviour cloning), `ppo` (MaskablePPO fine-tuning with a KL anchor and self-play league). |
| `scripts/foulplay_teacher.py` | Records [Foul Play](https://github.com/pmariglia/foul-play)'s decisions as BC data in our feature and action space. |
| `scripts/eval_real.py` | Win rate vs poke-env baselines with Wilson 95% intervals. |
| `scripts/eval_foulplay.py` | Win rate vs a locally run Foul Play. |

## Setup

```bash
git clone https://github.com/smogon/pokemon-showdown.git ~/pokemon-showdown
cd ~/pokemon-showdown && npm install && cp config/config-example.js config/config.js
node pokemon-showdown start --no-security 8000   # more ports: 8001, 8002, ...
```

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[rl]"
```

## Recipe

```bash
# 1. Teacher data from Foul Play (see scripts/foulplay_teacher.py for setup)
python scripts/foulplay_teacher.py collect --battles 3000 --instances 3 --port 8000 \
  --opponents smart,heuristic,policy,policy,smart,max_power,random \
  --policy models/real/bc_smart.zip --search-time-ms 100 --search-backend thread \
  --out models/real/foulplay_teacher.npz

# 2. Behaviour cloning (warm start from the heuristic-teacher policy)
python scripts/train_real.py collect --battles 40000 --epsilon 0.1 --out models/real/bc_smart_40k.npz
python scripts/train_real.py bc --data models/real/bc_smart_40k.npz --out models/real/bc_smart.zip
python scripts/train_real.py bc --data models/real/foulplay_teacher.npz --init models/real/bc_smart.zip \
  --bc-epochs 20 --bc-lr 5e-4 --out models/real/bc_foulplay.zip

# 3. Evaluate (1000 battles per opponent)
python scripts/eval_real.py --agent models/real/bc_foulplay.zip --opponents random,max_power,heuristic,smart --n 1000
```

## What we learned

- **The real-simulator heuristic is a strong baseline.** The smart heuristic
  beats poke-env's `SimpleHeuristicsPlayer` about 62% of the time.
- **PPO fine-tuning barely moved the policy.** Starting from the BC policy, 1.3M
  steps with a KL anchor and a self-play league did not improve on BC within
  the confidence interval. PPO on this game needs far more samples than a
  single machine produces in hours (published agents use hundreds of millions).
- **Imitating a search bot works much better.** Foul Play (MCTS on poke-engine)
  wins more than 90% against our heuristics. Cloning its decisions produced the
  largest single gain.
- **Evaluate with at least 1000 battles.** At 1000 battles the 95% interval is
  about ±3 points. Smaller runs can't separate most of the changes above.
