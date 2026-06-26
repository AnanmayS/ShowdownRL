# ShowdownRL: Training a Pokémon Battle AI with Reinforcement Learning

_A multi-part technical series on building a competitive Pokémon Showdown agent using PPO._

---

## Part 1: Observation Space Design — 106 Features and Why Each Matters

### Introduction

Building an RL agent for Pokémon battles means deciding what the agent sees. Unlike Atari games where pixels are the natural input, Pokémon's state is fundamentally symbolic: types, HP percentages, move properties, status conditions. How we encode this state into a fixed-size vector directly determines what the agent can learn.

ShowdownRL's observation space evolved through three major iterations: **simple** (14 floats), **rich** (46 floats), and **team-aware** (106 floats). Each iteration added critical information the agent needed to reach human-competitive play.

### The Simple Observation (v1–v2)

The first observation space encoded exactly 14 values:

```
[own_hp, opp_hp, move0_bp, move0_acc, move0_eff, move1_bp, move1_acc, move1_eff, ...]
```

- `own_hp` / `opp_hp`: Current HP fraction (0.0–1.0)
- For each of 4 moves: `base_power`, `accuracy`, `type_effectiveness`

This was enough for the agent to learn basic strategy — use high-damage moves, prefer super-effective hits — but it had fundamental blind spots:

1. **No move-type awareness.** The agent couldn't distinguish a Fire move from a Water move; it only saw the raw type-effectiveness multiplier against current opponent types.
2. **No understanding of non-attacking moves.** Recovery, status, and setup moves all looked identical to the agent (base_power=0, acc=1.0, eff=0).
3. **No team context.** With bench Pokémon absent from observations, the agent couldn't plan switches or consider type coverage across its team.

### The Rich Observation (v3–v5)

The rich observation added 8 derived features per move (the "why" behind each move):

```
[move_i features]:
  - raw_score:   base_power × accuracy × effectiveness (capped at 4.0)
  - stab:        does this move share a type with the active Pokémon? (0/1)
  - super_effective: is this move >1× effective? (0/1)
  - resisted:    is this move <1× effective? (0/1)
  - finish_flag: would expected damage KO the opponent? (0/1)
  - is_recovery: is this a healing move? (0/1)
  - is_setup:    is this a boosting move? (0/1)
  - is_status:   is this a status move? (0/1)
```

This was a major leap. The agent could now:

- **Know when to finish**: the `finish_flag` lets it recognize KO range
- **Understand move roles**: recovery moves have different value near 35% HP vs. 85%
- **Evaluate type matchups directly**: STAB and effectiveness flags abstract away the type chart math

With 4 moves × 8 rich features = 32 additional values (46 total), the agent trained on `rich` observations achieved a 24.6% win rate against the `type_aware` heuristic — still below the 75% baseline, but significantly better than the random-policy baseline of ~32%.

### The Team-Aware Observation (v6+)

The biggest gap remained: the agent had no awareness of its bench Pokémon. In real Pokémon battles, switching is a core strategic action. The team-aware observation added **20 features per bench slot** (up to 3 bench Pokémon):

```
[bench_i features]:
  - hp:          current HP fraction
  - type_flags:  18 one-hot booleans (one per Pokémon type)
  - has_recovery: does this bench Pokémon know recovery?
```

The 18 type flags are particularly important. They let the agent evaluate defensive type matchups — e.g., "my bench has a Water type that resists Fire, so I should switch to absorb the incoming Fire move." This brought the observation to 106 dimensions:

```
14 (simple) + 32 (rich) + 60 (3 bench × 20) = 106
```

### Design Rationale

| Feature | Why it's there |
|---|---|
| HP fractions | Normalized to [0,1] for neural network stability; makes damage relative |
| Base power | Raw offensive potential; needed even with multiplier to distinguish weak/strong moves |
| Accuracy | Critical for risk assessment (e.g., 70% accurate move vs 100%) |
| Type effectiveness | Compresses 18×18 type chart into one multiplier |
| STAB flag | Same-type attack bonus (1.5×) is a key strategic signal |
| Move role flags | Without these, recovery moves look like zero-damage useless actions |
| Finish flag | Bypasses the need to learn "damage > enemy HP → good" from scratch |
| Bench type flags | 18 one-hot encoding is sparse but complete; no false negatives |

### What Didn't Work

- **Raw type names as integers**: Encoding "Fire" as 3 and "Water" as 8 created spurious ordinal relationships. One-hot encoding was essential.
- **Omitting accuracy**: Early simple observations excluded accuracy because "all moves hit in the simulator." Once we added accuracy, the agent learned to prefer reliable moves.
- **Normalizing all features to [0,1]**: Some features like `raw_score` needed capping (at 4.0) rather than min-max normalization, because the distribution is heavily skewed toward low values.

---

## Part 2: Reward Engineering — What Worked, What Didn't, and the Dead Ends

### The Core Challenge

Reward design in Pokémon battles is deceptively hard. The terminal reward (win = +1, lose = −1) is sparse and binary — the agent plays 20+ turns before seeing any feedback. Dense shaping rewards are necessary but dangerous: a poorly shaped reward can teach the agent to grind HP without ever winning.

### Reward Components

#### Terminal Reward (Final)
```
win:  +1.0
lose: −1.0
timeout (no winner after max_turns): −0.75
```

The time-out penalty is critical. Without it, the agent could learn to stall indefinitely by alternating recovery moves, never winning but also never losing. The −0.75 is less punitive than a loss, but still strong enough to discourage passive play.

#### HP Delta Reward (Per Turn)
```
reward += opp_hp_decrease − own_hp_decrease − 0.01 (time penalty)
```

This is the engine of the shaping reward. Each turn, the agent is rewarded for dealing damage and penalized for taking it. The −0.01 per turn creates implicit pressure to end battles quickly.

**Why this works**: It creates a dense reward signal that correlates with winning. Every point of damage matters. The agent learns to maximize damage output and minimize incoming damage.

**Why it's risky**: If the reward coefficient is too high, the agent can learn to sac Pokémon for a few points of chip damage. We kept the coefficient at 1.0 (raw delta) to avoid this.

#### Move-Type Bonuses

| Action | Condition | Bonus | Penalty |
|---|---|---|---|
| Attack (finish) | Expected damage ≥ opponent HP | +0.18 | — |
| Recovery | Own HP ≤ 55% | +0.18 per effective heal | −0.12 if healthy |
| Setup | Own HP ≥ 45%, opponent ≥ 35%, boost < 1.8× | +0.03 | −0.10 if misused |
| Status | Opponent HP ≥ 35%, opponent boost > 0.45× | +0.02 | −0.08 if misused |
| Switch | — | — | −0.01 |

These bonuses were calibrated through extensive trial and error:

**The finishing bonus** (+0.18) was the single most impactful addition. Without it, the agent would sometimes choose a low-damage status move when a KO was guaranteed. The bonus is deliberately timed: it activates on the *perfect* expected-damage ≥ remaining-HP check, not retroactively. This taught the agent to recognize KO range.

**Recovery bonus**: Initially we rewarded recovery unconditionally (+0.1). The agent learned to spam Recovery every turn. Adding the HP threshold (≤55%) and the penalty for overhealing (using recovery above 55% HP) fixed this.

**Setup moves** were the hardest to tune. A single Swords Dance doubles damage output, but the turn spent setting up means taking a hit. The +0.03 bonus is intentionally small — the reward for setup is primarily in the increased future HP delta, not the immediate bonus. The penalty (−0.10) prevents setting up when low on HP or the opponent is already weakened.

#### Switch Penalty

```
reward -= 0.01 per switch
```

Switches are powerful but costly (the incoming Pokémon takes a hit). We made switching free in terms of HP delta (the switch animation doesn't deal damage in the simulator), but added a small per-switch penalty to prevent infinite switching loops.

### Dead Ends

#### 1. Pure Terminal Reward
Training with only win/loss rewards (no shaping). After 500K timesteps, the agent had learned nothing — it was still choosing moves at random. The reward signal was too sparse.

#### 2. HP Delta Only (No Bonuses)
The agent learned to deal damage and avoid damage, but couldn't distinguish between a finishing blow and chip damage. It would sometimes leave the opponent at 1 HP and switch out. The finishing bonus was essential.

#### 3. Positive-Only Shaping
We tried rewarding good behavior without penalizing bad behavior (e.g., +0.05 for any hit, no penalty for healing at full HP). The agent learned to stall — infinite loops of weak attacks and recovery. Negative penalties for clearly bad actions are necessary.

#### 4. Normalized HP Delta
We tried rewarding `(opp_hp_decrease − own_hp_decrease) / max_hp` to normalize by Pokémon max HP. In practice, this reduced the reward magnitude for OHKO scenarios and made training unstable. Raw deltas worked better.

#### 5. Status Move Overhaul
Initially, all status moves (Toxic, Thunder Wave, etc.) were treated identically with a flat +0.05 bonus. The agent learned to spam Toxic regardless of matchup. Adding the opponent-HP and boost-penalty thresholds (opponent must have >35% HP and >0.45× boost) made status usage strategic.

### Final Reward Signature

```
r(t) = ΔHP_opp − ΔHP_own − 0.01
       + bonus_move_type(role, state)
       + switch_penalty
       + terminal_bonus

where terminal_bonus = +1 (win), −1 (loss), −0.75 (timeout)
```

The -0.01 per-turn cost and the −0.75 timeout penalty together ensure the agent learns to win efficiently. The shaping bonuses guide the agent toward optimal move selection patterns.

---

## Part 3: PPO Tuning — Hyperparameters, Training Stability, and the Sim-to-Real Gap

### Architecture

ShowdownRL uses Proximal Policy Optimization (PPO) from Stable-Baselines3, with maskable action spaces via SB3-Contrib. The policy network is a standard MLP:

```
Input (106 features) → 256 → 128 → Output (7 actions: 4 moves + 3 bench switches)
```

Both the policy and value networks share the same architecture (no separate feature extractor).

### Hyperparameter Search

We used Optuna for hyperparameter optimization across 50 trials. Key findings:

| Parameter | Search Range | Best Value | Why |
|---|---|---|---|
| Learning rate | 1e-5 to 1e-3 | 3e-4 | Higher rates caused divergence; lower rates were too slow |
| Batch size | 64–2048 | 512 | Trade-off between sample efficiency and gradient noise |
| n_epochs | 3–20 | 10 | More epochs led to overfitting on stale data |
| γ (discount) | 0.90–0.999 | 0.99 | Long battles (20+ turns) need high discount |
| GAE λ | 0.90–0.99 | 0.95 | Standard value for advantage estimation |
| clip_range | 0.1–0.3 | 0.2 | Default PPO clipping worked best |
| ent_coef | 0.0–0.05 | 0.01 | Small entropy bonus prevents premature convergence |

### Training Stability

Three critical lessons:

#### 1. Action Masking Is Non-Negotiable
Without action masking, the agent wasted training time learning that it can't select "Recover" when all moves are attacking moves. MaskablePPO eliminates this noise by zeroing invalid actions before softmax.

```
Valid action mask:
- Attack moves: always valid (if expected_damage > 0)
- Recovery moves: valid only when HP ≤ 85%
- Status moves: valid when opponent HP ≥ 25% and boost > 0.45×
- Setup moves: valid when own HP ≥ 35% and boost < 1.8×
- Switch: valid when bench Pokémon has HP > 0
```

#### 2. Curriculum Learning
Starting the agent against a random opponent and gradually increasing opponent difficulty produced much more stable learning than training against `type_aware` directly:

```
Phase 1 (0–50K steps): Random opponent → basic move selection
Phase 2 (50K–200K): Max-damage opponent → learn type matchups  
Phase 3 (200K+): Type-aware opponent → learn switching and strategy
```

We implemented this through the `mixed` opponent policy, which randomly selects from the three opponent types with weighted probabilities:
- Random: 15%
- Max damage: 25%
- Type aware: 60%

#### 3. Reward Scaling
PPO expects rewards with variance around 1.0. Our rewards were initially too small (per-turn delta was typically ±0.05). We experimented with reward scaling but ultimately found that leaving rewards unscaled and relying on PPO's advantage normalization worked best. The key insight: the *relative* values matter more than the *absolute* values.

### The Sim-to-Real Gap

Training happens in a simplified simulator (`SimplePokemonMoveEnv`) but the policy is deployed on the real Pokémon Showdown website via Playwright. The gap between simulation and reality introduced several issues:

#### Simulator Limitations
1. **No status conditions.** The simulator doesn't model burn, paralysis, sleep, or freeze. The real game does. A trained agent might make moves that work in simulation but fail in real battles.
2. **Simplified damage.** The simulator uses `bp × acc × eff × 0.25` for damage. Real Pokémon uses a complex formula involving level, stats, EVs, IVs, items, and abilities. The agent learns approximate damage expectations that may be wrong in real play.
3. **No abilities or items.** No Intimidate, no Choice Scarf, no Leftovers. These are core strategic elements in real Pokémon.
4. **Turn-synchronous.** Real Pokémon Showdown has speed-determined move order, priority, and double-targeting. The simulator is strictly turn-based: agent moves first, then opponent.

#### Bridging the Gap

We addressed these through:

1. **Conservative move selection**: The live policy bridge uses `ranked_moves` from the heuristic policy as a fallback. If the PPO model selects a move that the heuristic considers terrible (e.g., using a Normal move against a Ghost type), we fall back to the heuristic's top choice.
2. **Inference-time action masking**: The live bridge builds action masks from the actual available moves on the website DOM, not from simulator state. If a move slot is disabled (PP disabled, trapped by Mean Look), it's masked out.
3. **Data validation**: We log all turn-state snapshots in debug mode and compare predicted vs. actual outcomes to identify simulator-reality mismatches.

---

## Part 4: Live Battle Integration — Playwright, Action Parsing, and WebM Recording

### Architecture

The live player runs ShowdownRL as a CLI tool that opens a real Chromium browser via Playwright, navigates to `play.pokemonshowdown.com`, logs in, queues a Random Battle, and plays moves in real time.

```
showdownrl live --policy ppo --model-path models/maskable_ppo_v11_conservative_3M.zip
```

### Browser Automation Pipeline

```
1. Launch Chromium (visible, slow-mo for observability)
2. Navigate to https://play.pokemonshowdown.com/
3. Wait for lobby to load (.userbar selector)
4. Click "Sign in" → enter username/password
5. Navigate to Random Battle format → click "Battle!"
6. Wait for .battle selector to appear
7. Loop:
   a. Read turn state from DOM (GET_TURN_STATE JS injection)
   b. Read available moves from .movemenu buttons
   c. Build observation → run policy → get action
   d. Click the chosen move/switch button
   e. Wait ~750ms for opponent's move
   f. Repeat until battle ends
8. Detect result → log battle record → repeat or exit
```

### DOM-Based State Extraction

We extract battle state by injecting JavaScript into the browser page:

**GET_TURN_STATE** reads:
- Active Pokémon name, HP%, status, types
- Opponent Pokémon name, HP%, status, types
- Available moves (name, type, category, base power)
- Switch options (name, HP%, status, types)
- Last 12 battle log messages

This is a fundamentally different approach from using the Showdown protocol API. Advantages:
- **No protocol parsing needed** — works even if the website's internal API changes
- **Visible debugging** — you can watch the agent play and see exactly what it sees
- **Resilient** — doesn't break if the website adds new features

Disadvantages:
- **Slower** — each state read takes ~100ms of JS evaluation time
- **Fragile to DOM changes** — CSS selectors need updating when the website redesigns

### The Click Pipeline

Once the policy selects an action, we execute it through a three-stage process:

1. **Get Click Target**: Find the correct button in the DOM (`.movemenu button[data-move="..."]` for moves, `.switchmenu button` for switches)
2. **Scroll Into View**: Ensure the button is visible
3. **Click with Marker**: Playwright clicks the element. We also inject a visual click marker (CSS animation) so the user can see what was clicked.

```
SHOW_CLICK JS:
  - Creates a pulsing red circle at click position
  - Shows a label (move name / switch target)
  - Animates outward over 760ms
```

This visual feedback is essential for user trust. Watching an invisible bot play is unsettling; seeing "click → Earthquake" with a visual marker makes the process understandable.

### WebM Recording

ShowdownRL records battles as WebM videos using Playwright's built-in screencast:

```python
await page.screencast(path="battle.webm")
```

Key design decisions:
- **WebM over MP4**: WebM is patent-free and natively supported by Chromium's screencast API
- **Full-screen recording**: We record the entire viewport (1280×800), not just the battle div, to capture context (ladder rating, timer, chat)
- **Configurable quality**: Default is 30fps at medium quality; --slow-mo-ms adds delay between actions for clarity

Videos are saved to:
- Installed via pipx: `~/Movies/ShowdownRL/`
- Running from repo: `./results/`

### Stats Pipeline

Every battle produces a JSONL record:

```json
{
  "started_at": "2025-06-26T14:30:00+00:00",
  "ended_at": "2025-06-26T14:32:15+00:00",
  "result": "win",
  "turns": 18,
  "format": "Random Battle",
  "policy": "ppo",
  "model_path": "models/maskable_ppo_v11_conservative_3M.zip",
  "selected_moves": [
    {"name": "Earthquake", "type": "ground"},
    {"name": "Stealth Rock", "type": "rock"}
  ],
  "selected_switches": [],
  "forced_switches": 1,
  "start_rating": 1200,
  "end_rating": 1223,
  "errors": [],
  "video_path": "results/battle-2025-06-26T14-30-00.webm",
  "visible_result_text": "You won the battle!"
}
```

Stats are stored locally and never uploaded. The `showdownrl stats` command reads from the same JSONL file and can output terminal summaries, HTML reports, CSV/JSON exports, and daily trends.

### Debug Mode

With `--debug-policy`, ShowdownRL saves a turn-state snapshot after every move:

```
stats/debug-turns/battle-01-turn-012-20250626T143000.json
```

Each snapshot contains:
- Full observation vector (106 floats)
- Move scores from the heuristic policy
- PPO action output (with fallback reason if triggered)
- Raw DOM state (HP, types, available moves)

This is invaluable for debugging why the agent made a particular decision. You can replay a battle turn by turn and see exactly what the AI saw.

### Self-Play and Continuous Improvement

The live-and-learn cycle works like this:

1. **Play**: Run `showdownrl live --policy ppo` to collect battle logs
2. **Analyze**: `showdownrl stats --trend` shows win-rate trends over time
3. **Generate training data**: The debug snapshots can be converted into Gymnasium episodes
4. **Retrain**: `python scripts/train_ppo.py` with new data
5. **Evaluate**: `python scripts/evaluate_model.py` against baseline policies
6. **Deploy**: `showdownrl live --policy ppo --model-path models/new_model.zip`

---

## Conclusion

ShowdownRL demonstrates that a relatively simple PPO agent with a carefully designed observation space and reward function can achieve competitive play in Pokémon battles. The key lessons:

1. **Observation design matters more than architecture.** The 106-feature team-aware observation was the single biggest performance driver.
2. **Reward shaping requires constraints.** Unconditional bonuses lead to degenerate strategies; conditional bonuses with penalties teach nuanced behavior.
3. **Action masking is essential.** Without it, the agent wastes training time learning about invalid actions.
4. **The sim-to-real gap is manageable** with conservative fallbacks and inference-time masking.
5. **Visible browser automation makes RL accessible.** Watching the agent play in a real browser builds trust and makes debugging intuitive.

The current best model (MaskablePPO v11 conservative 3M) achieves ~79% win rate against the `type_aware` heuristic opponent in simulation. On live Pokémon Showdown, performance varies with format and opponent skill, but the agent consistently demonstrates competent play: recognizing KO range, making intelligent switches, and using status moves strategically.

### Future Work

- **Ability awareness**: Encoding abilities into the observation could dramatically improve performance
- **Open-ended format support**: Currently optimized for Random Battle; OU and VGC would require different observation spaces
- **Self-play training**: Training against copies of itself could teach the agent to exploit its own weaknesses
- **Cross-format transfer**: Can a policy trained on Random Battle generalize to other formats?

---

*ShowdownRL is open source at [github.com/AnanmayS/ShowdownRL](https://github.com/AnanmayS/ShowdownRL). All training code, evaluation scripts, and trained models are available in the repository.*
