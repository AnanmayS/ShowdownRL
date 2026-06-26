/**
 * Parse benchmark CSV and battle JSONL logs for the ShowdownRL dashboard.
 *
 * Benchmark CSV format (docs/benchmarks/current_evaluation.csv):
 *   scenario,policy,episodes,wins,losses,draws,win_rate,non_loss_rate,average_reward,average_turns,...
 *
 * Battle JSONL format (battles.jsonl):
 *   One JSON object per line: { started_at, ended_at, result, turns, format, policy,
 *   model_path, selected_moves, forced_switches, errors, ... }
 */

export function parseBenchmarkCSV(text) {
  const lines = text.trim().split("\n");
  if (lines.length < 2) return [];
  // Skip header
  const header = lines[0].split(",").map((h) => h.trim());
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    const vals = line.split(",");
    const row = {};
    header.forEach((h, idx) => {
      row[h] = vals[idx] !== undefined ? vals[idx].trim() : "";
    });
    rows.push(row);
  }
  return rows;
}

export function parseBattleLogs(text) {
  const lines = text.trim().split("\n");
  const records = [];
  for (const line of lines) {
    if (!line.trim()) continue;
    try {
      const record = JSON.parse(line);
      if (typeof record === "object" && record !== null) {
        records.push(record);
      }
    } catch {
      // skip corrupt lines
    }
  }
  return records;
}

/**
 * Parse a matchup from scenario strings like "rich_type_aware_seed42"
 * Returns { mechanics, opponent_policy, seed }
 */
export function parseScenario(scenario) {
  const match = scenario.match(
    /^(rich|typed|toy)_(type_aware|max_damage|random|mixed)_seed(\d+)$/i
  );
  if (match) {
    const [, mechanics, opponentPolicy, seed] = match;
    return { mechanics, opponentPolicy, seed };
  }
  return { mechanics: "unknown", opponentPolicy: "unknown", seed: "0" };
}

/**
 * Compute win rate over time (by date) from battle logs.
 * Returns array of { date, wins, losses, total, winRate }
 */
export function winRateOverTime(records) {
  const byDate = {};
  for (const r of records) {
    const dateStr = (r.started_at || "").split("T")[0];
    if (!dateStr) continue;
    if (!byDate[dateStr]) byDate[dateStr] = { wins: 0, losses: 0, total: 0 };
    const result = (r.result || "").toLowerCase();
    if (result === "win") byDate[dateStr].wins++;
    else if (result === "loss") byDate[dateStr].losses++;
    byDate[dateStr].total++;
  }
  return Object.entries(byDate)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([date, stats]) => ({
      date,
      wins: stats.wins,
      losses: stats.losses,
      total: stats.total,
      winRate:
        stats.wins + stats.losses > 0
          ? stats.wins / (stats.wins + stats.losses)
          : 0,
    }));
}

/**
 * Compute match-up win rates from benchmark data.
 * Groups by opponent policy (archetype).
 */
export function matchupWinRates(benchmarkRows) {
  const groups = {};
  for (const row of benchmarkRows) {
    const { opponentPolicy } = parseScenario(row.scenario || "");
    const policy = opponentPolicy || "unknown";
    if (!groups[policy]) groups[policy] = { wins: 0, losses: 0, total: 0 };
    groups[policy].wins += parseInt(row.wins) || 0;
    groups[policy].losses += parseInt(row.losses) || 0;
    groups[policy].total += parseInt(row.episodes) || 0;
  }
  return Object.entries(groups).map(([policy, stats]) => ({
    policy,
    wins: stats.wins,
    losses: stats.losses,
    total: stats.total,
    winRate:
      stats.wins + stats.losses > 0
        ? stats.wins / (stats.wins + stats.losses)
        : 0,
  }));
}

/**
 * Identify common misplays: moves that correlate with losses.
 * Groups selected_moves by result and returns the most common
 * moves in losses that aren't as common in wins.
 */
export function commonMisplayMoves(records) {
  const moveInWin = {};
  const moveInLoss = {};
  for (const r of records) {
    const moves = r.selected_moves || [];
    const result = (r.result || "").toLowerCase();
    const target = result === "win" ? moveInWin : moveInLoss;
    for (const m of moves) {
      const name = typeof m === "object" ? m.name || "?" : String(m);
      target[name] = (target[name] || 0) + 1;
    }
  }
  // Find moves over-represented in losses
  const misplays = [];
  const allMoves = new Set([
    ...Object.keys(moveInLoss),
    ...Object.keys(moveInWin),
  ]);
  for (const move of allMoves) {
    const lossCount = moveInLoss[move] || 0;
    const winCount = moveInWin[move] || 0;
    const totalLoss = Object.values(moveInLoss).reduce((a, b) => a + b, 0) || 1;
    const totalWin = Object.values(moveInWin).reduce((a, b) => a + b, 0) || 1;
    const lossRate = lossCount / totalLoss;
    const winRate = winCount / totalWin;
    const ratio = winRate > 0 ? lossRate / winRate : lossRate;
    if (lossCount > 2 && ratio > 1.5) {
      misplays.push({ move, lossCount, winCount, lossRate, winRate, ratio });
    }
  }
  return misplays.sort((a, b) => b.ratio - a.ratio);
}

/**
 * Policy version comparison: group benchmark rows by policy name.
 * Returns array of { policy, winRate, avgReward, avgTurns, episodes }
 */
export function policyComparison(benchmarkRows) {
  const groups = {};
  for (const row of benchmarkRows) {
    const policy = row.policy || "unknown";
    if (!groups[policy])
      groups[policy] = {
        wins: 0,
        losses: 0,
        total: 0,
        rewardSum: 0,
        turnSum: 0,
      };
    groups[policy].wins += parseInt(row.wins) || 0;
    groups[policy].losses += parseInt(row.losses) || 0;
    groups[policy].total += parseInt(row.episodes) || 0;
    groups[policy].rewardSum +=
      parseFloat(row.average_reward || 0) * (parseInt(row.episodes) || 0);
    groups[policy].turnSum +=
      parseFloat(row.average_turns || 0) * (parseInt(row.episodes) || 0);
  }
  return Object.entries(groups).map(([policy, stats]) => ({
    policy: shortPolicyLabel(policy),
    wins: stats.wins,
    losses: stats.losses,
    winRate:
      stats.wins + stats.losses > 0
        ? stats.wins / (stats.wins + stats.losses)
        : 0,
    avgReward: stats.total > 0 ? stats.rewardSum / stats.total : 0,
    avgTurns: stats.total > 0 ? stats.turnSum / stats.total : 0,
    episodes: stats.total,
  }));
}

function shortPolicyLabel(policy) {
  const labels = {
    random_policy: "Random",
    max_damage_policy: "Max Damage",
    type_aware_policy: "Type Aware",
    maskable_ppo_v11_conservative_3M: "Maskable PPO v11",
    maskable_ppo_move_selection_v6_rich: "Maskable PPO v6",
    ppo_move_selection_v5_rich_finetuned: "PPO v5 Finetuned",
    ppo_move_selection_v4_rich: "PPO v4 Rich",
    ppo_move_selection_v3_rich: "PPO v3 Rich",
    ppo_move_selection_v2_typed: "PPO v2 Typed",
  };
  return labels[policy] || policy.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());
}
