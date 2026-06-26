import React, { useState, useCallback, useMemo } from "react";
import WinRateOverTime from "./components/WinRateOverTime";
import MatchupWinRates from "./components/MatchupWinRates";
import CommonMisplay from "./components/CommonMisplay";
import PolicyComparison from "./components/PolicyComparison";
import BattleReplayViewer from "./components/BattleReplayViewer";
import {
  parseBenchmarkCSV,
  parseBattleLogs,
  winRateOverTime,
  matchupWinRates,
  commonMisplayMoves,
  policyComparison,
} from "./data/loader";

const BENCHMARK_DEFAULT = "/benchmarks/current_evaluation.csv";
const BATTLE_LOG_HELP =
  "Select a battles.jsonl file from your ShowdownRL stats directory (e.g. ~/Library/Application Support/ShowdownRL/stats/battles.jsonl)";

export default function App() {
  const [benchmarkRows, setBenchmarkRows] = useState([]);
  const [battleRecords, setBattleRecords] = useState([]);
  const [replayIndex, setReplayIndex] = useState(0);
  const [benchmarkMode, setBenchmarkMode] = useState("builtin");

  const handleBenchmarkFile = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const rows = parseBenchmarkCSV(ev.target.result);
      setBenchmarkRows(rows);
    };
    reader.readAsText(file);
  }, []);

  const handleBattleFile = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const records = parseBattleLogs(ev.target.result);
      setBattleRecords(records);
    };
    reader.readAsText(file);
  }, []);

  // Load built-in benchmark CSV on mount
  React.useEffect(() => {
    if (benchmarkMode === "builtin" && benchmarkRows.length === 0) {
      fetch(BENCHMARK_DEFAULT)
        .then((res) => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.text();
        })
        .then((text) => {
          const rows = parseBenchmarkCSV(text);
          setBenchmarkRows(rows);
        })
        .catch(() => {
          // built-in not available (e.g. running without server)
        });
    }
  }, [benchmarkMode, benchmarkRows.length]);

  const winRateData = useMemo(
    () => winRateOverTime(battleRecords),
    [battleRecords]
  );
  const matchupData = useMemo(
    () => matchupWinRates(benchmarkRows),
    [benchmarkRows]
  );
  const misplayData = useMemo(
    () => commonMisplayMoves(battleRecords),
    [battleRecords]
  );
  const policyData = useMemo(
    () => policyComparison(benchmarkRows),
    [benchmarkRows]
  );

  const hasBenchmark = benchmarkRows.length > 0;
  const hasBattles = battleRecords.length > 0;

  return (
    <div className="app">
      <h1>ShowdownRL Dashboard</h1>
      <p className="subtitle">
        Battle history and analytics for the Pokemon Showdown RL agent
      </p>

      <div className="controls">
        <label>
          Benchmark data:
          <select
            value={benchmarkMode}
            onChange={(e) => setBenchmarkMode(e.target.value)}
          >
            <option value="builtin">Built-in (current_evaluation.csv)</option>
            <option value="custom">Upload custom CSV</option>
          </select>
        </label>
        {benchmarkMode === "custom" && (
          <label>
            <input
              type="file"
              accept=".csv"
              onChange={handleBenchmarkFile}
            />
          </label>
        )}
        <label>
          Battle logs:
          <input
            type="file"
            accept=".jsonl,.json"
            onChange={handleBattleFile}
            title={BATTLE_LOG_HELP}
          />
        </label>
      </div>

      <div className="grid">
        {/* Summary stats (only if battle logs loaded) */}
        {hasBattles && (
          <div className="card full">
            <h2>Summary</h2>
            <div className="summary-stats">
              <div className="stat-box">
                <div className="value">{battleRecords.length}</div>
                <div className="label">Battles</div>
              </div>
              <div className="stat-box">
                <div className="value">
                  {(
                    (battleRecords.filter((r) => (r.result || "").toLowerCase() === "win")
                      .length /
                      Math.max(
                        1,
                        battleRecords.filter((r) =>
                          ["win", "loss"].includes((r.result || "").toLowerCase())
                        ).length
                      )) *
                    100
                  ).toFixed(1)}
                  %
                </div>
                <div className="label">Win Rate</div>
              </div>
              <div className="stat-box">
                <div className="value">
                  {(
                    battleRecords.reduce(
                      (s, r) => s + (parseInt(r.turns) || 0),
                      0
                    ) / Math.max(1, battleRecords.length)
                  ).toFixed(1)}
                </div>
                <div className="label">Avg Turns</div>
              </div>
              <div className="stat-box">
                <div className="value">
                  {
                    battleRecords.filter(
                      (r) => (r.result || "").toLowerCase() === "win"
                    ).length
                  }
                  -
                  {
                    battleRecords.filter(
                      (r) => (r.result || "").toLowerCase() === "loss"
                    ).length
                  }
                </div>
                <div className="label">W-L</div>
              </div>
            </div>
          </div>
        )}

        {/* Win Rate Over Time */}
        <div className="card full">
          <h2>Win Rate Over Time</h2>
          {hasBattles ? (
            <WinRateOverTime data={winRateData} />
          ) : (
            <p className="empty-state">
              Upload a battles.jsonl file to see win rate trends
            </p>
          )}
        </div>

        {/* Matchup Win Rates */}
        <div className="card">
          <h2>Matchup Win Rates</h2>
          {hasBenchmark ? (
            <MatchupWinRates data={matchupData} />
          ) : (
            <p className="empty-state">
              Load benchmark data to see matchup win rates
            </p>
          )}
        </div>

        {/* Common Misplays */}
        <div className="card">
          <h2>Common Misplays</h2>
          {hasBattles ? (
            <CommonMisplay data={misplayData} />
          ) : (
            <p className="empty-state">
              Upload battle logs to analyze misplays
            </p>
          )}
        </div>

        {/* Policy Comparison */}
        <div className="card full">
          <h2>Policy Version Comparison</h2>
          {hasBenchmark ? (
            <PolicyComparison data={policyData} />
          ) : (
            <p className="empty-state">
              Load benchmark data to compare policy versions
            </p>
          )}
        </div>

        {/* Battle Replay Viewer */}
        <div className="card full">
          <h2>Battle Replay Viewer</h2>
          {hasBattles ? (
            <BattleReplayViewer
              records={battleRecords}
              index={replayIndex}
              onIndexChange={setReplayIndex}
            />
          ) : (
            <p className="empty-state">
              Upload battle logs to view battle replays
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
