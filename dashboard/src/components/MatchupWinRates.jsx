import React from "react";

const OPPONENT_COLORS = {
  random: "#94a3b8",
  max_damage: "#fb923c",
  type_aware: "#818cf8",
  mixed: "#f472b6",
};

export default function MatchupWinRates({ data }) {
  if (!data || data.length === 0) {
    return <p className="empty-state">No matchup data available</p>;
  }

  return (
    <table>
      <thead>
        <tr>
          <th>Opponent</th>
          <th>Record</th>
          <th>Win Rate</th>
        </tr>
      </thead>
      <tbody>
        {data.map((item) => (
          <tr key={item.policy}>
            <td>
              <span
                className="tag"
                style={{
                  background: `${
                    OPPONENT_COLORS[item.policy] || "#64748b"
                  }22`,
                  color: OPPONENT_COLORS[item.policy] || "#64748b",
                }}
              >
                {item.policy.replace(/_/g, " ")}
              </span>
            </td>
            <td>
              {item.wins}-{item.losses}
            </td>
            <td>
              <div className="bar-cell">
                <div className="bar-bg">
                  <div
                    className="bar-fill"
                    style={{
                      width: `${(item.winRate * 100).toFixed(0)}%`,
                      background: OPPONENT_COLORS[item.policy] || "#64748b",
                    }}
                  />
                </div>
                <span>{(item.winRate * 100).toFixed(1)}%</span>
              </div>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
