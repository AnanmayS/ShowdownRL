import React from "react";

export default function CommonMisplay({ data }) {
  if (!data || data.length === 0) {
    return (
      <p className="empty-state">
        <p>No misplay patterns detected</p>
        <p style={{ fontSize: "0.85rem", marginTop: 4 }}>
          More battle data needed for analysis
        </p>
      </p>
    );
  }

  // Show top 10 worst-performing moves
  const topMisplay = data.slice(0, 10);

  return (
    <div>
      <p style={{ color: "#94a3b8", fontSize: "0.85rem", marginBottom: 12 }}>
        Moves over-represented in losses vs. wins (ratio &gt; 1.5x)
      </p>
      <table>
        <thead>
          <tr>
            <th>Move</th>
            <th>Loss Rate</th>
            <th>Win Rate</th>
            <th>Ratio</th>
          </tr>
        </thead>
        <tbody>
          {topMisplay.map((item) => (
            <tr key={item.move}>
              <td>
                <strong>{item.move}</strong>
              </td>
              <td>{(item.lossRate * 100).toFixed(1)}%</td>
              <td>{(item.winRate * 100).toFixed(1)}%</td>
              <td>
                <span
                  className="tag"
                  style={{
                    background:
                      item.ratio > 2
                        ? "rgba(248,113,113,0.15)"
                        : "rgba(251,146,60,0.15)",
                    color: item.ratio > 2 ? "#f87171" : "#fb923c",
                  }}
                >
                  {item.ratio.toFixed(1)}x
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
