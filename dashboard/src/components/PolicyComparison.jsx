import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

export default function PolicyComparison({ data }) {
  if (!data || data.length === 0) {
    return <p className="empty-state">No benchmark data available</p>;
  }

  // Chart-friendly format
  const chartData = data.map((d) => ({
    name: d.policy.length > 14 ? d.policy.slice(0, 12) + "…" : d.policy,
    "Win Rate": +(d.winRate * 100).toFixed(1),
    "Avg Reward": +d.avgReward.toFixed(2),
    "Avg Turns": +d.avgTurns.toFixed(1),
  }));

  return (
    <div>
      <table style={{ marginBottom: 16 }}>
        <thead>
          <tr>
            <th>Policy</th>
            <th>Record</th>
            <th>Win Rate</th>
            <th>Avg Reward</th>
            <th>Avg Turns</th>
            <th>Episodes</th>
          </tr>
        </thead>
        <tbody>
          {data
            .sort((a, b) => b.winRate - a.winRate)
            .map((item) => (
              <tr key={item.policy}>
                <td>
                  <strong style={{ fontSize: "0.8rem" }}>{item.policy}</strong>
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
                          background:
                            item.winRate > 0.5 ? "#4ade80" : "#f87171",
                        }}
                      />
                    </div>
                    <span>{(item.winRate * 100).toFixed(1)}%</span>
                  </div>
                </td>
                <td>{item.avgReward.toFixed(3)}</td>
                <td>{item.avgTurns.toFixed(1)}</td>
                <td>{item.episodes}</td>
              </tr>
            ))}
        </tbody>
      </table>

      <ResponsiveContainer width="100%" height={250}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis
            dataKey="name"
            stroke="#94a3b8"
            fontSize={11}
            angle={-20}
            textAnchor="end"
            height={60}
          />
          <YAxis stroke="#94a3b8" fontSize={12} />
          <Tooltip
            contentStyle={{
              background: "#1e293b",
              border: "1px solid #334155",
              borderRadius: 8,
              color: "#e2e8f0",
            }}
          />
          <Legend />
          <Bar
            dataKey="Win Rate"
            fill="#38bdf8"
            radius={[4, 4, 0, 0]}
            name="Win Rate %"
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
