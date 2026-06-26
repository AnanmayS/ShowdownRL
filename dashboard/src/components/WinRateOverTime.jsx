import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export default function WinRateOverTime({ data }) {
  if (!data || data.length === 0) {
    return <p className="empty-state">No battle data available</p>;
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis
          dataKey="date"
          stroke="#94a3b8"
          fontSize={12}
          tickFormatter={(v) => v?.slice(5) || v}
        />
        <YAxis
          domain={[0, 1]}
          stroke="#94a3b8"
          fontSize={12}
          tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
        />
        <Tooltip
          contentStyle={{
            background: "#1e293b",
            border: "1px solid #334155",
            borderRadius: 8,
            color: "#e2e8f0",
          }}
          formatter={(value) => [`${(value * 100).toFixed(1)}%`, "Win Rate"]}
        />
        <Line
          type="monotone"
          dataKey="winRate"
          stroke="#38bdf8"
          strokeWidth={2}
          dot={{ fill: "#38bdf8", r: 3 }}
          activeDot={{ r: 5 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
