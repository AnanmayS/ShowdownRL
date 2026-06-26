import React, { useMemo } from "react";

export default function BattleReplayViewer({
  records,
  index,
  onIndexChange,
}) {
  const battle = records[index];

  const replayLog = useMemo(() => {
    if (!battle) return [];
    const lines = [];
    lines.push({
      text: `Battle #${index + 1} — ${battle.started_at || "?"}`,
      cls: "turn-hdr",
    });
    lines.push({
      text: `Result: ${battle.result || "unknown"} | Format: ${
        battle.format || "?"
      } | Turns: ${battle.turns || "?"}`,
      cls: "",
    });
    if (battle.policy) {
      lines.push({
        text: `Policy: ${battle.policy}${battle.model_path ? ` (${battle.model_path})` : ""}`,
        cls: "",
      });
    }
    if (battle.start_rating || battle.end_rating) {
      lines.push({
        text: `Rating: ${battle.start_rating || "?"} → ${battle.end_rating || "?"}`,
        cls: "",
      });
    }
    lines.push({ text: "─".repeat(40), cls: "" });

    // Selected moves
    const moves = battle.selected_moves || [];
    if (moves.length > 0) {
      for (let i = 0; i < moves.length; i++) {
        const m = typeof moves[i] === "object" ? moves[i] : { name: moves[i] };
        const turnNum = i + 1;
        lines.push({
          text: `Turn ${turnNum}: selected ${m.name || "?"}`,
          cls: "",
        });
      }
    } else {
      lines.push({ text: "No move-level detail recorded", cls: "" });
    }

    if (battle.forced_switches && parseInt(battle.forced_switches) > 0) {
      lines.push({
        text: `Forced switches: ${battle.forced_switches}`,
        cls: "",
      });
    }
    if (battle.errors && battle.errors.length > 0) {
      lines.push({ text: "Errors:", cls: "" });
      for (const err of battle.errors) {
        lines.push({ text: `  ⚠ ${err}`, cls: "damage" });
      }
    }
    if (battle.visible_result_text) {
      lines.push({
        text: `Visible result: ${battle.visible_result_text}`,
        cls: "ko",
      });
    }
    return lines;
  }, [battle, index, records.length]);

  if (!battle) {
    return <p className="empty-state">No battle selected</p>;
  }

  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 12,
        }}
      >
        <div>
          <span
            className={`tag ${(battle.result || "").toLowerCase() === "win" ? "tag-win" : (battle.result || "").toLowerCase() === "loss" ? "tag-loss" : ""}`}
          >
            {battle.result || "unknown"}
          </span>
          <span style={{ marginLeft: 8, color: "#94a3b8", fontSize: "0.85rem" }}>
            Battle {index + 1} of {records.length}
          </span>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button
            onClick={() => onIndexChange(Math.max(0, index - 1))}
            disabled={index <= 0}
            style={navBtnStyle}
          >
            ← Prev
          </button>
          <button
            onClick={() =>
              onIndexChange(Math.min(records.length - 1, index + 1))
            }
            disabled={index >= records.length - 1}
            style={navBtnStyle}
          >
            Next →
          </button>
        </div>
      </div>

      <div className="replay-log">
        {replayLog.map((line, i) => (
          <div key={i} className={line.cls}>
            {line.text}
          </div>
        ))}
      </div>
    </div>
  );
}

const navBtnStyle = {
  background: "#334155",
  color: "#e2e8f0",
  border: "1px solid #475569",
  borderRadius: 6,
  padding: "6px 14px",
  cursor: "pointer",
  fontSize: "0.85rem",
};
