"""Local battle statistics for ShowdownRL."""

from __future__ import annotations

import json
import statistics
import webbrowser
from collections import Counter
from datetime import date, datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from showdownrl.config import default_stats_dir

BATTLE_LOG_NAME = "battles.jsonl"
HTML_REPORT_NAME = "stats.html"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def battle_log_path(stats_dir: Path | None = None) -> Path:
    return (stats_dir or default_stats_dir()) / BATTLE_LOG_NAME


def html_report_path(stats_dir: Path | None = None) -> Path:
    return (stats_dir or default_stats_dir()) / HTML_REPORT_NAME


def append_battle_record(record: dict[str, Any], stats_dir: Path | None = None) -> Path:
    path = battle_log_path(stats_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def load_battle_records(stats_dir: Path | None = None) -> tuple[list[dict[str, Any]], int]:
    path = battle_log_path(stats_dir)
    if not path.exists():
        return [], 0

    records: list[dict[str, Any]] = []
    corrupt = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            corrupt += 1
            continue
        if isinstance(value, dict):
            records.append(value)
        else:
            corrupt += 1
    return records, corrupt


def parse_since(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def record_date(record: dict[str, Any]) -> date | None:
    raw = str(record.get("started_at") or record.get("ended_at") or "")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def filter_records(
    records: list[dict[str, Any]],
    *,
    since: date | None = None,
    format_name: str = "",
) -> list[dict[str, Any]]:
    filtered = []
    wanted_format = format_name.strip().lower()
    for record in records:
        if since:
            current_date = record_date(record)
            if current_date is None or current_date < since:
                continue
        if wanted_format:
            current_format = str(record.get("format") or "").lower()
            if wanted_format not in current_format:
                continue
        filtered.append(record)
    return filtered


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(1 for item in records if item.get("result") == "win")
    losses = sum(1 for item in records if item.get("result") == "loss")
    errors = sum(1 for item in records if item.get("result") == "error")
    unknowns = len(records) - wins - losses - errors
    decided = wins + losses
    turns = [int(item.get("turns") or 0) for item in records if int(item.get("turns") or 0) > 0]
    forced_switches = sum(int(item.get("forced_switches") or 0) for item in records)

    move_counter: Counter[str] = Counter()
    for item in records:
        for move in item.get("selected_moves") or []:
            if isinstance(move, dict) and move.get("name"):
                move_counter[str(move["name"])] += 1

    return {
        "total": len(records),
        "wins": wins,
        "losses": losses,
        "unknowns": unknowns,
        "errors": errors,
        "win_rate": wins / decided if decided else 0.0,
        "average_turns": statistics.fmean(turns) if turns else 0.0,
        "forced_switches": forced_switches,
        "most_used_moves": move_counter.most_common(10),
    }


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def terminal_summary(
    records: list[dict[str, Any]],
    *,
    corrupt_count: int = 0,
    stats_dir: Path | None = None,
) -> str:
    summary = summarize_records(records)
    lines = [
        "ShowdownRL stats",
        f"Log file: {battle_log_path(stats_dir)}",
        f"Total battles: {summary['total']}",
        f"Record: {summary['wins']}-{summary['losses']} "
        f"({summary['unknowns']} unknown, {summary['errors']} errors)",
        f"Win rate: {format_percent(summary['win_rate'])}",
        f"Average turns: {summary['average_turns']:.1f}",
        f"Forced switches: {summary['forced_switches']}",
    ]
    if corrupt_count:
        lines.append(f"Skipped corrupt log lines: {corrupt_count}")

    lines.append("")
    lines.append("Most-used moves:")
    if summary["most_used_moves"]:
        for move, count in summary["most_used_moves"]:
            lines.append(f"  {move}: {count}")
    else:
        lines.append("  none yet")

    lines.append("")
    lines.append("Last 10 battles:")
    recent = records[-10:]
    if not recent:
        lines.append("  no battles logged yet")
    for item in recent:
        started = str(item.get("started_at") or "?").replace("+00:00", "Z")
        result = str(item.get("result") or "unknown")
        turns = item.get("turns") or 0
        format_name = item.get("format") or "current format"
        lines.append(f"  {started} | {result} | {turns} turns | {format_name}")
    return "\n".join(lines)


def svg_bar(label: str, value: float, color: str) -> str:
    width = max(0, min(100, value))
    return (
        '<div class="bar-row">'
        f'<span>{escape(label)}</span>'
        '<svg viewBox="0 0 100 10" role="img" aria-label="'
        f'{escape(label)} {width:.1f}%">'
        f'<rect width="100" height="10" rx="2" fill="#e5e7eb"/>'
        f'<rect width="{width:.3f}" height="10" rx="2" fill="{color}"/>'
        "</svg>"
        f"<b>{width:.1f}%</b>"
        "</div>"
    )


def html_report(records: list[dict[str, Any]], *, corrupt_count: int = 0) -> str:
    summary = summarize_records(records)
    total = max(1, summary["total"])
    move_rows = "\n".join(
        f"<tr><td>{escape(move)}</td><td>{count}</td></tr>"
        for move, count in summary["most_used_moves"]
    ) or '<tr><td colspan="2">No moves logged yet</td></tr>'
    battle_rows = "\n".join(
        "<tr>"
        f"<td>{escape(str(item.get('started_at') or ''))}</td>"
        f"<td>{escape(str(item.get('result') or 'unknown'))}</td>"
        f"<td>{escape(str(item.get('turns') or 0))}</td>"
        f"<td>{escape(str(item.get('format') or 'current format'))}</td>"
        f"<td>{escape(str(item.get('visible_result_text') or ''))}</td>"
        "</tr>"
        for item in records[-50:]
    ) or '<tr><td colspan="5">No battles logged yet</td></tr>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ShowdownRL Stats</title>
  <style>
    body {{ font: 16px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #182026; }}
    h1, h2 {{ line-height: 1.15; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }}
    .metric {{ border: 1px solid #d8e0dc; border-radius: 8px; padding: 14px; background: #f8faf9; }}
    .metric b {{ display: block; font-size: 28px; }}
    .bar-row {{ display: grid; grid-template-columns: 120px 1fr 56px; gap: 12px; align-items: center; margin: 10px 0; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f6f5; }}
    .note {{ color: #5b6770; }}
  </style>
</head>
<body>
  <h1>ShowdownRL Stats</h1>
  <p class="note">Local-only report generated from this machine's battle log. No credentials are included.</p>
  <div class="grid">
    <div class="metric"><span>Total battles</span><b>{summary['total']}</b></div>
    <div class="metric"><span>Record</span><b>{summary['wins']}-{summary['losses']}</b></div>
    <div class="metric"><span>Win rate</span><b>{format_percent(summary['win_rate'])}</b></div>
    <div class="metric"><span>Average turns</span><b>{summary['average_turns']:.1f}</b></div>
  </div>
  <h2>Result Breakdown</h2>
  {svg_bar("Wins", summary["wins"] / total * 100, "#2f6f73")}
  {svg_bar("Losses", summary["losses"] / total * 100, "#b94b4b")}
  {svg_bar("Unknown", summary["unknowns"] / total * 100, "#61737f")}
  {svg_bar("Errors", summary["errors"] / total * 100, "#d28b36")}
  <h2>Most-used Moves</h2>
  <table><thead><tr><th>Move</th><th>Uses</th></tr></thead><tbody>{move_rows}</tbody></table>
  <h2>Recent Battles</h2>
  <table><thead><tr><th>Started</th><th>Result</th><th>Turns</th><th>Format</th><th>Visible result</th></tr></thead><tbody>{battle_rows}</tbody></table>
  <p class="note">Skipped corrupt log lines: {corrupt_count}</p>
</body>
</html>
"""


def write_html_report(records: list[dict[str, Any]], *, stats_dir: Path | None = None, corrupt_count: int = 0) -> Path:
    path = html_report_path(stats_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_report(records, corrupt_count=corrupt_count), encoding="utf-8")
    return path


def open_html_report(path: Path) -> None:
    webbrowser.open(path.resolve().as_uri())
