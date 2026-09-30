"""Visible Pokemon Showdown browser automation.

Decisions are made from the raw battle protocol, not from the rendered page:
an init script hooks the web client's message handler (``app.receive`` in the
classic client, ``PS.receive`` in the preact client) and records every
``>battle-...`` message. ``showdownrl.protocol_battle`` replays those
messages through poke-env's own parser, so the policy sees the same
``Battle`` state (and ``battle_features`` observation) as in training. The
browser is only used to display the battle and to click the chosen button.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from showdownrl.config import DEFAULT_SITE, default_record_dir
from showdownrl.stats import append_battle_record, utc_now_iso, write_debug_snapshot

if TYPE_CHECKING:  # heavy imports (poke-env, torch) stay lazy for non-live commands
    from showdownrl.policy_bridge import Decision, LivePolicy
    from showdownrl.protocol_battle import BattleRoomTracker, ClickPlan, ProtocolBattle

CONNECTED = "() => document.querySelector('button[name=openSounds], .userbar') !== null"
IN_BATTLE = "() => document.querySelector('.battle') !== null"

# Send a chat command (e.g. "/timer on") to a battle room through the client.
SEND_ROOM_COMMAND = """
({room, text}) => {
  try {
    if (window.app && typeof window.app.send === 'function') { window.app.send(text, room); return 'app'; }
    const ps = window.PS || (typeof PS !== 'undefined' ? PS : null);
    if (ps && typeof ps.send === 'function') { ps.send(text, room); return 'PS'; }
  } catch (e) { return 'error: ' + e; }
  return 'none';
}
"""

# SockJS readyState of the client's connection (1 = open), or -1 if unknown.
CONNECTION_STATE = "() => (window.app && window.app.socket) ? window.app.socket.readyState : -1"
REJOIN_ROOM = """
(room) => {
  if (!window.app) return 'no app';
  if (app.rooms && app.rooms[room]) return 'already joined';
  app.joinRoom(room);
  return 'joined';
}
"""
CONNECTION_CHECK_SECONDS = 10

# Opponents who stop moving are only forced to act by the battle timer.
STALL_WARN_SECONDS = 240
STALL_ABANDON_SECONDS = 20 * 60

CLICK_MARKER = """
() => {
    if (document.getElementById('showdownrl-click-style')) return;
    const style = document.createElement('style');
    style.id = 'showdownrl-click-style';
    style.textContent = `
      .showdownrl-click-marker {
        position: fixed;
        width: 34px;
        height: 34px;
        margin-left: -17px;
        margin-top: -17px;
        border: 3px solid #ff375f;
        border-radius: 999px;
        pointer-events: none;
        z-index: 2147483647;
        box-shadow: 0 0 0 6px rgba(255,55,95,.16);
        animation: showdownrl-click-pop .72s ease-out forwards;
      }
      .showdownrl-click-label {
        position: fixed;
        transform: translate(16px, -34px);
        padding: 4px 7px;
        border-radius: 5px;
        background: #111827;
        color: white;
        font: 12px/1.25 system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        pointer-events: none;
        z-index: 2147483647;
        max-width: 240px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        animation: showdownrl-click-pop .72s ease-out forwards;
      }
      @keyframes showdownrl-click-pop {
        0% { opacity: .95; transform: scale(.7); }
        70% { opacity: .8; }
        100% { opacity: 0; transform: scale(1.45); }
      }
    `;
    document.head.appendChild(style);
}
"""

SHOW_CLICK = """
({x, y, label}) => {
    const marker = document.createElement('div');
    marker.className = 'showdownrl-click-marker';
    marker.style.left = `${x}px`;
    marker.style.top = `${y}px`;
    document.body.appendChild(marker);

    if (label) {
        const text = document.createElement('div');
        text.className = 'showdownrl-click-label';
        text.textContent = label;
        text.style.left = `${x}px`;
        text.style.top = `${y}px`;
        document.body.appendChild(text);
        setTimeout(() => text.remove(), 760);
    }
    setTimeout(() => marker.remove(), 760);
}
"""

VISIBLE_RESULT = """
() => {
    const nodes = Array.from(document.querySelectorAll('.broadcast-green,.broadcast-red,.battle-history'));
    for (const node of nodes) {
        const text = (node.textContent || '').replace(/\\s+/g, ' ').trim();
        if (/\\b(won the battle|you won|you lost|forfeited)\\b/i.test(text)) return text;
    }
    return null;
}
"""

GET_LADDER_RATING = """
() => {
    const clean = value => (value || '').replace(/\\s+/g, ' ').trim();

    // 1. Check broadcast nodes — most reliable after-battle rating display
    // Typical: "Your rating: 1050 → 1070 (+20)" or "Rating: 1000 → 1020"
    const broadcasts = document.querySelectorAll('.broadcast-green, .broadcast-red');
    for (const node of broadcasts) {
        const text = clean(node.textContent);
        const arrow = text.match(/rating:\\s*(\\d{3,5})\\s*[→➡].*?\\b(\\d{3,5})\\b/i);
        if (arrow) return Number(arrow[2]);      // after → rating
        const single = text.match(/rating:\\s*(\\d{3,5})\\b/i);
        if (single) return Number(single[1]);
    }

    // 2. Check battle-history for rating lines
    // Formats: "Rating: 1000 → 1020" or "Your rating is now 1050"
    for (const node of document.querySelectorAll('.battle-history > *, .battle-log > *')) {
        const text = clean(node.textContent);
        // Two-number arrow: "Rating 1000 → 1020" or "rating 1000 -> 1020"
        const arrow = text.match(/rating\\D{0,8}(\\d{3,5})\\D{0,8}(?:→|➡|->)\\D{0,8}(\\d{3,5})\\b/i);
        if (arrow) return Number(arrow[2]);
        const current = text.match(/(?:your\\s+)?rating\\D{0,24}(\\d{3,5})\\b/i);
        if (current) return Number(current[1]);
    }

    // 3. Fallback: userbar sometimes shows rating in room header
    const room = document.querySelector('.room');
    if (room) {
        const text = clean(room.textContent);
        const match = text.match(/[ELO\\s]*rating[\\s:]*\\b(\\d{3,5})\\b/i);
        if (match) return Number(match[1]);
    }
    return null;
}
"""

# Installed with context.add_init_script so it runs before the client boots.
# Both Showdown clients route every server message through one method:
# classic client (play.pokemonshowdown.com today): window.app.receive(data)
# preact client: PS.receive(msg) (fed by a WebSocket in a Worker, which is
# why hooking the client beats sniffing the page's websocket frames).
PROTOCOL_HOOK = """
(() => {
    if (window.__showdownrl) return;
    const store = window.__showdownrl = {messages: [], hooked: [], errors: 0};
    const record = data => {
        try {
            const text = typeof data === 'string' ? data : String(data);
            if (text.startsWith('>battle-')) store.messages.push(text);
        } catch (err) {
            store.errors++;
        }
    };
    const wrap = (target, name) => {
        if (!target || typeof target.receive !== 'function' || target.receive.__showdownrl) return;
        const original = target.receive;
        const wrapped = function (data) {
            record(data);
            return original.apply(this, arguments);
        };
        wrapped.__showdownrl = true;
        target.receive = wrapped;
        if (!store.hooked.includes(name)) store.hooked.push(name);
    };
    const tick = () => {
        try { wrap(typeof app !== 'undefined' ? app : window.app, 'app'); } catch (err) {}
        try { wrap(typeof PS !== 'undefined' ? PS : window.PS, 'PS'); } catch (err) {}
    };
    tick();
    setInterval(tick, 50);
})();
"""

READ_PROTOCOL = """
(start) => {
    const store = window.__showdownrl;
    if (!store) return {installed: false, hooked: [], total: 0, messages: []};
    return {
        installed: true,
        hooked: store.hooked.slice(),
        total: store.messages.length,
        messages: store.messages.slice(start),
    };
}
"""

# Fallback when the receive hook is missing: the client keeps each battle's
# protocol log in room.battle.stepQueue and the last request in room.request.
ROOM_SNAPSHOT = """
() => {
    const out = [];
    const clone = value => {
        try { return value ? JSON.parse(JSON.stringify(value)) : null; } catch (err) { return null; }
    };
    const add = (id, room, focused) => {
        if (!room || !room.battle || !String(id).startsWith('battle-')) return;
        const queue = Array.isArray(room.battle.stepQueue) ? room.battle.stepQueue : [];
        out.push({id: String(id), lines: queue.slice(), request: clone(room.request), focused: !!focused});
    };
    const classic = typeof app !== 'undefined' ? app : window.app;
    if (classic && classic.rooms) {
        for (const id in classic.rooms) add(id, classic.rooms[id], classic.curRoom === classic.rooms[id]);
    }
    const preact = typeof PS !== 'undefined' ? PS : window.PS;
    if (preact && preact.rooms) {
        for (const id in preact.rooms) add(id, preact.rooms[id], preact.room === preact.rooms[id]);
    }
    return out;
}
"""

# Visible, enabled battle controls. The classic client disables buttons with
# the disabled attribute; the preact client uses class="disabled"/aria-disabled.
CONTROLS = """
() => {
    const usable = el => el.offsetParent !== null && !el.disabled
        && !el.classList.contains('disabled') && el.getAttribute('aria-disabled') !== 'true';
    const count = selector => Array.from(document.querySelectorAll(selector)).filter(usable).length;
    return {
        moves: count('.movemenu button'),
        switches: count('.switchmenu button'),
        menus: count("button[data-cmd='/movemenu'], button[data-cmd='/switchmenu']"),
    };
}
"""

# Classic client popups are .ps-overlay; preact popups are .ps-popup rooms.
POPUP_OPEN = """
() => Array.from(document.querySelectorAll('.ps-overlay, .ps-popup'))
    .some(el => el.offsetParent !== null || getComputedStyle(el).position === 'fixed')
"""

CLOSE_POPUP = """
() => {
    const classic = typeof app !== 'undefined' ? app : window.app;
    if (classic && typeof classic.closePopup === 'function') {
        for (let i = 0; i < 5 && classic.popups && classic.popups.length; i++) classic.closePopup();
    }
    const preact = typeof PS !== 'undefined' ? PS : window.PS;
    if (preact && typeof preact.closePopup === 'function') preact.closePopup();
}
"""

@dataclass
class LiveOptions:
    username: str = ""
    password: str = ""
    guest: bool = False
    site: str = DEFAULT_SITE
    format_name: str = ""
    record: bool = False
    record_dir: Path | None = None
    keep_open: bool = False
    login_only: bool = False
    check_ui_only: bool = False
    max_turns: int = 200
    max_battles: int = 1
    max_time_minutes: float | None = None
    policy: str = "heuristic"
    model_path: Path | None = None
    search_samples: int = 4
    search_time_ms: int = 100
    stats_enabled: bool = True
    stats_dir: Path | None = None
    debug_policy: bool = False
    slow_mo_ms: int = 250
    headless: bool = False
    click_delay: float = 0.75
    viewport_width: int = 1280
    viewport_height: int = 800


@dataclass(frozen=True)
class SelectorHealth:
    name: str
    ok: bool
    required: bool
    count: int | None = None
    note: str = ""


async def wait_cond(page: Any, js: str, timeout: float = 30.0) -> bool:
    for _ in range(int(timeout * 2)):
        try:
            if await page.evaluate(js):
                return True
        except Exception:
            pass
        await asyncio.sleep(0.5)
    return False


async def first_visible(locator: Any, timeout_ms: int = 5000) -> Any | None:
    deadline = asyncio.get_running_loop().time() + timeout_ms / 1000
    while asyncio.get_running_loop().time() < deadline:
        count = await locator.count()
        for index in range(count):
            candidate = locator.nth(index)
            try:
                if await candidate.is_visible():
                    return candidate
            except Exception:
                continue
        await asyncio.sleep(0.2)
    return None


async def click_locator(page: Any, locator: Any, label: str, delay: float = 0.75, timeout_ms: int = 5000) -> bool:
    target = await first_visible(locator, timeout_ms=timeout_ms)
    if not target:
        return False

    # The element can vanish between first_visible and here (e.g. Showdown auto-login
    # replacing the Choose Name button); bound these waits instead of Playwright's 30s default.
    try:
        await target.scroll_into_view_if_needed(timeout=timeout_ms)
        box = await target.bounding_box(timeout=timeout_ms)
    except Exception:
        return False
    if not box:
        return False

    x = box["x"] + box["width"] / 2
    y = box["y"] + box["height"] / 2
    await page.evaluate(SHOW_CLICK, {"x": x, "y": y, "label": label})
    await page.mouse.move(x, y, steps=12)
    await page.mouse.down()
    await asyncio.sleep(0.08)
    await page.mouse.up()
    await asyncio.sleep(delay)
    return True


def button_with_text(page: Any, pattern: str) -> Any:
    return page.locator("button").filter(has_text=re.compile(pattern, re.I))


async def visible_count(locator: Any) -> int:
    count = await locator.count()
    visible = 0
    for index in range(count):
        try:
            if await locator.nth(index).is_visible():
                visible += 1
        except Exception:
            continue
    return visible


def selector_health_lines(checks: list[SelectorHealth]) -> list[str]:
    lines = []
    for check in checks:
        status = "ok" if check.ok else ("fail" if check.required else "info")
        count = "" if check.count is None else f" ({check.count} visible)"
        note = f" - {check.note}" if check.note else ""
        lines.append(f"  [{status}] {check.name}{count}{note}")
    return lines


async def collect_selector_health(page: Any) -> list[SelectorHealth]:
    try:
        connected = bool(await page.evaluate(CONNECTED))
    except Exception:
        connected = False

    choose_name_count = await visible_count(button_with_text(page, r"choose name"))
    battle_count = await visible_count(button_with_text(page, r"battle!|find a battle|find a random opponent"))
    battle_panel_count = await visible_count(page.locator(".battle"))
    move_button_count = await visible_count(page.locator(".movemenu button:not([disabled])"))
    switch_button_count = await visible_count(page.locator(".switchmenu button:not([disabled])"))
    result_node_count = await page.locator(".broadcast-green,.broadcast-red,.battle-history").count()
    try:
        protocol = await page.evaluate(READ_PROTOCOL, 0)
        hooked = list(protocol.get("hooked") or [])
    except Exception:
        hooked = []

    return [
        SelectorHealth("lobby connection", connected, True, note="public lobby loaded"),
        SelectorHealth("protocol hook", bool(hooked), True, note=f"client receive hooked: {', '.join(hooked) or 'none'}"),
        SelectorHealth("choose-name button", choose_name_count > 0, True, choose_name_count),
        SelectorHealth("battle queue button", battle_count > 0, True, battle_count),
        SelectorHealth("battle panel", battle_panel_count > 0, False, battle_panel_count, "expected after queueing"),
        SelectorHealth("move buttons", move_button_count > 0, False, move_button_count, "expected during a turn"),
        SelectorHealth("switch buttons", switch_button_count > 0, False, switch_button_count, "expected on forced switches"),
        SelectorHealth("result log nodes", result_node_count > 0, False, result_node_count, "expected during/after battle"),
    ]


async def fill_first_visible(locator: Any, text: str) -> bool:
    target = await first_visible(locator)
    if not target:
        return False
    await target.fill(text)
    return True


async def login(page: Any, username: str, password: str, guest: bool, click_delay: float) -> str:
    userbar = page.locator(".userbar").filter(has_text=re.compile(re.escape(username), re.I))
    # After a reload Showdown restores the session asynchronously, so give the userbar a
    # moment to show our name before deciding we need to log in.
    if username and await first_visible(userbar, timeout_ms=6000):
        return "already logged in"

    choose_name = page.locator("button[name='login']")
    if not await click_locator(page, choose_name, "Choose Name", click_delay):
        if username and await userbar.count():
            return "already logged in"
        return "choose-name button not found"

    name_inputs = page.locator("input[name='username'], input[type='text'], input:not([type])")
    if not await fill_first_visible(name_inputs, username):
        return "username input not found"

    confirm_name = page.locator("button[type='submit']").filter(
        has_text=re.compile(r"^choose name$|^ok$|confirm|submit|join", re.I)
    )
    if not await click_locator(page, confirm_name, f"Set name: {username}", click_delay):
        await page.keyboard.press("Enter")
        await asyncio.sleep(click_delay)

    await asyncio.sleep(1)
    if username and await userbar.count():
        return "logged in"

    if guest:
        if await first_visible(page.locator("input[type='password']"), timeout_ms=1000):
            return "guest name requires a password; choose another --username"
        return "guest name submitted"

    password_input = await first_visible(page.locator("input[type='password']"), timeout_ms=5000)
    if not password_input:
        return "name submitted without password prompt or matching userbar"

    if not password:
        return "password prompt shown, but no password was provided"

    await password_input.fill(password)
    login_buttons = page.locator("button[type='submit']").filter(
        has_text=re.compile(r"log in|login|submit|ok|confirm", re.I)
    )
    if not await click_locator(page, login_buttons, "Log In", click_delay):
        await page.keyboard.press("Enter")
        await asyncio.sleep(click_delay)

    for _ in range(12):
        if username and await userbar.count():
            return "logged in"
        await asyncio.sleep(0.5)
    return "login submitted"


async def maybe_select_format(page: Any, format_name: str, click_delay: float) -> None:
    if not format_name:
        return

    format_button = page.locator("button, .select").filter(has_text=re.compile(r"format|random|battle", re.I))
    if not await click_locator(page, format_button, "Format", click_delay):
        print("  Could not open the format picker; continuing with the current format.", flush=True)
        return

    option = page.locator("button, li, .selectMenu li, .formatselect").filter(
        has_text=re.compile(re.escape(format_name), re.I)
    )
    if await click_locator(page, option, format_name, click_delay):
        print(f"  Selected format: {format_name}", flush=True)
    else:
        print(f"  Could not find format '{format_name}'; continuing with the current format.", flush=True)


async def click_queue_battle(page: Any, click_delay: float) -> bool:
    queue_buttons = button_with_text(page, r"battle!|find a random opponent|find a battle|look for a battle")
    return await click_locator(page, queue_buttons, "Queue Battle", click_delay)


async def click_team_preview(page: Any, click_delay: float) -> bool:
    battle_buttons = page.locator(".battle button").filter(has_text=re.compile(r"battle!|ready!|fight!", re.I))
    return await click_locator(page, battle_buttons, "Confirm Team", click_delay)


class BrowserProtocol:
    """Feed the web client's battle protocol into poke-env ``Battle`` objects."""

    def __init__(self, username: str):
        from showdownrl.protocol_battle import BattleRoomTracker

        self.tracker: BattleRoomTracker = BattleRoomTracker(username)
        self.cursor = 0
        self.mode = "stream"
        self.hooked: list[str] = []
        self._warned_snapshot = False

    async def refresh(self, page: Any) -> ProtocolBattle | None:
        """Pull new protocol from the page; return the battle we are playing."""
        try:
            data = await page.evaluate(READ_PROTOCOL, self.cursor)
        except Exception:
            data = None
        if data and int(data.get("total") or 0) < self.cursor:
            # The page was reloaded (e.g. between battles): the store restarted.
            self.cursor = 0
            data = await page.evaluate(READ_PROTOCOL, 0)
        if data and data.get("hooked"):
            self.mode = "stream"
            self.hooked = list(data["hooked"])
            messages = list(data.get("messages") or [])
            self.cursor += len(messages)
            self.tracker.ingest(messages)
            return self.tracker.current()
        return await self._refresh_from_snapshot(page)

    async def _refresh_from_snapshot(self, page: Any) -> ProtocolBattle | None:
        self.mode = "snapshot"
        if not self._warned_snapshot:
            print("  Protocol hook unavailable; rebuilding battles from the client's room log.", flush=True)
            self._warned_snapshot = True
        try:
            rooms = await page.evaluate(ROOM_SNAPSHOT)
        except Exception:
            return None
        if not rooms:
            return None
        room = next((r for r in rooms if r.get("focused")), rooms[-1])
        state = self.tracker.ingest_snapshot(room["id"], room.get("lines") or [], room.get("request"))
        return state if state.battle is not None else None

    def mark_answered(self, state: ProtocolBattle) -> None:
        self.tracker.mark_answered(state)


async def connection_lost(page: Any) -> bool:
    try:
        return int(await page.evaluate(CONNECTION_STATE)) in (2, 3)  # SockJS CLOSING / CLOSED
    except Exception:  # noqa: BLE001 - an unreadable page is handled by the stall watchdog
        return False


async def reconnect_to_battle(page: Any, options: "LiveOptions", room: str) -> None:
    """Reload the client, log back in and rejoin ``room`` (Showdown replays its log)."""
    try:
        await page.goto(options.site, wait_until="domcontentloaded", timeout=45_000)
        await page.evaluate(CLICK_MARKER)
        await asyncio.sleep(3)
        status = await login(page, options.username, options.password, options.guest, options.click_delay)
        print(f"  Login status after reconnect: {status}", flush=True)
        if room:
            await asyncio.sleep(2)
            print(f"  Rejoin {room}: {await page.evaluate(REJOIN_ROOM, room)}", flush=True)
    except Exception as exc:  # noqa: BLE001 - keep the battle loop alive; the watchdog retries
        print(f"  Reconnect failed: {exc}", flush=True)


async def save_stall_screenshot(page: Any, options: "LiveOptions", room: str, turn: int) -> None:
    from showdownrl.config import default_stats_dir

    try:
        directory = (options.stats_dir or default_stats_dir()) / "stalls"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{room or 'battle'}-turn{turn}-{int(time.time())}.png"
        await page.screenshot(path=str(path))
        print(f"  Stall screenshot: {path}", flush=True)
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        print(f"  Could not save stall screenshot: {exc}", flush=True)


async def send_room_command(page: Any, room: str, text: str) -> str:
    try:
        return await page.evaluate(SEND_ROOM_COMMAND, {"room": room, "text": text})
    except Exception as exc:  # noqa: BLE001 - best effort; never stop the battle loop
        return f"error: {exc}"


async def controls_ready(page: Any) -> bool:
    try:
        controls = await page.evaluate(CONTROLS)
    except Exception:
        return False
    return bool(controls and (controls.get("moves") or controls.get("switches") or controls.get("menus")))


async def dismiss_popups(page: Any) -> None:
    """Close client popups (e.g. PM/challenge notices) that would eat clicks."""
    try:
        if not await page.evaluate(POPUP_OPEN):
            return
        await page.keyboard.press("Escape")
        await asyncio.sleep(0.2)
        if await page.evaluate(POPUP_OPEN):
            await page.evaluate(CLOSE_POPUP)
            await asyncio.sleep(0.2)
    except Exception:
        pass


async def wait_choice_sent(page: Any, state: ProtocolBattle, request_seq: int, timeout: float = 4.0) -> bool:
    """After a click, the client hides the choice buttons (or a new request lands)."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if state.request_seq != request_seq or state.finished:
            return True
        try:
            controls = await page.evaluate(CONTROLS)
        except Exception:
            return True
        if not (controls.get("moves") or controls.get("switches")):
            return True
        await asyncio.sleep(0.25)
    return False


async def execute_click_plan(page: Any, plan: ClickPlan, delay: float) -> bool:
    """Click the button(s) for one decision; True once the choice is clicked."""
    if plan.kind not in ("move", "switch") or not plan.selectors:
        return False
    await dismiss_popups(page)
    target = page.locator(", ".join(plan.selectors))
    if not await first_visible(target, timeout_ms=1500):
        # Preact client in narrow layouts hides the menus behind Battle/Switch.
        opener = "/movemenu" if plan.kind == "move" else "/switchmenu"
        await click_locator(page, page.locator(f"button[data-cmd='{opener}']"), plan.kind.title(), delay, 1000)
    if plan.terastallize:
        box = await first_visible(page.locator(", ".join(plan.tera_selectors)), timeout_ms=1500)
        if box is not None:
            try:
                if not await box.is_checked():
                    await click_locator(page, box, "Terastallize", delay, 1000)
                if not await box.is_checked():
                    await box.check()
            except Exception:
                pass
    return await click_locator(page, target, plan.label, delay, 3000)


async def click_any_control(page: Any, delay: float) -> bool:
    """Last resort when a planned button is missing: click any enabled choice."""
    for selector in (".movemenu button:not([disabled]):not(.disabled)", ".switchmenu button:not([disabled]):not(.disabled)"):
        if await click_locator(page, page.locator(selector), "Fallback choice", delay, 1000):
            return True
    return False


def infer_result(visible_text: str | None, username: str) -> str:
    text = (visible_text or "").lower()
    user = username.lower()
    if not text:
        return "unknown"
    if user and f"{user} won" in text:
        return "win"
    if user and (f"{user} lost" in text or f"{user} forfeited" in text):
        return "loss"
    if "you won" in text:
        return "win"
    if "you lost" in text or "you forfeited" in text:
        return "loss"
    if "forfeited" in text and (not user or user not in text):
        return "win"
    if "won" in text or "lost" in text or "forfeited" in text:
        return "unknown"
    return "unknown"


def new_battle_record(options: LiveOptions, battle_number: int) -> dict[str, Any]:
    return {
        "started_at": utc_now_iso(),
        "ended_at": "",
        "battle_number": battle_number,
        "username_mode": "guest" if options.guest else "account",
        "site_url": options.site,
        "format": options.format_name or "current format",
        "result": "unknown",
        "turns": 0,
        "selected_moves": [],
        "selected_switches": [],
        "forced_switches": 0,
        "policy": options.policy,
        "model_path": str(options.model_path or ""),
        "start_rating": None,
        "end_rating": None,
        "errors": [],
        "video_path": "",
        "visible_result_text": "",
    }


def finish_battle_record(record: dict[str, Any], *, visible_text: str | None, username: str) -> None:
    record["ended_at"] = utc_now_iso()
    record["visible_result_text"] = visible_text or ""
    if record.get("result") not in {"error", "win", "loss"}:
        record["result"] = infer_result(visible_text, username)


async def safe_ladder_rating(page: Any) -> int | None:
    try:
        value = await page.evaluate(GET_LADDER_RATING)
    except Exception:
        return None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def save_battle_records(records: list[dict[str, Any]], options: LiveOptions, video_path: Path | None) -> None:
    if not options.stats_enabled or not records:
        return
    for record in records:
        if video_path and video_path.exists():
            record["video_path"] = str(video_path)
        path = append_battle_record(record, options.stats_dir)
    print(f"\nStats saved: {path}", flush=True)


def debug_turn_snapshot(
    *,
    options: LiveOptions,
    battle_number: int,
    turn: int,
    summary: dict[str, Any],
    order: str,
    plan: dict[str, Any] | None = None,
    policy_source: str = "smart",
    fallback_reason: str = "",
    action: int | None = None,
    protocol_mode: str = "stream",
    parse_errors: list[str] | None = None,
) -> dict[str, Any]:
    """Redacted per-decision snapshot (no usernames or credentials)."""
    summary = {key: value for key, value in summary.items() if key not in {"username", "player_username"}}
    return {
        "captured_at": utc_now_iso(),
        "battle_number": battle_number,
        "turn": turn,
        "policy": options.policy,
        "policy_source": policy_source,
        "fallback_reason": fallback_reason,
        "site_url": options.site,
        "format": options.format_name or "current format",
        "username_mode": "guest" if options.guest else "account",
        "protocol_mode": protocol_mode,
        "state": summary,
        "order": order,
        "action": action,
        "click": plan or {},
        "parse_errors": list(parse_errors or [])[-5:],
    }


def describe_state(state: ProtocolBattle) -> str:
    battle = state.battle
    if battle is None:
        return ""
    me, opp = battle.active_pokemon, battle.opponent_active_pokemon
    ours = f"{me.species} {round(me.current_hp_fraction * 100)}%" if me else "?"
    theirs = f"{opp.species} {round(opp.current_hp_fraction * 100)}%" if opp else "?"
    return f"{ours} vs {theirs}"


def decision_for(state: ProtocolBattle, policy: LivePolicy | None, attempt: int) -> Decision:
    """First try: model (or heuristic). Retries after a rejected choice degrade."""
    from poke_env.player import Player

    from showdownrl.policy_bridge import Decision, choose_order, heuristic_decision

    battle = state.battle
    assert battle is not None
    if attempt == 0:
        return choose_order(battle, policy)
    reason = f"retry {attempt}: {state.last_error or 'previous choice was not registered'}"
    if attempt == 1:
        return heuristic_decision(battle, reason)
    return Decision(order=Player.choose_random_singles_move(battle), source="random", fallback_reason=reason)


def record_decision(record: dict[str, Any], state: ProtocolBattle, decision: Decision, plan: ClickPlan) -> None:
    from poke_env.battle import Move

    battle = state.battle
    assert battle is not None
    record["turns"] = max(int(record.get("turns") or 0), battle.turn)
    common = {
        "turn": battle.turn,
        "policy_source": decision.source,
        "fallback_reason": decision.fallback_reason,
        "action": decision.action,
    }
    if plan.kind == "move" and isinstance(decision.order.order, Move):
        move = decision.order.order
        record["selected_moves"].append(
            {
                **common,
                "name": plan.label.replace(" + Tera", ""),
                "id": move.id,
                "type": move.type.name.title() if move.type else "",
                "terastallize": plan.terastallize,
            }
        )
    elif plan.kind == "switch":
        forced = bool(battle.force_switch)
        record["forced_switches"] += int(forced)
        record["selected_switches"].append({**common, "name": plan.species, "forced": forced})


async def play_battle(
    page: Any,
    options: LiveOptions,
    record: dict[str, Any],
    protocol: BrowserProtocol,
    policy: LivePolicy | None,
    battle_number: int,
    deadline: float | None,
) -> tuple[str | None, bool]:
    """Play one battle from protocol state. Returns (visible result, stopped by time)."""
    from dataclasses import asdict

    from showdownrl.protocol_battle import battle_summary, order_to_click

    loop = asyncio.get_running_loop()
    # Rooms that already ended (e.g. the previous battle) must not end this one.
    stale = {tag for tag, room in protocol.tracker.rooms.items() if room.finished}
    decisions = 0
    attempts: dict[tuple[str, int], int] = {}
    reported_errors = 0
    timer_rooms: set[str] = set()
    progress_key: Any = None
    last_progress = loop.time()
    last_warn = last_progress
    last_conn_check = last_progress
    current_room = ""
    while decisions < options.max_turns:
        if loop.time() - last_conn_check >= CONNECTION_CHECK_SECONDS:
            last_conn_check = loop.time()
            if await connection_lost(page):
                print("  Connection to Showdown lost; reloading and rejoining the battle.", flush=True)
                record["errors"].append("connection lost; reconnected")
                await reconnect_to_battle(page, options, current_room)
                attempts.clear()
                timer_rooms.discard(current_room)
                continue
        if deadline is not None and loop.time() >= deadline:
            record["errors"].append(f"stopped after max time {options.max_time_minutes:g} minutes")
            print(f"\n  Stopped after --max-time={options.max_time_minutes:g} minutes.", flush=True)
            return None, True

        state = await protocol.refresh(page)
        if state is not None and state.battle_tag in stale:
            state = None
        if state is not None and len(state.parse_errors) > reported_errors:
            for error in state.parse_errors[reported_errors:]:
                print(f"  Protocol parse warning: {error}", flush=True)
            reported_errors = len(state.parse_errors)

        if state is not None and not state.finished:
            current_room = state.battle_tag or current_room
            if state.battle_tag and state.battle_tag not in timer_rooms:
                timer_rooms.add(state.battle_tag)
                via = await send_room_command(page, state.battle_tag, "/timer on")
                print(f"  Battle timer on ({via}).", flush=True)
            key = (state.battle_tag, state.request_seq, state.battle.turn)
            now = loop.time()
            if key != progress_key:
                progress_key, last_progress, last_warn = key, now, now
            elif now - last_progress >= STALL_ABANDON_SECONDS:
                print(f"\n  No progress for {STALL_ABANDON_SECONDS // 60} minutes; leaving this battle.", flush=True)
                record["errors"].append(f"stalled at turn {state.battle.turn}")
                return None, False
            elif now - last_warn >= STALL_WARN_SECONDS:
                last_warn = now
                waiting = "our move" if state.needs_decision else "the opponent"
                ready = await controls_ready(page)
                conn = await page.evaluate(CONNECTION_STATE)
                print(f"  No progress for {int(now - last_progress)}s at turn {state.battle.turn} "
                      f"(waiting on {waiting}; controls ready: {ready}; connection state {conn}); "
                      "re-sending /timer on.", flush=True)
                await save_stall_screenshot(page, options, state.battle_tag, state.battle.turn)
                await send_room_command(page, state.battle_tag, "/timer on")

        if state is not None and state.finished:
            battle = state.battle
            if battle.won is True:
                record["result"] = "win"
            elif battle.won is False:
                record["result"] = "loss"
            record["turns"] = max(int(record.get("turns") or 0), battle.turn)
            print(f"\n  Battle finished ({record['result']}) after turn {battle.turn}.", flush=True)
            return await page.evaluate(VISIBLE_RESULT), False

        if state is not None and state.request and state.request.get("teamPreview"):
            if await click_team_preview(page, options.click_delay):
                print("  Team preview confirmed.", flush=True)
                protocol.mark_answered(state)

        if state is not None and state.needs_decision:
            if not await controls_ready(page):
                await asyncio.sleep(0.4)
                continue
            key = (state.battle_tag, state.request_seq)
            attempt = attempts.get(key, 0)
            attempts[key] = attempt + 1
            decision = decision_for(state, policy, attempt)
            plan = order_to_click(decision.order, state.request)
            battle = state.battle
            tag = f"{decision.source}" + (f", {decision.fallback_reason}" if decision.fallback_reason else "")
            print(f"  Turn {battle.turn}: {plan.label or decision.order} [{tag}] | {describe_state(state)}", flush=True)
            if options.debug_policy:
                snapshot_path = write_debug_snapshot(
                    debug_turn_snapshot(
                        options=options,
                        battle_number=battle_number,
                        turn=battle.turn,
                        summary=battle_summary(battle),
                        order=str(decision.order),
                        plan=asdict(plan),
                        policy_source=decision.source,
                        fallback_reason=decision.fallback_reason,
                        action=decision.action,
                        protocol_mode=protocol.mode,
                        parse_errors=state.parse_errors,
                    ),
                    options.stats_dir,
                )
                print(f"  Debug snapshot: {snapshot_path}", flush=True)

            request_seq = state.request_seq
            clicked = await execute_click_plan(page, plan, options.click_delay)
            if not clicked and attempt >= 2:
                print(f"  Could not find the button for {plan.label!r}; using any available choice.", flush=True)
                clicked = await click_any_control(page, options.click_delay)
            if clicked and await wait_choice_sent(page, state, request_seq):
                if state.request_seq == request_seq:
                    protocol.mark_answered(state)
                record_decision(record, state, decision, plan)
                decisions += 1
            else:
                # Nothing was sent (missing button, popup, re-render): retry.
                record["errors"].append(f"turn {battle.turn}: {plan.kind} {plan.label!r} was not sent (try {attempt + 1})")
                print(f"  Choice {plan.label!r} was not registered; retrying.", flush=True)
                await asyncio.sleep(0.5)
            continue

        visible_result = await page.evaluate(VISIBLE_RESULT)
        if visible_result:
            return visible_result, False
        if not await page.evaluate(IN_BATTLE):
            print("\n  Battle panel closed.", flush=True)
            return None, False
        await asyncio.sleep(0.5)

    print(f"\n  Stopped after --max-turns={options.max_turns}.", flush=True)
    record["errors"].append(f"stopped after max turns {options.max_turns}")
    return None, False


async def run_live(options: LiveOptions) -> int:
    record_dir = options.record_dir or default_record_dir()
    if options.record:
        record_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 58, flush=True)
    print("  ShowdownRL - Live Browser Battle", flush=True)
    account_label = "not needed for UI check" if options.check_ui_only else f"{options.username}{' (guest)' if options.guest else ''}"
    print(f"  Account: {account_label}", flush=True)
    print(f"  Policy: {options.policy}", flush=True)
    if options.headless:
        print("  Running headless (no browser window).", flush=True)
    else:
        print("  A visible browser will open. Watch the AI pointer markers.", flush=True)
    print("=" * 58, flush=True)

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("Playwright is not installed. Run `showdownrl setup`.", flush=True)
        return 2

    video_path: Path | None = None
    battle_records: list[dict[str, Any]] = []
    active_record: dict[str, Any] | None = None
    policy: LivePolicy | None = None
    protocol: BrowserProtocol | None = None
    if not options.check_ui_only:
        try:
            protocol = BrowserProtocol(options.username)
        except ImportError as exc:
            print(f"Live play needs poke-env: pip install -e '.[rl]' ({exc})", flush=True)
            return 2
    if options.policy == "search":
        from showdownrl.policy_bridge import LivePolicy, PolicyLoadError, SearchLivePolicy

        try:
            policy = SearchLivePolicy(options.model_path, n_samples=options.search_samples,
                                      time_ms=options.search_time_ms)
            print(f"  Search: {options.search_samples} samples x {options.search_time_ms} ms, "
                  f"fallback model {policy.model_path}", flush=True)
        except PolicyLoadError as exc:
            print(f"  Search unavailable ({exc}); trying the PPO model.", flush=True)
            try:
                policy = LivePolicy(options.model_path)
                print(f"  Loaded PPO model: {policy.model_path}", flush=True)
            except PolicyLoadError as exc2:
                print(f"  PPO unavailable; using the smart damage-calc heuristic. {exc2}", flush=True)
    elif options.policy == "ppo":
        from showdownrl.policy_bridge import LivePolicy, PolicyLoadError

        try:
            policy = LivePolicy(options.model_path)
            print(f"  Loaded PPO model: {policy.model_path}", flush=True)
        except PolicyLoadError as exc:
            print(f"  PPO unavailable; using the smart damage-calc heuristic. {exc}", flush=True)
    else:
        print("  Using the smart damage-calc heuristic.", flush=True)
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=options.headless,
                slow_mo=options.slow_mo_ms,
                args=["--no-sandbox"],
            )
            context_kwargs: dict[str, Any] = {
                "viewport": {"width": options.viewport_width, "height": options.viewport_height},
            }
            if options.record:
                context_kwargs.update(
                    {
                        "record_video_dir": str(record_dir),
                        "record_video_size": {"width": options.viewport_width, "height": options.viewport_height},
                    }
                )
            context = await browser.new_context(**context_kwargs)
            await context.add_init_script(PROTOCOL_HOOK)
            page = await context.new_page()

            print("\n[1] Loading Pokemon Showdown...", flush=True)
            await page.goto(options.site, wait_until="domcontentloaded", timeout=45_000)
            await page.evaluate(CLICK_MARKER)
            if not await wait_cond(page, CONNECTED, 45):
                print("  Warning: the lobby did not finish connecting before the timeout.", flush=True)
            await asyncio.sleep(1.5)

            if options.check_ui_only:
                await first_visible(button_with_text(page, r"choose name"), timeout_ms=8000)
                await first_visible(button_with_text(page, r"battle!|find a battle|find a random opponent"), timeout_ms=8000)
                checks = await collect_selector_health(page)
                print("\n[check-ui-only]", flush=True)
                for line in selector_health_lines(checks):
                    print(line, flush=True)
                await context.close()
                await browser.close()
                return 0 if all(check.ok for check in checks if check.required) else 1

            print(f"\n[2] Signing in as {options.username}...", flush=True)
            login_result = await login(page, options.username, options.password, options.guest, options.click_delay)
            print(f"  Login status: {login_result}", flush=True)
            if "password prompt shown" in login_result or "requires a password" in login_result:
                await context.close()
                await browser.close()
                print("Login failed. Try `showdownrl logout && showdownrl setup`.", flush=True)
                return 2

            if options.login_only:
                print("\n[login-only] Stopping before queueing a battle.", flush=True)
                await context.close()
                await browser.close()
                return 0

            deadline: float | None = None
            if options.max_time_minutes and options.max_time_minutes > 0:
                deadline = asyncio.get_running_loop().time() + options.max_time_minutes * 60

            for battle_number in range(1, max(1, options.max_battles) + 1):
                if deadline is not None and asyncio.get_running_loop().time() >= deadline:
                    print(f"\nReached --max-time={options.max_time_minutes:g} minutes before queueing another battle.", flush=True)
                    break

                record = new_battle_record(options, battle_number)
                if policy:
                    record["model_path"] = str(policy.model_path)
                active_record = record
                record["start_rating"] = await safe_ladder_rating(page)
                print(f"\n[3] Queueing for battle {battle_number}/{max(1, options.max_battles)}...", flush=True)
                await maybe_select_format(page, options.format_name, options.click_delay)
                if not await click_queue_battle(page, options.click_delay):
                    print("  Battle button was not visible. Try `showdownrl doctor`.", flush=True)
                    record["errors"].append("battle button was not visible")
                print("  Waiting for an opponent...", flush=True)

                if not await wait_cond(page, IN_BATTLE, 75):
                    print("  No battle yet; retrying the queue click once.", flush=True)
                    await click_queue_battle(page, options.click_delay)
                    await wait_cond(page, IN_BATTLE, 75)

                if not await page.evaluate(IN_BATTLE):
                    print("  Could not detect an active battle. Try `showdownrl doctor`.", flush=True)
                    record["result"] = "error"
                    record["errors"].append("could not detect an active battle")
                    finish_battle_record(record, visible_text=None, username=options.username)
                    battle_records.append(record)
                    active_record = None
                    if options.keep_open:
                        print("  Browser left open because --keep-open was set.", flush=True)
                        await asyncio.Event().wait()
                    if battle_number < max(1, options.max_battles):
                        await page.goto(options.site, wait_until="domcontentloaded", timeout=45_000)
                        await page.evaluate(CLICK_MARKER)
                        await asyncio.sleep(2)
                        login_result = await login(page, options.username, options.password, options.guest, options.click_delay)
                        print(f"  Login status before next battle: {login_result}", flush=True)
                        continue
                    break

                print("\n=== BATTLE STARTED ===\n", flush=True)
                assert protocol is not None
                visible_result, stopped_by_time = await play_battle(
                    page, options, record, protocol, policy, battle_number, deadline
                )
                if visible_result:
                    print(f"\n  {visible_result}", flush=True)

                if visible_result is None:
                    visible_result = await page.evaluate(VISIBLE_RESULT)
                record["end_rating"] = await safe_ladder_rating(page)
                finish_battle_record(record, visible_text=visible_result, username=options.username)
                battle_records.append(record)
                active_record = None
                print("\n=== BATTLE LOOP ENDED ===", flush=True)
                if stopped_by_time:
                    break
                if battle_number < max(1, options.max_battles):
                    await page.goto(options.site, wait_until="domcontentloaded", timeout=45_000)
                    await page.evaluate(CLICK_MARKER)
                    await asyncio.sleep(2)
                    login_result = await login(page, options.username, options.password, options.guest, options.click_delay)
                    print(f"  Login status before next battle: {login_result}", flush=True)
            await asyncio.sleep(2)

            if options.keep_open and not options.record:
                print("\nBrowser left open. Press Ctrl+C when you are done.", flush=True)
                await asyncio.Event().wait()

            page_video = page.video
            await context.close()
            if page_video:
                raw_video = Path(await page_video.path())
                video_path = record_dir / "showdown_live_battle.webm"
                if raw_video.exists():
                    if video_path.exists():
                        video_path.unlink()
                    shutil.move(str(raw_video), str(video_path))
            save_battle_records(battle_records, options, video_path)
            if options.keep_open:
                print("  Recording contexts must close before Playwright can save video.", flush=True)
            await browser.close()
    except (Exception, KeyboardInterrupt) as exc:
        message = str(exc)
        if "Executable doesn't exist" in message or "playwright install" in message:
            print("Chromium is missing. Run `showdownrl setup`.", flush=True)
            return 2
        if active_record:
            active_record["result"] = "error"
            active_record["errors"].append(message)
            finish_battle_record(active_record, visible_text=None, username=options.username)
            battle_records.append(active_record)
            save_battle_records(battle_records, options, video_path)
        print(f"ShowdownRL failed: {exc}", flush=True)
        print(f"Site: {options.site}", flush=True)
        print("Try `showdownrl doctor` for diagnostics.", flush=True)
        return 1

    if video_path and video_path.exists():
        print(f"\nVideo saved: {video_path} ({video_path.stat().st_size / 1024:.0f} KB)", flush=True)
    return 0
