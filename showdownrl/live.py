"""Visible Pokemon Showdown browser automation."""

from __future__ import annotations

import asyncio
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from showdownrl.ai import score_move
from showdownrl.config import DEFAULT_SITE, default_record_dir

CONNECTED = "() => document.querySelector('button[name=openSounds], .userbar') !== null"
IN_BATTLE = "() => document.querySelector('.battle') !== null"

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

GET_MOVES = """
() => {
    const menu = document.querySelector('.movemenu');
    if (!menu) return [];
    return Array.from(menu.querySelectorAll('button:not([disabled])')).map((button, index) => ({
        index,
        name: (button.getAttribute('data-move') || button.textContent || '').trim(),
        type: (button.getAttribute('data-movetype') || '').trim(),
        text: (button.textContent || '').replace(/\\s+/g, ' ').trim(),
    }));
}
"""

VISIBLE_RESULT = """
() => {
    const nodes = Array.from(document.querySelectorAll('.broadcast-green,.broadcast-red,.battle-history'));
    for (const node of nodes) {
        const text = (node.textContent || '').replace(/\\s+/g, ' ').trim();
        if (/\\b(won|lost|forfeited)\\b/i.test(text)) return text;
    }
    return null;
}
"""


@dataclass
class LiveOptions:
    username: str = ""
    password: str = ""
    guest: bool = False
    site: str = DEFAULT_SITE
    format_name: str = ""
    record: bool = True
    record_dir: Path | None = None
    keep_open: bool = False
    login_only: bool = False
    check_ui_only: bool = False
    max_turns: int = 200
    slow_mo_ms: int = 250
    click_delay: float = 0.75
    viewport_width: int = 1280
    viewport_height: int = 800


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


async def click_locator(page: Any, locator: Any, label: str, delay: float = 0.75) -> bool:
    target = await first_visible(locator)
    if not target:
        return False

    await target.scroll_into_view_if_needed()
    box = await target.bounding_box()
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


async def fill_first_visible(locator: Any, text: str) -> bool:
    target = await first_visible(locator)
    if not target:
        return False
    await target.fill(text)
    return True


async def login(page: Any, username: str, password: str, guest: bool, click_delay: float) -> str:
    userbar = page.locator(".userbar").filter(has_text=re.compile(re.escape(username), re.I))
    if username and await userbar.count():
        return "already logged in"

    choose_name = page.locator("button[name='login']")
    if not await click_locator(page, choose_name, "Choose Name", click_delay):
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


async def click_switch(page: Any, click_delay: float) -> bool:
    switches = page.locator(".switchmenu button:not([disabled])")
    return await click_locator(page, switches, "Switch", click_delay)


async def click_move(page: Any, move_index: int, label: str, click_delay: float) -> bool:
    moves = page.locator(".movemenu button:not([disabled])")
    return await click_locator(page, moves.nth(move_index), label, click_delay)


async def run_live(options: LiveOptions) -> int:
    record_dir = options.record_dir or default_record_dir()
    if options.record:
        record_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 58, flush=True)
    print("  ShowdownRL - Live Browser Battle", flush=True)
    account_label = "not needed for UI check" if options.check_ui_only else f"{options.username}{' (guest)' if options.guest else ''}"
    print(f"  Account: {account_label}", flush=True)
    print("  A visible browser will open. Watch the AI pointer markers.", flush=True)
    print("=" * 58, flush=True)

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("Playwright is not installed. Run `showdownrl setup`.", flush=True)
        return 2

    video_path: Path | None = None
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=False,
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
            page = await context.new_page()

            print("\n[1] Loading Pokemon Showdown...", flush=True)
            await page.goto(options.site, wait_until="domcontentloaded", timeout=45_000)
            await page.evaluate(CLICK_MARKER)
            if not await wait_cond(page, CONNECTED, 45):
                print("  Warning: the lobby did not finish connecting before the timeout.", flush=True)
            await asyncio.sleep(1.5)

            if options.check_ui_only:
                choose_name = button_with_text(page, r"choose name")
                battle = button_with_text(page, r"battle!|find a battle|find a random opponent")
                await first_visible(choose_name, timeout_ms=8000)
                await first_visible(battle, timeout_ms=8000)
                choose_name_count = await choose_name.count()
                battle_count = await battle.count()
                print("\n[check-ui-only]", flush=True)
                print(f"  Choose Name buttons visible/matched: {choose_name_count}", flush=True)
                print(f"  Battle buttons visible/matched: {battle_count}", flush=True)
                await context.close()
                await browser.close()
                return 0 if choose_name_count and battle_count else 1

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

            print("\n[3] Queueing for a battle...", flush=True)
            await maybe_select_format(page, options.format_name, options.click_delay)
            if not await click_queue_battle(page, options.click_delay):
                print("  Battle button was not visible. Try `showdownrl doctor`.", flush=True)
            print("  Waiting for an opponent...", flush=True)

            if not await wait_cond(page, IN_BATTLE, 75):
                print("  No battle yet; retrying the queue click once.", flush=True)
                await click_queue_battle(page, options.click_delay)
                await wait_cond(page, IN_BATTLE, 75)

            if not await page.evaluate(IN_BATTLE):
                print("  Could not detect an active battle. Try `showdownrl doctor`.", flush=True)
                if options.keep_open:
                    print("  Browser left open because --keep-open was set.", flush=True)
                    await asyncio.Event().wait()
                await context.close()
                await browser.close()
                return 1

            print("\n=== BATTLE STARTED ===\n", flush=True)
            for turn in range(1, options.max_turns + 1):
                if await click_team_preview(page, options.click_delay):
                    print("  Team preview confirmed.", flush=True)
                    await asyncio.sleep(2.5)
                    continue

                moves = await page.evaluate(GET_MOVES)
                if moves:
                    best = max(moves, key=score_move)
                    print(
                        f"  Turn {turn}: {best['name']} ({best['type'] or 'unknown'}) "
                        f"from {[m['name'] for m in moves]}",
                        flush=True,
                    )
                    if await click_move(page, int(best["index"]), best["name"], options.click_delay):
                        await asyncio.sleep(3.25)
                        continue

                if await page.locator(".switchmenu button:not([disabled])").count():
                    if await click_switch(page, options.click_delay):
                        print("  Forced switch clicked.", flush=True)
                        await asyncio.sleep(2.5)
                        continue

                result = await page.evaluate(VISIBLE_RESULT)
                if result:
                    print(f"\n  {result}", flush=True)
                    break

                if not await page.evaluate(IN_BATTLE):
                    print("\n  Battle panel closed.", flush=True)
                    break

                await asyncio.sleep(1)
            else:
                print(f"\n  Stopped after --max-turns={options.max_turns}.", flush=True)

            print("\n=== BATTLE LOOP ENDED ===", flush=True)
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
            if options.keep_open:
                print("  Recording contexts must close before Playwright can save video.", flush=True)
            await browser.close()
    except Exception as exc:
        message = str(exc)
        if "Executable doesn't exist" in message or "playwright install" in message:
            print("Chromium is missing. Run `showdownrl setup`.", flush=True)
            return 2
        print(f"ShowdownRL failed: {exc}", flush=True)
        print(f"Site: {options.site}", flush=True)
        print("Try `showdownrl doctor` for diagnostics.", flush=True)
        return 1

    if video_path and video_path.exists():
        print(f"\nVideo saved: {video_path} ({video_path.stat().st_size / 1024:.0f} KB)", flush=True)
    return 0
