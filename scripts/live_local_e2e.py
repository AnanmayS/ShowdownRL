#!/usr/bin/env python
"""End-to-end check of the live decision path against a LOCAL Showdown server.

Opens the official web client at https://localhost.psim.us/ (it connects to
ws://localhost:8000), lets a poke-env SimpleHeuristicsPlayer challenge the
browser's auto-assigned guest, accepts the challenge through the client's
command channel, and then plays with ``showdownrl.live.play_battle`` - the same
protocol-driven loop ``showdownrl live`` uses - clicking the real buttons.
No account login and no public-server battles are involved.

Usage::

    python scripts/live_local_e2e.py --battles 2 [--model models/real/x.zip] [--headed]
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_env.player import SimpleHeuristicsPlayer  # noqa: E402
from poke_env.ps_client import AccountConfiguration  # noqa: E402

from showdownrl.live import (  # noqa: E402
    CLICK_MARKER,
    PROTOCOL_HOOK,
    BrowserProtocol,
    LiveOptions,
    new_battle_record,
    play_battle,
)
from showdownrl.policy_bridge import LivePolicy  # noqa: E402
from showdownrl.real_env import BATTLE_FORMAT, server_configuration  # noqa: E402

# Chrome blocks a public origin (localhost.psim.us) from opening ws://localhost
# unless Local Network Access checks are disabled.
CHROME_ARGS = ["--disable-features=LocalNetworkAccessChecks,BlockInsecurePrivateNetworkRequests"]


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--battles", type=int, default=1)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=Path, help="MaskablePPO model; default is the smart heuristic")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--channel", default="chrome", help="Playwright browser channel ('' for bundled Chromium)")
    args = parser.parse_args()

    from playwright.async_api import async_playwright

    policy = LivePolicy(args.model) if args.model else None
    opponent = SimpleHeuristicsPlayer(
        account_configuration=AccountConfiguration(f"srl e2e {random.randrange(10000)}", None),
        battle_format=BATTLE_FORMAT,
        server_configuration=server_configuration(args.port),
        log_level=40,
    )
    wins = 0
    async with async_playwright() as playwright:
        launch = {"headless": not args.headed, "args": CHROME_ARGS}
        if args.channel:
            launch["channel"] = args.channel
        browser = await playwright.chromium.launch(**launch)
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        await context.add_init_script(PROTOCOL_HOOK)
        page = await context.new_page()
        site = "https://localhost.psim.us/" if args.port == 8000 else f"https://localhost--{args.port}.psim.us/"
        await page.goto(site, wait_until="domcontentloaded", timeout=45_000)
        await page.evaluate(CLICK_MARKER)
        await page.wait_for_function("() => window.app && app.user && app.user.get('userid')", timeout=45_000)
        guest = await page.evaluate("() => app.user.get('name')")
        print(f"browser user: {guest}", flush=True)

        options = LiveOptions(username=guest, guest=True, click_delay=0.2, max_turns=400)
        protocol = BrowserProtocol(guest)
        for number in range(1, args.battles + 1):
            challenge = asyncio.ensure_future(opponent.send_challenges(guest, n_challenges=1))
            accepted = False
            for tick in range(60):
                await asyncio.sleep(0.5)
                if not accepted and (tick >= 3 or await page.evaluate("() => !!document.querySelector('.challenge')")):
                    await page.evaluate(f"() => app.send('/accept {opponent.username}')")
                    accepted = True
                if await page.evaluate("() => !!document.querySelector('.battle')"):
                    break
            record = new_battle_record(options, number)
            await play_battle(page, options, record, protocol, policy, number, None)
            await challenge
            wins += record["result"] == "win"
            tera = sum(bool(move.get("terastallize")) for move in record["selected_moves"])
            print(
                f"battle {number}: {record['result']} turns={record['turns']} moves={len(record['selected_moves'])} "
                f"switches={len(record['selected_switches'])} (forced {record['forced_switches']}) tera={tera} "
                f"errors={record['errors']}",
                flush=True,
            )
            await page.evaluate("() => { for (const id in app.rooms) if (id.startsWith('battle-')) app.leaveRoom(id); }")
        await browser.close()
    print(f"won {wins}/{args.battles}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
