from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from typing import Any

from showdownrl.live import (
    PROTOCOL_HOOK,
    READ_PROTOCOL,
    ROOM_SNAPSHOT,
    BrowserProtocol,
    LiveOptions,
    SelectorHealth,
    click_locator,
    debug_turn_snapshot,
    infer_result,
    login,
    selector_health_lines,
)


class LiveResultTests(unittest.TestCase):
    def test_infers_result_without_storing_credentials(self) -> None:
        self.assertEqual(infer_result("aquerro won the battle!", "aquerro"), "win")
        self.assertEqual(infer_result("aquerro forfeited.", "aquerro"), "loss")
        self.assertEqual(infer_result("Opponent forfeited.", "aquerro"), "win")
        self.assertEqual(infer_result("The battle ended.", "aquerro"), "unknown")

    def test_debug_turn_snapshot_is_redacted(self) -> None:
        options = LiveOptions(
            username="aquerro",
            password="secret-password",
            guest=False,
            site="https://play.pokemonshowdown.com/",
            format_name="Random Battle",
        )
        snapshot = debug_turn_snapshot(
            options=options,
            battle_number=1,
            turn=2,
            summary={
                "role": "p1",
                "turn": 2,
                "active": "pikachu",
                "opponent": "gyarados",
                "moves": ["thunderbolt"],
                "switches": [],
                "username": "aquerro",
            },
            order="/choose move thunderbolt",
            plan={"kind": "move", "slot": 1, "label": "Thunderbolt"},
            policy_source="ppo",
            action=6,
        )

        self.assertEqual(snapshot["username_mode"], "account")
        self.assertEqual(snapshot["state"]["active"], "pikachu")
        self.assertEqual(snapshot["order"], "/choose move thunderbolt")
        self.assertNotIn("username", snapshot)
        self.assertNotIn("password", snapshot)
        self.assertNotIn("aquerro", repr(snapshot))
        self.assertNotIn("secret-password", repr(snapshot))

    def test_protocol_hook_wraps_both_clients(self) -> None:
        # Classic client exposes window.app.receive, the preact client PS.receive.
        self.assertIn("'app'", PROTOCOL_HOOK)
        self.assertIn("'PS'", PROTOCOL_HOOK)
        self.assertIn(">battle-", PROTOCOL_HOOK)
        self.assertIn("messages.slice(start)", READ_PROTOCOL)

    def test_selector_health_lines_mark_required_failures(self) -> None:
        lines = selector_health_lines(
            [
                SelectorHealth("choose-name button", True, True, 1),
                SelectorHealth("battle queue button", False, True, 0),
                SelectorHealth("move buttons", False, False, 0, "expected during a turn"),
            ]
        )

        self.assertIn("[ok] choose-name button (1 visible)", lines[0])
        self.assertIn("[fail] battle queue button (0 visible)", lines[1])
        self.assertIn("[info] move buttons (0 visible) - expected during a turn", lines[2])


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "protocol_p2.json"


class FakePage:
    """Stands in for a Playwright page running the receive hook."""

    def __init__(self, messages: list[str], hooked: bool = True):
        self.messages = messages
        self.hooked = hooked
        self.snapshot: list[dict[str, Any]] = []

    async def evaluate(self, script: str, arg: Any = None) -> Any:
        if script == READ_PROTOCOL:
            if not self.hooked:
                return {"installed": True, "hooked": [], "total": 0, "messages": []}
            return {"installed": True, "hooked": ["app"], "total": len(self.messages),
                    "messages": self.messages[arg:]}
        if script == ROOM_SNAPSHOT:
            return self.snapshot
        raise AssertionError("unexpected script")


class BrowserProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.first = self.fixture["decisions"][0]["after_message"]

    def test_stream_mode_reads_incrementally_and_ignores_other_rooms(self) -> None:
        messages = ["|updateuser| Guest|0|1", ">lobby\n|c|~|hello"] + self.fixture["messages"][: self.first + 1]
        page = FakePage(messages[:3])
        protocol = BrowserProtocol("")

        async def run():
            first = await protocol.refresh(page)
            page.messages = messages
            second = await protocol.refresh(page)
            return first, second

        first, second = asyncio.run(run())
        self.assertIsNone(first)  # only the |init| message so far: side unknown
        self.assertIsNotNone(second)
        self.assertEqual(protocol.mode, "stream")
        self.assertEqual(protocol.cursor, len(messages))
        self.assertTrue(second.needs_decision)
        self.assertEqual(second.battle.player_role, "p2")
        protocol.mark_answered(second)
        self.assertFalse(second.needs_decision)

    def test_reload_resets_cursor(self) -> None:
        page = FakePage(self.fixture["messages"][: self.first + 1])
        protocol = BrowserProtocol("")
        asyncio.run(protocol.refresh(page))
        page.messages = []  # page reloaded: the in-page store starts empty again
        asyncio.run(protocol.refresh(page))
        self.assertEqual(protocol.cursor, 0)

    def test_snapshot_fallback_when_hook_missing(self) -> None:
        lines, request = [], None
        for message in self.fixture["messages"][: self.first + 1]:
            for line in message.split("\n")[1:]:
                if line.startswith("|request|"):
                    request = json.loads(line[len("|request|"):])
                else:
                    lines.append(line)
        page = FakePage([], hooked=False)
        page.snapshot = [{"id": self.fixture["battle_tag"], "lines": lines, "request": request, "focused": True}]
        protocol = BrowserProtocol("")
        state = asyncio.run(protocol.refresh(page))
        self.assertEqual(protocol.mode, "snapshot")
        self.assertTrue(state.needs_decision)
        protocol.mark_answered(state)
        # The next poll rebuilds from scratch but remembers the answered request.
        again = asyncio.run(protocol.refresh(page))
        self.assertFalse(again.needs_decision)


class FakeLocator:
    """Locator whose element becomes visible after `appears_after` count() polls."""

    def __init__(self, appears_after: int = 0, vanishes: bool = False):
        self.appears_after = appears_after
        self.vanishes = vanishes
        self.polls = 0
        self.clicked = False

    def filter(self, **_: Any) -> "FakeLocator":
        return self

    def nth(self, _: int) -> "FakeLocator":
        return self

    async def count(self) -> int:
        self.polls += 1
        return 1 if self.polls > self.appears_after else 0

    async def is_visible(self) -> bool:
        return True

    async def scroll_into_view_if_needed(self, timeout: float | None = None) -> None:
        pass

    async def bounding_box(self, timeout: float | None = None) -> dict[str, float] | None:
        if self.vanishes:
            raise TimeoutError("Locator.bounding_box: Timeout exceeded.")
        return {"x": 0, "y": 0, "width": 10, "height": 10}


class FakeLoginPage:
    def __init__(self, locators: dict[str, FakeLocator]):
        self.locators = locators

    def locator(self, selector: str) -> FakeLocator:
        return self.locators[selector]


class LiveLoginTests(unittest.TestCase):
    def test_click_locator_returns_false_when_element_vanishes(self) -> None:
        page = FakeLoginPage({})
        clicked = asyncio.run(click_locator(page, FakeLocator(vanishes=True), "Choose Name", delay=0))
        self.assertFalse(clicked)

    def test_login_waits_for_restored_session_after_reload(self) -> None:
        # Userbar shows our name only after a few polls, like Showdown's async session restore.
        userbar = FakeLocator(appears_after=3)
        choose_name = FakeLocator(vanishes=True)
        page = FakeLoginPage({".userbar": userbar, "button[name='login']": choose_name})
        result = asyncio.run(login(page, "arosTar", "", guest=False, click_delay=0))
        self.assertEqual(result, "already logged in")
        self.assertEqual(choose_name.polls, 0)


if __name__ == "__main__":
    unittest.main()
