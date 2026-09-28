#!/usr/bin/env python
"""Run Foul Play with per-battle protocol logging (a wrapper; Foul Play is not modified).

This file is executed with **Foul Play's** interpreter (``~/foul-play/.venv/bin/python``)
and imports nothing from ShowdownRL. It monkeypatches
``fp.websocket_client.PSWebsocketClient.receive_message`` / ``send_message`` so that
every websocket frame Foul Play receives for a battle room and every message it sends
to a battle room is appended, in order, to ``$FP_BATTLE_LOG_DIR/<battle-tag>.jsonl``,
then runs Foul Play's normal entry point with the same CLI arguments.

Each JSONL line is one of::

    {"dir": "meta", "username": "...", "t": ...}   # first line of every file
    {"dir": "in",   "msg": "<raw websocket frame>", "t": ...}
    {"dir": "out",  "msg": "<raw sent message, e.g. 'battle-x|/choose move 2 terastallize|7'>", "t": ...}

When the battle ends (``|win|`` / ``|tie|`` seen) the file is closed and a sibling
``<battle-tag>.done`` marker is created, so readers can tell finished logs from live ones.

If ``FP_BATTLE_LOG_DIR`` is unset the wrapper behaves exactly like ``run.py``.

Usage (cwd must be the Foul Play checkout, or set FOULPLAY_DIR)::

    FP_BATTLE_LOG_DIR=/tmp/fplogs ~/foul-play/.venv/bin/python scripts/foulplay_logged_run.py \\
        --websocket-uri ws://localhost:8000/showdown/websocket --ps-username fp1 \\
        --bot-mode accept_challenge --pokemon-format gen9randombattle
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

FP_DIR = Path(os.environ.get("FOULPLAY_DIR", os.getcwd())).expanduser().resolve()
sys.path.insert(0, str(FP_DIR))

from fp.websocket_client import PSWebsocketClient  # noqa: E402

logger = logging.getLogger(__name__)


class BattleLogger:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.files: dict[str, object] = {}
        self.finished: set[str] = set()  # tags whose battle ended (ignore trailing frames)
        self.username = ""

    def _file(self, tag: str):
        f = self.files.get(tag)
        if f is None:
            f = open(self.log_dir / f"{tag}.jsonl", "a", encoding="utf-8")
            f.write(json.dumps({"dir": "meta", "username": self.username, "t": time.time()}) + "\n")
            self.files[tag] = f
        return f

    def _write(self, tag: str, rec: dict) -> None:
        f = self._file(tag)
        f.write(json.dumps(rec) + "\n")
        f.flush()

    def received(self, msg: str) -> None:
        if not msg.startswith(">battle-"):
            return
        tag = msg.split("\n", 1)[0][1:].strip()
        if tag in self.finished:
            return
        self._write(tag, {"dir": "in", "msg": msg, "t": time.time()})
        if any(line.startswith("|win|") or line == "|tie" or line.startswith("|tie|")
               for line in msg.split("\n")[1:]):
            self.close(tag)

    def sent(self, room: str, message: str) -> None:
        if not room.startswith("battle-") or room not in self.files:
            return
        self._write(room, {"dir": "out", "msg": message, "t": time.time()})

    def close(self, tag: str) -> None:
        self.finished.add(tag)
        f = self.files.pop(tag, None)
        if f is not None:
            f.close()
            (self.log_dir / f"{tag}.done").touch()


def install(log_dir: Path) -> BattleLogger:
    blog = BattleLogger(log_dir)
    orig_recv = PSWebsocketClient.receive_message
    orig_send = PSWebsocketClient.send_message

    async def receive_message(self):
        msg = await orig_recv(self)
        blog.username = self.username or blog.username
        try:
            blog.received(msg)
        except Exception:  # logging must never break the bot
            logger.warning("battle log write failed:\n%s", traceback.format_exc())
        return msg

    async def send_message(self, room, message_list):
        await orig_send(self, room, message_list)
        try:
            blog.sent(room, room + "|" + "|".join(message_list))
        except Exception:
            logger.warning("battle log write failed:\n%s", traceback.format_exc())

    PSWebsocketClient.receive_message = receive_message
    PSWebsocketClient.send_message = send_message
    return blog


def use_thread_search() -> None:
    """Run the MCTS worlds in a worker thread instead of a fresh ProcessPoolExecutor.

    Foul Play creates a new process pool (spawn start method on macOS) for every
    decision; with ``--search-parallelism 1`` the sampled worlds are searched one after
    another in that single worker anyway, so a 1-thread pool gives the same search
    (same worlds, same time budget) without re-importing Foul Play in a new
    interpreter each turn. Opt-in via ``FP_THREAD_SEARCH=1``."""
    import concurrent.futures

    import fp.search.main as search_main

    search_main.ProcessPoolExecutor = concurrent.futures.ThreadPoolExecutor


def main() -> None:
    log_dir = os.environ.get("FP_BATTLE_LOG_DIR")
    if log_dir:
        install(Path(log_dir).expanduser())
    if os.environ.get("FP_THREAD_SEARCH") == "1":
        use_thread_search()
    os.chdir(FP_DIR)  # Foul Play resolves some data paths relative to cwd
    from fp.main import run_foul_play

    try:
        asyncio.run(run_foul_play())
    except Exception:
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
