#!/usr/bin/env python
"""Compatibility wrapper for the packaged ShowdownRL live command."""

from __future__ import annotations

import sys

from showdownrl.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["live", *sys.argv[1:]]))
