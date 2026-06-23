"""Command-line interface for ShowdownRL."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import platform
import subprocess
import sys
from pathlib import Path

from showdownrl.config import CONFIG_FILE, DEFAULT_SITE, UserConfig, delete_config, load_config, merged_config, save_config
from showdownrl.live import LiveOptions, run_live


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="showdownrl",
        description="Watch an AI play Pokemon Showdown in a visible browser.",
    )
    parser.add_argument("--version", action="version", version="showdownrl 0.1.0")

    subparsers = parser.add_subparsers(dest="command", required=True)

    setup = subparsers.add_parser("setup", help="Install browser support and save local account settings.")
    setup.add_argument("--guest", action="store_true", help="Configure guest mode without a password.")
    setup.add_argument("--username", help="Pokemon Showdown username or guest name.")
    setup.add_argument("--password", help="Pokemon Showdown password. Prefer the interactive prompt.")
    setup.add_argument("--site", default=DEFAULT_SITE, help="Pokemon Showdown URL.")
    setup.add_argument("--skip-browser-install", action="store_true", help="Do not run playwright install chromium.")

    check = subparsers.add_parser("check", help="Verify config, browser, login, and website selectors.")
    add_common_live_args(check)
    check.add_argument("--skip-login", action="store_true", help="Only check the public lobby selectors.")

    live = subparsers.add_parser("live", help="Launch the visible AI player.")
    add_common_live_args(live)
    live.add_argument("--format", default="", help="Optional format text to select before queueing.")
    live.add_argument("--no-record", action="store_true", help="Do not save a WebM recording.")
    live.add_argument("--check-ui-only", action="store_true", help=argparse.SUPPRESS)
    live.add_argument("--record-dir", type=Path, help="Directory for the WebM recording.")
    live.add_argument("--keep-open", action="store_true", help="Leave the browser open after the battle loop ends.")
    live.add_argument("--max-turns", type=int, default=200, help="Maximum AI action cycles before stopping.")
    live.add_argument("--slow-mo-ms", type=int, default=250, help="Browser slow-motion delay in milliseconds.")
    live.add_argument("--click-delay", type=float, default=0.75, help="Pause after visible clicks.")
    live.add_argument("--viewport-width", type=int, default=1280)
    live.add_argument("--viewport-height", type=int, default=800)

    logout = subparsers.add_parser("logout", help="Delete saved local ShowdownRL credentials.")
    logout.add_argument("--yes", action="store_true", help="Do not ask for confirmation.")

    subparsers.add_parser("doctor", help="Print diagnostics without exposing secrets.")
    return parser


def add_common_live_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--username", help="Override saved username.")
    parser.add_argument("--password", help="Override saved password.")
    parser.add_argument("--guest", action="store_true", help="Use guest mode for this run.")
    parser.add_argument("--site", help="Pokemon Showdown URL.")
    parser.add_argument("--login-only", action="store_true", help="Stop before queueing a battle.")


def install_chromium() -> int:
    print("\nInstalling Playwright Chromium. This can take a few minutes...", flush=True)
    result = subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=False)
    if result.returncode != 0:
        print("Chromium install failed. You can retry with `showdownrl setup`.", flush=True)
    return result.returncode


def setup_command(args: argparse.Namespace) -> int:
    print("ShowdownRL setup")
    print("This configures a local browser bot for Pokemon Showdown.")
    print("Your password is only used to log in to Pokemon Showdown and is stored locally.")
    print(f"Config file: {CONFIG_FILE}")

    guest = bool(args.guest)
    username = args.username
    password = args.password

    if not username:
        prompt = "Guest name" if guest else "Pokemon Showdown username"
        username = input(f"{prompt}: ").strip()
    if not username:
        print("Setup cancelled: username is required.")
        return 2

    if not guest and password is None:
        password = getpass.getpass("Pokemon Showdown password: ")
    if not guest and not password:
        print("Setup cancelled: password is required unless --guest is used.")
        return 2

    saved = save_config(UserConfig(username=username, password=password or "", guest=guest, site=args.site))
    print(f"Saved local config: {saved}")

    if args.skip_browser_install:
        print("Skipped browser install. Run `python -m playwright install chromium` if needed.")
        return 0
    return install_chromium()


def config_from_args(args: argparse.Namespace) -> UserConfig:
    password = args.password if hasattr(args, "password") else None
    username = args.username if hasattr(args, "username") else None
    site = args.site if hasattr(args, "site") else None
    guest = True if getattr(args, "guest", False) else None
    return merged_config(username=username, password=password, guest=guest, site=site)


def options_from_args(args: argparse.Namespace, *, check_ui_only: bool = False) -> LiveOptions:
    config = config_from_args(args)
    return LiveOptions(
        username=config.username,
        password=config.password,
        guest=config.guest,
        site=config.site,
        format_name=getattr(args, "format", ""),
        record=not getattr(args, "no_record", False),
        record_dir=getattr(args, "record_dir", None),
        keep_open=getattr(args, "keep_open", False),
        login_only=getattr(args, "login_only", False),
        check_ui_only=check_ui_only,
        max_turns=getattr(args, "max_turns", 200),
        slow_mo_ms=getattr(args, "slow_mo_ms", 250),
        click_delay=getattr(args, "click_delay", 0.75),
        viewport_width=getattr(args, "viewport_width", 1280),
        viewport_height=getattr(args, "viewport_height", 800),
    )


def validate_login_options(options: LiveOptions) -> int:
    if options.check_ui_only:
        return 0
    if not options.username:
        print("Missing credentials. Run `showdownrl setup` or use `--guest`.")
        return 2
    if not options.guest and not options.password:
        print("Missing credentials. Run `showdownrl setup` or use `--guest`.")
        return 2
    return 0


def check_command(args: argparse.Namespace) -> int:
    if args.skip_login:
        options = options_from_args(args, check_ui_only=True)
    else:
        options = options_from_args(args)
        options.login_only = True
        options.record = False
        options.slow_mo_ms = 0
    validation = validate_login_options(options)
    if validation:
        return validation
    return asyncio.run(run_live(options))


def live_command(args: argparse.Namespace) -> int:
    options = options_from_args(args, check_ui_only=getattr(args, "check_ui_only", False))
    validation = validate_login_options(options)
    if validation:
        return validation
    return asyncio.run(run_live(options))


def logout_command(args: argparse.Namespace) -> int:
    if CONFIG_FILE.exists() and not args.yes:
        answer = input(f"Delete saved ShowdownRL config at {CONFIG_FILE}? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Nothing deleted.")
            return 0
    deleted = delete_config()
    print("Deleted saved ShowdownRL credentials." if deleted else "No saved ShowdownRL config found.")
    return 0


def doctor_command() -> int:
    config = load_config()
    print("ShowdownRL doctor")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Config path: {CONFIG_FILE}")
    print(f"Config exists: {CONFIG_FILE.exists()}")
    print(f"Configured username: {'yes' if config.username else 'no'}")
    print(f"Configured password: {'yes' if config.password else 'no'}")
    print(f"Guest mode: {'yes' if config.guest else 'no'}")
    print(f"Site: {config.site or DEFAULT_SITE}")
    try:
        import playwright  # noqa: F401

        print("Playwright import: ok")
    except Exception as exc:
        print(f"Playwright import: failed ({exc})")
        print("Run `showdownrl setup`.")
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "setup":
        return setup_command(args)
    if args.command == "check":
        return check_command(args)
    if args.command == "live":
        return live_command(args)
    if args.command == "logout":
        return logout_command(args)
    if args.command == "doctor":
        return doctor_command()
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
