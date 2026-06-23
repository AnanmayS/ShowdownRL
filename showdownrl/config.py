"""Local configuration for the ShowdownRL CLI."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

APP_NAME = "ShowdownRL"
DEFAULT_SITE = "https://play.pokemonshowdown.com/"


def _user_config_dir() -> Path:
    try:
        from platformdirs import user_config_dir

        return Path(user_config_dir(APP_NAME, appauthor=False))
    except Exception:
        if os.name == "posix" and Path.home().joinpath("Library").exists():
            return Path.home() / "Library" / "Application Support" / APP_NAME
        return Path.home() / ".config" / "showdownrl"


def _user_videos_dir() -> Path:
    try:
        from platformdirs import user_videos_dir

        return Path(user_videos_dir())
    except Exception:
        movies = Path.home() / "Movies"
        return movies if movies.exists() else Path.home()


def _user_data_dir() -> Path:
    try:
        from platformdirs import user_data_dir

        return Path(user_data_dir(APP_NAME, appauthor=False))
    except Exception:
        if os.name == "posix" and Path.home().joinpath("Library").exists():
            return Path.home() / "Library" / "Application Support" / APP_NAME
        return Path.home() / ".local" / "share" / "showdownrl"


CONFIG_DIR = _user_config_dir()
CONFIG_FILE = CONFIG_DIR / "config.env"
DATA_DIR = _user_data_dir()
DEFAULT_STATS_DIR = DATA_DIR / "stats"


@dataclass
class UserConfig:
    username: str = ""
    password: str = ""
    guest: bool = False
    site: str = DEFAULT_SITE

    @property
    def has_login(self) -> bool:
        return self.guest or bool(self.username and self.password)


def parse_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def load_config(path: Path = CONFIG_FILE) -> UserConfig:
    values = parse_env_file(path)
    return UserConfig(
        username=values.get("PS_USERNAME", ""),
        password=values.get("PS_PASSWORD", ""),
        guest=parse_bool(values.get("PS_GUEST")),
        site=values.get("PS_SITE", DEFAULT_SITE) or DEFAULT_SITE,
    )


def merged_config(
    *,
    username: str | None = None,
    password: str | None = None,
    guest: bool | None = None,
    site: str | None = None,
    path: Path = CONFIG_FILE,
) -> UserConfig:
    saved = load_config(path)
    env_username = os.environ.get("PS_USERNAME")
    env_password = os.environ.get("PS_PASSWORD") or os.environ.get("PS_PW")
    env_guest = os.environ.get("PS_GUEST")
    env_site = os.environ.get("PS_SITE")

    return UserConfig(
        username=username or env_username or saved.username,
        password=password if password is not None else (env_password if env_password is not None else saved.password),
        guest=guest if guest is not None else (parse_bool(env_guest) if env_guest is not None else saved.guest),
        site=site or env_site or saved.site or DEFAULT_SITE,
    )


def save_config(config: UserConfig, path: Path = CONFIG_FILE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Local ShowdownRL config. Do not commit this file.",
        f"PS_USERNAME={config.username}",
        f"PS_GUEST={'true' if config.guest else 'false'}",
        f"PS_SITE={config.site or DEFAULT_SITE}",
    ]
    if not config.guest and config.password:
        lines.append(f"PS_PASSWORD={config.password}")

    path.write_text("\n".join(lines) + "\n")
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    return path


def delete_config(path: Path = CONFIG_FILE) -> bool:
    if not path.exists():
        return False
    path.unlink()
    return True


def default_record_dir(cwd: Path | None = None) -> Path:
    cwd = cwd or Path.cwd()
    if (cwd / "showdownrl").is_dir() and (cwd / "README.md").exists():
        return cwd / "results"
    return _user_videos_dir() / APP_NAME


def default_stats_dir() -> Path:
    return DEFAULT_STATS_DIR
