from __future__ import annotations

import os
from pathlib import Path


def dotenv_value(name: str, dotenv_path: Path = Path(".env")) -> str | None:
    try:
        lines = dotenv_path.read_text().splitlines()
    except FileNotFoundError:
        return None

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].lstrip()
        key, separator, raw_value = stripped.partition("=")
        if separator != "=" or key.strip() != name:
            continue
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return value or None
    return None


def env_or_dotenv_path(name: str, dotenv_path: Path = Path(".env")) -> Path | None:
    value = os.environ.get(name) or dotenv_value(name, dotenv_path)
    return Path(value).expanduser() if value else None


def require_env_or_dotenv_path(name: str, label: str, value: str | Path | None = None) -> str:
    path = Path(value).expanduser() if value else env_or_dotenv_path(name)
    if path is None:
        raise SystemExit(f"{label} required; pass --rom-path or set {name} in the environment or .env")
    return str(path)
