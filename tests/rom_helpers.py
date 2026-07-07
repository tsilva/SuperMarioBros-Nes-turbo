from __future__ import annotations

from pathlib import Path

import pytest

from supermariobrosnes_turbo import ROM_PATH_ENV_VAR, default_rom_path


def require_rom() -> Path:
    rom_path = default_rom_path()
    if rom_path is None:
        pytest.fail(f"set {ROM_PATH_ENV_VAR} to run ROM-dependent tests")
    if not rom_path.exists():
        pytest.fail(f"{ROM_PATH_ENV_VAR} does not exist: {rom_path}")
    return rom_path
