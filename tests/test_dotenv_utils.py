from __future__ import annotations

from pathlib import Path

import pytest

from scripts.dotenv_utils import dotenv_value, env_or_dotenv_path
from scripts.run_pypi_stable_retro_turbo_benchmark import parse_args as parse_stable_pypi_args
from scripts.run_pypi_supermariobrosnes_turbo_benchmark import parse_args as parse_smb_pypi_args
from supermariobrosnes_turbo import default_rom_path


def test_dotenv_value_supports_export_and_quotes(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "# local secrets\n"
        "export SMB_ROM_PATH='~/roms/SuperMarioBros.nes'\n"
    )

    assert dotenv_value("SMB_ROM_PATH", dotenv) == "~/roms/SuperMarioBros.nes"


def test_env_or_dotenv_path_prefers_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("SMB_ROM_PATH=/from/dotenv.nes\n")
    monkeypatch.setenv("SMB_ROM_PATH", "/from/env.nes")

    assert env_or_dotenv_path("SMB_ROM_PATH", dotenv) == Path("/from/env.nes")


def test_package_default_rom_path_reads_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SMB_ROM_PATH", raising=False)
    (tmp_path / ".env").write_text("SMB_ROM_PATH=/tmp/from-dotenv.nes\n")

    assert default_rom_path() == Path("/tmp/from-dotenv.nes")


def test_pypi_benchmark_wrappers_read_rom_path_from_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SMB_ROM_PATH", raising=False)
    (tmp_path / ".env").write_text('SMB_ROM_PATH="/tmp/from-dotenv.nes"\n')

    assert parse_stable_pypi_args(["--version", "1.0.0"]).rom_path == "/tmp/from-dotenv.nes"
    assert parse_smb_pypi_args(["--version", "1.0.0"]).rom_path == "/tmp/from-dotenv.nes"
