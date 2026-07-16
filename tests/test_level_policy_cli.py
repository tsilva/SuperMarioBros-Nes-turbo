from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from supermariobrosnes_turbo.jerk import (
    normalize_level_name,
    policy_path_for_level,
    run_directory_for_level,
)
from train import build_parser as build_train_parser


ROOT = Path(__file__).resolve().parents[1]


def load_level_player():
    spec = importlib.util.spec_from_file_location("level_policy_cli", ROOT / "play.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_level_deterministically_selects_run_and_policy_names() -> None:
    assert normalize_level_name("Level1-1") == "Level1-1"
    assert run_directory_for_level("Level1-1") == Path("runs/Level1-1-jerk")
    assert policy_path_for_level("Level1-1") == Path("runs/Level1-1-jerk/Level1-1.zip")


@pytest.mark.parametrize("level", ["level1-1", "Level1", "../Level1-1", "Level0-1"])
def test_invalid_level_names_cannot_escape_policy_directory(level: str) -> None:
    with pytest.raises(ValueError, match="invalid level name"):
        policy_path_for_level(level)


def test_train_command_takes_level_as_its_positional_key() -> None:
    args = build_train_parser().parse_args(["Level1-1"])

    assert args.level == "Level1-1"
    assert args.output is None


def test_play_command_resolves_level_policy_and_forwards_options(
    tmp_path: Path,
) -> None:
    player = load_level_player()
    policy = tmp_path / "Level1-1-jerk" / "Level1-1.zip"
    policy.parent.mkdir(parents=True)
    policy.touch()

    argv = player.playback_argv(
        "Level1-1",
        ["--backend", "native"],
        runs_root=tmp_path,
    )

    assert argv == [
        str(policy),
        "--state",
        "Level1-1",
        "--backend",
        "native",
    ]


def test_play_command_explains_how_to_create_a_missing_policy(tmp_path: Path) -> None:
    player = load_level_player()

    with pytest.raises(FileNotFoundError, match=r"train\.py Level1-1"):
        player.resolve_level_policy("Level1-1", runs_root=tmp_path)
