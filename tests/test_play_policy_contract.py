from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import play_policy  # noqa: E402


def test_plain_checkpoint_uses_training_preprocessing_contract() -> None:
    args = play_policy.parse_args(["policy.pt"])

    play_policy.apply_checkpoint_defaults(args, Path("policy.pt"))

    assert args.backend == "native"
    assert args.max_pool_frames is False
    assert args.crop_mode == "mask"


def test_legacy_checkpoint_keeps_stable_retro_contract() -> None:
    args = play_policy.parse_args(["policy.zip"])

    play_policy.apply_checkpoint_defaults(args, Path("policy.zip"))

    assert args.backend == "stable-retro"
    assert args.max_pool_frames is True
    assert args.crop_mode == "remove"


def test_stable_retro_rejects_unrepresentable_mask_crop() -> None:
    args = play_policy.parse_args(
        ["policy.pt", "--backend", "stable-retro", "--crop-mode", "mask"]
    )

    with pytest.raises(ValueError, match="cannot reproduce"):
        play_policy.apply_checkpoint_defaults(args, Path("policy.pt"))
