from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import play_policy  # noqa: E402


def test_jerk_checkpoint_uses_native_lightweight_contract() -> None:
    args = play_policy.parse_args(["policy.json"])

    play_policy.apply_checkpoint_defaults(args, Path("policy.json"))

    assert args.backend == "native"
    assert args.max_pool_frames is False
    assert args.crop_mode == "remove"


def test_explicit_stable_retro_backend_is_preserved() -> None:
    args = play_policy.parse_args(["policy.json", "--backend", "stable-retro"])

    play_policy.apply_checkpoint_defaults(args, Path("policy.json"))

    assert args.backend == "stable-retro"
    assert args.max_pool_frames is False
    assert args.crop_mode == "remove"


def test_stable_retro_supports_mask_crop_through_shared_preprocessing() -> None:
    args = play_policy.parse_args(
        ["policy.json", "--backend", "stable-retro", "--crop-mode", "mask"]
    )

    play_policy.apply_checkpoint_defaults(args, Path("policy.json"))

    assert args.backend == "stable-retro"
    assert args.crop_mode == "mask"
