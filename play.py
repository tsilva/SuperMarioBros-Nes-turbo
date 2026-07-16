#!/usr/bin/env python3
"""Play the deterministic JERK policy trained for a named Mario level."""

from __future__ import annotations

import argparse
from pathlib import Path

from supermariobrosnes_turbo.jerk import normalize_level_name, policy_path_for_level


def resolve_level_policy(level: str, *, runs_root: str | Path = "runs") -> Path:
    path = policy_path_for_level(level, runs_root=runs_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"no policy trained for {normalize_level_name(level)} at {path}; "
            f"run `uv run python train.py {normalize_level_name(level)}` first"
        )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("level", help="Level policy to play, for example Level1-1")
    parser.add_argument(
        "playback_args",
        nargs=argparse.REMAINDER,
        help="Additional options forwarded to scripts/play_policy.py",
    )
    return parser


def playback_argv(
    level: str,
    playback_args: list[str],
    *,
    runs_root: str | Path = "runs",
) -> list[str]:
    name = normalize_level_name(level)
    if "--state" in playback_args:
        raise ValueError("play.py derives --state from the level; do not pass --state")
    return [
        str(resolve_level_policy(name, runs_root=runs_root)),
        "--state",
        name,
        *playback_args,
    ]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        policy_args = playback_argv(args.level, args.playback_args)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    from scripts import play_policy

    parsed = play_policy.parse_args(policy_args)
    try:
        play_policy.SdlPolicyPlayer(parsed).run()
    except play_policy.SdlUnavailableError as exc:
        raise SystemExit(f"SDL backend unavailable: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
