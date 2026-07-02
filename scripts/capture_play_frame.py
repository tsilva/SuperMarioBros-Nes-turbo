from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from play import latest_frame, png_from_frame
from supermariobrosnes_turbo import Actions, SuperMarioBrosNesTurboVecEnv, default_rom_path, resolve_required_rom_path


DEFAULT_ROM = default_rom_path()
NES_BUTTONS = ("B", None, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A")
BUTTON_TO_INDEX = {name: index for index, name in enumerate(NES_BUTTONS) if name is not None}
ACTION_BUTTONS = {
    "noop": (),
    "start": ("START",),
    "right_b": ("RIGHT", "B"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rom-path",
        type=Path,
        default=DEFAULT_ROM,
        help="Path to the SMB NES ROM. Defaults to SMB_ROM_PATH when set.",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/play-frame.png"))
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--pre-start-frames", type=int, default=120)
    parser.add_argument("--start-frames", type=int, default=30)
    parser.add_argument("--right-frames", type=int, default=90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = SuperMarioBrosNesTurboVecEnv(
        "SuperMarioBros-Nes-v0",
        rom_path=resolve_required_rom_path(args.rom_path),
        num_envs=1,
        use_restricted_actions=Actions.ALL,
        frame_skip=1,
        obs_grayscale=False,
        frame_stack=1,
        obs_resize=(240, 256),
    )
    obs = env.reset()[0]

    def step_one(action_name: str) -> np.ndarray:
        actions = np.zeros((1, len(NES_BUTTONS)), dtype=np.uint8)
        for button in ACTION_BUTTONS[action_name]:
            actions[0, BUTTON_TO_INDEX[button]] = 1
        return env.step_fast(actions)[0][0]

    for _ in range(args.pre_start_frames):
        obs = step_one("noop")
    for _ in range(args.start_frames):
        obs = step_one("start")
    for _ in range(60):
        obs = step_one("noop")
    for _ in range(args.right_frames):
        obs = step_one("right_b")

    frame = latest_frame(obs)
    if args.scale > 1:
        frame = np.repeat(np.repeat(frame, args.scale, axis=0), args.scale, axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(png_from_frame(np.ascontiguousarray(frame)))
    print(args.output)


if __name__ == "__main__":
    main()
