from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from supermariobrosnes_turbo import (
    ACTION_SETS,
    Actions,
    SuperMarioBrosNesTurboVecEnv,
    default_rom_path,
    resolve_required_rom_path,
)

NES_BUTTONS = ("B", None, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A")
BUTTON_TO_INDEX = {name: index for index, name in enumerate(NES_BUTTONS) if name is not None}
ACTION_BUTTONS = {
    "noop": (),
    "right": ("RIGHT",),
    "right_b": ("RIGHT", "B"),
    "right_a": ("RIGHT", "A"),
    "right_a_b": ("RIGHT", "A", "B"),
    "a": ("A",),
    "left": ("LEFT",),
    "start": ("START",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rom-path",
        type=Path,
        default=default_rom_path(),
        help="Path to the SMB NES ROM. Defaults to SMB_ROM_PATH when set.",
    )
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--resize-width", type=int, default=84)
    parser.add_argument("--resize-height", type=int, default=84)
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--state", default=None)
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--action-set", choices=sorted(ACTION_SETS), default="simple")
    parser.add_argument("--action", default="noop")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    action_names = ACTION_SETS[args.action_set]
    if args.action not in action_names:
        raise ValueError(f"--action must be one of: {', '.join(action_names)}")
    env = SuperMarioBrosNesTurboVecEnv(
        "SuperMarioBros-Nes-v0",
        state=args.state,
        rom_path=resolve_required_rom_path(args.rom_path),
        num_envs=args.num_envs,
        use_restricted_actions=Actions.ALL,
        frame_skip=args.frame_skip,
        obs_grayscale=not args.rgb,
        frame_stack=args.frame_stack,
        obs_resize=(args.resize_height, args.resize_width),
        obs_layout="chw",
        obs_resize_algorithm="area",
    )
    obs = env.reset()
    actions = np.zeros((args.num_envs, len(NES_BUTTONS)), dtype=np.uint8)
    for button in ACTION_BUTTONS[args.action]:
        actions[:, BUTTON_TO_INDEX[button]] = 1

    for _ in range(args.warmup):
        env.step_fast(actions)

    start = time.perf_counter()
    for _ in range(args.steps):
        env.step_fast(actions)
    elapsed = time.perf_counter() - start

    batch_sps = args.steps / elapsed
    env_sps = batch_sps * args.num_envs
    frame_sps = env_sps * args.frame_skip
    obs_gib_per_s = (obs.nbytes * batch_sps) / (1024**3)

    print(f"obs_shape={obs.shape} obs_dtype={obs.dtype} obs_mib={obs.nbytes / (1024**2):.2f}")
    print(f"elapsed_s={elapsed:.6f}")
    print(f"batch_steps_per_sec={batch_sps:.1f}")
    print(f"env_steps_per_sec={env_sps:.1f}")
    print(f"emulated_frames_per_sec={frame_sps:.1f}")
    print(f"obs_buffer_gib_per_sec={obs_gib_per_s:.2f}")


if __name__ == "__main__":
    main()
