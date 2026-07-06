#!/usr/bin/env python3
"""Benchmark the published stable-retro-turbo vector env on the SMB workload."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import statistics
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np

try:
    from benchmark_rom import validate_rom_hash
except ModuleNotFoundError:
    from scripts.benchmark_rom import validate_rom_hash


DEFAULT_GAME = "SuperMarioBros-Nes-v0"
ROM_PATH_ENV_VAR = "ROM_PATH"
DEFAULT_ROM = (
    Path(os.environ[ROM_PATH_ENV_VAR]).expanduser()
    if ROM_PATH_ENV_VAR in os.environ
    else None
)
DEFAULT_STATES = ("Level1-1", "Level1-2", "Level1-3", "Level1-4")
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


def resolve_required_rom_path(path: Path | None = None) -> Path:
    if path is None:
        path = dotenv_rom_path()
    if path is None:
        raise ValueError(
            f"ROM path required; pass --rom-path or set {ROM_PATH_ENV_VAR} in the environment or .env"
        )
    expanded = path.expanduser()
    if not expanded.exists():
        raise ValueError(f"ROM path does not exist: {expanded}")
    if not expanded.is_file():
        raise ValueError(f"ROM path is not a file: {expanded}")
    return expanded.resolve()


def resolve_verified_rom_path(path: Path | None = None) -> Path:
    resolved = resolve_required_rom_path(path)
    validate_rom_hash(resolved)
    return resolved


def sha256_path(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def dotenv_rom_path(dotenv_path: Path = Path(".env")) -> Path | None:
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
        if separator != "=" or key.strip() != ROM_PATH_ENV_VAR:
            continue
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return Path(value).expanduser() if value else None
    return None


def install_sb3_vecenv_shim_if_needed() -> None:
    if "stable_baselines3.common.vec_env" in sys.modules:
        return
    try:
        has_vec_env = importlib.util.find_spec("stable_baselines3.common.vec_env") is not None
    except (ModuleNotFoundError, ValueError):
        has_vec_env = False
    if has_vec_env:
        return

    stable_baselines3 = types.ModuleType("stable_baselines3")
    common = types.ModuleType("stable_baselines3.common")
    vec_env = types.ModuleType("stable_baselines3.common.vec_env")

    class VecEnv:
        def __init__(self, num_envs: int, observation_space: Any, action_space: Any) -> None:
            self.num_envs = int(num_envs)
            self.observation_space = observation_space
            self.action_space = action_space
            self._seeds = [None for _ in range(self.num_envs)]
            self._options = [{} for _ in range(self.num_envs)]
            self.reset_infos = [{} for _ in range(self.num_envs)]

        def seed(self, seed: int | None = None) -> list[int | None]:
            self._seeds = (
                [None for _ in range(self.num_envs)]
                if seed is None
                else [int(seed) + index for index in range(self.num_envs)]
            )
            return list(self._seeds)

        def step(self, actions: Any):
            self.step_async(actions)
            return self.step_wait()

        def _reset_seeds(self) -> None:
            self._seeds = [None for _ in range(self.num_envs)]

        def _reset_options(self) -> None:
            self._options = [{} for _ in range(self.num_envs)]

        def _get_indices(self, indices: Any = None) -> list[int]:
            if indices is None:
                return list(range(self.num_envs))
            if isinstance(indices, int):
                return [indices]
            return [int(index) for index in indices]

    vec_env.VecEnv = VecEnv
    common.vec_env = vec_env
    stable_baselines3.common = common
    sys.modules["stable_baselines3"] = stable_baselines3
    sys.modules["stable_baselines3.common"] = common
    sys.modules["stable_baselines3.common.vec_env"] = vec_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rom-path",
        type=Path,
        default=DEFAULT_ROM,
        help="Path to the SMB NES ROM. Defaults to ROM_PATH from the environment or .env.",
    )
    parser.add_argument("--game", default=DEFAULT_GAME)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-threads", type=int, default=12)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--crop-top", type=int, default=32)
    parser.add_argument("--crop-bottom", type=int, default=0)
    parser.add_argument("--resize-width", type=int, default=84)
    parser.add_argument("--resize-height", type=int, default=84)
    parser.add_argument("--states", default=",".join(DEFAULT_STATES))
    parser.add_argument("--action", choices=sorted(ACTION_BUTTONS), default="noop")
    parser.add_argument("--obs-copy", default="safe_view")
    parser.add_argument("--obs-resize-algorithm", default="area")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("num_envs", "num_threads", "steps", "repeats", "frame_skip", "frame_stack"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("warmup", "crop_top", "crop_bottom"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")


def parse_states(value: str) -> tuple[str, ...]:
    states = tuple(part.strip() for part in value.split(","))
    if not states or not all(states):
        raise ValueError("--states must be a comma-separated list without empty entries")
    return states


def lane_states(num_envs: int, states: tuple[str, ...]) -> list[str]:
    return [states[index % len(states)] for index in range(num_envs)]


def retro_button_names(retro, rom_path: Path) -> tuple[str | None, ...]:
    system = retro.get_romfile_system(str(rom_path))
    core = retro.get_system_info(system)
    return tuple(None if name is None else str(name).upper() for name in core["buttons"])


def action_mask(action_name: str, buttons: tuple[str | None, ...]) -> np.ndarray:
    button_to_index = {name: index for index, name in enumerate(buttons) if name is not None}
    mask = np.zeros((len(buttons),), dtype=np.uint8)
    for button in ACTION_BUTTONS[action_name]:
        mask[button_to_index[button]] = 1
    return mask


def fill_actions(num_envs: int, action_name: str, buttons: tuple[str | None, ...]) -> np.ndarray:
    return np.repeat(action_mask(action_name, buttons)[None, :], num_envs, axis=0)


def step_repeated(env: Any, actions: np.ndarray, count: int) -> None:
    for _ in range(count):
        env.step(actions)


def run_once(env: Any, actions: np.ndarray, args: argparse.Namespace) -> dict[str, float]:
    start = time.perf_counter()
    step_repeated(env, actions, args.steps)
    elapsed = time.perf_counter() - start
    batch_sps = args.steps / elapsed
    env_sps = batch_sps * args.num_envs
    return {
        "elapsed_s": elapsed,
        "batch_steps_per_sec": batch_sps,
        "env_steps_per_sec": env_sps,
        "emulated_frames_per_sec": env_sps * args.frame_skip,
    }


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def build_result(
    args: argparse.Namespace,
    version: str,
    obs: np.ndarray,
    runs: list[dict[str, float]],
    states: tuple[str, ...],
    rom_path: Path,
) -> dict[str, Any]:
    batch_sps = [run["batch_steps_per_sec"] for run in runs]
    env_sps = [run["env_steps_per_sec"] for run in runs]
    frame_sps = [run["emulated_frames_per_sec"] for run in runs]
    elapsed = [run["elapsed_s"] for run in runs]
    mean_batch_sps = statistics.fmean(batch_sps)
    return {
        "package": {
            "name": "stable-retro-turbo",
            "version": version,
            "import": "stable_retro",
        },
        "config": {
            "rom_path": str(rom_path),
            "rom_sha256": sha256_path(rom_path),
            "game": args.game,
            "num_envs": args.num_envs,
            "num_threads": args.num_threads,
            "steps": args.steps,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "frame_skip": args.frame_skip,
            "frame_stack": args.frame_stack,
            "grayscale": not args.rgb,
            "crop_top": args.crop_top,
            "crop_bottom": args.crop_bottom,
            "resize_width": args.resize_width,
            "resize_height": args.resize_height,
            "states": list(states),
            "lane_states": lane_states(args.num_envs, states),
            "action": args.action,
            "obs_copy": args.obs_copy,
            "obs_resize_algorithm": args.obs_resize_algorithm,
        },
        "observation": {
            "shape": list(obs.shape),
            "dtype": str(obs.dtype),
            "bytes": int(obs.nbytes),
            "mib": obs.nbytes / (1024**2),
        },
        "runs": runs,
        "summary": {
            "elapsed_s": summarize(elapsed),
            "batch_steps_per_sec": summarize(batch_sps),
            "env_steps_per_sec": summarize(env_sps),
            "emulated_frames_per_sec": summarize(frame_sps),
            "obs_buffer_gib_per_sec": (obs.nbytes * mean_batch_sps) / (1024**3),
        },
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    install_sb3_vecenv_shim_if_needed()

    import stable_retro as retro

    version = importlib.metadata.version("stable-retro-turbo")
    states = parse_states(args.states)
    rom_path = resolve_verified_rom_path(args.rom_path)
    crop = None
    if args.crop_top or args.crop_bottom:
        crop = (args.crop_top, args.crop_bottom, 0, 0)
    env_class = getattr(retro, "Retro" "Vec" "Env")
    env = env_class(
        args.game,
        state=lane_states(args.num_envs, states),
        num_envs=args.num_envs,
        num_threads=args.num_threads,
        rom_path=str(rom_path),
        render_mode="rgb_array",
        use_restricted_actions=retro.Actions.ALL,
        obs_crop=crop,
        obs_resize=(args.resize_height, args.resize_width),
        obs_grayscale=not args.rgb,
        obs_resize_algorithm=args.obs_resize_algorithm,
        obs_layout="chw",
        obs_copy=args.obs_copy,
        frame_skip=args.frame_skip,
        frame_stack=args.frame_stack,
        maxpool_last_two=False,
        noop_reset_max=0,
        sticky_action_prob=0.0,
        reward_clip=False,
        info_filter="none",
        done_on=None,
    )
    try:
        obs = env.reset()
        actions = fill_actions(args.num_envs, args.action, retro_button_names(retro, rom_path))
        step_repeated(env, actions, args.warmup)
        runs = [run_once(env, actions, args) for _ in range(args.repeats)]
        result = build_result(args, version, obs, runs, states, rom_path)
    finally:
        env.close()

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        summary = result["summary"]["env_steps_per_sec"]
        print(
            f"stable-retro-turbo=={version} "
            f"mean_env_steps_per_sec={summary['mean']:.1f} "
            f"best_env_steps_per_sec={summary['max']:.1f}"
        )


if __name__ == "__main__":
    main()
