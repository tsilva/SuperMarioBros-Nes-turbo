from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from supermariobrosnes_turbo import (
    ACTION_SETS,
    CORE_ACTION_MEANINGS,
    Actions,
    SuperMarioBrosNesTurboVecEnv,
    default_rom_path,
    resolve_required_rom_path,
)

try:
    from benchmark_rom import validate_rom_hash
except ModuleNotFoundError:
    from scripts.benchmark_rom import validate_rom_hash


DEFAULT_ROM = default_rom_path()
DEFAULT_STATES = ("Level1-1", "Level1-2", "Level1-3", "Level1-4")
DEFAULT_BENCHMARK_ACTIONS = ("noop", "right", "right_b", "right_a")
DEFAULT_ACTION_SEED = 0
DEFAULT_MIN_START_LOAD_LIMIT = 4.0
DEFAULT_START_LOAD_CPU_FRACTION = 0.5
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
PACKAGE_NAME = "supermariobrosnes-turbo"
IMPORT_PACKAGE = "supermariobrosnes_turbo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark no-GUI Super Mario Bros vector-env steps per second."
    )
    parser.add_argument(
        "--rom-path",
        type=Path,
        default=DEFAULT_ROM,
        help="Path to the SMB NES ROM. Defaults to ROM_PATH from the environment or .env.",
    )
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--crop-top", type=int, default=32)
    parser.add_argument("--crop-bottom", type=int, default=0)
    parser.add_argument("--resize-width", type=int, default=84)
    parser.add_argument("--resize-height", type=int, default=84)
    parser.add_argument("--action-set", choices=sorted(ACTION_SETS), default="simple")
    parser.add_argument(
        "--actions",
        default=None,
        help=(
            "Comma-separated actions sampled per vector step. Defaults to "
            f"{','.join(DEFAULT_BENCHMARK_ACTIONS)}."
        ),
    )
    parser.add_argument(
        "--action",
        choices=CORE_ACTION_MEANINGS,
        default=None,
        help="Legacy single-action override. If set, --actions is ignored.",
    )
    parser.add_argument("--action-seed", type=int, default=DEFAULT_ACTION_SEED)
    parser.add_argument("--state", default=None)
    parser.add_argument(
        "--states",
        default=None,
        help=(
            "Comma-separated stable-retro states assigned round-robin across lanes, "
            f"default: {','.join(DEFAULT_STATES)} unless --state is provided."
        ),
    )
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument(
        "--include-info",
        action="store_true",
        default=True,
        help="Deprecated compatibility flag; benchmark_sps always uses step().",
    )
    parser.add_argument("--terminate-on-flag", action="store_true")
    parser.add_argument("--no-start-game", action="store_true")
    parser.add_argument("--pre-start-steps", type=int, default=30)
    parser.add_argument("--start-steps", type=int, default=8)
    parser.add_argument("--post-start-steps", type=int, default=30)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--max-start-load",
        type=float,
        default=None,
        help=(
            "Fail before benchmarking if 1-minute load exceeds this value. "
            "Default: max(4.0, logical_cpus * 0.5)."
        ),
    )
    parser.add_argument(
        "--skip-load-preflight",
        action="store_true",
        help="Disable the startup CPU load guard.",
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        default=None,
        help="Enable local Rust hot-path profiling and write benchmark+profile JSON.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_fields = (
        "num_envs",
        "steps",
        "repeats",
        "frame_skip",
        "frame_stack",
        "resize_width",
        "resize_height",
    )
    for field in positive_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    non_negative_fields = (
        "warmup",
        "crop_top",
        "crop_bottom",
        "pre_start_steps",
        "start_steps",
        "post_start_steps",
    )
    for field in non_negative_fields:
        if getattr(args, field) < 0:
            raise ValueError(f"--{field.replace('_', '-')} must be non-negative")
    if args.max_start_load is not None and args.max_start_load <= 0:
        raise ValueError("--max-start-load must be positive")
    action_meanings = ACTION_SETS[args.action_set]
    for action in selected_actions_for_args(args):
        if action not in action_meanings:
            raise ValueError(
                f"action {action!r} is not in action_set={args.action_set!r}; "
                f"valid actions: {', '.join(action_meanings)}"
            )
    if args.state is not None and args.states is not None:
        raise ValueError("--state and --states are mutually exclusive")


def parse_states(states: str | None) -> tuple[str, ...] | None:
    if states is None:
        return None
    parsed = tuple(state.strip() for state in states.split(","))
    if not parsed or not all(parsed):
        raise ValueError("--states must be a comma-separated list without empty entries")
    return parsed


def parse_actions(actions: str | None) -> tuple[str, ...]:
    if actions is None:
        return DEFAULT_BENCHMARK_ACTIONS
    parsed = tuple(action.strip() for action in actions.split(","))
    if not parsed or not all(parsed):
        raise ValueError("--actions must be a comma-separated list without empty entries")
    return parsed


def selected_actions_for_args(args: argparse.Namespace) -> tuple[str, ...]:
    if args.action is not None:
        return (args.action,)
    return parse_actions(args.actions)


def initial_states_for_args(args: argparse.Namespace) -> tuple[str, ...] | None:
    if args.state is not None:
        return None
    return parse_states(args.states) if args.states is not None else DEFAULT_STATES


def lane_states(num_envs: int, states: tuple[str, ...] | None) -> list[str] | None:
    if states is None:
        return None
    return [states[index % len(states)] for index in range(num_envs)]


def benchmark_state(args: argparse.Namespace) -> str | list[str] | None:
    if args.parsed_states is None:
        return args.state
    return lane_states(args.num_envs, args.parsed_states)


def has_initial_state(args: argparse.Namespace) -> bool:
    return args.state is not None or args.parsed_states is not None


def default_max_start_load() -> float:
    return max(
        DEFAULT_MIN_START_LOAD_LIMIT,
        (os.cpu_count() or 1) * DEFAULT_START_LOAD_CPU_FRACTION,
    )


def load_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.skip_load_preflight:
        return {
            "enabled": False,
            "start_1min": None,
            "max_start_load": None,
            "load_ok": True,
        }
    max_start_load = args.max_start_load
    if max_start_load is None:
        max_start_load = default_max_start_load()
    try:
        start_1min = os.getloadavg()[0]
    except (AttributeError, OSError) as exc:
        return {
            "enabled": True,
            "start_1min": None,
            "max_start_load": max_start_load,
            "load_ok": True,
            "unavailable_reason": str(exc),
        }
    load_ok = start_1min < max_start_load
    result: dict[str, Any] = {
        "enabled": True,
        "start_1min": start_1min,
        "max_start_load": max_start_load,
        "load_ok": load_ok,
    }
    if not load_ok:
        raise SystemExit(
            f"Refusing to benchmark: 1-minute load {start_1min:.2f} meets or exceeds "
            f"--max-start-load {max_start_load:.2f}. Use --skip-load-preflight "
            "to override."
        )
    return result


def sha256_path(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def package_metadata() -> dict[str, str | None]:
    try:
        version = importlib.metadata.version(PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {
        "name": PACKAGE_NAME,
        "version": version,
        "import": IMPORT_PACKAGE,
    }


def resolve_verified_rom_path(path: str | Path | None = None) -> Path:
    try:
        resolved = resolve_required_rom_path(path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not resolved.exists():
        raise SystemExit(f"ROM path does not exist: {resolved}")
    if not resolved.is_file():
        raise SystemExit(f"ROM path is not a file: {resolved}")
    validate_rom_hash(resolved)
    return resolved


def rayon_num_threads() -> int | str | None:
    raw = os.environ.get("RAYON_NUM_THREADS")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return raw


def fill_action(num_envs: int, action_name: str, action_meanings: tuple[str, ...]) -> np.ndarray:
    del action_meanings
    actions = np.zeros((num_envs, len(NES_BUTTONS)), dtype=np.uint8)
    for button in ACTION_BUTTONS[action_name]:
        actions[:, BUTTON_TO_INDEX[button]] = 1
    return actions


def action_templates(
    num_envs: int,
    action_names: Sequence[str],
    action_meanings: tuple[str, ...],
) -> tuple[np.ndarray, ...]:
    return tuple(fill_action(num_envs, action_name, action_meanings) for action_name in action_names)


def sampled_action_sequence(
    templates: Sequence[np.ndarray],
    count: int,
    seed: int,
) -> tuple[np.ndarray, ...]:
    if count <= 0:
        return ()
    if len(templates) == 1:
        return tuple(templates[0] for _ in range(count))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(templates), size=count)
    return tuple(templates[int(index)] for index in indices)


def step_env(env: SuperMarioBrosNesTurboVecEnv, actions: np.ndarray, include_info: bool) -> None:
    del include_info
    env.step(actions)


def step_repeated(
    env: SuperMarioBrosNesTurboVecEnv,
    actions: np.ndarray,
    count: int,
    include_info: bool,
) -> None:
    for _ in range(count):
        step_env(env, actions, include_info)


def step_action_sequence(
    env: SuperMarioBrosNesTurboVecEnv,
    actions: Sequence[np.ndarray],
    include_info: bool,
) -> None:
    for action in actions:
        step_env(env, action, include_info)


def prepare_game(
    env: SuperMarioBrosNesTurboVecEnv,
    args: argparse.Namespace,
    action_meanings: tuple[str, ...],
) -> None:
    env.reset()
    if args.no_start_game or has_initial_state(args) or "start" not in action_meanings:
        return
    noop = fill_action(args.num_envs, "noop", action_meanings)
    start = fill_action(args.num_envs, "start", action_meanings)
    step_repeated(env, noop, args.pre_start_steps, args.include_info)
    step_repeated(env, start, args.start_steps, args.include_info)
    step_repeated(env, noop, args.post_start_steps, args.include_info)


def run_once(
    env: SuperMarioBrosNesTurboVecEnv,
    actions: Sequence[np.ndarray],
    args: argparse.Namespace,
) -> dict[str, float]:
    start = time.perf_counter()
    step_action_sequence(env, actions, args.include_info)
    elapsed = time.perf_counter() - start
    batch_sps = args.steps / elapsed
    env_sps = batch_sps * args.num_envs
    frame_sps = env_sps * args.frame_skip
    return {
        "elapsed_s": elapsed,
        "batch_steps_per_sec": batch_sps,
        "env_steps_per_sec": env_sps,
        "emulated_frames_per_sec": frame_sps,
    }


def summarize(values: list[float]) -> dict[str, float]:
    result = {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }
    result["stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return result


def build_result(
    args: argparse.Namespace,
    obs: np.ndarray,
    runs: list[dict[str, float]],
    active_states: tuple[str | None, ...],
    load: dict[str, Any],
    rom_path: Path,
) -> dict[str, Any]:
    batch_sps = [run["batch_steps_per_sec"] for run in runs]
    env_sps = [run["env_steps_per_sec"] for run in runs]
    frame_sps = [run["emulated_frames_per_sec"] for run in runs]
    elapsed = [run["elapsed_s"] for run in runs]
    mean_batch_sps = statistics.fmean(batch_sps)
    return {
        "package": package_metadata(),
        "config": {
            "rom_path": str(rom_path),
            "rom_sha256": sha256_path(rom_path),
            "rayon_num_threads": rayon_num_threads(),
            "num_envs": args.num_envs,
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
            "obs_resize_algorithm": "area",
            "action_set": args.action_set,
            "action": args.action,
            "actions": list(args.parsed_actions),
            "action_seed": args.action_seed,
            "state": args.state,
            "states": list(args.parsed_states) if args.parsed_states is not None else None,
            "lane_states": list(active_states) if has_initial_state(args) else None,
            "state_dir": str(args.state_dir) if args.state_dir is not None else None,
            "include_info": True,
            "terminate_on_flag": args.terminate_on_flag,
            "start_game": (
                not args.no_start_game
                and not has_initial_state(args)
                and "start" in ACTION_SETS[args.action_set]
            ),
        },
        "observation": {
            "shape": list(obs.shape),
            "dtype": str(obs.dtype),
            "bytes": int(obs.nbytes),
            "mib": obs.nbytes / (1024**2),
        },
        "load": load,
        "runs": runs,
        "summary": {
            "elapsed_s": summarize(elapsed),
            "batch_steps_per_sec": summarize(batch_sps),
            "env_steps_per_sec": summarize(env_sps),
            "emulated_frames_per_sec": summarize(frame_sps),
            "obs_buffer_gib_per_sec": (obs.nbytes * mean_batch_sps) / (1024**3),
        },
    }


def print_human(result: dict[str, Any]) -> None:
    config = result["config"]
    obs = result["observation"]
    summary = result["summary"]
    print(
        "config="
        f"num_envs={config['num_envs']} steps={config['steps']} repeats={config['repeats']} "
        f"frame_skip={config['frame_skip']} frame_stack={config['frame_stack']} "
        f"grayscale={config['grayscale']} crop=({config['crop_top']},{config['crop_bottom']}) "
        f"resize={config['resize_width']}x{config['resize_height']} "
        f"action_set={config['action_set']} actions={config['actions']} "
        f"action_seed={config['action_seed']} "
        f"state={config['state']} states={config['states']} "
        f"include_info={config['include_info']}"
    )
    if config["lane_states"] is not None:
        print(f"lane_states={config['lane_states']}")
    load = result["load"]
    if load["enabled"] and load["start_1min"] is not None:
        print(
            "load_preflight="
            f"start_1min={load['start_1min']:.2f} "
            f"max_start_load={load['max_start_load']:.2f} "
            f"load_ok={load['load_ok']}"
        )
    elif load["enabled"]:
        print(f"load_preflight=unavailable load_ok={load['load_ok']}")
    else:
        print("load_preflight=disabled")
    print(
        f"obs_shape={tuple(obs['shape'])} obs_dtype={obs['dtype']} "
        f"obs_mib={obs['mib']:.2f}"
    )
    for idx, run in enumerate(result["runs"], start=1):
        print(
            f"run={idx} elapsed_s={run['elapsed_s']:.6f} "
            f"batch_steps_per_sec={run['batch_steps_per_sec']:.1f} "
            f"env_steps_per_sec={run['env_steps_per_sec']:.1f} "
            f"emulated_frames_per_sec={run['emulated_frames_per_sec']:.1f}"
        )
    print(
        "summary="
        f"env_steps_per_sec_mean={summary['env_steps_per_sec']['mean']:.1f} "
        f"env_steps_per_sec_stdev={summary['env_steps_per_sec']['stdev']:.1f} "
        f"best_env_steps_per_sec={summary['env_steps_per_sec']['max']:.1f} "
        f"emulated_frames_per_sec_mean={summary['emulated_frames_per_sec']['mean']:.1f} "
        f"obs_buffer_gib_per_sec={summary['obs_buffer_gib_per_sec']:.2f}"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.parsed_states = initial_states_for_args(args)
    args.parsed_actions = selected_actions_for_args(args)
    rom_path = resolve_verified_rom_path(args.rom_path)
    load = load_preflight(args)
    action_set = args.action_set
    action_meanings = ACTION_SETS[action_set]
    if args.state_dir is not None:
        os.environ["SUPERMARIOBROSNES_FASTENV_STATE_DIR"] = str(args.state_dir)
    env = SuperMarioBrosNesTurboVecEnv(
        "SuperMarioBros-Nes-v0",
        state=benchmark_state(args),
        rom_path=rom_path,
        num_envs=args.num_envs,
        use_restricted_actions=Actions.ALL,
        frame_skip=args.frame_skip,
        obs_grayscale=not args.rgb,
        frame_stack=args.frame_stack,
        obs_crop=(args.crop_top, args.crop_bottom, 0, 0),
        obs_resize=(args.resize_height, args.resize_width),
        obs_resize_algorithm="area",
        obs_layout="chw",
    )
    if args.profile_output is not None:
        env.enable_profiler()
    obs, _infos = env.reset()
    active_states = env.active_states()
    templates = action_templates(args.num_envs, args.parsed_actions, action_meanings)
    warmup_actions = sampled_action_sequence(templates, args.warmup, args.action_seed + 1)
    measured_actions = sampled_action_sequence(templates, args.steps, args.action_seed)
    prepare_game(env, args, action_meanings)
    step_action_sequence(env, warmup_actions, args.include_info)
    if args.profile_output is not None:
        env.reset_profiler()
    runs = [run_once(env, measured_actions, args) for _ in range(args.repeats)]
    result = build_result(args, obs, runs, active_states, load, rom_path)
    if args.profile_output is not None:
        result["profiler"] = env.profiler_snapshot()

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.profile_output is not None:
        args.profile_output.parent.mkdir(parents=True, exist_ok=True)
        args.profile_output.write_text(json.dumps(result, indent=2) + "\n")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_human(result)


if __name__ == "__main__":
    main()
