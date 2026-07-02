from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces

from ._supermariobrosnes_turbo import FastMarioVecEnv

try:
    from stable_baselines3.common.vec_env import VecEnv as _SB3VecEnv
except ImportError:

    class _SB3VecEnv:
        def __init__(self, num_envs: int, observation_space: spaces.Space, action_space: spaces.Space):
            self.num_envs = num_envs
            self.observation_space = observation_space
            self.action_space = action_space
            self.reset_infos = [{} for _ in range(num_envs)]
            self._seeds = [None for _ in range(num_envs)]
            self._options = [{} for _ in range(num_envs)]

        def _get_indices(self, indices: None | int | Iterable[int]) -> Iterable[int]:
            if indices is None:
                return range(self.num_envs)
            if isinstance(indices, int):
                return [indices]
            return indices

        def seed(self, seed: int | None = None) -> Sequence[int | None]:
            if seed is None:
                seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))
            self._seeds = [seed + index for index in range(self.num_envs)]
            return self._seeds

        def set_options(self, options: list[dict[str, Any]] | dict[str, Any] | None = None) -> None:
            if options is None:
                options = {}
            if isinstance(options, dict):
                self._options = deepcopy([options] * self.num_envs)
            else:
                self._options = deepcopy(options)

        def _reset_seeds(self) -> None:
            self._seeds = [None for _ in range(self.num_envs)]

        def _reset_options(self) -> None:
            self._options = [{} for _ in range(self.num_envs)]

        def step(self, actions: np.ndarray):
            self.step_async(actions)
            return self.step_wait()


CORE_ACTION_MEANINGS = ("noop", "right", "right_b", "right_a", "right_a_b", "a", "left", "start")
ACTION_SETS = {
    "simple": ("noop", "right", "right_b", "right_a", "right_a_b", "a", "left"),
    "right": ("right", "right_b", "right_a", "right_a_b"),
    "full": CORE_ACTION_MEANINGS,
}
ACTION_MEANINGS = ACTION_SETS["simple"]
DEFAULT_STABLE_RETRO_GAME = "SuperMarioBros-Nes-v0"
GZIP_MAGIC = b"\x1f\x8b"
INFO_KEYS = (
    "x_pos",
    "coins",
    "levelHi",
    "levelLo",
    "lives",
    "score",
    "scrolling",
    "time",
    "xscrollHi",
    "xscrollLo",
)
StateSpec = str | Path | bytes | bytearray | memoryview
DoneOnInfoSpec = Mapping[str, Sequence[Any]]
DoneOnInfoRule = tuple[str, tuple[str, ...], str]


def _expand_rom_path(path: str | Path) -> str:
    return str(Path(path).expanduser())


def _stable_retro_state_dir() -> Path | None:
    try:
        import stable_retro.data  # type: ignore[import-not-found]
    except ImportError:
        return None

    try:
        state_path = stable_retro.data.get_file_path(
            DEFAULT_STABLE_RETRO_GAME,
            "Level1-1.state",
            stable_retro.data.Integrations.ALL,
        )
    except Exception:
        return None
    if not state_path:
        return None
    return Path(state_path).parent


def _sibling_stable_retro_state_dir() -> Path | None:
    game_path = Path("stable_retro/data/stable") / DEFAULT_STABLE_RETRO_GAME
    for parent in Path(__file__).resolve().parents:
        candidate = parent.parent / "stable-retro-turbo" / game_path
        if candidate.exists():
            return candidate
    return None


def _candidate_state_dirs(state_dir: str | Path | None = None) -> list[Path]:
    candidates: list[Path | None] = []
    if state_dir is not None:
        candidates.append(Path(state_dir).expanduser())
    env_dir = os.environ.get("SUPERMARIOBROSNES_FASTENV_STATE_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.append(_stable_retro_state_dir())
    candidates.append(_sibling_stable_retro_state_dir())

    dirs: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.resolve()
        if resolved not in seen and resolved.exists():
            dirs.append(resolved)
            seen.add(resolved)
    return dirs


def list_available_states(state_dir: str | Path | None = None) -> tuple[str, ...]:
    """Return available stable-retro Super Mario Bros state names."""
    states: set[str] = set()
    for directory in _candidate_state_dirs(state_dir):
        states.update(
            path.stem for path in directory.glob("*.state") if not path.name.startswith("_")
        )
    return tuple(sorted(states))


def _resolve_state_path(state: str | Path, state_dir: str | Path | None = None) -> Path:
    raw_path = Path(state).expanduser()
    if raw_path.exists():
        return raw_path

    state_name = str(state)
    state_file = state_name if state_name.endswith(".state") else f"{state_name}.state"
    for directory in _candidate_state_dirs(state_dir):
        candidate = directory / state_file
        if candidate.exists():
            return candidate

    dirs = ", ".join(str(path) for path in _candidate_state_dirs(state_dir)) or "<none>"
    raise FileNotFoundError(
        f"could not resolve state {state_name!r}; checked direct path and state dirs: {dirs}"
    )


def _load_initial_state(
    state: StateSpec,
    state_dir: str | Path | None = None,
) -> bytes:
    if isinstance(state, (bytes, bytearray, memoryview)):
        raw = bytes(state)
    else:
        raw = _resolve_state_path(state, state_dir).read_bytes()
    if raw.startswith(GZIP_MAGIC):
        return gzip.decompress(raw)
    return raw


def _state_label(state: StateSpec, fallback: str) -> str:
    if isinstance(state, (bytes, bytearray, memoryview)):
        return fallback
    return str(state)


def _normalize_initial_state_config(
    state: StateSpec | Sequence[StateSpec] | Mapping[StateSpec, float] | None,
    state_dir: str | Path | None,
    num_envs: int,
) -> tuple[list[bytes], tuple[str, ...], list[float] | None]:
    if state is None:
        return [], (), None

    if isinstance(state, Mapping):
        if not state:
            raise ValueError("weighted state mapping must contain at least one state")
        initial_states: list[bytes] = []
        state_names: list[str] = []
        state_weights: list[float] = []
        for index, (state_value, weight_value) in enumerate(state.items()):
            weight = float(weight_value)
            if not np.isfinite(weight) or weight <= 0.0:
                raise ValueError("weighted state values must be positive finite numbers")
            initial_states.append(_load_initial_state(state_value, state_dir))
            state_names.append(_state_label(state_value, f"state-{index}"))
            state_weights.append(weight)
        total_weight = float(np.sum(state_weights))
        return initial_states, tuple(state_names), [weight / total_weight for weight in state_weights]

    if isinstance(state, Sequence) and not isinstance(state, (str, bytes, bytearray, memoryview)):
        states = list(state)
        if len(states) != num_envs:
            raise ValueError(
                "state sequences must provide exactly one state per env slot: "
                f"got {len(states)} states for num_envs={num_envs}"
            )
        if not states:
            raise ValueError("state sequence must contain at least one state")
        return (
            [_load_initial_state(state_value, state_dir) for state_value in states],
            tuple(_state_label(state_value, f"state-{index}") for index, state_value in enumerate(states)),
            None,
        )

    return [_load_initial_state(state, state_dir)], (_state_label(state, "state-0"),), None


def _resolve_action_set(action_set: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(action_set, str):
        try:
            return ACTION_SETS[action_set]
        except KeyError as exc:
            valid = ", ".join(sorted(ACTION_SETS))
            raise ValueError(f"unknown action_set {action_set!r}; valid values: {valid}") from exc

    actions = tuple(str(action) for action in action_set)
    if not actions:
        raise ValueError("action_set must contain at least one action")
    unknown = [action for action in actions if action not in CORE_ACTION_MEANINGS]
    if unknown:
        valid = ", ".join(CORE_ACTION_MEANINGS)
        raise ValueError(f"unknown action(s) {unknown!r}; valid actions: {valid}")
    return actions


def _core_action_ids(action_meanings: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [CORE_ACTION_MEANINGS.index(action) for action in action_meanings],
        dtype=np.uint8,
    )


def _normalize_done_on_info(
    done_on_info: DoneOnInfoSpec | None,
    terminate_on_life_loss: bool,
    terminate_on_level_change: bool,
) -> tuple[DoneOnInfoRule, ...]:
    rules: dict[str, tuple[tuple[str, ...], str]] = {}
    if done_on_info is not None:
        if not isinstance(done_on_info, Mapping):
            raise ValueError("done_on_info must be a mapping of rule names to (key_or_keys, op)")
        for raw_name, spec in done_on_info.items():
            name = str(raw_name)
            if not name:
                raise ValueError("done_on_info rule names must not be empty")
            if (
                not isinstance(spec, Sequence)
                or isinstance(spec, (str, bytes, bytearray))
                or len(spec) != 2
            ):
                raise ValueError("done_on_info values must be (key_or_keys, op) pairs")
            raw_keys, raw_op = spec
            op = str(raw_op)
            if op not in {"change", "increase", "decrease"}:
                raise ValueError("done_on_info ops must be 'change', 'increase', or 'decrease'")
            if isinstance(raw_keys, str):
                keys = (raw_keys,)
            elif isinstance(raw_keys, Sequence) and not isinstance(raw_keys, (bytes, bytearray)):
                keys = tuple(str(key) for key in raw_keys)
            else:
                raise ValueError("done_on_info keys must be a string or sequence of strings")
            if not keys or any(not key for key in keys):
                raise ValueError("done_on_info rules must reference at least one key")
            unknown = [key for key in keys if key not in INFO_KEYS]
            if unknown:
                valid = ", ".join(INFO_KEYS)
                raise ValueError(f"unknown done_on_info key(s) {unknown!r}; valid keys: {valid}")
            rules[name] = (keys, op)

    if terminate_on_life_loss and "life_loss" not in rules:
        rules["life_loss"] = (("lives",), "decrease")
    if terminate_on_level_change and "level_change" not in rules:
        rules["level_change"] = (("levelHi", "levelLo"), "change")

    return tuple((name, keys, op) for name, (keys, op) in rules.items())


class SuperMarioBrosVecEnv(_SB3VecEnv):
    """SB3-compatible vectorized Mario environment with the hot loop in Rust.

    `step_wait()` follows the Stable Baselines3 `VecEnv` contract and returns
    `(obs, rewards, dones, infos)`. Use `step_wait_gymnasium()` when code needs
    separate `terminated` and `truncated` arrays.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        rom_path: str | Path = "~/Desktop/roms/SuperMarioBros.nes",
        num_envs: int = 1,
        frame_skip: int = 4,
        grayscale: bool = True,
        frame_stack: int = 4,
        frame_maxpool: bool = False,
        terminate_on_flag: bool = True,
        crop_top: int = 0,
        crop_bottom: int = 0,
        resize_width: int = 84,
        resize_height: int = 84,
        state: StateSpec | Sequence[StateSpec] | Mapping[StateSpec, float] | None = None,
        state_dir: str | Path | None = None,
        action_set: str | Sequence[str] = "simple",
        seed: int | None = None,
        terminate_on_life_loss: bool = False,
        terminate_on_level_change: bool = False,
        done_on_info: DoneOnInfoSpec | None = None,
        noop_reset_max: int = 0,
        sticky_action_prob: float = 0.0,
    ) -> None:
        noop_reset_max = int(noop_reset_max)
        sticky_action_prob = float(sticky_action_prob)
        if noop_reset_max < 0:
            raise ValueError("noop_reset_max must be non-negative")
        if not 0.0 <= sticky_action_prob <= 1.0:
            raise ValueError("sticky_action_prob must be between 0.0 and 1.0")
        self.action_meanings = _resolve_action_set(action_set)
        self.action_set = action_set if isinstance(action_set, str) else "custom"
        self._core_action_ids = _core_action_ids(self.action_meanings)
        self._state_collection = isinstance(state, Mapping) or (
            isinstance(state, Sequence) and not isinstance(state, (str, bytes, bytearray, memoryview))
        )
        done_on_info_rules = _normalize_done_on_info(
            done_on_info,
            bool(terminate_on_life_loss),
            bool(terminate_on_level_change),
        )
        initial_states, initial_state_names, initial_state_weights = _normalize_initial_state_config(
            state,
            state_dir,
            num_envs,
        )
        self._core = FastMarioVecEnv(
            _expand_rom_path(rom_path),
            num_envs,
            frame_skip,
            grayscale,
            frame_stack,
            terminate_on_flag,
            crop_top,
            crop_bottom,
            resize_width,
            resize_height,
            initial_states,
            list(initial_state_names),
            initial_state_weights,
            0 if seed is None else int(seed),
            bool(terminate_on_life_loss),
            bool(terminate_on_level_change),
            [(name, list(keys), op) for name, keys, op in done_on_info_rules],
            bool(frame_maxpool),
            noop_reset_max,
            sticky_action_prob,
        )
        self.initial_state_names = tuple(self._core.initial_state_names)
        self.num_envs = self._core.num_envs
        self.frame_skip = self._core.frame_skip
        self.grayscale = self._core.grayscale
        self.frame_stack = self._core.frame_stack
        self.frame_maxpool = self._core.frame_maxpool
        self.terminate_on_flag = terminate_on_flag
        self.terminate_on_life_loss = bool(terminate_on_life_loss)
        self.terminate_on_level_change = bool(terminate_on_level_change)
        self.done_on_info_rules = done_on_info_rules
        self.noop_reset_max = self._core.noop_reset_max
        self.sticky_action_prob = self._core.sticky_action_prob
        self.crop_top = self._core.crop_top
        self.crop_bottom = self._core.crop_bottom
        self.resize_width = self._core.resize_width
        self.resize_height = self._core.resize_height
        self.single_action_space = spaces.Discrete(len(self.action_meanings))
        self.vector_action_space = spaces.MultiDiscrete([len(self.action_meanings)] * self.num_envs)
        observation_space = spaces.Box(
            low=0,
            high=255,
            shape=self._core.obs_shape()[1:],
            dtype=np.uint8,
        )
        self.render_mode = None
        super().__init__(
            num_envs=self.num_envs,
            observation_space=observation_space,
            action_space=self.single_action_space,
        )

        self._actions = np.zeros((self.num_envs,), dtype=np.uint8)
        self._obs = np.empty(self._core.obs_shape(), dtype=np.uint8)
        self._rewards = np.empty((self.num_envs,), dtype=np.float32)
        self._terminated = np.empty((self.num_envs,), dtype=np.bool_)
        self._truncated = np.empty((self.num_envs,), dtype=np.bool_)
        self._x_pos = np.empty((self.num_envs,), dtype=np.uint16)
        self._coins = np.empty((self.num_envs,), dtype=np.uint8)
        self._level_hi = np.empty((self.num_envs,), dtype=np.int16)
        self._level_lo = np.empty((self.num_envs,), dtype=np.int16)
        self._lives = np.empty((self.num_envs,), dtype=np.int16)
        self._score = np.empty((self.num_envs,), dtype=np.uint32)
        self._scrolling = np.empty((self.num_envs,), dtype=np.int16)
        self._time = np.empty((self.num_envs,), dtype=np.uint16)
        self._xscroll_hi = np.empty((self.num_envs,), dtype=np.uint8)
        self._xscroll_lo = np.empty((self.num_envs,), dtype=np.uint8)
        self._active_state_indices = np.empty((self.num_envs,), dtype=np.int32)
        self._done_on_info: list[dict[str, dict[str, Any]]] = [
            {} for _ in range(self.num_envs)
        ]
        self._terminal_observations: list[np.ndarray | None] = [
            None for _ in range(self.num_envs)
        ]
        self._write_active_state_indices()

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> np.ndarray:
        if seed is not None:
            self.seed(seed)
        if options is not None:
            self.set_options(options)
        pending_seed = next((value for value in self._seeds if value is not None), None)
        if pending_seed is not None:
            self._core.seed(int(pending_seed))
        self._core.reset_into(self._obs)
        self._rewards.fill(0)
        self._terminated.fill(False)
        self._truncated.fill(False)
        self._done_on_info = [{} for _ in range(self.num_envs)]
        self._terminal_observations = [None for _ in range(self.num_envs)]
        self._write_active_state_indices()
        self._write_info_arrays()
        self.reset_infos = [self._reset_info_dict(index) for index in range(self.num_envs)]
        self._reset_seeds()
        self._reset_options()
        return self._obs

    def seed(self, seed: int | None = None) -> list[int | None]:
        return list(super().seed(seed))

    def enable_profiler(self) -> None:
        self._core.enable_profiler()

    def reset_profiler(self) -> None:
        self._core.reset_profiler()

    def disable_profiler(self) -> None:
        self._core.disable_profiler()

    def profiler_snapshot(self, top_n: int = 64) -> dict[str, Any]:
        return json.loads(self._core.profiler_snapshot(int(top_n)))

    def step_async(self, actions: np.ndarray) -> None:
        actions_arr = np.asarray(actions, dtype=np.uint8)
        if actions_arr.shape != (self.num_envs,):
            raise ValueError(f"actions must have shape {(self.num_envs,)}, got {actions_arr.shape}")
        if actions_arr.size and int(actions_arr.max()) >= len(self.action_meanings):
            raise ValueError(
                f"actions must be in [0, {len(self.action_meanings) - 1}] "
                f"for action_set={self.action_set!r}"
            )
        np.copyto(self._actions, self._core_action_ids[actions_arr])

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        obs, rewards, terminated, truncated = self.step_wait_fast()
        dones = np.logical_or(terminated, truncated)
        infos = [self._info_dict(index) for index in range(self.num_envs)]
        return obs, rewards, dones, infos

    def step_wait_gymnasium(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        obs, rewards, terminated, truncated = self.step_wait_fast()
        infos = [self._info_dict(index) for index in range(self.num_envs)]
        return obs, rewards, terminated, truncated, infos

    def step_wait_fast(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Step the whole batch without allocating per-env info dictionaries."""
        self._core.step_into(
            self._actions,
            self._obs,
            self._rewards,
            self._terminated,
            self._truncated,
            self._x_pos,
            self._coins,
            self._level_hi,
            self._level_lo,
            self._lives,
            self._score,
            self._scrolling,
            self._time,
            self._xscroll_hi,
            self._xscroll_lo,
        )
        if np.any(self._terminated) or np.any(self._truncated):
            self._write_active_state_indices()
        self._write_done_on_info()
        self._write_terminal_observations()
        return self._obs, self._rewards, self._terminated, self._truncated

    def _write_done_on_info(self) -> None:
        reports = self._core.done_on_info()
        self._done_on_info = []
        for lane_reports in reports:
            lane_done_on_info = {}
            for name, keys, op, prev, next_values in lane_reports:
                lane_done_on_info[str(name)] = {
                    "op": str(op),
                    "keys": list(keys),
                    "prev": list(prev),
                    "next": list(next_values),
                }
            self._done_on_info.append(lane_done_on_info)

    def _write_terminal_observations(self) -> None:
        reports = self._core.terminal_observations()
        obs_shape = self._obs.shape[1:]
        self._terminal_observations = []
        for report in reports:
            if report is None:
                self._terminal_observations.append(None)
                continue
            if isinstance(report, (bytes, bytearray, memoryview)):
                terminal_obs = np.frombuffer(report, dtype=np.uint8).reshape(obs_shape).copy()
            else:
                terminal_obs = np.asarray(report, dtype=np.uint8).reshape(obs_shape)
            self._terminal_observations.append(terminal_obs)

    def step_gymnasium(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        self.step_async(actions)
        return self.step_wait_gymnasium()

    def step_fast(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.step_async(actions)
        return self.step_wait_fast()

    def _write_info_arrays(self) -> None:
        self._core.info_into(
            self._x_pos,
            self._coins,
            self._level_hi,
            self._level_lo,
            self._lives,
            self._score,
            self._scrolling,
            self._time,
            self._xscroll_hi,
            self._xscroll_lo,
        )

    def _write_active_state_indices(self) -> None:
        self._active_state_indices[:] = np.asarray(
            self._core.active_state_indices(),
            dtype=np.int32,
        )

    def _info_dict(self, index: int) -> dict[str, Any]:
        info = {
            "coins": int(self._coins[index]),
            "levelHi": int(self._level_hi[index]),
            "levelLo": int(self._level_lo[index]),
            "lives": int(self._lives[index]),
            "score": int(self._score[index]),
            "scrolling": int(self._scrolling[index]),
            "time": int(self._time[index]),
            "xscrollHi": int(self._xscroll_hi[index]),
            "xscrollLo": int(self._xscroll_lo[index]),
        }
        if self._done_on_info[index]:
            info["done_on_info"] = self._done_on_info[index]
        if bool(self._terminated[index]) or bool(self._truncated[index]):
            terminal_observation = self._terminal_observations[index]
            if terminal_observation is not None:
                info["terminal_observation"] = terminal_observation
            info["reset_info"] = self._reset_info_dict(index)
            info["TimeLimit.truncated"] = bool(self._truncated[index])
        return info

    def _reset_info_dict(self, index: int) -> dict[str, Any]:
        if not self._state_collection:
            return {}
        if len(self.initial_state_names) == 0:
            return {}
        state_index = int(self._active_state_indices[index])
        if state_index < 0:
            return {}
        state = self.initial_state_names[state_index]
        return {
            "state": state,
            "start_state": state,
        }

    @property
    def x_pos(self) -> np.ndarray:
        return self._x_pos

    @property
    def coins(self) -> np.ndarray:
        return self._coins

    @property
    def level_hi(self) -> np.ndarray:
        return self._level_hi

    @property
    def level_lo(self) -> np.ndarray:
        return self._level_lo

    @property
    def lives(self) -> np.ndarray:
        return self._lives

    @property
    def score(self) -> np.ndarray:
        return self._score

    @property
    def scrolling(self) -> np.ndarray:
        return self._scrolling

    @property
    def time(self) -> np.ndarray:
        return self._time

    @property
    def xscroll_hi(self) -> np.ndarray:
        return self._xscroll_hi

    @property
    def xscroll_lo(self) -> np.ndarray:
        return self._xscroll_lo

    def active_state_indices(self) -> np.ndarray:
        view = self._active_state_indices.view()
        view.setflags(write=False)
        return view

    def active_states(self) -> tuple[str | None, ...]:
        names = self.initial_state_names
        return tuple(
            None if int(index) < 0 else names[int(index)]
            for index in self._active_state_indices
        )

    def close(self) -> None:
        pass

    def get_attr(self, attr_name: str, indices: None | int | Iterable[int] = None) -> list[Any]:
        if not hasattr(self, attr_name):
            raise AttributeError(attr_name)
        value = getattr(self, attr_name)
        return [self._lane_attr_value(value, index) for index in self._get_indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices: None | int | Iterable[int] = None) -> None:
        if hasattr(self, attr_name):
            current = getattr(self, attr_name)
            if isinstance(current, np.ndarray) and current.shape[:1] == (self.num_envs,):
                for index in self._get_indices(indices):
                    current[int(index)] = value
                return
        selected = list(self._get_indices(indices))
        if selected != list(range(self.num_envs)):
            raise AttributeError(f"cannot set per-lane attribute {attr_name!r}")
        setattr(self, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: None | int | Iterable[int] = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        if not hasattr(self, method_name):
            raise AttributeError(method_name)
        method = getattr(self, method_name)
        if not callable(method):
            raise AttributeError(f"{method_name!r} is not callable")
        result = method(*method_args, **method_kwargs)
        return [result for _ in self._get_indices(indices)]

    def env_is_wrapped(
        self,
        wrapper_class: type[Any],
        indices: None | int | Iterable[int] = None,
    ) -> list[bool]:
        return [False for _ in self._get_indices(indices)]

    def get_images(self) -> Sequence[np.ndarray | None]:
        return [None for _ in range(self.num_envs)]

    def _lane_attr_value(self, value: Any, index: int) -> Any:
        if isinstance(value, np.ndarray) and value.shape[:1] == (self.num_envs,):
            item = value[int(index)]
            return item.item() if isinstance(item, np.generic) else item
        if isinstance(value, list) and len(value) == self.num_envs:
            return value[int(index)]
        if isinstance(value, tuple) and len(value) == self.num_envs and value != self.action_meanings:
            return value[int(index)]
        return value
