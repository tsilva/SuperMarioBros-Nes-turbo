from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ._supermarioemu import FastMarioVecEnv

ACTION_MEANINGS = ("noop", "right", "right_b", "right_a", "right_a_b", "a", "left", "start")


def _expand_rom_path(path: str | Path) -> str:
    return str(Path(path).expanduser())


class SuperMarioBrosVecEnv:
    """Vectorized Mario environment with the hot loop in Rust.

    The important API is `step_wait()`: it performs one Python/Rust crossing for
    the whole batch, with frame skip, grayscale, and frame stacking already done
    before the observation buffer reaches Python.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        rom_path: str | Path = "~/Desktop/roms/SuperMarioBros.nes",
        num_envs: int = 1,
        frame_skip: int = 4,
        grayscale: bool = True,
        frame_stack: int = 4,
        terminate_on_flag: bool = True,
        crop_top: int = 0,
        crop_bottom: int = 0,
        resize_width: int = 84,
        resize_height: int = 84,
    ) -> None:
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
        )
        self.num_envs = self._core.num_envs
        self.frame_skip = self._core.frame_skip
        self.grayscale = self._core.grayscale
        self.frame_stack = self._core.frame_stack
        self.terminate_on_flag = terminate_on_flag
        self.crop_top = self._core.crop_top
        self.crop_bottom = self._core.crop_bottom
        self.resize_width = self._core.resize_width
        self.resize_height = self._core.resize_height
        self.single_action_space = spaces.Discrete(len(ACTION_MEANINGS))
        self.action_space = spaces.MultiDiscrete([len(ACTION_MEANINGS)] * self.num_envs)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=self._core.obs_shape()[1:],
            dtype=np.uint8,
        )

        self._actions = np.zeros((self.num_envs,), dtype=np.uint8)
        self._obs = np.empty(self._core.obs_shape(), dtype=np.uint8)
        self._rewards = np.empty((self.num_envs,), dtype=np.float32)
        self._terminated = np.empty((self.num_envs,), dtype=np.bool_)
        self._truncated = np.empty((self.num_envs,), dtype=np.bool_)
        self._x_pos = np.empty((self.num_envs,), dtype=np.uint16)
        self._lives = np.empty((self.num_envs,), dtype=np.uint8)

    def reset(self) -> np.ndarray:
        self._core.reset_into(self._obs)
        self._rewards.fill(0)
        self._terminated.fill(False)
        self._truncated.fill(False)
        self._x_pos.fill(0)
        self._lives.fill(3)
        return self._obs

    def step_async(self, actions: np.ndarray) -> None:
        actions_arr = np.asarray(actions, dtype=np.uint8)
        if actions_arr.shape != (self.num_envs,):
            raise ValueError(f"actions must have shape {(self.num_envs,)}, got {actions_arr.shape}")
        np.copyto(self._actions, actions_arr)

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        obs, rewards, terminated, truncated = self.step_wait_fast()
        infos = [{"x_pos": int(x), "lives": int(l)} for x, l in zip(self._x_pos, self._lives)]
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
            self._lives,
        )
        return self._obs, self._rewards, self._terminated, self._truncated

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        self.step_async(actions)
        return self.step_wait()

    def step_fast(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.step_async(actions)
        return self.step_wait_fast()

    @property
    def x_pos(self) -> np.ndarray:
        return self._x_pos

    @property
    def lives(self) -> np.ndarray:
        return self._lives

    def close(self) -> None:
        pass


class SuperMarioBrosEnv(gym.Env[np.ndarray, int]):
    """Single-env Gymnasium wrapper over the Rust vectorized core."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        rom_path: str | Path = "~/Desktop/roms/SuperMarioBros.nes",
        frame_skip: int = 4,
        grayscale: bool = True,
        frame_stack: int = 4,
        terminate_on_flag: bool = True,
        crop_top: int = 0,
        crop_bottom: int = 0,
        resize_width: int = 84,
        resize_height: int = 84,
    ) -> None:
        self._vec = SuperMarioBrosVecEnv(
            rom_path=rom_path,
            num_envs=1,
            frame_skip=frame_skip,
            grayscale=grayscale,
            frame_stack=frame_stack,
            terminate_on_flag=terminate_on_flag,
            crop_top=crop_top,
            crop_bottom=crop_bottom,
            resize_width=resize_width,
            resize_height=resize_height,
        )
        self.action_space = self._vec.single_action_space
        self.observation_space = self._vec.observation_space

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        obs = self._vec.reset()
        return obs[0], {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, rewards, terminated, truncated, infos = self._vec.step(np.asarray([action], dtype=np.uint8))
        return obs[0], float(rewards[0]), bool(terminated[0]), bool(truncated[0]), infos[0]

    def close(self) -> None:
        self._vec.close()
