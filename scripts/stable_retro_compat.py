from __future__ import annotations

import importlib.util
import sys
import types
from typing import Any


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
