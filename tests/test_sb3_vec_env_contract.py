from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from supermariobrosnes_turbo import ACTION_MEANINGS, SuperMarioBrosVecEnv


DEFAULT_ROM = Path("~/Desktop/roms/NES/mapper-000-NROM/SuperMarioBros-Nes-v0.nes")


def require_rom() -> Path:
    rom_path = DEFAULT_ROM.expanduser()
    if not rom_path.exists():
        pytest.skip(f"local SuperMarioBros-Nes ROM is missing: {rom_path}")
    return rom_path


def make_env(rom_path: Path, **kwargs) -> SuperMarioBrosVecEnv:
    return SuperMarioBrosVecEnv(
        rom_path=rom_path,
        num_envs=kwargs.pop("num_envs", 2),
        frame_skip=kwargs.pop("frame_skip", 4),
        frame_stack=kwargs.pop("frame_stack", 1),
        grayscale=True,
        crop_top=32,
        crop_bottom=0,
        resize_width=84,
        resize_height=84,
        state=kwargs.pop("state", "Level1-1"),
        action_set="simple",
        seed=kwargs.pop("seed", 123),
        terminate_on_flag=False,
        **kwargs,
    )


def test_super_mario_vec_env_is_sb3_vec_env_type() -> None:
    assert issubclass(SuperMarioBrosVecEnv, VecEnv)


def test_sb3_step_contract_and_reset_infos() -> None:
    env = make_env(require_rom())
    try:
        assert isinstance(env, VecEnv)
        assert isinstance(env.action_space, spaces.Discrete)
        assert isinstance(env.vector_action_space, spaces.MultiDiscrete)

        obs = env.reset()
        assert obs.shape == (2, 1, 84, 84)
        assert env.reset_infos == [{}, {}]

        actions = np.zeros((env.num_envs,), dtype=np.uint8)
        obs, rewards, dones, infos = env.step(actions)
        assert obs.shape == (2, 1, 84, 84)
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        assert dones.dtype == np.bool_
        assert len(infos) == 2
        assert "xscrollHi" in infos[0]

        gym_step = env.step_gymnasium(actions)
        assert len(gym_step) == 5
    finally:
        env.close()


def test_sb3_terminal_infos_include_terminal_observation_and_reset_info() -> None:
    env = make_env(
        require_rom(),
        num_envs=1,
        done_on_info={"x_progress": ("x_pos", "increase")},
    )
    right = ACTION_MEANINGS.index("right")
    actions = np.asarray([right], dtype=np.uint8)
    try:
        env.reset()
        for _ in range(300):
            _obs, _rewards, dones, infos = env.step(actions)
            if bool(dones[0]):
                info = infos[0]
                assert "terminal_observation" in info
                assert info["terminal_observation"].shape == (1, 84, 84)
                assert info["reset_info"] == {}
                break
        else:
            pytest.fail("x_pos did not increase enough to trigger done_on_info")
    finally:
        env.close()


def test_sb3_reset_infos_preserve_multi_state_labels() -> None:
    env = make_env(
        require_rom(),
        state=["Level1-1", "Level1-2"],
        num_envs=2,
    )
    try:
        env.reset()
        assert env.reset_infos == [
            {"state": "Level1-1", "start_state": "Level1-1"},
            {"state": "Level1-2", "start_state": "Level1-2"},
        ]
    finally:
        env.close()
