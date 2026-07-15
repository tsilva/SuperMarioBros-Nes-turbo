from __future__ import annotations

import importlib.metadata
import inspect

import numpy as np
import pytest

from rom_helpers import require_rom
from scripts.benchmark_sps import (
    GAME,
    PreprocessingConfig,
    create_stable_retro_vector_env,
    named_action_mask,
)
from supermariobrosnes_turbo import (
    Actions,
    NES_BUTTONS,
    SuperMarioBrosNesTurboVecEnv,
    action_batch,
)
from supermariobrosnes_turbo import _supermariobrosnes_turbo as native


def test_oracle_is_upstream_stable_retro() -> None:
    assert importlib.metadata.version("stable-retro") == "1.0.1"


def test_native_binding_removed_lifecycle_and_policy_mutators() -> None:
    for name in (
        "done_on_info",
        "terminal_observations",
        "terminal_infos",
        "set_initial_states",
    ):
        assert not hasattr(native._RetroVecEnv, name)


def test_public_signature_preserves_vector_features() -> None:
    params = inspect.signature(SuperMarioBrosNesTurboVecEnv).parameters
    for name in (
        "state",
        "obs_copy",
        "obs_resize",
        "obs_crop",
        "obs_grayscale",
        "frame_skip",
        "frame_stack",
        "maxpool_last_two",
        "noop_reset_max",
        "sticky_action_prob",
        "reward_clip",
        "info_filter",
    ):
        assert name in params
    for name in ("done_on", "autoreset_mode"):
        assert name not in params


@pytest.mark.retro_oracle
@pytest.mark.parametrize("num_envs", [1, 4])
def test_upstream_oracle_exact_short_sequence_parity(num_envs: int) -> None:
    rom_path = require_rom()
    states = [f"Level1-{index + 1}" for index in range(num_envs)]
    preprocessing = PreprocessingConfig(4, 4, True, 32, 0, "mask", 84, 84)
    retro_env = create_stable_retro_vector_env(
        rom_path=rom_path,
        lane_state_names=states,
        preprocessing=preprocessing,
        asynchronous=True,
    )
    fast_env = SuperMarioBrosNesTurboVecEnv(
        GAME,
        state=states,
        rom_path=rom_path,
        num_envs=num_envs,
        use_restricted_actions=Actions.ALL,
        frame_skip=4,
        frame_stack=4,
        obs_grayscale=True,
        obs_crop=(32, 0, 0, 0),
        obs_crop_mode="mask",
        obs_resize=(84, 84),
        obs_resize_algorithm="area",
        obs_layout="chw",
        maxpool_last_two=False,
    )
    action_names = ("noop", "right", "right_b", "right_a") * 4
    try:
        retro_obs, _ = retro_env.reset()
        fast_obs, _ = fast_env.reset()
        assert retro_obs.shape == fast_obs.shape == (num_envs, 4, 84, 84)
        assert retro_obs.dtype == fast_obs.dtype == np.uint8
        np.testing.assert_array_equal(retro_obs, fast_obs)
        for action_name in action_names:
            retro_action = np.repeat(
                named_action_mask(action_name, retro_env.buttons)[None, :], num_envs, axis=0
            )
            fast_action = action_batch(action_name, num_envs)
            retro_obs, retro_rewards, retro_terminated, retro_truncated, _ = retro_env.step(
                retro_action
            )
            fast_obs, fast_rewards, fast_terminated, fast_truncated, _ = fast_env.step(
                fast_action
            )
            np.testing.assert_array_equal(retro_obs, fast_obs)
            np.testing.assert_array_equal(retro_rewards, fast_rewards)
            np.testing.assert_array_equal(retro_terminated, fast_terminated)
            np.testing.assert_array_equal(retro_truncated, fast_truncated)
    finally:
        retro_env.close()
        fast_env.close()


def test_action_and_layout_runtime_smoke() -> None:
    env = SuperMarioBrosNesTurboVecEnv(
        GAME,
        state="Level1-1",
        rom_path=require_rom(),
        num_envs=1,
        use_restricted_actions=Actions.ALL,
        obs_layout="hwc",
        obs_grayscale=False,
        obs_resize=(96, 112),
        obs_crop=(16, 8, 4, 4),
        frame_skip=2,
        frame_stack=2,
        maxpool_last_two=True,
        noop_reset_max=2,
        sticky_action_prob=0.1,
    )
    try:
        obs, _ = env.reset(seed=5)
        assert obs.shape == (1, 96, 112, 6)
        actions = np.zeros((1, len(NES_BUTTONS)), dtype=np.uint8)
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        assert next_obs.shape == obs.shape
        assert rewards.shape == terminated.shape == truncated.shape == (1,)
        assert "lives" in infos and "levelHi" in infos and "levelLo" in infos
    finally:
        env.close()
