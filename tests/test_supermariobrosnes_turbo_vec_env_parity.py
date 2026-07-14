from __future__ import annotations

import importlib.metadata
import inspect

import numpy as np
import pytest

from rom_helpers import require_rom
from scripts import compare_supermariobrosnes_turbo_vec_env as compare
from supermariobrosnes_turbo import Actions, NES_BUTTONS, SuperMarioBrosNesTurboVecEnv
from supermariobrosnes_turbo import _supermariobrosnes_turbo as native


def test_oracle_version_is_post30() -> None:
    assert compare.EXPECTED_STABLE_RETRO_VERSION == "1.0.1.post30"


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
def test_post30_oracle_raw_signal_and_observation_parity() -> None:
    require_rom()
    assert importlib.metadata.version("stable-retro-turbo") == "1.0.1.post30"
    config = compare.ComparisonConfig(
        rom_path=require_rom(),
        stable_retro_path=None,
        game=compare.DEFAULT_STABLE_RETRO_GAME,
        state="Level1-1",
        num_envs=1,
        env_threads=1,
        steps=16,
        seed=123,
        frame_skip=4,
        frame_stack=1,
        grayscale=True,
        crop_top=32,
        crop_bottom=0,
        resize_width=84,
        resize_height=84,
        action_set="simple",
        frame_maxpool=False,
        noop_reset_max=0,
        sticky_action_prob=0.0,
        obs_copy="copy",
        terminate_on_flag=False,
        include_obs=True,
        include_rewards=True,
        include_dones=True,
        include_infos=True,
        stop_on_done=True,
        fixed_action="noop",
        output_json=None,
        allow_version_mismatch=False,
        preprocessing_matrix=False,
        termination_matrix=False,
    )
    result = compare.run_comparison(config)
    assert result["status"] == "ok"
    assert result["compared_steps"] == 16


def test_action_and_layout_runtime_smoke() -> None:
    env = SuperMarioBrosNesTurboVecEnv(
        "SuperMarioBros-Nes-v0",
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
