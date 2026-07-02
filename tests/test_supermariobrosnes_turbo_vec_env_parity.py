from __future__ import annotations

import importlib.metadata
import inspect

import pytest
import numpy as np
from gymnasium import spaces

from scripts import compare_supermariobrosnes_turbo_vec_env as compare
from supermariobrosnes_turbo import (
    Actions,
    Integrations,
    Observations,
    State,
    SuperMarioBrosNesTurboVecEnv,
    SuperMarioBrosVecEnv,
)
from rom_helpers import require_rom

NES_BUTTONS = ("B", None, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A")
BUTTON_TO_INDEX = {name: index for index, name in enumerate(NES_BUTTONS) if name is not None}


def require_stable_retro_oracle() -> None:
    require_rom()
    try:
        version = importlib.metadata.version("stable-retro-turbo")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip(
            "stable-retro-turbo oracle is not installed; run `uv sync --extra dev` "
            "under Python 3.14",
        )
    assert version == compare.EXPECTED_STABLE_RETRO_VERSION


def test_native_vec_env_name_is_public() -> None:
    assert SuperMarioBrosNesTurboVecEnv.__name__ == "SuperMarioBrosNesTurboVecEnv"


def test_native_turbo_vec_env_defaults_match_stable_retro_turbo_signature() -> None:
    native_signature = inspect.signature(SuperMarioBrosNesTurboVecEnv)
    assert list(native_signature.parameters) == [
        "game",
        "state",
        "scenario",
        "info",
        "use_restricted_actions",
        "record",
        "players",
        "inttype",
        "obs_type",
        "render_mode",
        "num_envs",
        "num_threads",
        "rom_path",
        "obs_copy",
        "obs_resize",
        "obs_crop",
        "obs_grayscale",
        "obs_resize_algorithm",
        "obs_layout",
        "frame_skip",
        "frame_stack",
        "maxpool_last_two",
        "noop_reset_max",
        "sticky_action_prob",
        "reward_clip",
        "info_filter",
        "done_on",
        "copy_observations",
        "info_mode",
        "info_keys",
        "done_on_info",
        "unsafe_zero_copy",
    ]

    native_defaults = {
        name: parameter.default
        for name, parameter in native_signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }

    assert native_defaults["scenario"] is None
    assert native_defaults["info"] is None
    assert native_defaults["record"] is False
    assert native_defaults["players"] == 1
    assert native_defaults["render_mode"] == "human"
    assert native_defaults["num_envs"] == 1
    assert native_defaults["num_threads"] is None
    assert native_defaults["rom_path"] is None
    assert native_defaults["obs_copy"] == "copy"
    assert native_defaults["obs_resize"] is None
    assert native_defaults["obs_crop"] is None
    assert native_defaults["obs_grayscale"] is False
    assert native_defaults["obs_resize_algorithm"] == "nearest"
    assert native_defaults["obs_layout"] == "hwc"
    assert native_defaults["frame_skip"] == 1
    assert native_defaults["frame_stack"] == 1
    assert native_defaults["maxpool_last_two"] is False
    assert native_defaults["noop_reset_max"] == 0
    assert native_defaults["sticky_action_prob"] == 0.0
    assert native_defaults["reward_clip"] is False
    assert native_defaults["info_filter"] == "all"
    assert native_defaults["done_on"] is None

    assert native_defaults["state"] is State.DEFAULT
    assert native_defaults["state"].name == "DEFAULT"
    assert native_defaults["state"].value == -1
    assert native_defaults["use_restricted_actions"] is Actions.FILTERED
    assert native_defaults["use_restricted_actions"].name == "FILTERED"
    assert native_defaults["use_restricted_actions"].value == 1
    assert native_defaults["inttype"] is Integrations.STABLE
    assert native_defaults["inttype"].name == "STABLE"
    assert native_defaults["inttype"].value == 1
    assert native_defaults["obs_type"] is Observations.IMAGE
    assert native_defaults["obs_type"].name == "IMAGE"
    assert native_defaults["obs_type"].value == 0

    sentinel_defaults = (
        "copy_observations",
        "info_mode",
        "info_keys",
        "done_on_info",
        "unsafe_zero_copy",
    )
    for name in sentinel_defaults:
        assert type(native_defaults[name]) is object


def test_native_turbo_vec_env_rejects_non_stable_retro_alias_keywords() -> None:
    with pytest.raises(TypeError, match="frame_maxpool"):
        SuperMarioBrosNesTurboVecEnv("SuperMarioBros-Nes-v0", frame_maxpool=False)
    with pytest.raises(TypeError, match="reset_noops"):
        SuperMarioBrosNesTurboVecEnv("SuperMarioBros-Nes-v0", reset_noops=0)
    with pytest.raises(TypeError, match="action_sticky_prob"):
        SuperMarioBrosNesTurboVecEnv("SuperMarioBros-Nes-v0", action_sticky_prob=0.0)


@pytest.mark.retro_oracle
def test_stable_retro_vector_env_constructs_with_oracle_keyword_surface() -> None:
    require_stable_retro_oracle()
    import stable_retro

    rom_path = require_rom()
    env_class = getattr(stable_retro, "Retro" "Vec" "Env")
    env = env_class(
        compare.DEFAULT_STABLE_RETRO_GAME,
        state="Level1-1",
        num_envs=1,
        num_threads=1,
        rom_path=str(rom_path),
        render_mode="rgb_array",
        use_restricted_actions=stable_retro.Actions.ALL,
        obs_crop=(32, 0, 0, 0),
        obs_resize=(84, 84),
        obs_grayscale=True,
        obs_resize_algorithm="area",
        obs_layout="chw",
        obs_copy="safe_view",
        frame_skip=4,
        frame_stack=4,
        maxpool_last_two=True,
        noop_reset_max=0,
        sticky_action_prob=0.0,
        reward_clip=False,
        info_filter="all",
        done_on={
            "life_loss": ("lives", "decrease"),
            "level_change": (("levelHi", "levelLo"), "change"),
        },
    )
    try:
        assert env.num_envs == 1
        assert getattr(env, "obs_copy", None) == "safe_view"
    finally:
        env.close()


def test_native_turbo_vec_env_accepts_smb_keyword_surface() -> None:
    rom_path = require_rom()

    env = SuperMarioBrosNesTurboVecEnv(
        compare.DEFAULT_STABLE_RETRO_GAME,
        state="Level1-1",
        num_envs=1,
        num_threads=1,
        rom_path=str(rom_path),
        render_mode="rgb_array",
        use_restricted_actions=Actions.ALL,
        obs_crop=(32, 0, 0, 0),
        obs_resize=(84, 84),
        obs_grayscale=True,
        obs_resize_algorithm="area",
        obs_layout="chw",
        obs_copy="safe_view",
        frame_skip=4,
        frame_stack=4,
        maxpool_last_two=True,
        noop_reset_max=0,
        sticky_action_prob=0.0,
        reward_clip=False,
        info_filter="terminal",
        done_on=["life_loss"],
    )
    try:
        assert env.num_envs == 1
        assert env.num_threads == 1
        assert env.obs_copy == "safe_view"
        assert env.copy_observations is False
        assert env.unsafe_zero_copy is False
        assert isinstance(env.action_space, spaces.MultiBinary)
        obs = env.reset()
        assert obs.shape == (1, 4, 84, 84)
        masks = np.zeros((1, env.num_buttons), dtype=np.uint8)
        obs, rewards, dones, infos = env.step(masks)
        assert obs.shape == (1, 4, 84, 84)
        assert rewards.shape == (1,)
        assert dones.shape == (1,)
        assert infos == [{}]
    finally:
        env.close()


def test_native_turbo_vec_env_hwc_layout_and_safe_view_survives_next_step() -> None:
    rom_path = require_rom()

    env = SuperMarioBrosNesTurboVecEnv(
        compare.DEFAULT_STABLE_RETRO_GAME,
        state="Level1-1",
        num_envs=1,
        rom_path=str(rom_path),
        render_mode="rgb_array",
        use_restricted_actions=Actions.ALL,
        obs_crop=(32, 0, 0, 0),
        obs_resize=(84, 84),
        obs_grayscale=True,
        obs_resize_algorithm="nearest",
        obs_layout="hwc",
        obs_copy="safe_view",
        frame_skip=1,
        frame_stack=1,
        info_filter="none",
    )
    try:
        first = env.reset()
        first_snapshot = first.copy()
        masks = np.zeros((1, env.num_buttons), dtype=np.uint8)
        masks[0, BUTTON_TO_INDEX["RIGHT"]] = 1
        second, _rewards, _dones, infos = env.step(masks)
        assert first.shape == (1, 84, 84, 1)
        assert second.shape == (1, 84, 84, 1)
        np.testing.assert_array_equal(first, first_snapshot)
        assert infos == [{}]
    finally:
        env.close()


def test_native_turbo_vec_env_reward_clip_and_info_filter_all() -> None:
    rom_path = require_rom()

    env = SuperMarioBrosNesTurboVecEnv(
        compare.DEFAULT_STABLE_RETRO_GAME,
        state="Level1-1",
        num_envs=1,
        rom_path=str(rom_path),
        render_mode="rgb_array",
        use_restricted_actions=Actions.ALL,
        obs_crop=(32, 0, 0, 0),
        obs_resize=(84, 84),
        obs_grayscale=True,
        obs_layout="chw",
        reward_clip=(0.0, 0.0),
        info_filter={"mode": "all", "keys": ("lives", "xscrollHi")},
        frame_skip=4,
    )
    try:
        env.reset()
        masks = np.zeros((1, env.num_buttons), dtype=np.uint8)
        masks[0, BUTTON_TO_INDEX["RIGHT"]] = 1
        _obs, rewards, _dones, infos = env.step(masks)
        assert rewards.tolist() == [0.0]
        assert set(infos[0]) <= {"lives", "xscrollHi"}
        assert "lives" in infos[0]
    finally:
        env.close()


def test_native_turbo_vec_env_actions_all_mask_matches_discrete_fast_env() -> None:
    rom_path = require_rom()

    retro_env = SuperMarioBrosNesTurboVecEnv(
        compare.DEFAULT_STABLE_RETRO_GAME,
        state="Level1-1",
        num_envs=1,
        rom_path=str(rom_path),
        render_mode="rgb_array",
        use_restricted_actions=Actions.ALL,
        obs_crop=(32, 0, 0, 0),
        obs_resize=(84, 84),
        obs_grayscale=True,
        obs_resize_algorithm="area",
        obs_layout="chw",
        frame_skip=1,
        frame_stack=1,
        info_filter="none",
    )
    fast_env = SuperMarioBrosVecEnv(
        rom_path=rom_path,
        num_envs=1,
        frame_skip=1,
        grayscale=True,
        frame_stack=1,
        crop_top=32,
        resize_width=84,
        resize_height=84,
        state="Level1-1",
        action_set="full",
        terminate_on_flag=False,
    )
    try:
        np.testing.assert_array_equal(retro_env.reset(), fast_env.reset())
        masks = np.zeros((1, retro_env.num_buttons), dtype=np.uint8)
        masks[0, BUTTON_TO_INDEX["RIGHT"]] = 1
        masks[0, BUTTON_TO_INDEX["A"]] = 1
        retro_obs, retro_rewards, retro_dones, _infos = retro_env.step(masks)
        fast_obs, fast_rewards, fast_dones, _infos = fast_env.step(
            np.asarray([compare.ACTION_SETS["full"].index("right_a")], dtype=np.uint8),
        )
        np.testing.assert_array_equal(retro_obs, fast_obs)
        np.testing.assert_array_equal(retro_rewards, fast_rewards)
        np.testing.assert_array_equal(retro_dones, fast_dones)
    finally:
        retro_env.close()
        fast_env.close()
