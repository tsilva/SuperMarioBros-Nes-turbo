from __future__ import annotations

from pathlib import Path

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from supermariobrosnes_turbo import ACTION_MEANINGS, SuperMarioBrosVecEnv
from rom_helpers import require_rom


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


def test_active_state_indices_are_read_only_and_track_state_labels() -> None:
    env = make_env(
        require_rom(),
        state=["Level1-1", "Level1-2"],
        num_envs=2,
    )
    try:
        env.reset()
        active_state_indices = env.active_state_indices()

        np.testing.assert_array_equal(active_state_indices, np.asarray([0, 1], dtype=np.int32))
        assert active_state_indices.flags.writeable is False
        with pytest.raises(ValueError, match="read-only"):
            active_state_indices[0] = 1
        assert env.active_states() == ("Level1-1", "Level1-2")
    finally:
        env.close()


def test_sb3_helper_methods_respect_lane_indices() -> None:
    env = make_env(require_rom(), num_envs=2)
    try:
        env.reset()
        x_pos_before = env.x_pos.copy()

        assert env.get_attr("x_pos", indices=0) == [int(x_pos_before[0])]
        env.set_attr("x_pos", 123, indices=[1])
        assert env.get_attr("x_pos") == [int(x_pos_before[0]), 123]

        active_states = env.active_states()
        assert env.env_method("active_states", indices=[0, 1]) == [active_states, active_states]
        assert env.env_is_wrapped(VecEnv, indices=[0, 1]) == [False, False]
        assert env.get_images() == [None, None]

        env.set_attr("custom_attr", "shared")
        assert env.get_attr("custom_attr", indices=[0, 1]) == ["shared", "shared"]
        with pytest.raises(AttributeError, match="missing_attr"):
            env.get_attr("missing_attr")
        with pytest.raises(AttributeError, match="cannot set per-lane attribute"):
            env.set_attr("other_custom_attr", "lane-only", indices=[0])
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


def test_weighted_state_sampling_survives_lane_local_autoreset() -> None:
    env = make_env(
        require_rom(),
        state={"Level1-1": 0.5, "Level1-2": 0.5},
        num_envs=4,
        done_on_info={"x_progress": ("x_pos", "increase")},
    )
    right = ACTION_MEANINGS.index("right")
    noop = ACTION_MEANINGS.index("noop")
    actions = np.asarray([right, noop, noop, noop], dtype=np.uint8)
    valid_states = {"Level1-1", "Level1-2"}

    try:
        env.reset()
        before_states = env.active_states()
        assert set(before_states) <= valid_states
        assert all(info["state"] in valid_states for info in env.reset_infos)

        for _ in range(300):
            _obs, _rewards, dones, infos = env.step(actions)
            done_lanes = np.flatnonzero(dones).tolist()
            if not done_lanes:
                continue

            assert done_lanes == [0]
            after_states = env.active_states()
            assert set(after_states) <= valid_states
            assert after_states[1:] == before_states[1:]

            done_info = infos[0]
            assert done_info["reset_info"] == {
                "state": after_states[0],
                "start_state": after_states[0],
            }
            assert "terminal_observation" in done_info
            assert all("reset_info" not in info for info in infos[1:])
            break
        else:
            pytest.fail("x_pos did not increase enough to trigger lane-local autoreset")
    finally:
        env.close()


def test_terminal_info_filter_only_reports_done_lanes() -> None:
    env = make_env(
        require_rom(),
        num_envs=2,
        done_on_info={"x_progress": ("x_pos", "increase")},
        info_filter="terminal",
    )
    right = ACTION_MEANINGS.index("right")
    noop = ACTION_MEANINGS.index("noop")
    actions = np.asarray([right, noop], dtype=np.uint8)

    try:
        env.reset()
        for _ in range(300):
            _obs, _rewards, dones, infos = env.step(actions)
            if not bool(dones[0]):
                continue

            assert dones.tolist() == [True, False]
            assert "reset_info" in infos[0]
            assert "terminal_observation" in infos[0]
            assert infos[1] == {}
            break
        else:
            pytest.fail("x_pos did not increase enough to trigger terminal-only info")
    finally:
        env.close()


def test_wrapper_option_validation_runs_before_rom_load() -> None:
    missing_rom = "/definitely/missing/SuperMarioBros.nes"
    base_kwargs = {"rom_path": missing_rom, "state": None}

    with pytest.raises(ValueError, match="obs_copy"):
        SuperMarioBrosVecEnv(**base_kwargs, obs_copy=True)
    with pytest.raises(ValueError, match="cannot pass both obs_copy"):
        SuperMarioBrosVecEnv(**base_kwargs, obs_copy="safe_view", copy_observations=False)
    with pytest.raises(ValueError, match="unsafe_zero_copy"):
        SuperMarioBrosVecEnv(**base_kwargs, copy_observations=True, unsafe_zero_copy=True)
    with pytest.raises(ValueError, match="info_filter mode"):
        SuperMarioBrosVecEnv(**base_kwargs, info_filter="sometimes")
    with pytest.raises(ValueError, match="info_filter keys"):
        SuperMarioBrosVecEnv(**base_kwargs, info_filter={"keys": "lives"})
    with pytest.raises(ValueError, match="reward_clip low"):
        SuperMarioBrosVecEnv(**base_kwargs, reward_clip=(1.0, 0.0))
    with pytest.raises(ValueError, match="obs_layout"):
        SuperMarioBrosVecEnv(**base_kwargs, obs_layout="nhwc")
    with pytest.raises(ValueError, match="obs_resize_algorithm"):
        SuperMarioBrosVecEnv(**base_kwargs, obs_resize_algorithm="lanczos")
    with pytest.raises(ValueError, match="unknown configured event"):
        SuperMarioBrosVecEnv(**base_kwargs, done_on=["bad_event"])
    with pytest.raises(ValueError, match="cannot pass both done_on"):
        SuperMarioBrosVecEnv(**base_kwargs, done_on=["life_loss"], done_on_info={})
