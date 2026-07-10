from __future__ import annotations

from pathlib import Path
import inspect
import shutil

import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, VectorEnv

from supermariobrosnes_turbo import (
    ACTION_BUTTONS,
    BUTTON_TO_INDEX,
    Actions,
    NES_BUTTONS,
    SuperMarioBrosNesTurboVecEnv,
    list_available_states,
)
from rom_helpers import require_rom
from supermariobrosnes_turbo.env import _normalize_initial_state_config, _resolve_state_path


EXPECTED_PACKAGED_STATES = {
    "Level1-1",
    "Level1-1-99lives",
    "Level1-2",
    "Level1-3",
    "Level1-4",
    "Level2-1",
    "Level2-1-clouds",
    "Level2-1-clouds-easy",
    "Level2-2",
    "Level2-3",
    "Level2-4",
    "Level3-1",
    "Level3-2",
    "Level3-3",
    "Level3-4",
    "Level4-1",
    "Level4-2",
    "Level4-3",
    "Level4-4",
    "Level5-1",
    "Level5-2",
    "Level5-3",
    "Level5-4",
    "Level6-1",
    "Level6-2",
    "Level6-3",
    "Level6-4",
    "Level7-1",
    "Level7-2",
    "Level7-3",
    "Level7-4",
    "Level8-1",
    "Level8-2",
    "Level8-3",
    "Level8-4",
}


def make_action_batch(num_envs: int, names: str | list[str]) -> np.ndarray:
    if isinstance(names, str):
        names = [names] * num_envs
    masks = np.zeros((num_envs, len(NES_BUTTONS)), dtype=np.uint8)
    for env_idx, name in enumerate(names):
        for button in ACTION_BUTTONS[name]:
            masks[env_idx, BUTTON_TO_INDEX[button]] = 1
    return masks


def make_env(rom_path: Path, **kwargs) -> SuperMarioBrosNesTurboVecEnv:
    seed = kwargs.pop("seed", 123)
    env = SuperMarioBrosNesTurboVecEnv(
        "SuperMarioBros-Nes-v0",
        state=kwargs.pop("state", "Level1-1"),
        rom_path=rom_path,
        num_envs=kwargs.pop("num_envs", 2),
        use_restricted_actions=Actions.ALL,
        frame_skip=kwargs.pop("frame_skip", 4),
        frame_stack=kwargs.pop("frame_stack", 1),
        obs_grayscale=True,
        obs_crop=(32, 0, 0, 0),
        obs_resize=(84, 84),
        obs_resize_algorithm="area",
        obs_layout=kwargs.pop("obs_layout", "chw"),
        **kwargs,
    )
    env.seed(seed)
    return env


def lane_has(infos: dict[str, object], key: str, lane: int) -> bool:
    mask = infos.get(f"_{key}")
    return bool(mask is not None and np.asarray(mask, dtype=np.bool_)[lane])


def lane_value(infos: dict[str, object], key: str, lane: int) -> object:
    assert lane_has(infos, key, lane), f"lane {lane} missing info key {key!r}: {infos}"
    values = infos[key]
    return values[lane]  # type: ignore[index]


def nested_lane_value(infos: dict[str, object], path: tuple[str, ...], lane: int) -> object:
    current: object = infos
    for key in path[:-1]:
        assert isinstance(current, dict)
        current = current[key]
    assert isinstance(current, dict)
    return lane_value(current, path[-1], lane)


def final_done_on_info(infos: dict[str, object], lane: int, event: str) -> dict[str, object]:
    current: object = infos
    for key in ("final_info", "done_on_info", event):
        assert isinstance(current, dict)
        current = current[key]
    assert isinstance(current, dict)
    return {
        key: lane_value(current, key, lane)
        for key in current
        if not key.startswith("_")
    }


def test_super_mario_vector_env_is_gymnasium_vector_env_type() -> None:
    assert issubclass(SuperMarioBrosNesTurboVecEnv, VectorEnv)
    assert SuperMarioBrosNesTurboVecEnv.__name__ == "SuperMarioBrosNesTurboVecEnv"
    assert not issubclass(SuperMarioBrosNesTurboVecEnv, tuple)


def test_runtime_env_module_does_not_import_stable_baselines3() -> None:
    import supermariobrosnes_turbo.env as env_module

    assert "stable_baselines3" not in inspect.getsource(env_module)


def test_packaged_state_inventory_includes_all_levels() -> None:
    states = set(list_available_states())

    assert EXPECTED_PACKAGED_STATES <= states
    assert "supermariobrosnes_turbo/data/SuperMarioBros-Nes-v0/Level8-4.state" in str(
        _resolve_state_path("Level8-4")
    )


def test_state_dir_env_resolves_named_initial_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state_dir = tmp_path / "states"
    state_dir.mkdir()
    shutil.copy2(_resolve_state_path("Level1-1"), state_dir / "Level1-1.state")
    monkeypatch.setenv("SUPERMARIOBROSNES_FASTENV_STATE_DIR", str(state_dir))

    initial_states, state_names, state_weights = _normalize_initial_state_config(
        "Level1-1",
        None,
        num_envs=1,
    )

    assert len(initial_states) == 1
    assert state_names == ("Level1-1",)
    assert state_weights is None


def test_constructor_state_accepts_string_list_and_dict() -> None:
    single_states, single_names, single_weights = _normalize_initial_state_config(
        "Level1-1",
        None,
        num_envs=2,
    )
    list_states, list_names, list_weights = _normalize_initial_state_config(
        ["Level1-1", "Level1-2"],
        None,
        num_envs=2,
    )
    dict_states, dict_names, dict_weights = _normalize_initial_state_config(
        {"Level1-1": 0.25, "Level1-2": 0.75},
        None,
        num_envs=2,
    )

    assert len(single_states) == 1
    assert single_names == ("Level1-1",)
    assert single_weights is None
    assert len(list_states) == 2
    assert list_names == ("Level1-1", "Level1-2")
    assert list_weights is None
    assert len(dict_states) == 2
    assert dict_names == ("Level1-1", "Level1-2")
    assert dict_weights == [0.25, 0.75]


def test_weighted_state_validation_allows_zero_but_rejects_negative_and_all_zero() -> None:
    _states, names, weights = _normalize_initial_state_config(
        {"Level1-1": 0.0, "Level1-2": 3.0},
        None,
        num_envs=2,
    )
    assert names == ("Level1-1", "Level1-2")
    assert weights == [0.0, 1.0]

    with pytest.raises(ValueError, match="non-negative finite"):
        _normalize_initial_state_config(
            {"Level1-1": -0.1, "Level1-2": 1.0},
            None,
            num_envs=2,
        )
    with pytest.raises(ValueError, match="positive finite"):
        _normalize_initial_state_config(
            {"Level1-1": 0.0, "Level1-2": 0.0},
            None,
            num_envs=2,
        )


def test_gymnasium_reset_step_contract_and_spaces() -> None:
    env = make_env(require_rom())
    try:
        assert isinstance(env.action_space, spaces.MultiBinary)
        assert isinstance(env.single_action_space, spaces.MultiBinary)
        assert env.single_observation_space.shape == (1, 84, 84)
        assert env.observation_space.shape == (2, 1, 84, 84)
        assert env.metadata["autoreset_mode"] is AutoresetMode.SAME_STEP

        obs, infos = env.reset(seed=123)
        assert obs.shape == (2, 1, 84, 84)
        assert infos == {}

        actions = make_action_batch(env.num_envs, "noop")
        obs, rewards, terminations, truncations, infos = env.step(actions)
        assert obs.shape == (2, 1, 84, 84)
        assert rewards.shape == (2,)
        assert terminations.shape == (2,)
        assert terminations.dtype == np.bool_
        assert truncations.shape == (2,)
        assert truncations.dtype == np.bool_
        assert "x_pos" in infos
        assert lane_has(infos, "x_pos", 0)
        assert lane_has(infos, "x_pos", 1)
        assert "xscrollHi" in infos
        assert lane_has(infos, "xscrollHi", 0)
        assert lane_has(infos, "xscrollHi", 1)
    finally:
        env.close()


def test_disabled_autoreset_returns_terminal_transition_until_masked_reset() -> None:
    manual_env = make_env(
        require_rom(),
        num_envs=1,
        done_on={"x_progress": ("x_pos", "increase")},
        autoreset_mode=AutoresetMode.DISABLED,
    )
    same_step_env = make_env(
        require_rom(),
        num_envs=1,
        done_on={"x_progress": ("x_pos", "increase")},
    )
    actions = make_action_batch(1, "right")
    try:
        manual_env.reset(seed=[123])
        same_step_env.reset(seed=[123])
        assert manual_env.autoreset_mode is AutoresetMode.DISABLED
        assert manual_env.metadata["autoreset_mode"] is AutoresetMode.DISABLED
        assert same_step_env.metadata["autoreset_mode"] is AutoresetMode.SAME_STEP

        for _ in range(300):
            manual_result = manual_env.step(actions)
            same_step_result = same_step_env.step(actions)
            if bool(manual_result[2][0] or manual_result[3][0]):
                break
        else:
            pytest.fail("x_pos did not increase enough to trigger disabled-mode termination")

        manual_obs, _, terminated, truncated, infos = manual_result
        same_step_infos = same_step_result[4]
        assert terminated.tolist() == [True]
        assert truncated.tolist() == [False]
        assert not lane_has(infos, "final_obs", 0)
        assert not lane_has(infos, "final_info", 0)
        assert bool(lane_value(infos, "terminated", 0)) is True
        np.testing.assert_array_equal(manual_obs[0], lane_value(same_step_infos, "final_obs", 0))

        with pytest.raises(RuntimeError, match="pending reset"):
            manual_env.step(actions)

        reset_obs, reset_infos = manual_env.reset(
            options={"reset_mask": np.array([True], dtype=np.bool_)},
        )
        assert lane_has(reset_infos, "x_pos", 0)
        assert reset_obs.shape == manual_obs.shape
        manual_env.step(actions)
    finally:
        manual_env.close()
        same_step_env.close()


def test_masked_reset_preserves_unselected_lanes_and_allows_active_lane_reset() -> None:
    kwargs = {
        "num_envs": 4,
        "state": ["Level1-1"] * 4,
        "frame_stack": 4,
        "autoreset_mode": AutoresetMode.DISABLED,
        "noop_reset_max": 3,
        "sticky_action_prob": 0.5,
    }
    env = make_env(require_rom(), **kwargs)
    control = make_env(require_rom(), **kwargs)
    actions = make_action_batch(4, ["right", "right_b", "noop", "right_a"])
    try:
        obs, _ = env.reset(seed=[11, 22, 33, 44])
        control_obs, _ = control.reset(seed=[11, 22, 33, 44])
        np.testing.assert_array_equal(obs, control_obs)
        obs, *_ = env.step(actions)
        control_obs, *_ = control.step(actions)
        np.testing.assert_array_equal(obs, control_obs)

        terminal_copies = obs[[1, 3]].copy()
        unselected_ram = [bytes(env._core._debug_ram(index)) for index in (0, 2)]
        options = {"reset_mask": np.array([False, True, False, True], dtype=np.bool_)}
        options_copy = options["reset_mask"].copy()
        reset_obs, reset_infos = env.reset(options=options)
        np.testing.assert_array_equal(options["reset_mask"], options_copy)
        np.testing.assert_array_equal(reset_obs[[0, 2]], obs[[0, 2]])
        assert [bytes(env._core._debug_ram(index)) for index in (0, 2)] == unselected_ram
        assert not np.array_equal(reset_obs[[1, 3]], terminal_copies)
        assert not lane_has(reset_infos, "x_pos", 0)
        assert lane_has(reset_infos, "x_pos", 1)
        assert not lane_has(reset_infos, "x_pos", 2)
        assert lane_has(reset_infos, "x_pos", 3)

        for _ in range(20):
            result = env.step(actions)
            control_result = control.step(actions)
            np.testing.assert_array_equal(result[0][[0, 2]], control_result[0][[0, 2]])
            np.testing.assert_array_equal(result[1][[0, 2]], control_result[1][[0, 2]])
    finally:
        env.close()
        control.close()


def test_masked_reset_explicit_start_indices_and_seed_determinism() -> None:
    kwargs = {
        "num_envs": 2,
        "state": {"Level1-1": 0.5, "Level1-2": 0.5},
        "noop_reset_max": 5,
        "sticky_action_prob": 0.5,
        "autoreset_mode": AutoresetMode.DISABLED,
    }
    env = make_env(require_rom(), **kwargs)
    twin = make_env(require_rom(), **kwargs)
    mask = np.array([True, False], dtype=np.bool_)
    starts = np.array([1, 999], dtype=np.int32)
    try:
        zero_seed, _ = env.reset(seed=0)
        twin_zero_seed, _ = twin.reset(seed=[0, 1])
        np.testing.assert_array_equal(zero_seed, twin_zero_seed)

        first, _ = env.reset(seed=123)
        twin_first, _ = twin.reset(seed=[123, 124])
        np.testing.assert_array_equal(first, twin_first)

        before_lane_one = first[1].copy()
        reset_obs, infos = env.reset(
            seed=[777, 999],
            options={"reset_mask": mask, "start_indices": starts},
        )
        assert env.active_states()[0] == "Level1-2"
        np.testing.assert_array_equal(reset_obs[1], before_lane_one)
        assert lane_value(infos, "start_state", 0) == "Level1-2"
        assert not lane_has(infos, "start_state", 1)

        sampled_obs, _ = env.reset(
            seed=[888, None],
            options={
                "reset_mask": mask,
                "start_indices": np.array([-1, -1], dtype=np.int32),
            },
        )
        assert env.active_states()[0] in {"Level1-1", "Level1-2"}
        np.testing.assert_array_equal(sampled_obs[1], before_lane_one)
    finally:
        env.close()
        twin.close()


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"reset_mask": [True, False]}, TypeError, "NumPy array"),
        ({"reset_mask": np.array([True], dtype=np.bool_)}, ValueError, "shape"),
        ({"reset_mask": np.array([1, 0], dtype=np.uint8)}, TypeError, "dtype"),
        ({"reset_mask": np.array([False, False], dtype=np.bool_)}, ValueError, "at least one"),
        ({"reset_mask": np.array([True, False], dtype=np.bool_), "unknown": True}, ValueError, "unsupported reset option"),
        (
            {
                "reset_mask": np.array([True, False], dtype=np.bool_),
                "start_indices": np.array([99, -1], dtype=np.int32),
            },
            ValueError,
            "valid initial-state catalog index",
        ),
    ],
)
def test_manual_lifecycle_reset_validation(options, error, message) -> None:
    env = make_env(require_rom(), num_envs=2, state=["Level1-1", "Level1-2"])
    try:
        with pytest.raises(error, match=message):
            env.reset(options=options)
    finally:
        env.close()


def test_manual_lifecycle_rejects_bad_seed_length_and_next_step() -> None:
    with pytest.raises(ValueError, match="SAME_STEP or AutoresetMode.DISABLED"):
        make_env(require_rom(), autoreset_mode=AutoresetMode.NEXT_STEP)

    env = make_env(require_rom(), num_envs=2)
    try:
        with pytest.raises(ValueError, match="seed sequence length"):
            env.reset(seed=[123])
    finally:
        env.close()


def test_active_state_indices_are_read_only_and_track_state_labels() -> None:
    env = make_env(
        require_rom(),
        state=["Level1-1", "Level1-2"],
        num_envs=2,
    )
    try:
        _obs, infos = env.reset()
        active_state_indices = env.active_state_indices()

        np.testing.assert_array_equal(active_state_indices, np.asarray([0, 1], dtype=np.int32))
        assert active_state_indices.flags.writeable is False
        with pytest.raises(ValueError, match="read-only"):
            active_state_indices[0] = 1
        assert env.active_states() == ("Level1-1", "Level1-2")
        assert lane_value(infos, "state", 0) == "Level1-1"
        assert lane_value(infos, "state", 1) == "Level1-2"
    finally:
        env.close()


def test_rgb_array_rendering_keeps_policy_observation_preprocessed() -> None:
    env = make_env(
        require_rom(),
        num_envs=1,
        frame_stack=4,
        render_mode="rgb_array",
    )
    try:
        obs, _infos = env.reset()

        assert obs.shape == (1, 4, 84, 84)
        assert env.single_observation_space.shape == (4, 84, 84)

        images = env.get_images()
        assert len(images) == 1
        frame = images[0]
        assert frame is not None
        assert frame.shape == (224, 240, 3)
        assert frame.dtype == np.uint8

        rendered = env.render()
        assert rendered is not None
        assert rendered.shape == (224, 240, 3)
        assert rendered.dtype == np.uint8
    finally:
        env.close()


def test_rgb_array_rendering_returns_one_image_per_lane() -> None:
    env = make_env(
        require_rom(),
        num_envs=2,
        frame_stack=4,
        render_mode="rgb_array",
    )
    try:
        obs, _infos = env.reset()

        assert obs.shape == (2, 4, 84, 84)
        images = env.get_images()
        assert len(images) == 2
        for frame in images:
            assert frame is not None
            assert frame.shape == (224, 240, 3)
            assert frame.dtype == np.uint8

        rendered = env.render()
        assert rendered is not None
        assert rendered.shape == (448, 240, 3)
        assert rendered.dtype == np.uint8
    finally:
        env.close()


def test_done_lane_includes_final_obs_and_final_info() -> None:
    env = make_env(
        require_rom(),
        num_envs=1,
        done_on={"x_progress": ("x_pos", "increase")},
    )
    actions = make_action_batch(env.num_envs, "right")
    try:
        env.reset()
        for _ in range(300):
            obs, _rewards, terminations, truncations, infos = env.step(actions)
            if bool(terminations[0] or truncations[0]):
                assert bool(terminations[0])
                assert not bool(truncations[0])
                final_obs = lane_value(infos, "final_obs", 0)
                assert isinstance(final_obs, np.ndarray)
                assert final_obs.shape == (1, 84, 84)
                assert final_obs.shape == obs.shape[1:]
                assert bool(nested_lane_value(infos, ("final_info", "terminated"), 0)) is True
                x_progress = final_done_on_info(infos, 0, "x_progress")
                assert nested_lane_value(infos, ("final_info", "x_pos"), 0) == x_progress["next"][0]
                assert nested_lane_value(infos, ("final_info", "score"), 0) == lane_value(
                    infos,
                    "score",
                    0,
                )
                assert "terminal_observation" not in infos
                assert "TimeLimit.truncated" not in infos
                break
        else:
            pytest.fail("x_pos did not increase enough to trigger done_on_info")
    finally:
        env.close()


def test_same_step_final_info_only_reports_terminal_lanes() -> None:
    env = make_env(
        require_rom(),
        num_envs=2,
        done_on={"x_progress": ("x_pos", "increase")},
    )
    actions = make_action_batch(env.num_envs, ["right", "noop"])

    try:
        env.reset()
        for _ in range(300):
            _obs, _rewards, terminations, truncations, infos = env.step(actions)
            if not bool(terminations[0] or truncations[0]):
                continue

            assert terminations.tolist() == [True, False]
            assert truncations.tolist() == [False, False]
            assert lane_has(infos, "final_obs", 0)
            assert not lane_has(infos, "final_obs", 1)
            assert lane_has(infos, "final_info", 0)
            assert not lane_has(infos, "final_info", 1)
            x_progress = final_done_on_info(infos, 0, "x_progress")
            assert nested_lane_value(infos, ("final_info", "x_pos"), 0) == x_progress["next"][0]
            assert lane_has(infos["final_info"], "score", 0)
            assert not lane_has(infos["final_info"], "score", 1)
            break
        else:
            pytest.fail("x_pos did not increase enough to trigger lane-local terminal info")
    finally:
        env.close()


def test_reset_info_preserves_multi_state_labels() -> None:
    env = make_env(
        require_rom(),
        state=["Level1-1", "Level1-2"],
        num_envs=2,
    )
    try:
        _obs, infos = env.reset()
        assert lane_value(infos, "state", 0) == "Level1-1"
        assert lane_value(infos, "start_state", 0) == "Level1-1"
        assert lane_value(infos, "state", 1) == "Level1-2"
        assert lane_value(infos, "start_state", 1) == "Level1-2"
    finally:
        env.close()


def test_weighted_state_sampling_survives_lane_local_autoreset() -> None:
    env = make_env(
        require_rom(),
        state={"Level1-1": 0.5, "Level1-2": 0.5},
        num_envs=4,
        done_on={"x_progress": ("x_pos", "increase")},
    )
    actions = make_action_batch(env.num_envs, ["right", "noop", "noop", "noop"])
    valid_states = {"Level1-1", "Level1-2"}

    try:
        _obs, reset_info = env.reset()
        before_states = env.active_states()
        assert set(before_states) <= valid_states
        assert all(lane_value(reset_info, "state", lane) in valid_states for lane in range(env.num_envs))

        for _ in range(300):
            _obs, _rewards, terminations, truncations, infos = env.step(actions)
            done_lanes = np.flatnonzero(np.logical_or(terminations, truncations)).tolist()
            if not done_lanes:
                continue

            assert done_lanes == [0]
            after_states = env.active_states()
            assert set(after_states) <= valid_states
            assert after_states[1:] == before_states[1:]

            assert lane_value(infos, "state", 0) == after_states[0]
            assert lane_value(infos, "start_state", 0) == after_states[0]
            assert lane_has(infos, "final_obs", 0)
            assert not lane_has(infos, "state", 1)
            break
        else:
            pytest.fail("x_pos did not increase enough to trigger lane-local autoreset")
    finally:
        env.close()


def test_set_state_policy_updates_sampling_on_next_reset() -> None:
    env = make_env(
        require_rom(),
        state={"Level1-1": 1.0, "Level1-2": 1.0},
        num_envs=4,
    )
    try:
        env.seed(20260706)
        env.reset()
        assert set(env.active_states()) <= {"Level1-1", "Level1-2"}

        assert not hasattr(env, "set_state")
        env.set_state_policy({"Level1-1": 0.0, "Level1-2": 1.0})
        assert env.state_sampling_weights() == {"Level1-1": 0.0, "Level1-2": 1.0}
        assert set(env.active_states()) <= {"Level1-1", "Level1-2"}

        env.reset()
        assert env.active_states() == ("Level1-2",) * env.num_envs
    finally:
        env.close()


def test_set_state_policy_string_and_list_match_constructor_state_forms() -> None:
    env = make_env(
        require_rom(),
        state={"Level1-1": 1.0, "Level1-2": 1.0, "Level1-3": 1.0},
        num_envs=4,
    )
    try:
        env.reset()

        env.set_state_policy("Level1-3")
        env.reset()
        assert env.initial_state_names == ("Level1-3",)
        np.testing.assert_array_equal(
            env.active_state_indices(),
            np.zeros(env.num_envs, dtype=np.int32),
        )
        assert env.active_states() == ("Level1-3",) * env.num_envs

        fixed_states = ["Level1-1", "Level1-2", "Level1-1", "Level1-2"]
        env.set_state_policy(fixed_states)
        env.reset()
        assert env.initial_state_names == tuple(fixed_states)
        assert env.active_states() == tuple(fixed_states)
    finally:
        env.close()


def test_set_state_policy_does_not_change_active_lanes_before_boundary() -> None:
    env = make_env(
        require_rom(),
        state=["Level1-1", "Level1-2"],
        num_envs=2,
    )
    try:
        env.reset()
        before_states = env.active_states()
        before_indices = env.active_state_indices().copy()

        env.set_state_policy("Level1-3")

        assert env.active_states() == before_states
        np.testing.assert_array_equal(env.active_state_indices(), before_indices)
    finally:
        env.close()


def test_set_state_policy_applies_to_lane_autoreset_only_after_done() -> None:
    env = make_env(
        require_rom(),
        state=["Level1-1", "Level1-1"],
        num_envs=2,
        done_on={"x_progress": ("x_pos", "increase")},
    )
    actions = make_action_batch(env.num_envs, ["right", "noop"])
    try:
        env.reset()
        assert env.active_states() == ("Level1-1", "Level1-1")

        env.set_state_policy({"Level1-2": 1.0})
        assert env.active_states() == ("Level1-1", "Level1-1")

        for _ in range(300):
            _obs, _rewards, terminations, truncations, infos = env.step(actions)
            if not bool(terminations[0] or truncations[0]):
                continue

            assert terminations.tolist() == [True, False]
            assert truncations.tolist() == [False, False]
            assert env.active_states() == ("Level1-2", "Level1-1")
            assert lane_value(infos, "state", 0) == "Level1-2"
            assert lane_value(infos, "start_state", 0) == "Level1-2"
            assert not lane_has(infos, "state", 1)
            break
        else:
            pytest.fail("x_pos did not increase enough to trigger lane-local autoreset")
    finally:
        env.close()


def test_terminal_info_filter_only_reports_done_lanes() -> None:
    env = make_env(
        require_rom(),
        num_envs=2,
        done_on={"x_progress": ("x_pos", "increase")},
        info_filter="terminal",
    )
    actions = make_action_batch(env.num_envs, ["right", "noop"])

    try:
        env.reset()
        for _ in range(300):
            _obs, _rewards, terminations, truncations, infos = env.step(actions)
            if not bool(terminations[0] or truncations[0]):
                continue

            assert terminations.tolist() == [True, False]
            assert lane_has(infos, "final_obs", 0)
            assert not lane_has(infos, "final_obs", 1)
            assert not lane_has(infos, "xscrollHi", 1)
            break
        else:
            pytest.fail("x_pos did not increase enough to trigger terminal-only info")
    finally:
        env.close()


def test_regular_step_infos_use_direct_vector_arrays() -> None:
    env = make_env(
        require_rom(),
        num_envs=2,
        info_filter={"keys": ["lives", "time"]},
    )
    actions = make_action_batch(env.num_envs, ["noop", "right"])

    def fail_info_dict(_index: int) -> dict[str, object]:
        raise AssertionError("regular step should not build per-lane info dicts")

    try:
        env.reset()
        env._info_dict = fail_info_dict  # type: ignore[method-assign]
        _obs, _rewards, terminations, truncations, infos = env.step(actions)

        assert not np.any(terminations)
        assert not np.any(truncations)
        assert sorted(key for key in infos if not key.startswith("_")) == ["lives", "time"]
        assert lane_has(infos, "lives", 0)
        assert lane_has(infos, "lives", 1)
        assert lane_has(infos, "time", 0)
        assert lane_has(infos, "time", 1)
        assert infos["lives"].shape == (2,)
        assert infos["time"].shape == (2,)
    finally:
        env.close()


@pytest.mark.parametrize(
    ("obs_layout", "expected_single_shape"),
    [
        ("chw", (1, 84, 84)),
        ("hwc", (84, 84, 1)),
    ],
)
def test_final_observation_matches_public_layout(obs_layout: str, expected_single_shape: tuple[int, ...]) -> None:
    env = make_env(
        require_rom(),
        num_envs=1,
        obs_layout=obs_layout,
        done_on={"x_progress": ("x_pos", "increase")},
    )
    actions = make_action_batch(env.num_envs, "right")
    try:
        obs, _infos = env.reset()
        assert obs.shape == (1, *expected_single_shape)
        assert env.single_observation_space.shape == expected_single_shape
        assert env.observation_space.shape == (1, *expected_single_shape)

        for _ in range(300):
            obs, _rewards, terminations, truncations, infos = env.step(actions)
            if not bool(terminations[0] or truncations[0]):
                continue
            final_obs = lane_value(infos, "final_obs", 0)
            assert isinstance(final_obs, np.ndarray)
            assert final_obs.shape == expected_single_shape
            assert final_obs.dtype == obs.dtype
            break
        else:
            pytest.fail("x_pos did not increase enough to trigger done_on_info")
    finally:
        env.close()


def test_safe_view_preserves_rollout_observations_across_next_step() -> None:
    env = make_env(
        require_rom(),
        num_envs=1,
        obs_layout="hwc",
        obs_copy="safe_view",
        frame_skip=1,
        frame_stack=1,
        info_filter="none",
    )
    try:
        first, _infos = env.reset()
        first_snapshot = first.copy()
        masks = make_action_batch(env.num_envs, "right")
        second, _rewards, _terminations, _truncations, infos = env.step(masks)
        assert first.shape == (1, 84, 84, 1)
        assert second.shape == (1, 84, 84, 1)
        np.testing.assert_array_equal(first, first_snapshot)
        assert infos == {}
    finally:
        env.close()


def test_named_done_on_life_loss_payload_is_in_final_info() -> None:
    env = make_env(
        require_rom(),
        num_envs=1,
        done_on=["life_loss"],
    )
    try:
        env.reset()
        masks = make_action_batch(env.num_envs, "noop")

        for step in range(1, 3000):
            _obs, _rewards, terminations, truncations, infos = env.step(masks)
            if not bool(terminations[0] or truncations[0]):
                continue
            assert step == 2456
            assert final_done_on_info(infos, 0, "life_loss") == {
                "trigger": "lives_decrease",
                "op": "decrease",
                "compare": "reset",
                "keys": ["lives"],
                "variables": ["lives"],
                "prev": [2],
                "next": [1],
            }
            break
        else:
            pytest.fail("life_loss done_on rule did not fire before game-over")
    finally:
        env.close()


def test_named_done_on_level_change_payload_is_in_final_info() -> None:
    env = make_env(
        require_rom(),
        state="Level1-1",
        num_envs=1,
        done_on=["level_change"],
    )
    try:
        env.reset()
        right = make_action_batch(env.num_envs, "right")
        for _step in range(1, 4500):
            _obs, _rewards, terminations, truncations, infos = env.step(right)
            if not bool(terminations[0] or truncations[0]):
                continue
            final_info = infos.get("final_info", {})
            if not isinstance(final_info, dict):
                continue
            done_on_infos = final_info.get("done_on_info", {})
            if not isinstance(done_on_infos, dict) or "level_change" not in done_on_infos:
                continue
            payload = final_done_on_info(infos, 0, "level_change")
            assert payload["trigger"] == "level_bytes_changed"
            assert payload["op"] == "change"
            assert payload["keys"] == ["levelHi", "levelLo"]
            assert payload["variables"] == ["levelHi", "levelLo"]
            break
        else:
            pytest.skip("level_change did not fire within bounded probe")
    finally:
        env.close()


def test_wrapper_option_validation_runs_before_rom_load() -> None:
    missing_rom = "/definitely/missing/SuperMarioBros.nes"
    base_kwargs = {"game": "SuperMarioBros-Nes-v0", "rom_path": missing_rom, "state": None}

    with pytest.raises(ValueError, match="obs_copy"):
        SuperMarioBrosNesTurboVecEnv(**base_kwargs, obs_copy=True)
    with pytest.raises(TypeError, match="copy_observations"):
        SuperMarioBrosNesTurboVecEnv(**base_kwargs, copy_observations=False)
    with pytest.raises(TypeError, match="unsafe_zero_copy"):
        SuperMarioBrosNesTurboVecEnv(**base_kwargs, unsafe_zero_copy=True)
    with pytest.raises(ValueError, match="info_filter mode"):
        SuperMarioBrosNesTurboVecEnv(**base_kwargs, info_filter="sometimes")
    with pytest.raises(ValueError, match="info_filter keys"):
        SuperMarioBrosNesTurboVecEnv(**base_kwargs, info_filter={"keys": "lives"})
    with pytest.raises(ValueError, match="reward_clip low"):
        SuperMarioBrosNesTurboVecEnv(**base_kwargs, reward_clip=(1.0, 0.0))
    with pytest.raises(ValueError, match="obs_layout"):
        SuperMarioBrosNesTurboVecEnv(**base_kwargs, obs_layout="nhwc")
    with pytest.raises(ValueError, match="obs_resize_algorithm"):
        SuperMarioBrosNesTurboVecEnv(**base_kwargs, obs_resize_algorithm="lanczos")
    with pytest.raises(ValueError, match="obs_crop_mode"):
        SuperMarioBrosNesTurboVecEnv(**base_kwargs, obs_crop_mode="hide")
    with pytest.raises(ValueError, match="obs_crop_fill"):
        SuperMarioBrosNesTurboVecEnv(**base_kwargs, obs_crop_fill=256)
    with pytest.raises(ValueError, match="unknown configured event"):
        SuperMarioBrosNesTurboVecEnv(**base_kwargs, done_on=["bad_event"])
    with pytest.raises(TypeError, match="unexpected_param"):
        SuperMarioBrosNesTurboVecEnv(**base_kwargs, unexpected_param=True)
    with pytest.raises(TypeError, match="done_on_info"):
        SuperMarioBrosNesTurboVecEnv(**base_kwargs, done_on_info={})
    with pytest.raises(TypeError, match="states"):
        SuperMarioBrosNesTurboVecEnv(**base_kwargs, states=["Level1-1"])
    with pytest.raises(TypeError, match="state_probs"):
        SuperMarioBrosNesTurboVecEnv(**base_kwargs, state_probs=[1.0])
