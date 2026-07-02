from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from supermariobrosnes_turbo import ACTION_MEANINGS, SuperMarioBrosVecEnv
from rom_helpers import require_rom


def make_env(
    rom_path: Path,
    *,
    num_envs: int = 4,
    frame_skip: int = 1,
    frame_stack: int = 1,
    seed: int = 1,
    noop_reset_max: int = 0,
    sticky_action_prob: float = 0.0,
) -> SuperMarioBrosVecEnv:
    return SuperMarioBrosVecEnv(
        rom_path=rom_path,
        num_envs=num_envs,
        frame_skip=frame_skip,
        frame_stack=frame_stack,
        grayscale=True,
        crop_top=32,
        crop_bottom=0,
        resize_width=84,
        resize_height=84,
        state="Level1-1",
        action_set="simple",
        seed=seed,
        terminate_on_flag=False,
        noop_reset_max=noop_reset_max,
        sticky_action_prob=sticky_action_prob,
    )


def assert_fast_step_equal(
    actual: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    expected: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    for actual_array, expected_array in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_array, expected_array)


def test_noop_and_sticky_validation_runs_before_rom_load() -> None:
    missing_rom = "/definitely/missing/SuperMarioBros.nes"
    with pytest.raises(ValueError, match="noop_reset_max"):
        SuperMarioBrosVecEnv(rom_path=missing_rom, noop_reset_max=-1)
    with pytest.raises(ValueError, match="sticky_action_prob"):
        SuperMarioBrosVecEnv(rom_path=missing_rom, sticky_action_prob=-0.01)
    with pytest.raises(ValueError, match="sticky_action_prob"):
        SuperMarioBrosVecEnv(rom_path=missing_rom, sticky_action_prob=1.01)


def test_noop_and_sticky_properties_are_exposed() -> None:
    rom_path = require_rom()
    env = make_env(rom_path, noop_reset_max=2, sticky_action_prob=0.25)

    assert env.noop_reset_max == 2
    assert env.sticky_action_prob == pytest.approx(0.25)


def test_noop_reset_max_one_matches_explicit_single_noop_step() -> None:
    rom_path = require_rom()
    noop = ACTION_MEANINGS.index("noop")
    baseline = make_env(rom_path, num_envs=1, seed=1, noop_reset_max=0)
    reset_noop = make_env(rom_path, num_envs=1, seed=1, noop_reset_max=1)

    baseline.reset()
    expected_obs = baseline.step_fast(np.asarray([noop], dtype=np.uint8))[0]
    actual_obs = reset_noop.reset()

    np.testing.assert_array_equal(actual_obs, expected_obs)


def test_noop_reset_is_seed_deterministic_across_envs() -> None:
    rom_path = require_rom()
    first = make_env(rom_path, num_envs=8, seed=17, noop_reset_max=2)
    second = make_env(rom_path, num_envs=8, seed=17, noop_reset_max=2)

    np.testing.assert_array_equal(first.reset(), second.reset())


def test_full_sticky_actions_reuse_initial_noop_action() -> None:
    rom_path = require_rom()
    noop = ACTION_MEANINGS.index("noop")
    right = ACTION_MEANINGS.index("right")
    sticky = make_env(rom_path, num_envs=4, seed=7, sticky_action_prob=1.0)
    explicit_noop = make_env(rom_path, num_envs=4, seed=7)

    np.testing.assert_array_equal(sticky.reset(), explicit_noop.reset())
    right_actions = np.full((4,), right, dtype=np.uint8)
    noop_actions = np.full((4,), noop, dtype=np.uint8)
    for _ in range(8):
        assert_fast_step_equal(sticky.step_fast(right_actions), explicit_noop.step_fast(noop_actions))


def test_stochastic_sticky_actions_are_seed_deterministic() -> None:
    rom_path = require_rom()
    first = make_env(rom_path, num_envs=8, seed=1234, sticky_action_prob=0.5)
    second = make_env(rom_path, num_envs=8, seed=1234, sticky_action_prob=0.5)
    action_trace = [
        np.asarray([(step + lane) % len(ACTION_MEANINGS) for lane in range(8)], dtype=np.uint8)
        for step in range(12)
    ]

    np.testing.assert_array_equal(first.reset(), second.reset())
    for actions in action_trace:
        assert_fast_step_equal(first.step_fast(actions), second.step_fast(actions))


def test_disabled_noop_and_sticky_match_default_env() -> None:
    rom_path = require_rom()
    default = make_env(rom_path, num_envs=8, seed=99)
    explicit_disabled = make_env(
        rom_path,
        num_envs=8,
        seed=99,
        noop_reset_max=0,
        sticky_action_prob=0.0,
    )
    action_trace = [
        np.asarray([(step * 3 + lane) % len(ACTION_MEANINGS) for lane in range(8)], dtype=np.uint8)
        for step in range(10)
    ]

    np.testing.assert_array_equal(default.reset(), explicit_disabled.reset())
    for actions in action_trace:
        assert_fast_step_equal(default.step_fast(actions), explicit_disabled.step_fast(actions))
