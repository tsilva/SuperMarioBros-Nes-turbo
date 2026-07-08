from __future__ import annotations

from pathlib import Path

import numpy as np

from supermariobrosnes_turbo import ACTION_MEANINGS, Actions, SuperMarioBrosNesTurboVecEnv
from rom_helpers import require_rom
GROUP_STATES = ("Level1-1", "Level1-2", "Level1-3", "Level1-4")
NES_BUTTONS = ("B", None, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A")
BUTTON_TO_INDEX = {name: index for index, name in enumerate(NES_BUTTONS) if name is not None}
ACTION_BUTTONS = {
    "noop": (),
    "right": ("RIGHT",),
}


def action_batch(name: str, num_envs: int) -> np.ndarray:
    masks = np.zeros((num_envs, len(NES_BUTTONS)), dtype=np.uint8)
    for button in ACTION_BUTTONS[name]:
        masks[:, BUTTON_TO_INDEX[button]] = 1
    return masks


def make_env(rom_path: Path, state: str | list[str], num_envs: int) -> SuperMarioBrosNesTurboVecEnv:
    return SuperMarioBrosNesTurboVecEnv(
        "SuperMarioBros-Nes-v0",
        state=state,
        rom_path=rom_path,
        num_envs=num_envs,
        use_restricted_actions=Actions.ALL,
        frame_skip=4,
        frame_stack=4,
        obs_grayscale=True,
        obs_crop=(32, 0, 0, 0),
        obs_resize=(84, 84),
        obs_resize_algorithm="area",
        obs_layout="chw",
    )


def assert_step_equal(
    actual: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    expected: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    for actual_array, expected_array in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_array, expected_array)


def step_arrays(
    env: SuperMarioBrosNesTurboVecEnv,
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs, rewards, terminated, truncated, _infos = env.step(actions)
    return obs, rewards, terminated, truncated


def reset_obs(env: SuperMarioBrosNesTurboVecEnv) -> np.ndarray:
    obs, _infos = env.reset()
    return obs


def test_repeated_state_groups_match_independent_lane_references() -> None:
    rom_path = require_rom()
    lane_states = [GROUP_STATES[index % len(GROUP_STATES)] for index in range(16)]
    grouped = make_env(rom_path, lane_states, num_envs=16)
    refs = [make_env(rom_path, state, num_envs=1) for state in GROUP_STATES]

    grouped_obs = reset_obs(grouped)
    ref_obs = [reset_obs(ref) for ref in refs]
    for lane, state in enumerate(lane_states):
        ref_index = GROUP_STATES.index(state)
        np.testing.assert_array_equal(grouped_obs[lane], ref_obs[ref_index][0])

    for action_name in ("noop", "noop", "right", "noop"):
        grouped_result = step_arrays(grouped, action_batch(action_name, 16))
        ref_results = [
            step_arrays(ref, action_batch(action_name, 1)) for ref in refs
        ]
        for lane, state in enumerate(lane_states):
            ref_index = GROUP_STATES.index(state)
            for actual_array, expected_array in zip(grouped_result, ref_results[ref_index], strict=True):
                np.testing.assert_array_equal(actual_array[lane], expected_array[0])


def test_repeated_state_groups_match_references_after_autoreset() -> None:
    rom_path = require_rom()
    lane_states = [GROUP_STATES[index % len(GROUP_STATES)] for index in range(16)]
    grouped = make_env(rom_path, lane_states, num_envs=16)
    refs = [make_env(rom_path, state, num_envs=1) for state in GROUP_STATES]

    reset_obs(grouped)
    for ref in refs:
        reset_obs(ref)

    saw_done = False
    for _ in range(900):
        grouped_result = step_arrays(grouped, action_batch("noop", 16))
        ref_results = [step_arrays(ref, action_batch("noop", 1)) for ref in refs]
        saw_done |= bool(np.any(grouped_result[2]) or np.any(grouped_result[3]))
        for lane, state in enumerate(lane_states):
            ref_index = GROUP_STATES.index(state)
            for actual_array, expected_array in zip(grouped_result, ref_results[ref_index], strict=True):
                np.testing.assert_array_equal(actual_array[lane], expected_array[0])
        if saw_done:
            return

    raise AssertionError("expected repeated-state noop rollout to trigger autoreset")


def test_grouped_lanes_materialize_before_divergent_actions() -> None:
    rom_path = require_rom()
    lane_states = [GROUP_STATES[index % len(GROUP_STATES)] for index in range(16)]
    grouped = make_env(rom_path, lane_states, num_envs=16)
    independent = make_env(rom_path, lane_states, num_envs=16)
    grouped.reset()
    independent.reset()

    actions = action_batch("noop", 16)
    actions[4] = action_batch("right", 1)[0]

    assert_step_equal(step_arrays(grouped, actions), step_arrays(independent, actions))
    assert_step_equal(step_arrays(grouped, actions), step_arrays(independent, actions))
