from __future__ import annotations

import numpy as np
import pytest

from supermariobrosnes_turbo.jerk import load_jerk_checkpoint, save_jerk_checkpoint
from train import EpisodeTrace, episode_boundary, exploit_probability


def test_level_change_does_not_end_episode() -> None:
    boundary = episode_boundary(
        previous_lives=2,
        current_lives=2,
        previous_level=(0, 0),
        current_level=(0, 1),
        episode_steps=1_000,
        max_episode_steps=4_500,
    )

    assert boundary.level_changed
    assert not boundary.life_loss
    assert not boundary.done


def test_life_loss_ends_episode() -> None:
    boundary = episode_boundary(
        previous_lives=2,
        current_lives=1,
        previous_level=(0, 0),
        current_level=(0, 0),
        episode_steps=100,
        max_episode_steps=4_500,
    )

    assert boundary.life_loss
    assert boundary.done


def test_trace_retains_first_prefix_that_reached_best_reward() -> None:
    trace = EpisodeTrace()
    for action, reward in ((1, 1.0), (3, 2.0), (6, -1.0), (1, 1.0)):
        trace.record(action, reward)

    assert trace.best_reward == pytest.approx(3.0)
    assert trace.best_sequence() == (1, 3)


def test_jerk_checkpoint_round_trip_and_noop_padding(tmp_path) -> None:
    path = save_jerk_checkpoint(
        tmp_path / "policy.json",
        ("right", "right_a"),
        timesteps=123,
        episodes=4,
        best_reward=99.0,
        metadata={"terminate_on_level_change": False},
    )

    policy = load_jerk_checkpoint(path)
    observations = np.zeros((1, 1, 1, 1), dtype=np.uint8)
    actions = [int(policy.predict(observations)[0][0]) for _ in range(3)]

    assert actions == [1, 3, 0]
    assert policy.timesteps == 123
    assert policy.metadata["terminate_on_level_change"] is False


def test_exploit_probability_increases_and_caps_at_one() -> None:
    assert exploit_probability(0, 1_000) == pytest.approx(0.25)
    assert exploit_probability(500, 1_000) == pytest.approx(0.75)
    assert exploit_probability(1_000, 1_000) == pytest.approx(1.0)
