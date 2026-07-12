from __future__ import annotations

import io
import json
import logging
import zipfile

import numpy as np
import pytest
import torch

from supermariobrosnes_turbo.ppo import (
    PlainPPOPolicy,
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from train import COMPLETION_RATE_METRIC, CompletionTracker, SUCCESS_WINDOW, TRAIN_DONE_ON


def record_outcomes(tracker: CompletionTracker, outcomes: list[bool]) -> None:
    tracker.record(
        np.ones(len(outcomes), dtype=bool),
        [{"level_complete": value} for value in outcomes],
    )


def test_training_terminates_on_life_loss_and_level_change() -> None:
    assert TRAIN_DONE_ON == ("life_loss", "level_change")


def test_completion_window_requires_exactly_100_successes(tmp_path) -> None:
    tracker = CompletionTracker(tmp_path / "metrics.jsonl")
    record_outcomes(tracker, [True] * (SUCCESS_WINDOW - 1))
    assert not tracker.solved
    record_outcomes(tracker, [True])
    assert tracker.solved
    assert tracker.rate == pytest.approx(1.0)


def test_any_failure_in_the_rolling_window_prevents_stop(tmp_path) -> None:
    tracker = CompletionTracker(tmp_path / "metrics.jsonl")
    record_outcomes(tracker, [True] * 99 + [False])
    assert not tracker.solved
    assert tracker.rate == pytest.approx(0.99)
    record_outcomes(tracker, [True] * 99)
    assert not tracker.solved
    record_outcomes(tracker, [True])
    assert tracker.solved


def test_completion_rate_uses_standard_logger_and_jsonl(tmp_path, caplog) -> None:
    path = tmp_path / "level_completion.jsonl"
    tracker = CompletionTracker(path)
    record_outcomes(tracker, [True, False, True])
    with caplog.at_level(logging.INFO, logger="ppo_train"):
        tracker.log(12_345)
    assert "completion_rate=0.666667" in caplog.text
    assert "window_completions=2/100" in caplog.text
    assert "window=3/100" not in caplog.text
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row[COMPLETION_RATE_METRIC] == pytest.approx(2 / 3)
    assert row["window_size"] == 3
    assert row["window_completions"] == 2


def test_plain_policy_checkpoint_round_trip(tmp_path) -> None:
    torch.manual_seed(7)
    policy = PlainPPOPolicy(input_channels=4, action_count=7)
    observations = np.zeros((2, 4, 84, 84), dtype=np.uint8)
    expected, _ = policy.predict(observations, deterministic=True)
    path = save_policy_checkpoint(tmp_path / "policy.pt", policy, timesteps=123)
    loaded = load_policy_checkpoint(path)
    actual, _ = loaded.predict(observations, deterministic=True)
    np.testing.assert_array_equal(actual, expected)


def test_legacy_policy_zip_loads_without_stable_baselines3(tmp_path) -> None:
    policy = PlainPPOPolicy(input_channels=4, action_count=7)
    buffer = io.BytesIO()
    torch.save(policy.state_dict(), buffer)
    path = tmp_path / "legacy.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("policy.pth", buffer.getvalue())

    loaded = load_policy_checkpoint(path)

    assert loaded.input_channels == 4
    assert loaded.action_count == 7
