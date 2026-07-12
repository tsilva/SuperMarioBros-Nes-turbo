from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from train import (
    COMPLETION_RATE_METRIC,
    SUCCESS_WINDOW,
    TRAIN_DONE_ON,
    WinnerScheduleAndStop,
)


class RecordingLogger:
    def __init__(self) -> None:
        self.values: dict[str, int | float] = {}
        self.dump_steps: list[int] = []

    def record(self, key: str, value: int | float) -> None:
        self.values[key] = value

    def dump(self, step: int) -> None:
        self.dump_steps.append(step)


def test_training_terminates_on_life_loss_and_level_change() -> None:
    assert TRAIN_DONE_ON == ("life_loss", "level_change")


def test_completion_window_requires_exactly_100_successes() -> None:
    callback = WinnerScheduleAndStop(verbose=0)

    assert not callback.record_outcomes([True] * (SUCCESS_WINDOW - 1))
    assert callback.record_outcomes([True])

    payload = callback.metric_payload()
    assert payload[COMPLETION_RATE_METRIC] == pytest.approx(1.0)


def test_any_failure_in_the_rolling_window_prevents_stop() -> None:
    callback = WinnerScheduleAndStop(verbose=0)

    assert not callback.record_outcomes([True] * 99 + [False])
    assert callback.metric_payload()[COMPLETION_RATE_METRIC] == pytest.approx(0.99)
    assert not callback.record_outcomes([True] * 99)
    assert callback.record_outcomes([True])


def test_completion_metrics_are_logged_and_persisted(tmp_path) -> None:
    metrics_path = tmp_path / "level_completion.jsonl"
    logger = RecordingLogger()
    callback = WinnerScheduleAndStop(metrics_path=metrics_path, verbose=0)
    callback.model = SimpleNamespace(logger=logger)
    callback.num_timesteps = 12_345
    callback._on_training_start()
    callback.record_outcomes([True, False, True])

    callback._log_metrics()

    assert logger.values[COMPLETION_RATE_METRIC] == pytest.approx(2 / 3)
    row = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert row["timesteps"] == 12_345
    assert row["window_size"] == 3
    assert row["window_completions"] == 2
    assert row["window_rate"] == pytest.approx(2 / 3)


def test_callback_logs_final_rate_before_requesting_stop(tmp_path) -> None:
    metrics_path = tmp_path / "level_completion.jsonl"
    logger = RecordingLogger()
    callback = WinnerScheduleAndStop(metrics_path=metrics_path, verbose=0)
    callback.model = SimpleNamespace(logger=logger)
    callback.num_timesteps = 456_789
    callback._on_training_start()
    callback.record_outcomes([True] * 99)
    callback.locals = {
        "dones": [True],
        "infos": [{"level_complete": True, "termination_reason": "level_change"}],
    }

    assert not callback._on_step()
    assert logger.values[COMPLETION_RATE_METRIC] == pytest.approx(1.0)
    assert logger.dump_steps == [456_789]
    row = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert row["window_size"] == SUCCESS_WINDOW
    assert row["window_rate"] == pytest.approx(1.0)
    assert row["stopped"] is True


def test_rollout_prints_exact_metric_name_during_training(capsys) -> None:
    logger = RecordingLogger()
    callback = WinnerScheduleAndStop(verbose=0)
    callback.model = SimpleNamespace(logger=logger)
    callback.record_outcomes([True, False])

    callback._on_rollout_end()

    output = capsys.readouterr().out
    assert f"{COMPLETION_RATE_METRIC}=0.500000" in output
    assert "window=2/100" in output
