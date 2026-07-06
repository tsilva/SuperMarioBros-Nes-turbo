from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts.benchmark_stats import (
    comparison_convergence,
    load_invocation_medians,
    parse_float_list,
    single_ref_convergence,
)


def samples_for(medians: list[float]) -> list[float]:
    samples: list[float] = []
    for value in medians:
        samples.extend([value - 1.0, value, value + 1.0])
    return samples


def test_single_ref_convergence_stops_after_stable_checkpoints() -> None:
    medians = [
        1000.0,
        1002.0,
        1001.0,
        1003.0,
        1001.0,
        1002.0,
        1000.0,
        1001.0,
        1002.0,
        1001.0,
        1000.0,
    ]

    result = single_ref_convergence(medians, samples_for(medians))

    assert result["decision"] == "converged"
    assert result["should_stop"] is True
    assert result["validity_passed"] is True
    assert result["next_checkpoint"] == 15
    assert result["checkpoint_trace"][-1]["count"] == 11


def test_single_ref_convergence_continues_while_checkpoint_median_drifts() -> None:
    medians = [
        1000.0,
        1001.0,
        1002.0,
        1003.0,
        1004.0,
        1010.0,
        1011.0,
        1012.0,
        1018.0,
        1019.0,
        1020.0,
    ]

    result = single_ref_convergence(medians, samples_for(medians))

    assert result["decision"] == "continue"
    assert result["should_stop"] is False
    assert result["validity_passed"] is False
    assert result["next_checkpoint"] == 15
    assert result["validity_gates"]["checkpoint_median_span_below_0_25_percent"] is False


def test_single_ref_convergence_stops_when_median_stable_despite_warning_gates() -> None:
    medians = [
        16160.52,
        16129.77,
        16079.20,
        16494.38,
        16162.69,
        16434.90,
        16132.28,
        16146.40,
        16147.74,
        16140.30,
        16402.33,
        16109.06,
        16135.47,
        16138.91,
        16140.30,
    ]

    result = single_ref_convergence(
        medians,
        samples_for(medians),
        load_ok=False,
    )

    assert result["decision"] == "converged"
    assert result["should_stop"] is True
    assert result["stable_enough_to_stop"] is True
    assert result["validity_passed"] is False
    assert result["stop_gates"] == {
        "checkpoint_stability_window_met": True,
        "checkpoint_median_span_below_0_25_percent": True,
        "bootstrap_ci_width_below_0_5_percent": True,
    }
    assert result["validity_gates"]["load_ok"] is False


def test_comparison_convergence_uses_paired_ratio_checkpoints() -> None:
    pair_ratios = [
        1.052,
        1.051,
        1.053,
        1.052,
        1.051,
        1.052,
        1.053,
        1.052,
        1.051,
        1.052,
        1.053,
        1.052,
        1.051,
        1.052,
        1.053,
    ]

    result = comparison_convergence(pair_ratios)

    assert result["decision"] == "converged_candidate_win"
    assert result["should_stop"] is True
    assert result["validity_passed"] is True
    assert result["candidate_faster_pairs"] == len(pair_ratios)
    assert result["checkpoint_trace"][-1]["count"] == 15


def test_comparison_convergence_stops_for_stable_no_meaningful_win() -> None:
    pair_ratios = [
        1.004,
        1.003,
        1.005,
        1.004,
        1.003,
        1.004,
        1.005,
        1.004,
        1.003,
        1.004,
        1.005,
        1.004,
        1.003,
        1.004,
        1.005,
    ]

    result = comparison_convergence(pair_ratios)

    assert result["decision"] == "converged_no_meaningful_win"
    assert result["should_stop"] is True
    assert result["validity_passed"] is True
    assert result["no_meaningful_win_decision_gates"]["median_pair_ratio_below_1_01"] is True


def test_load_invocation_medians_rejects_invalid_raw_samples(tmp_path: Path) -> None:
    path = tmp_path / "raw.json"

    path.write_text("[]\n")
    with pytest.raises(ValueError, match="is not a JSON object"):
        load_invocation_medians([path])

    path.write_text(json.dumps({"runs": []}) + "\n")
    with pytest.raises(ValueError, match="has no runs"):
        load_invocation_medians([path])

    path.write_text(json.dumps({"runs": [{"env_steps_per_sec": 0.0}]}) + "\n")
    with pytest.raises(ValueError, match="non-positive env_steps_per_sec"):
        load_invocation_medians([path])

    path.write_text(json.dumps({"runs": [{"env_steps_per_sec": float("nan")}]}) + "\n")
    with pytest.raises(ValueError, match="non-finite env_steps_per_sec"):
        load_invocation_medians([path])


def test_parse_float_list_rejects_invalid_pair_ratios() -> None:
    assert parse_float_list("1.01, 1.02") == [1.01, 1.02]
    with pytest.raises(argparse.ArgumentTypeError, match="values must be positive"):
        parse_float_list("1.01, 0")
    with pytest.raises(argparse.ArgumentTypeError, match="values must be finite"):
        parse_float_list("1.01, nan")
