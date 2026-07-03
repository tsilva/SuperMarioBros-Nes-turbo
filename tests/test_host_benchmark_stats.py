from __future__ import annotations

from scripts.host_benchmark_stats import comparison_convergence, single_ref_convergence


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
