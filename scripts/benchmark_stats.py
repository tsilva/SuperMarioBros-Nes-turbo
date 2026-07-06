#!/usr/bin/env python3
"""Sequential convergence helpers for fixed local benchmark aggregates."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SINGLE_CHECKPOINTS = (5, 8, 11, 15, 21, 31)
DEFAULT_COMPARISON_CHECKPOINTS = (7, 11, 15, 21, 31)
DEFAULT_STABILITY_WINDOW = 3


def median(values: Iterable[float]) -> float:
    return statistics.median(list(values))


def mean(values: Iterable[float]) -> float:
    return statistics.fmean(list(values))


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("summary requires at least one value")
    avg = mean(values)
    sd = stdev(values)
    return {
        "mean": avg,
        "median": median(values),
        "stdev": sd,
        "min": min(values),
        "max": max(values),
        "cv": sd / avg if avg else 0.0,
    }


def bootstrap_ci_median(values: list[float], n: int = 20000, seed: int = 12345) -> list[float]:
    if not values:
        raise ValueError("bootstrap_ci_median requires at least one value")
    rng = random.Random(seed)
    boot = []
    for _ in range(n):
        boot.append(median([values[rng.randrange(len(values))] for _ in values]))
    boot.sort()
    return [boot[int(0.025 * n)], boot[int(0.975 * n) - 1]]


def outliers(values: list[float], *, key_prefix: str = "value") -> dict[str, list[int]]:
    if len(values) < 4:
        return {
            f"iqr_{key_prefix}_indices": [],
            f"mad_{key_prefix}_indices": [],
        }
    sorted_values = sorted(values)
    q1 = statistics.median(sorted_values[: len(sorted_values) // 2])
    q3 = statistics.median(sorted_values[(len(sorted_values) + 1) // 2 :])
    iqr = q3 - q1
    iqr_indices = [
        index
        for index, value in enumerate(values)
        if value < q1 - 1.5 * iqr or value > q3 + 1.5 * iqr
    ]
    med = median(values)
    mad = median([abs(value - med) for value in values])
    mad_indices = (
        []
        if mad == 0
        else [
            index
            for index, value in enumerate(values)
            if abs(value - med) / (1.4826 * mad) > 3.5
        ]
    )
    return {
        f"iqr_{key_prefix}_indices": iqr_indices,
        f"mad_{key_prefix}_indices": mad_indices,
    }


def no_outlier_flags(diagnostics: dict[str, list[int]]) -> bool:
    return all(not indices for indices in diagnostics.values())


def relative_width(low: float, high: float, center: float) -> float:
    return (high - low) / abs(center) if center else float("inf")


def checkpoint_trace(values: list[float], checkpoints: tuple[int, ...]) -> list[dict[str, float]]:
    trace = []
    for count in checkpoints:
        if len(values) >= count:
            trace.append({"count": count, "median": median(values[:count])})
    return trace


def stabilization_span(trace: list[dict[str, float]], window: int) -> float | None:
    if len(trace) < window:
        return None
    recent = [item["median"] for item in trace[-window:]]
    center = median(recent)
    return relative_width(min(recent), max(recent), center)


def next_checkpoint(sample_count: int, checkpoints: tuple[int, ...]) -> int | None:
    for checkpoint in checkpoints:
        if sample_count < checkpoint:
            return checkpoint
    return None


def single_ref_convergence(
    invocation_medians: list[float],
    all_samples: list[float],
    *,
    load_ok: bool = True,
    checkpoints: tuple[int, ...] = DEFAULT_SINGLE_CHECKPOINTS,
    stability_window: int = DEFAULT_STABILITY_WINDOW,
    stability_rel_span: float = 0.0025,
    ci_rel_width: float = 0.005,
    invocation_cv: float = 0.0075,
    all_sample_cv: float = 0.0125,
) -> dict[str, Any]:
    if not invocation_medians:
        raise ValueError("single_ref_convergence requires invocation medians")
    if not all_samples:
        raise ValueError("single_ref_convergence requires all raw samples")

    official_median = median(invocation_medians)
    med_summary = summary(invocation_medians)
    all_summary = summary(all_samples)
    ci = bootstrap_ci_median(invocation_medians)
    ci_width = relative_width(ci[0], ci[1], official_median)
    diagnostics = outliers(invocation_medians, key_prefix="invocation_median")
    trace = checkpoint_trace(invocation_medians, checkpoints)
    span = stabilization_span(trace, stability_window)
    stable = span is not None and span <= stability_rel_span

    gates = {
        "checkpoint_stability_window_met": span is not None,
        "checkpoint_median_span_below_0_25_percent": stable,
        "invocation_median_cv_below_0_75_percent": med_summary["cv"] < invocation_cv,
        "all_sample_cv_below_1_25_percent": all_summary["cv"] < all_sample_cv,
        "bootstrap_ci_width_below_0_5_percent": ci_width < ci_rel_width,
        "no_iqr_mad_outliers": no_outlier_flags(diagnostics),
        "load_ok": load_ok,
    }
    stop_gates = {
        "checkpoint_stability_window_met": gates["checkpoint_stability_window_met"],
        "checkpoint_median_span_below_0_25_percent": gates[
            "checkpoint_median_span_below_0_25_percent"
        ],
        "bootstrap_ci_width_below_0_5_percent": gates[
            "bootstrap_ci_width_below_0_5_percent"
        ],
    }
    validity_passed = all(gates.values())
    stable_enough_to_stop = all(stop_gates.values())
    max_samples_reached = len(invocation_medians) >= checkpoints[-1]
    should_stop = stable_enough_to_stop or max_samples_reached

    if stable_enough_to_stop:
        decision = "converged"
    elif max_samples_reached:
        decision = "max_samples_no_convergence"
    else:
        decision = "continue"

    return {
        "protocol": "sequential_single_ref_convergence",
        "measured_invocation_count": len(invocation_medians),
        "checkpoint_counts": list(checkpoints),
        "next_checkpoint": next_checkpoint(len(invocation_medians), checkpoints),
        "official_median_sps": official_median,
        "mean_invocation_median_sps": mean(invocation_medians),
        "bootstrap_ci95_invocation_median_sps": ci,
        "bootstrap_ci95_relative_width": ci_width,
        "run_median_summary": med_summary,
        "all_sample_summary": all_summary,
        "checkpoint_trace": trace,
        "checkpoint_stability_window": stability_window,
        "checkpoint_stability_relative_span": span,
        "outlier_diagnostics": diagnostics,
        "stop_gates": stop_gates,
        "stable_enough_to_stop": stable_enough_to_stop,
        "validity_gates": gates,
        "validity_passed": validity_passed,
        "should_stop": should_stop,
        "decision": decision,
    }


def comparison_convergence(
    pair_ratios: list[float],
    *,
    load_ok: bool = True,
    checkpoints: tuple[int, ...] = DEFAULT_COMPARISON_CHECKPOINTS,
    stability_window: int = DEFAULT_STABILITY_WINDOW,
    stability_rel_span: float = 0.0025,
    ci_rel_width: float = 0.005,
    pair_ratio_cv: float = 0.015,
    win_ratio: float = 1.03,
    no_meaningful_win_ratio: float = 1.01,
) -> dict[str, Any]:
    if not pair_ratios:
        raise ValueError("comparison_convergence requires pair ratios")

    official_ratio = median(pair_ratios)
    ratio_summary = summary(pair_ratios)
    ci = bootstrap_ci_median(pair_ratios)
    ci_width = relative_width(ci[0], ci[1], official_ratio)
    diagnostics = outliers(pair_ratios, key_prefix="pair_ratio")
    trace = checkpoint_trace(pair_ratios, checkpoints)
    span = stabilization_span(trace, stability_window)
    stable = span is not None and span <= stability_rel_span

    candidate_faster_pairs = sum(1 for ratio in pair_ratios if ratio > 1.0)
    needed_faster_pairs = max(
        int(len(pair_ratios) * 0.75 + 0.999999),
        8 if len(pair_ratios) >= 11 else 0,
    )
    stability_gates = {
        "checkpoint_stability_window_met": span is not None,
        "checkpoint_median_span_below_0_25_percent": stable,
        "pair_ratio_cv_below_1_5_percent": ratio_summary["cv"] < pair_ratio_cv,
        "bootstrap_ci_width_below_0_5_percent": ci_width < ci_rel_width,
        "no_iqr_mad_outliers": no_outlier_flags(diagnostics),
        "load_ok": load_ok,
    }
    positive_decision_gates = {
        "median_pair_ratio_at_least_1_03": official_ratio >= win_ratio,
        "bootstrap_ci95_lower_above_1_00": ci[0] > 1.0,
        "candidate_faster_pairs_sufficient": candidate_faster_pairs >= needed_faster_pairs,
    }
    no_meaningful_win_decision_gates = {
        "median_pair_ratio_below_1_01": official_ratio < no_meaningful_win_ratio,
        "bootstrap_ci95_upper_below_1_03": ci[1] < win_ratio,
    }
    positive_decision = all(positive_decision_gates.values())
    no_meaningful_win_decision = all(no_meaningful_win_decision_gates.values())
    validity_passed = all(stability_gates.values()) and (
        positive_decision or no_meaningful_win_decision
    )
    max_samples_reached = len(pair_ratios) >= checkpoints[-1]
    should_stop = validity_passed or max_samples_reached

    if all(stability_gates.values()) and positive_decision:
        decision = "converged_candidate_win"
    elif all(stability_gates.values()) and no_meaningful_win_decision:
        decision = "converged_no_meaningful_win"
    elif max_samples_reached:
        decision = "max_samples_no_convergence"
    else:
        decision = "continue"

    return {
        "protocol": "sequential_paired_ratio_convergence",
        "measured_pairs": len(pair_ratios),
        "checkpoint_counts": list(checkpoints),
        "next_checkpoint": next_checkpoint(len(pair_ratios), checkpoints),
        "median_pair_ratio": official_ratio,
        "mean_pair_ratio": mean(pair_ratios),
        "pair_ratio_bootstrap_ci95": ci,
        "pair_ratio_bootstrap_ci95_relative_width": ci_width,
        "pair_ratio_summary": ratio_summary,
        "candidate_faster_pairs": candidate_faster_pairs,
        "candidate_faster_pairs_required_for_win": needed_faster_pairs,
        "checkpoint_trace": trace,
        "checkpoint_stability_window": stability_window,
        "checkpoint_stability_relative_span": span,
        "outlier_diagnostics": diagnostics,
        "stability_gates": stability_gates,
        "positive_decision_gates": positive_decision_gates,
        "no_meaningful_win_decision_gates": no_meaningful_win_decision_gates,
        "validity_gates": {
            **stability_gates,
            "positive_or_no_meaningful_win_decision": positive_decision
            or no_meaningful_win_decision,
        },
        "validity_passed": validity_passed,
        "should_stop": should_stop,
        "decision": decision,
    }


def load_invocation_medians(paths: list[Path]) -> tuple[list[float], list[float]]:
    invocation_medians = []
    all_samples = []
    for path in paths:
        payload = json.loads(path.read_text())
        samples = [float(run["env_steps_per_sec"]) for run in payload["runs"]]
        if not samples:
            raise ValueError(f"{path} has no runs")
        invocation_medians.append(median(samples))
        all_samples.extend(samples)
    return invocation_medians, all_samples


def parse_float_list(raw: str) -> list[float]:
    values = [float(value) for value in raw.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    single = subparsers.add_parser("single", help="Evaluate single-ref convergence.")
    single.add_argument("raw_json", nargs="*", type=Path)
    single.add_argument("--load-ok", action=argparse.BooleanOptionalAction, default=True)
    single.add_argument("--output-json", type=Path)

    compare = subparsers.add_parser("compare", help="Evaluate paired-ratio convergence.")
    compare.add_argument("--pair-ratios", required=True, type=parse_float_list)
    compare.add_argument("--load-ok", action=argparse.BooleanOptionalAction, default=True)
    compare.add_argument("--output-json", type=Path)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "single":
        if not args.raw_json:
            raise SystemExit("single mode requires measured invocation JSON files")
        invocation_medians, all_samples = load_invocation_medians(args.raw_json)
        result = single_ref_convergence(
            invocation_medians,
            all_samples,
            load_ok=args.load_ok,
        )
    else:
        result = comparison_convergence(args.pair_ratios, load_ok=args.load_ok)

    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
