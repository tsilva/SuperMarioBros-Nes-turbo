#!/usr/bin/env python3
"""Small controller for the autoresearch-speed loop."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from dotenv_utils import require_arg_or_env_or_dotenv_path
    from run_git_ref_benchmark import (
        AUTORESEARCH_ROOT_ENV,
        RESULTS_TSV_COLUMNS,
    )
except ModuleNotFoundError:
    from scripts.dotenv_utils import require_arg_or_env_or_dotenv_path
    from scripts.run_git_ref_benchmark import (
        AUTORESEARCH_ROOT_ENV,
        RESULTS_TSV_COLUMNS,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPO_ROOT / "scripts" / "run_git_ref_benchmark.py"
BENCHMARK_SPS_SCRIPT = REPO_ROOT / "scripts" / "benchmark_sps.py"
BENCHMARK_STATES = "Level1-1,Level1-2,Level1-3,Level1-4"


def autoresearch_root(value: str | Path | None = None) -> Path:
    return require_arg_or_env_or_dotenv_path(
        AUTORESEARCH_ROOT_ENV,
        "autoresearch root",
        value,
        must_be_dir=True,
    )


def build_benchmark_command(kind: str, refs: list[str], extra_args: list[str]) -> list[str]:
    if len(refs) != 2:
        raise SystemExit(f"{kind} requires baseline_ref and candidate_ref")
    command = [sys.executable, str(BENCHMARK_SCRIPT), refs[0], refs[1]]
    if kind == "screen":
        command += [
            "--steps",
            "5000",
            "--repeats",
            "1",
            "--warmups",
            "0",
            "--max-measured-invocations",
            "3",
        ]
    elif kind == "accept":
        command += ["--steps", "50000", "--repeats", "3"]
    else:
        raise SystemExit(f"unknown benchmark kind: {kind}")
    return command + extra_args


def build_diagnose_command(root: Path, *, profile: bool) -> list[str]:
    output = root / "benchmarks" / (
        "local-profile-benchmark.json" if profile else "local-diagnosis.json"
    )
    command = [
        sys.executable,
        str(BENCHMARK_SPS_SCRIPT),
        "--num-envs",
        "16",
        "--steps",
        "2000" if profile else "1000",
        "--repeats",
        "1",
        "--warmup",
        "100" if profile else "20",
        "--frame-skip",
        "4",
        "--frame-stack",
        "4",
        "--crop-top",
        "32",
        "--crop-bottom",
        "0",
        "--resize-width",
        "84",
        "--resize-height",
        "84",
        "--states",
        BENCHMARK_STATES,
        "--action-set",
        "simple",
        "--action",
        "noop",
        "--no-start-game",
    ]
    if profile:
        command += ["--profile-output", str(root / "benchmarks" / "local-profile.json")]
    return command + ["--json", "--output-json", str(output)]


def run_command(
    command: list[str],
    *,
    dry_run: bool,
    env_defaults: dict[str, str] | None = None,
) -> int:
    env = os.environ.copy()
    prefix = []
    for key, value in (env_defaults or {}).items():
        if key not in env:
            env[key] = value
            prefix.append(f"{key}={shlex.quote(value)}")
    command_text = shlex.join(command)
    print(" ".join(prefix + [command_text]))
    if dry_run:
        return 0
    return subprocess.run(command, check=False, env=env).returncode


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON file is not an object: {path}")
    return payload


def gate_truths(payload: dict[str, Any], key: str) -> bool:
    gates = payload.get(key)
    return isinstance(gates, dict) and all(value is True for value in gates.values())


def infer_status(aggregate: dict[str, Any]) -> str:
    tier = aggregate.get("benchmark_tier")
    decision = aggregate.get("decision")
    limit_stop_reason = aggregate.get("limit_stop_reason")
    ratio = aggregate.get("median_pair_ratio")
    if tier == "local_triage":
        if isinstance(ratio, int | float) and ratio >= 1.03:
            return "triage_promote"
        if isinstance(ratio, int | float) and ratio < 1.01:
            return "triage_discard"
        return "inconclusive"
    if tier != "local_acceptance":
        return "inconclusive"

    load_ok = aggregate.get("load_gate_passed") is True or aggregate.get(
        "load_gate_ignored_for_validity"
    ) is True
    if (
        decision == "converged_candidate_win"
        and aggregate.get("validity_passed") is True
        and load_ok
    ):
        return "keep"

    ci = aggregate.get("pair_ratio_bootstrap_ci95")
    ci_low_ok = isinstance(ci, list) and len(ci) >= 1 and float(ci[0]) >= 1.0
    faster = aggregate.get("candidate_faster_pairs")
    needed = aggregate.get("candidate_faster_pairs_required_for_win")
    faster_ok = isinstance(faster, int) and isinstance(needed, int) and faster >= needed
    if (
        limit_stop_reason is None
        and load_ok
        and gate_truths(aggregate, "stability_gates")
        and isinstance(ratio, int | float)
        and ratio > 1.0
        and ci_low_ok
        and faster_ok
    ):
        return "keep_small_gain"

    if decision == "converged_no_meaningful_win" or (
        isinstance(ratio, int | float) and ratio <= 1.0
    ):
        return "discard"
    return "inconclusive"


def json_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def row_from_event(event: dict[str, Any]) -> list[str]:
    return [json_cell(event.get(column)) for column in RESULTS_TSV_COLUMNS]


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def append_tsv(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a") as handle:
        if write_header:
            handle.write("\t".join(RESULTS_TSV_COLUMNS) + "\n")
        handle.write("\t".join(row_from_event(event)) + "\n")


def event_from_aggregate(
    aggregate_path: Path,
    *,
    status: str | None,
    description: str,
    artifact: str | None,
) -> dict[str, Any]:
    aggregate = load_json(aggregate_path)
    refs = aggregate.get("refs") if isinstance(aggregate.get("refs"), dict) else {}
    shas = aggregate.get("shas") if isinstance(aggregate.get("shas"), dict) else {}
    inferred_status = status or infer_status(aggregate)
    event = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "event": "benchmark_result",
        "epoch": aggregate.get("run_name"),
        "commit": shas.get("candidate") or shas.get("ref"),
        "baseline_commit": shas.get("baseline"),
        "ref": refs.get("candidate") or refs.get("ref"),
        "baseline_ref": refs.get("baseline"),
        "status": inferred_status,
        "description": description,
        "artifact": artifact or str(aggregate_path),
        "aggregate_path": str(aggregate_path),
    }
    for column in RESULTS_TSV_COLUMNS:
        if column not in event and column in aggregate:
            event[column] = aggregate[column]
    limits = aggregate.get("benchmark_limits")
    if "benchmark_limits" not in event and limits is not None:
        event["benchmark_limits"] = limits
    return event


def record(args: argparse.Namespace) -> int:
    root = autoresearch_root(args.autoresearch_root)
    event = event_from_aggregate(
        args.aggregate,
        status=args.status,
        description=args.description or "",
        artifact=args.artifact,
    )
    append_jsonl(root / "events.jsonl", event)
    append_tsv(root / "results.tsv", event)
    print(
        json.dumps(
            {
                "status": event["status"],
                "event": str(root / "events.jsonl"),
                "results": str(root / "results.tsv"),
            },
            indent=2,
        )
    )
    return 0


def checks(args: argparse.Namespace) -> int:
    commands = [
        ["cargo", "fmt", "--check"],
        ["cargo", "check", "--release"],
        [sys.executable, "-m", "maturin", "develop", "--release"],
        ["make", "test"],
    ]
    for command in commands:
        print(shlex.join(command))
        if not args.dry_run:
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                return completed.returncode
    return 0


def benchmark_extra_args(args: argparse.Namespace) -> list[str]:
    extra_args = list(args.extra_args)
    if "--dry-run" in extra_args:
        args.dry_run = True
        extra_args = [arg for arg in extra_args if arg != "--dry-run"]
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    return extra_args


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    diagnose_parser = subparsers.add_parser("diagnose")
    diagnose_parser.add_argument("--autoresearch-root")
    diagnose_parser.add_argument("--profile", action="store_true")
    diagnose_parser.add_argument("--dry-run", action="store_true")

    for name in ("screen", "accept"):
        benchmark_parser = subparsers.add_parser(name)
        benchmark_parser.add_argument("baseline_ref")
        benchmark_parser.add_argument("candidate_ref")
        benchmark_parser.add_argument("--dry-run", action="store_true")
        benchmark_parser.add_argument("extra_args", nargs=argparse.REMAINDER)

    checks_parser = subparsers.add_parser("checks")
    checks_parser.add_argument("--dry-run", action="store_true")

    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("aggregate", type=Path)
    record_parser.add_argument("--autoresearch-root")
    record_parser.add_argument("--status")
    record_parser.add_argument("--description")
    record_parser.add_argument("--artifact")

    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.command == "diagnose":
        root = autoresearch_root(args.autoresearch_root)
        return run_command(
            build_diagnose_command(root, profile=args.profile),
            dry_run=args.dry_run,
            env_defaults={"RAYON_NUM_THREADS": "12"},
        )
    if args.command in {"screen", "accept"}:
        command = build_benchmark_command(
            args.command,
            [args.baseline_ref, args.candidate_ref],
            benchmark_extra_args(args),
        )
        return run_command(command, dry_run=args.dry_run)
    if args.command == "checks":
        return checks(args)
    if args.command == "record":
        return record(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
