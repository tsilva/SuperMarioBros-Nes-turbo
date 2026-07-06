#!/usr/bin/env python3
"""Run and cache the PyPI stable-retro-turbo local benchmark baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import statistics
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOCAL_ROOT = Path("/Users/tsilva/SuperMarioBros-Nes-turbo-benchmarks")
DEFAULT_STATES = ("Level1-1", "Level1-2", "Level1-3", "Level1-4")
PACKAGE = "stable-retro-turbo"
PYPI_JSON = f"https://pypi.org/pypi/{PACKAGE}/json"


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def pypi_latest() -> dict[str, Any]:
    with urllib.request.urlopen(PYPI_JSON, timeout=30) as response:
        data = json.load(response)
    version = data["info"]["version"]
    return {
        "version": version,
        "urls": data["releases"].get(version, []),
        "queried_url": PYPI_JSON,
    }


def workload(args: argparse.Namespace, version: str) -> dict[str, Any]:
    return {
        "package": PACKAGE,
        "version": version,
        "python": args.python,
        "rom_path": args.rom_path,
        "num_envs": args.num_envs,
        "num_threads": args.num_threads,
        "steps": args.steps,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "warmup_invocations": args.warmup_invocations,
        "measured_invocations": args.measured_invocations,
        "frame_skip": 4,
        "frame_stack": 4,
        "grayscale": True,
        "crop_top": 32,
        "crop_bottom": 0,
        "resize": [84, 84],
        "states": list(DEFAULT_STATES),
        "action": "noop",
        "obs_copy": "safe_view",
        "obs_resize_algorithm": "area",
    }


def stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def cache_paths(args: argparse.Namespace, version: str, workload_hash: str) -> tuple[Path, Path]:
    cache_dir = args.local_cache_root / version / workload_hash[:16]
    return cache_dir, cache_dir / "aggregate.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def median(values: list[float]) -> float:
    return statistics.median(values)


def mean(values: list[float]) -> float:
    return statistics.mean(values)


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summary(values: list[float]) -> dict[str, float]:
    avg = mean(values)
    sd = stdev(values)
    return {"mean": avg, "median": median(values), "stdev": sd, "min": min(values), "max": max(values), "cv": sd / avg if avg else 0.0}


def bootstrap_ci_median(values: list[float], n: int = 20000, seed: int = 12345) -> list[float]:
    rng = random.Random(seed)
    boot = []
    for _ in range(n):
        boot.append(median([values[rng.randrange(len(values))] for _ in values]))
    boot.sort()
    return [boot[int(0.025 * n)], boot[int(0.975 * n) - 1]]


def outliers(values: list[float]) -> dict[str, list[int]]:
    if len(values) < 4:
        return {"iqr_invocation_median_indices": [], "mad_invocation_median_indices": []}
    sorted_values = sorted(values)
    q1 = statistics.median(sorted_values[: len(sorted_values) // 2])
    q3 = statistics.median(sorted_values[(len(sorted_values) + 1) // 2 :])
    iqr = q3 - q1
    iqr_indices = [i for i, value in enumerate(values) if value < q1 - 1.5 * iqr or value > q3 + 1.5 * iqr]
    med = median(values)
    mad = median([abs(value - med) for value in values])
    mad_indices = [] if mad == 0 else [i for i, value in enumerate(values) if abs(value - med) / (1.4826 * mad) > 3.5]
    return {"iqr_invocation_median_indices": iqr_indices, "mad_invocation_median_indices": mad_indices}


def parse_load1(text: str | None) -> float | None:
    if not text or "load average:" not in text:
        return None
    return float(text.split("load average:", 1)[1].split(",", 1)[0].strip())


def make_local_run_name(version: str, workload_hash: str) -> str:
    safe_version = re.sub(r"[^A-Za-z0-9_.-]", "-", version)
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    return f"benchmark-pypi-stable-retro-turbo-{safe_version}-{stamp}-{workload_hash[:8]}"


def local_setup(args: argparse.Namespace, run_dir: Path, version: str) -> None:
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    (run_dir / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2("scripts/benchmark_stable_retro_turbo_pypi.py", run_dir / "scripts")
    setup = (
        f"cd {quote(str(run_dir))} && "
        f"uv venv --python {quote(args.python)} .venv && "
        f"uv pip install --python .venv/bin/python {quote(PACKAGE + '==' + version)}"
    )
    run(["bash", "-lc", setup])


def run_invocations(args: argparse.Namespace, run_dir: Path) -> None:
    base_cmd = (
        f"cd {quote(str(run_dir))} && "
        f"RAYON_NUM_THREADS={args.num_threads} .venv/bin/python scripts/benchmark_stable_retro_turbo_pypi.py "
        f"--rom-path {quote(args.rom_path)} "
        f"--num-envs {args.num_envs} --num-threads {args.num_threads} "
        f"--steps {args.steps} --repeats {args.repeats} --warmup {args.warmup} "
        f"--json --output-json "
    )
    run(["bash", "-lc", f"uptime > {quote(str(run_dir / 'raw' / 'load-before-measured.txt'))}"])
    for index in range(args.warmup_invocations):
        output = run_dir / "raw" / f"warmup-pypi-{index:02d}.json"
        run(["bash", "-lc", base_cmd + quote(str(output)) + f" > {quote(str(output) + '.stdout')}"])
    for index in range(args.measured_invocations):
        output = run_dir / "raw" / f"measured-pypi-{index:02d}.json"
        run(["bash", "-lc", base_cmd + quote(str(output)) + f" > {quote(str(output) + '.stdout')}"])
    run(["bash", "-lc", f"uptime > {quote(str(run_dir / 'raw' / 'load-after-measured.txt'))}"])


def copy_local_tree(run_dir: Path, cache_dir: Path) -> None:
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    (cache_dir / "raw").mkdir(parents=True, exist_ok=True)
    for path in sorted((run_dir / "raw").glob("*")):
        if path.suffix == ".json" and path.name.endswith(".stdout.json"):
            continue
        if path.suffix in {".json", ".txt"}:
            shutil.copy2(path, cache_dir / "raw" / path.name)


def aggregate(args: argparse.Namespace, cache_dir: Path, run_dir: Path, version_info: dict[str, Any], workload_payload: dict[str, Any]) -> dict[str, Any]:
    measured_files = [cache_dir / "raw" / f"measured-pypi-{index:02d}.json" for index in range(args.measured_invocations)]
    warmup_files = [cache_dir / "raw" / f"warmup-pypi-{index:02d}.json" for index in range(args.warmup_invocations)]
    measured = []
    all_samples = []
    for path in measured_files:
        result = load_json(path)
        samples = [run["env_steps_per_sec"] for run in result["runs"]]
        all_samples.extend(samples)
        measured.append({"file": str(path.relative_to(cache_dir)), "mean_env_steps_per_sec": mean(samples), "median_env_steps_per_sec": median(samples), "samples_env_steps_per_sec": samples})
    medians = [item["median_env_steps_per_sec"] for item in measured]
    ci = bootstrap_ci_median(medians)
    median_summary = summary(medians)
    all_sample_summary = summary(all_samples)
    diagnostics = outliers(medians)
    load_before = (cache_dir / "raw" / "load-before-measured.txt").read_text()
    load_after = (cache_dir / "raw" / "load-after-measured.txt").read_text()
    load_values = {
        "before": parse_load1(load_before),
        "after": parse_load1(load_after),
    }
    gates = {
        "invocation_median_cv_below_0_75_percent": median_summary["cv"] < 0.0075,
        "all_sample_cv_below_1_25_percent": all_sample_summary["cv"] < 0.0125,
        "bootstrap_ci_width_below_0_75_percent": (ci[1] - ci[0]) / median(medians) < 0.0075,
        "no_iqr_mad_outliers": not diagnostics["iqr_invocation_median_indices"] and not diagnostics["mad_invocation_median_indices"],
        "load_below_4": all(value is not None and value < 4 for value in load_values.values()),
    }
    aggregate_payload = {
        "mode": "pypi_stable_retro_turbo_fixed_local",
        "package": version_info,
        "workload": workload_payload,
        "workload_hash": stable_hash(workload_payload),
        "execution_target": "local_machine",
        "local_run_dir": str(run_dir),
        "measured_invocation_count": args.measured_invocations,
        "warmup_invocation_count": args.warmup_invocations,
        "measured_invocations": measured,
        "warmup_raw_files": [str(path.relative_to(cache_dir)) for path in warmup_files],
        "official_median_sps": median(medians),
        "mean_invocation_median_sps": mean(medians),
        "bootstrap_ci95_invocation_median_sps": ci,
        "run_median_summary": median_summary,
        "all_sample_summary": all_sample_summary,
        "outlier_diagnostics": diagnostics,
        "load_snapshots": {"texts": {"before": load_before, "after": load_after}, "load1_values": load_values},
        "validity_gates": gates,
        "validity_passed": all(gates.values()),
        "tier": "full_official",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (cache_dir / "aggregate.json").write_text(json.dumps(aggregate_payload, indent=2, sort_keys=True) + "\n")
    return aggregate_payload


def write_manifest_and_index(args: argparse.Namespace, cache_dir: Path, aggregate_payload: dict[str, Any]) -> None:
    files = []
    for path in sorted(cache_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            if path.suffix == ".json":
                load_json(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            files.append({"path": str(path.relative_to(cache_dir)), "bytes": path.stat().st_size, "sha256": digest})
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cache_dir": str(cache_dir),
        "copied_files": files,
        "retention": "local cache is durable; local pypi run dir is purged after copy",
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    index_path = args.local_cache_root / "index.jsonl"
    record = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "version": aggregate_payload["package"]["version"],
        "workload_hash": aggregate_payload["workload_hash"],
        "cache_dir": str(cache_dir),
        "official_median_sps": aggregate_payload["official_median_sps"],
        "mean_invocation_median_sps": aggregate_payload["mean_invocation_median_sps"],
        "validity_passed": aggregate_payload["validity_passed"],
    }
    existing = []
    if index_path.exists():
        for line in index_path.read_text().splitlines():
            if not line.strip():
                continue
            old = json.loads(line)
            if not (old.get("version") == record["version"] and old.get("workload_hash") == record["workload_hash"]):
                existing.append(json.dumps(old, sort_keys=True))
    with index_path.open("w") as handle:
        for line in existing:
            handle.write(line + "\n")
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def purge_local(run_dir: Path) -> None:
    try:
        run_dir.relative_to(LOCAL_ROOT / "runs")
    except ValueError as exc:
        raise SystemExit(f"Refusing to purge path outside benchmark runs: {run_dir}") from exc
    shutil.rmtree(run_dir, ignore_errors=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=None, help="PyPI version; defaults to latest from PyPI JSON.")
    parser.add_argument("--python", default="3.14")
    parser.add_argument("--rom-path", required=True, help="ROM path on the benchmark machine.")
    parser.add_argument("--local-cache-root", type=Path, default=Path("artifacts/benchmarks/local-results/pypi-stable-retro-turbo"))
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-threads", type=int, default=12)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--warmup-invocations", type=int, default=2)
    parser.add_argument("--measured-invocations", type=int, default=11)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-run-dir", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    version_info = pypi_latest()
    if args.version is not None:
        version_info["version"] = args.version
    workload_payload = workload(args, version_info["version"])
    workload_hash = stable_hash(workload_payload)
    cache_dir, aggregate_path = cache_paths(args, version_info["version"], workload_hash)
    if aggregate_path.exists() and not args.force:
        print(json.dumps({"cache_hit": True, "aggregate": str(aggregate_path), "workload_hash": workload_hash}, indent=2))
        return 0

    run_name = make_local_run_name(version_info["version"], workload_hash)
    run_dir = LOCAL_ROOT / "runs" / run_name
    try:
        local_setup(args, run_dir, version_info["version"])
        run_invocations(args, run_dir)
        copy_local_tree(run_dir, cache_dir)
        aggregate_payload = aggregate(args, cache_dir, run_dir, version_info, workload_payload)
        write_manifest_and_index(args, cache_dir, aggregate_payload)
    except BaseException:
        if not args.keep_run_dir:
            try:
                purge_local(run_dir)
            except Exception as cleanup_error:
                print(f"warning: failed to purge interrupted local run {run_dir}: {cleanup_error}", file=sys.stderr)
        raise
    if not args.keep_run_dir:
        purge_local(run_dir)
    print(json.dumps({"cache_hit": False, "aggregate": str(cache_dir / "aggregate.json"), "workload_hash": workload_hash, "validity_passed": aggregate_payload["validity_passed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
