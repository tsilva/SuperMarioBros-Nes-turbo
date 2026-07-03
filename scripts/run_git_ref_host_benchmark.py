#!/usr/bin/env python3
"""Run fixed-host git-ref benchmarks with sequential convergence."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Literal

try:
    from host_benchmark_stats import (
        DEFAULT_COMPARISON_CHECKPOINTS,
        DEFAULT_SINGLE_CHECKPOINTS,
        comparison_convergence,
        median,
        single_ref_convergence,
        summary,
    )
except ModuleNotFoundError:
    from scripts.host_benchmark_stats import (
        DEFAULT_COMPARISON_CHECKPOINTS,
        DEFAULT_SINGLE_CHECKPOINTS,
        comparison_convergence,
        median,
        single_ref_convergence,
        summary,
    )


REMOTE_ROOT = PurePosixPath("/home/tsilva/SuperMarioBros-Nes-turbo-host-bench")
LOCAL_ROOT = Path("/Users/tsilva/SuperMarioBros-Nes-turbo-host-bench-local")
REMOTE_STATE_DIR = REMOTE_ROOT / "states" / "SuperMarioBros-Nes-v0"
LOCAL_STATE_DIR = LOCAL_ROOT / "states" / "SuperMarioBros-Nes-v0"
STATE_NAMES = ("Level1-1", "Level1-2", "Level1-3", "Level1-4")
DEFAULT_STATE_SOURCE = Path(
    "/Users/tsilva/repos/tsilva/stable-retro-turbo/"
    "stable_retro/data/stable/SuperMarioBros-Nes-v0"
)
ARCHIVE_DIR = Path("artifacts/benchmarks/host-archives")
LOCAL_RESULTS_ROOT = Path("artifacts/benchmarks/host-results")

Target = Literal["remote", "local"]
Mode = Literal["single", "compare"]


@dataclass(frozen=True)
class BenchmarkRef:
    role: str
    ref: str
    sha: str
    archive: Path

    @property
    def short_sha(self) -> str:
        return self.sha[:12]


@dataclass(frozen=True)
class BenchmarkPlan:
    mode: Mode
    target: Target
    run_name: str
    run_dir: str
    refs: list[BenchmarkRef]
    rom_path: str
    state_dir: str
    checkpoints: tuple[int, ...]
    warmups: int
    measured_cap: int


def run(
    cmd: list[str],
    *,
    check: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=check,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def run_stream(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def quote(value: str | os.PathLike[str]) -> str:
    return shlex.quote(str(value))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_ref(ref: str) -> str:
    return run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"]).stdout.strip()


def default_main_ref() -> str:
    return run(["git", "rev-parse", "--verify", "main^{commit}"]).stdout.strip()


def archive_ref(role: str, sha: str, archive_dir: Path = ARCHIVE_DIR) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    path = archive_dir / f"{role}-{sha[:12]}.tar.gz"
    completed = subprocess.run(
        ["git", "archive", "--format=tar", sha],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            gz.write(completed.stdout)
    return path


def decide_mode(refs: list[str], *, single: bool) -> tuple[Mode, list[tuple[str, str]]]:
    if single:
        if len(refs) != 1:
            raise SystemExit("--single requires exactly one ref")
        return "single", [("ref", refs[0])]
    if len(refs) == 1:
        return "compare", [("baseline", default_main_ref()), ("candidate", refs[0])]
    if len(refs) == 2:
        return "compare", [("baseline", refs[0]), ("candidate", refs[1])]
    raise SystemExit("pass one ref with --single, one candidate ref, or baseline candidate")


def make_run_name(mode: Mode, target: Target, refs: list[BenchmarkRef]) -> str:
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    prefix = "local" if target == "local" else "host"
    if mode == "single":
        return f"{prefix}-single-{stamp}-R{refs[0].short_sha}"
    baseline = next(ref for ref in refs if ref.role == "baseline")
    candidate = next(ref for ref in refs if ref.role == "candidate")
    return f"{prefix}-compare-{stamp}-B{baseline.short_sha}-C{candidate.short_sha}"


def build_plan(args: argparse.Namespace) -> BenchmarkPlan:
    mode, role_refs = decide_mode(args.refs, single=args.single)
    refs = []
    for role, ref in role_refs:
        sha = ref if re.fullmatch(r"[0-9a-fA-F]{40}", ref) else resolve_ref(ref)
        archive = ARCHIVE_DIR / f"{role}-{sha[:12]}.tar.gz"
        refs.append(BenchmarkRef(role=role, ref=ref, sha=sha, archive=archive))

    target: Target = "local" if args.local else "remote"
    root = Path(args.run_root) if target == "local" else PurePosixPath(args.run_root)
    run_name = make_run_name(mode, target, refs)
    run_dir = str(root / "runs" / run_name)
    state_dir = str(args.state_dir)
    checkpoints = DEFAULT_SINGLE_CHECKPOINTS if mode == "single" else DEFAULT_COMPARISON_CHECKPOINTS
    warmups = args.warmups if args.warmups is not None else 2
    return BenchmarkPlan(
        mode=mode,
        target=target,
        run_name=run_name,
        run_dir=run_dir,
        refs=refs,
        rom_path=args.rom_path,
        state_dir=state_dir,
        checkpoints=checkpoints,
        warmups=warmups,
        measured_cap=checkpoints[-1],
    )


def ssh_base(args: argparse.Namespace) -> list[str]:
    cmd = ["ssh"]
    if args.host_key_alias:
        cmd += ["-o", f"HostKeyAlias={args.host_key_alias}"]
    cmd.append(args.ssh_target)
    return cmd


def rsync_rsh(args: argparse.Namespace) -> str:
    parts = ["ssh"]
    if args.host_key_alias:
        parts += ["-o", f"HostKeyAlias={args.host_key_alias}"]
    return " ".join(parts)


def target_run(args: argparse.Namespace, plan: BenchmarkPlan, shell: str) -> str:
    if plan.target == "remote":
        return run(ssh_base(args) + [shell]).stdout
    return run(["bash", "-lc", shell]).stdout


def target_run_stream(args: argparse.Namespace, plan: BenchmarkPlan, shell: str) -> None:
    if plan.target == "remote":
        run_stream(ssh_base(args) + [shell])
    else:
        run_stream(["bash", "-lc", shell])


def target_write(args: argparse.Namespace, plan: BenchmarkPlan, path: str, text: str) -> None:
    if plan.target == "remote":
        run(ssh_base(args) + [f"cat > {quote(path)}"], input_text=text)
    else:
        local = Path(path)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(text)


def target_read(args: argparse.Namespace, plan: BenchmarkPlan, path: str) -> str:
    if plan.target == "remote":
        return run(ssh_base(args) + [f"cat {quote(path)}"]).stdout
    return Path(path).read_text()


def target_exists(args: argparse.Namespace, plan: BenchmarkPlan, path: str) -> bool:
    cmd = f"test -e {quote(path)}"
    if plan.target == "remote":
        return run(ssh_base(args) + [cmd], check=False).returncode == 0
    return run(["bash", "-lc", cmd], check=False).returncode == 0


def parse_load1(uptime_text: str) -> float | None:
    if "load average:" not in uptime_text:
        return None
    try:
        return float(uptime_text.split("load average:", 1)[1].split(",", 1)[0].strip())
    except ValueError:
        return None


def capture_load(args: argparse.Namespace, plan: BenchmarkPlan, label: str) -> tuple[float | None, str]:
    raw_path = f"{plan.run_dir}/raw/load-{label}.txt"
    if plan.target == "remote":
        shell = (
            "set -e; "
            f"{{ hostname; uptime; nproc; lscpu | sed -n '1,40p'; "
            "ps -eo pid,pcpu,pmem,comm,args --sort=-pcpu | head -20; }} "
            f"> {quote(raw_path)}"
        )
    else:
        shell = (
            "set -e; "
            f"{{ hostname; uptime; "
            "sysctl -n hw.ncpu 2>/dev/null || nproc; "
            "sysctl -n machdep.cpu.brand_string 2>/dev/null || lscpu | sed -n '1,40p'; "
            "ps -Ao pid,pcpu,pmem,comm,args | sort -k2 -nr | head -20; }} "
            f"> {quote(raw_path)}"
        )
    target_run(args, plan, shell)
    text = target_read(args, plan, raw_path)
    return parse_load1(text), text


def host_load_ok(load_values: list[float | None], max_load: float) -> bool:
    return all(value is not None and value <= max_load for value in load_values)


def ensure_states(args: argparse.Namespace, plan: BenchmarkPlan) -> None:
    target_run(args, plan, f"mkdir -p {quote(plan.state_dir)}")
    missing = [
        name
        for name in STATE_NAMES
        if not target_exists(args, plan, f"{plan.state_dir}/{name}.state")
    ]
    if not missing:
        return
    for name in missing:
        source = args.state_source / f"{name}.state"
        if not source.exists():
            raise SystemExit(f"missing state source {source}")
    if plan.target == "remote":
        cmd = [
            "rsync",
            "-az",
            "-e",
            rsync_rsh(args),
            *[str(args.state_source / f"{name}.state") for name in missing],
            f"{args.ssh_target}:{plan.state_dir}/",
        ]
        run_stream(cmd)
    else:
        for name in missing:
            shutil.copy2(args.state_source / f"{name}.state", Path(plan.state_dir) / f"{name}.state")


def create_archives(plan: BenchmarkPlan) -> list[BenchmarkRef]:
    archived = []
    for ref in plan.refs:
        path = archive_ref(ref.role, ref.sha)
        archived.append(BenchmarkRef(ref.role, ref.ref, ref.sha, path))
    return archived


def prepare_sources(args: argparse.Namespace, plan: BenchmarkPlan) -> None:
    target_run(args, plan, f"mkdir -p {quote(plan.run_dir + '/archives')} {quote(plan.run_dir + '/raw')}")
    if plan.target == "remote":
        run_stream(
            [
                "rsync",
                "-az",
                "-e",
                rsync_rsh(args),
                *[str(ref.archive) for ref in plan.refs],
                f"{args.ssh_target}:{plan.run_dir}/archives/",
            ]
        )
    else:
        archives = Path(plan.run_dir) / "archives"
        archives.mkdir(parents=True, exist_ok=True)
        for ref in plan.refs:
            shutil.copy2(ref.archive, archives / ref.archive.name)

    for ref in plan.refs:
        source_dir = f"{plan.run_dir}/sources/{ref.role}"
        archive_path = f"{plan.run_dir}/archives/{ref.archive.name}"
        shell = (
            f"rm -rf {quote(source_dir)} && mkdir -p {quote(source_dir)} && "
            f"tar -xzf {quote(archive_path)} -C {quote(source_dir)} && "
            f"cd {quote(source_dir)} && uv sync --frozen --no-dev"
        )
        target_run_stream(args, plan, shell)


def benchmark_command(
    plan: BenchmarkPlan,
    role: str,
    output_name: str,
    *,
    steps: int,
    repeats: int,
) -> str:
    source_dir = f"{plan.run_dir}/sources/{role}"
    output = f"{plan.run_dir}/raw/{output_name}.json"
    states = ",".join(STATE_NAMES)
    return (
        f"cd {quote(source_dir)} && "
        "RAYON_NUM_THREADS=12 .venv/bin/python scripts/benchmark_sps.py "
        f"--rom-path {quote(plan.rom_path)} "
        f"--state-dir {quote(plan.state_dir)} "
        "--num-envs 16 "
        f"--steps {steps} --repeats {repeats} "
        f"--states {quote(states)} --action-set simple --action noop "
        f"--json --output-json {quote(output)} "
        f"> {quote(output + '.stdout.json')}"
    )


def run_invocation(
    args: argparse.Namespace,
    plan: BenchmarkPlan,
    role: str,
    output_name: str,
    *,
    steps: int,
    repeats: int,
) -> None:
    target_run_stream(args, plan, benchmark_command(plan, role, output_name, steps=steps, repeats=repeats))


def load_raw(args: argparse.Namespace, plan: BenchmarkPlan, name: str) -> dict[str, Any]:
    return json.loads(target_read(args, plan, f"{plan.run_dir}/raw/{name}.json"))


def invocation_median(payload: dict[str, Any]) -> float:
    return median([float(run["env_steps_per_sec"]) for run in payload["runs"]])


def invocation_mean(payload: dict[str, Any]) -> float:
    samples = [float(run["env_steps_per_sec"]) for run in payload["runs"]]
    return sum(samples) / len(samples)


def all_samples(payloads: list[dict[str, Any]]) -> list[float]:
    samples: list[float] = []
    for payload in payloads:
        samples.extend(float(run["env_steps_per_sec"]) for run in payload["runs"])
    return samples


def aggregate_single(
    args: argparse.Namespace,
    plan: BenchmarkPlan,
    *,
    measured_count: int,
    load_values: list[float | None],
) -> dict[str, Any]:
    payloads = [load_raw(args, plan, f"measured-ref-{index:02d}") for index in range(measured_count)]
    medians = [invocation_median(payload) for payload in payloads]
    samples = all_samples(payloads)
    convergence = single_ref_convergence(
        medians,
        samples,
        host_load_ok=host_load_ok(load_values, args.max_load),
        checkpoints=plan.checkpoints,
    )
    measured = [
        {
            "file": f"raw/measured-ref-{index:02d}.json",
            "mean_env_steps_per_sec": invocation_mean(payload),
            "median_env_steps_per_sec": invocation_median(payload),
            "samples_env_steps_per_sec": [
                float(run["env_steps_per_sec"]) for run in payload["runs"]
            ],
        }
        for index, payload in enumerate(payloads)
    ]
    ref = plan.refs[0]
    return {
        **base_aggregate(args, plan, load_values),
        "mode": "single_ref_fixed_host",
        "refs": {"ref": ref.ref},
        "shas": {"ref": ref.sha},
        "measured_invocations": measured,
        "warmup_raw_files": [
            f"raw/warmup-ref-{index:02d}.json" for index in range(plan.warmups)
        ],
        **convergence,
    }


def aggregate_compare(
    args: argparse.Namespace,
    plan: BenchmarkPlan,
    *,
    measured_count: int,
    load_values: list[float | None],
) -> dict[str, Any]:
    pairs = []
    baseline_medians = []
    candidate_medians = []
    baseline_samples = []
    candidate_samples = []
    for index in range(measured_count):
        baseline_payload = load_raw(args, plan, f"measured-baseline-{index:02d}")
        candidate_payload = load_raw(args, plan, f"measured-candidate-{index:02d}")
        baseline_median = invocation_median(baseline_payload)
        candidate_median = invocation_median(candidate_payload)
        baseline_medians.append(baseline_median)
        candidate_medians.append(candidate_median)
        baseline_samples.extend(float(run["env_steps_per_sec"]) for run in baseline_payload["runs"])
        candidate_samples.extend(float(run["env_steps_per_sec"]) for run in candidate_payload["runs"])
        pairs.append(
            {
                "pair_index": index,
                "baseline_file": f"raw/measured-baseline-{index:02d}.json",
                "candidate_file": f"raw/measured-candidate-{index:02d}.json",
                "baseline_median_env_steps_per_sec": baseline_median,
                "candidate_median_env_steps_per_sec": candidate_median,
                "pair_ratio": candidate_median / baseline_median,
            }
        )
    pair_ratios = [pair["pair_ratio"] for pair in pairs]
    convergence = comparison_convergence(
        pair_ratios,
        host_load_ok=host_load_ok(load_values, args.max_load),
        checkpoints=plan.checkpoints,
    )
    baseline = next(ref for ref in plan.refs if ref.role == "baseline")
    candidate = next(ref for ref in plan.refs if ref.role == "candidate")
    return {
        **base_aggregate(args, plan, load_values),
        "mode": "paired_compare_fixed_host",
        "refs": {"baseline": baseline.ref, "candidate": candidate.ref},
        "shas": {"baseline": baseline.sha, "candidate": candidate.sha},
        "measured_pair_details": pairs,
        "warmup_raw_files": [
            f"raw/warmup-baseline-{index:02d}.json"
            for index in range(plan.warmups)
        ]
        + [f"raw/warmup-candidate-{index:02d}.json" for index in range(plan.warmups)],
        "baseline_run_median_summary": summary(baseline_medians),
        "candidate_run_median_summary": summary(candidate_medians),
        "baseline_all_sample_summary": summary(baseline_samples),
        "candidate_all_sample_summary": summary(candidate_samples),
        "paired_gain_percent": (convergence["median_pair_ratio"] - 1.0) * 100.0,
        **convergence,
    }


def archive_hashes(plan: BenchmarkPlan) -> dict[str, str]:
    return {ref.role: sha256_path(ref.archive) for ref in plan.refs if ref.archive.exists()}


def target_file_hashes(args: argparse.Namespace, plan: BenchmarkPlan, paths: list[str]) -> dict[str, str | None]:
    hashes: dict[str, str | None] = {}
    for path in paths:
        if not target_exists(args, plan, path):
            hashes[path] = None
            continue
        if plan.target == "remote":
            hashes[path] = target_run(
                args,
                plan,
                f"sha256sum {quote(path)} | awk '{{print $1}}'",
            ).strip()
        else:
            hashes[path] = sha256_path(Path(path))
    return hashes


def base_aggregate(
    args: argparse.Namespace,
    plan: BenchmarkPlan,
    load_values: list[float | None],
) -> dict[str, Any]:
    state_paths = [f"{plan.state_dir}/{name}.state" for name in STATE_NAMES]
    file_hashes = target_file_hashes(args, plan, [plan.rom_path, *state_paths])
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execution_target": "local" if plan.target == "local" else "remote_beast_3_local",
        "ssh_route": (
            None
            if plan.target == "local"
            else {"host": args.ssh_target, "host_key_alias": args.host_key_alias}
        ),
        "target_run_dir": plan.run_dir,
        "run_name": plan.run_name,
        "local_git_status_short": run(["git", "status", "--short"]).stdout.splitlines(),
        "dirty_local_files_excluded": True,
        "workload": {
            "rayon_num_threads": 12,
            "num_envs": 16,
            "steps": args.steps,
            "repeats": args.repeats,
            "frame_skip": 4,
            "frame_stack": 4,
            "grayscale": True,
            "crop_top": 32,
            "resize": [84, 84],
            "states": list(STATE_NAMES),
            "action_set": "simple",
            "action": "noop",
        },
        "sequential_policy": {
            "warmups": plan.warmups,
            "checkpoints": list(plan.checkpoints),
            "max_measured_samples": plan.measured_cap,
        },
        "source_archive_sha256": archive_hashes(plan),
        "rom_sha256": file_hashes.get(plan.rom_path),
        "state_sha256": {name: file_hashes.get(f"{plan.state_dir}/{name}.state") for name in STATE_NAMES},
        "load_1min_values": load_values,
        "command": {
            "benchmark": "RAYON_NUM_THREADS=12 .venv/bin/python scripts/benchmark_sps.py ...",
            "stats_helper": "scripts/host_benchmark_stats.py",
        },
    }


def write_aggregate(args: argparse.Namespace, plan: BenchmarkPlan, aggregate: dict[str, Any]) -> None:
    text = json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    target_write(args, plan, f"{plan.run_dir}/aggregate.json", text)


def run_single(args: argparse.Namespace, plan: BenchmarkPlan) -> dict[str, Any]:
    run_invocation(args, plan, "ref", "smoke-ref", steps=1000, repeats=1)
    for index in range(plan.warmups):
        run_invocation(args, plan, "ref", f"warmup-ref-{index:02d}", steps=args.steps, repeats=args.repeats)

    load_values = [capture_load(args, plan, "before-measured")[0]]
    aggregate: dict[str, Any] = {}
    measured_count = 0
    for checkpoint in plan.checkpoints:
        while measured_count < checkpoint:
            run_invocation(
                args,
                plan,
                "ref",
                f"measured-ref-{measured_count:02d}",
                steps=args.steps,
                repeats=args.repeats,
            )
            measured_count += 1
        load_values.append(capture_load(args, plan, f"after-checkpoint-{checkpoint}")[0])
        aggregate = aggregate_single(args, plan, measured_count=measured_count, load_values=load_values)
        write_aggregate(args, plan, aggregate)
        if aggregate["should_stop"]:
            break
    return aggregate


def pair_order(index: int) -> tuple[str, str]:
    return ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")


def run_compare(args: argparse.Namespace, plan: BenchmarkPlan) -> dict[str, Any]:
    run_invocation(args, plan, "baseline", "smoke-baseline", steps=1000, repeats=1)
    run_invocation(args, plan, "candidate", "smoke-candidate", steps=1000, repeats=1)
    for index in range(plan.warmups):
        for role in pair_order(index):
            run_invocation(
                args,
                plan,
                role,
                f"warmup-{role}-{index:02d}",
                steps=args.steps,
                repeats=args.repeats,
            )

    load_values = [capture_load(args, plan, "before-measured")[0]]
    aggregate: dict[str, Any] = {}
    measured_count = 0
    for checkpoint in plan.checkpoints:
        while measured_count < checkpoint:
            for role in pair_order(measured_count):
                run_invocation(
                    args,
                    plan,
                    role,
                    f"measured-{role}-{measured_count:02d}",
                    steps=args.steps,
                    repeats=args.repeats,
                )
            measured_count += 1
        load_values.append(capture_load(args, plan, f"after-checkpoint-{checkpoint}")[0])
        aggregate = aggregate_compare(args, plan, measured_count=measured_count, load_values=load_values)
        write_aggregate(args, plan, aggregate)
        if aggregate["should_stop"]:
            break
    return aggregate


def finalize_local(plan: BenchmarkPlan) -> Path:
    run_dir = Path(plan.run_dir)
    local_dir = LOCAL_RESULTS_ROOT / plan.run_name
    if local_dir.exists():
        shutil.rmtree(local_dir)
    raw_dir = local_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "aggregate.json", local_dir / "aggregate.json")
    for path in sorted((run_dir / "raw").glob("*")):
        if path.suffix == ".json" and path.name.endswith(".stdout.json"):
            continue
        if path.suffix in {".json", ".txt"}:
            shutil.copy2(path, raw_dir / path.name)
    copied_files = []
    for path in sorted(local_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            if path.suffix == ".json":
                json.loads(path.read_text())
            copied_files.append(
                {
                    "path": str(path.relative_to(local_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_path(path),
                }
            )
    if not any(item["path"].startswith("raw/") for item in copied_files):
        raise SystemExit("No raw benchmark files were copied; refusing local cleanup")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "local_run_dir": str(run_dir),
        "local_result_dir": str(local_dir),
        "copied_files": copied_files,
        "cleanup_policy": {
            "local_bulk_removed": ["sources/", "archives/", "raw/*.stdout.json"],
        },
    }
    (local_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    aggregate = json.loads((local_dir / "aggregate.json").read_text())
    append_index(plan.run_name, local_dir, aggregate)
    shutil.rmtree(run_dir / "sources", ignore_errors=True)
    shutil.rmtree(run_dir / "archives", ignore_errors=True)
    for path in (run_dir / "raw").glob("*.stdout.json"):
        path.unlink()
    return local_dir


def append_index(run_name: str, local_dir: Path, aggregate: dict[str, Any]) -> None:
    LOCAL_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    record = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "local_result_dir": str(local_dir),
        "mode": aggregate.get("mode"),
        "refs": aggregate.get("refs"),
        "shas": aggregate.get("shas"),
        "official_median_sps": aggregate.get("official_median_sps"),
        "mean_invocation_median_sps": aggregate.get("mean_invocation_median_sps"),
        "median_pair_ratio": aggregate.get("median_pair_ratio"),
        "mean_pair_ratio": aggregate.get("mean_pair_ratio"),
        "validity_passed": aggregate.get("validity_passed"),
        "decision": aggregate.get("decision"),
    }
    index_path = LOCAL_RESULTS_ROOT / "index.jsonl"
    existing = []
    if index_path.exists():
        for line in index_path.read_text().splitlines():
            if not line.strip():
                continue
            parsed = json.loads(line)
            if parsed.get("run_name") != run_name:
                existing.append(json.dumps(parsed, sort_keys=True))
    with index_path.open("w") as handle:
        for line in existing:
            handle.write(line + "\n")
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def finalize_remote(args: argparse.Namespace, plan: BenchmarkPlan) -> None:
    if args.no_finalize:
        return
    cmd = [
        sys.executable,
        "scripts/finalize_host_benchmark.py",
        "--ssh-target",
        args.ssh_target,
        "--remote-run-dir",
        plan.run_dir,
        "--purge-local-archives",
    ]
    if args.host_key_alias:
        cmd += ["--host-key-alias", args.host_key_alias]
    if not args.keep_bulk:
        cmd.append("--purge-remote-bulk")
    run_stream(cmd)


def execute(args: argparse.Namespace, plan: BenchmarkPlan) -> dict[str, Any]:
    if args.dry_run:
        payload = {
            "dry_run": True,
            "plan": plan_to_json(plan),
            "git_status": run(["git", "status", "--short"]).stdout.splitlines(),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return payload

    archived_refs = create_archives(plan)
    plan = BenchmarkPlan(
        mode=plan.mode,
        target=plan.target,
        run_name=plan.run_name,
        run_dir=plan.run_dir,
        refs=archived_refs,
        rom_path=plan.rom_path,
        state_dir=plan.state_dir,
        checkpoints=plan.checkpoints,
        warmups=plan.warmups,
        measured_cap=plan.measured_cap,
    )
    target_run(args, plan, f"mkdir -p {quote(plan.run_dir + '/raw')}")
    ensure_states(args, plan)
    initial_load, _ = capture_load(args, plan, "before-setup")
    if initial_load is not None and initial_load > args.max_load and not args.force_busy:
        raise SystemExit(
            f"host load {initial_load:.2f} exceeds max {args.max_load:.2f}; "
            "rerun with --force-busy to override"
        )
    prepare_sources(args, plan)
    aggregate = run_single(args, plan) if plan.mode == "single" else run_compare(args, plan)
    capture_load(args, plan, "after-measured")
    if plan.target == "remote":
        finalize_remote(args, plan)
    elif not args.no_finalize:
        finalize_local(plan)
    return aggregate


def plan_to_json(plan: BenchmarkPlan) -> dict[str, Any]:
    return {
        "mode": plan.mode,
        "target": plan.target,
        "run_name": plan.run_name,
        "run_dir": plan.run_dir,
        "refs": [
            {"role": ref.role, "ref": ref.ref, "sha": ref.sha, "archive": str(ref.archive)}
            for ref in plan.refs
        ],
        "rom_path": plan.rom_path,
        "state_dir": plan.state_dir,
        "warmups": plan.warmups,
        "checkpoints": list(plan.checkpoints),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("refs", nargs="+", help="single ref, candidate ref, or baseline candidate")
    parser.add_argument("--single", action="store_true", help="Benchmark one ref only.")
    parser.add_argument("--local", action="store_true", help="Run on this machine instead of beast-3-local.")
    parser.add_argument("--ssh-target", default="beast-3-local")
    parser.add_argument("--host-key-alias")
    parser.add_argument("--rom-path", required=True)
    parser.add_argument("--state-dir")
    parser.add_argument("--state-source", type=Path, default=DEFAULT_STATE_SOURCE)
    parser.add_argument("--run-root")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=None)
    parser.add_argument("--max-load", type=float)
    parser.add_argument("--force-busy", action="store_true")
    parser.add_argument("--keep-bulk", action="store_true")
    parser.add_argument("--no-finalize", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.run_root is None:
        args.run_root = str(LOCAL_ROOT if args.local else REMOTE_ROOT)
    if args.state_dir is None:
        args.state_dir = str(LOCAL_STATE_DIR if args.local else REMOTE_STATE_DIR)
    if args.max_load is None:
        args.max_load = max(os.cpu_count() or 1, 1) / 3 if args.local else 4.0
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    plan = build_plan(args)
    execute(args, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
