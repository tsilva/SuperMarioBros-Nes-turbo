from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import shlex
import statistics
import subprocess
import tarfile
import time
from typing import Any

import modal


REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    UV_PROJECT_DIR = str(REPO_ROOT.relative_to(Path.cwd().resolve())) or "."
except ValueError:
    UV_PROJECT_DIR = str(REPO_ROOT)

REMOTE_ROM = "/tmp/SuperMarioBros-Nes-v0.nes"
REMOTE_STATE_DIR = "/tmp/SuperMarioBros-Nes-turbo-states"
REMOTE_COMPARE_ROOT = "/tmp/SuperMarioBros-Nes-turbo-compare"
DEFAULT_ROM = Path("~/Desktop/roms/NES/mapper-000-NROM/SuperMarioBros-Nes-v0.nes")
DEFAULT_STATES = ("Level1-1", "Level1-2", "Level1-3", "Level1-4")
DEFAULT_STATS_CACHE = Path("artifacts/benchmarks/modal-compare-stats-cache.json")
SKILL_PATH = REPO_ROOT / ".codex" / "skills" / "modal-benchmark" / "SKILL.md"
CACHE_SCHEMA_VERSION = 1
DEFAULT_REPLICAS = 7
DEFAULT_PAIRS_PER_REPLICA = 7
DEFAULT_STEPS = 20_000
DEFAULT_WARMUP_PAIRS_PER_REPLICA = 1
BOOTSTRAP_ITERATIONS = 20_000
BOOTSTRAP_SEED = 1729
ACCEPTANCE_MEDIAN_THRESHOLD = 1.10
ACCEPTANCE_CI_LOWER_THRESHOLD = 1.05
ACCEPTANCE_MIN_FASTER_REPLICAS = 5
CPU_REQUEST = 16.0
MEMORY_MB = 8192
MODAL_CPU_USD_PER_CORE_SEC = 0.0000131
MODAL_MEMORY_USD_PER_GIB_SEC = 0.00000222
MODAL_PRICING_SOURCE = "https://modal.com/pricing"
PYTHON_VERSION = "3.12"
PYTHON_BIN = "/.uv/.venv/bin/python"

app = modal.App("supermariobros-nes-turbo-cpu-paired-compare")

image = (
    modal.Image.from_registry("rust:1.88-bookworm", add_python=PYTHON_VERSION)
    .apt_install("make")
    .uv_sync(UV_PROJECT_DIR, extras=["dev"], frozen=True)
)


def run_git(*args: str, input_bytes: bytes | None = None) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout.decode().strip()


def run_git_bytes(*args: str) -> bytes:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout


def resolve_ref(ref: str) -> str:
    return run_git("rev-parse", "--verify", f"{ref}^{{commit}}")


def archive_ref(ref: str) -> dict[str, Any]:
    sha = resolve_ref(ref)
    tar_bytes = run_git_bytes("archive", "--format=tar", sha)
    archive_bytes = gzip.compress(tar_bytes, mtime=0)
    return {
        "ref": ref,
        "sha": sha,
        "archive_bytes": archive_bytes,
        "archive_sha256": sha256(archive_bytes),
        "archive_bytes_len": len(archive_bytes),
    }


def git_text(*args: str) -> str | None:
    try:
        return run_git(*args)
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_states(states: str) -> list[str]:
    parsed = [state.strip() for state in states.split(",")]
    if not parsed or not all(parsed):
        raise ValueError("--states must be a comma-separated list without empty entries")
    return parsed


def stable_retro_state_dir() -> Path | None:
    try:
        import stable_retro.data  # type: ignore[import-not-found]
    except ImportError:
        return None

    try:
        state_path = stable_retro.data.get_file_path(
            "SuperMarioBros-Nes-v0",
            "Level1-1.state",
            stable_retro.data.Integrations.ALL,
        )
    except Exception:
        return None
    if not state_path:
        return None
    return Path(state_path).parent


def sibling_stable_retro_state_dir() -> Path | None:
    candidate = (
        REPO_ROOT.parent
        / "stable-retro-turbo"
        / "stable_retro"
        / "data"
        / "stable"
        / "SuperMarioBros-Nes-v0"
    )
    return candidate if candidate.exists() else None


def candidate_state_dirs(state_dir: str) -> list[Path]:
    candidates: list[Path | None] = []
    if state_dir:
        candidates.append(Path(state_dir).expanduser())
    env_dir = os.environ.get("SUPERMARIOBROSNES_FASTENV_STATE_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.append(stable_retro_state_dir())
    candidates.append(sibling_stable_retro_state_dir())

    dirs: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.resolve()
        if resolved.exists() and resolved not in seen:
            dirs.append(resolved)
            seen.add(resolved)
    return dirs


def load_state_files(states: list[str], state_dir: str) -> dict[str, bytes]:
    dirs = candidate_state_dirs(state_dir)
    files: dict[str, bytes] = {}
    for state in states:
        filename = state if state.endswith(".state") else f"{state}.state"
        for directory in dirs:
            path = directory / filename
            if path.exists():
                files[state.removesuffix(".state")] = path.read_bytes()
                break
        else:
            checked = ", ".join(str(path) for path in dirs) or "<none>"
            raise FileNotFoundError(f"could not find {filename}; checked state dirs: {checked}")
    return files


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256(path.read_bytes())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def hash_json(data: Any) -> str:
    return sha256(canonical_json(data).encode())


def resolve_local_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved.resolve()


def tool_hashes() -> dict[str, dict[str, str]]:
    paths = {
        "benchmark_script": Path(__file__).resolve(),
        "modal_benchmark_skill": SKILL_PATH.resolve(),
    }
    return {
        name: {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }


def cache_context(
    config: dict[str, Any],
    state_names: list[str],
    rom_bytes: bytes,
    state_files: dict[str, bytes],
    hashes: dict[str, dict[str, str]],
) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "tool_hashes": hashes,
        "benchmark_config": {
            **config,
            "states": state_names,
            "repeats_per_sample": 1,
        },
        "inputs": {
            "rom_sha256": sha256(rom_bytes),
            "states": {
                name: sha256(state_bytes)
                for name, state_bytes in sorted(state_files.items())
            },
        },
    }


def cache_key(ref: dict[str, Any], context_sha256: str) -> str:
    return hash_json(
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "commit_sha": ref["sha"],
            "archive_sha256": ref["archive_sha256"],
            "benchmark_context_sha256": context_sha256,
        }
    )


def empty_stats_cache() -> dict[str, Any]:
    return {
        "kind": "modal_compare_commit_stats_cache",
        "schema_version": CACHE_SCHEMA_VERSION,
        "entries": {},
    }


def load_stats_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_stats_cache()
    data = json.loads(path.read_text())
    if (
        data.get("kind") != "modal_compare_commit_stats_cache"
        or data.get("schema_version") != CACHE_SCHEMA_VERSION
        or not isinstance(data.get("entries"), dict)
    ):
        raise ValueError(f"unsupported stats cache format: {path}")
    return data


def commit_samples(result: dict[str, Any], label: str) -> list[dict[str, Any]]:
    samples = []
    for replica in result["replicas"]:
        for pair in replica["pairs"]:
            if pair.get("discarded_as_warmup"):
                continue
            sample = pair[label]
            samples.append(
                {
                    "replica_index": replica["replica_index"],
                    "pair_index": pair["pair_index"],
                    "order": pair["order"],
                    "discarded_as_warmup": False,
                    "env_steps_per_sec": sample["summary"]["env_steps_per_sec"]["mean"],
                    "mixed_level_summary": sample["summary"]["env_steps_per_sec"],
                    "mixed_level_run_count": sample["summary_flat"]["run_count"],
                }
            )
    return samples


def commit_stats_entry(
    result: dict[str, Any],
    label: str,
    ref: dict[str, Any],
    context: dict[str, Any],
    context_sha256: str,
    key: str,
    output_path: Path | None,
) -> dict[str, Any]:
    samples = commit_samples(result, label)
    values = [sample["env_steps_per_sec"] for sample in samples]
    return {
        "cache_key": key,
        "recorded_at": utc_now(),
        "updated_at": utc_now(),
        "commit_sha": ref["sha"],
        "archive_sha256": ref["archive_sha256"],
        "archive_bytes_len": ref["archive_bytes_len"],
        "refs": [ref["ref"]],
        "benchmark_context_sha256": context_sha256,
        "benchmark_context": context,
        "stats": {
            "env_steps_per_sec": stat_summary(values),
            "sample_count": len(samples),
        },
        "samples": samples,
        "source_output_json": str(output_path) if output_path is not None else None,
    }


def update_stats_cache(
    path: Path,
    result: dict[str, Any],
    context: dict[str, Any],
    context_sha256: str,
    keys: dict[str, str],
    output_path: Path | None,
) -> dict[str, Any]:
    cache = load_stats_cache(path)
    entries = cache["entries"]
    written: dict[str, Any] = {}
    for label in ("baseline", "candidate"):
        ref = result["refs"][label]
        key = keys[label]
        previous = entries.get(key)
        entry = commit_stats_entry(result, label, ref, context, context_sha256, key, output_path)
        if previous:
            entry["recorded_at"] = previous.get("recorded_at", entry["recorded_at"])
            entry["refs"] = sorted(set(previous.get("refs", [])) | {ref["ref"]})
            entry["previous_source_output_json"] = previous.get("source_output_json")
        entries[key] = entry
        written[label] = {
            "cache_key": key,
            "hit_before_run": previous is not None,
            "sample_count": entry["stats"]["sample_count"],
        }

    cache["updated_at"] = utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
    return {
        "path": str(path),
        "benchmark_context_sha256": context_sha256,
        "entries": written,
    }


def benchmark_args(config: dict[str, Any]) -> list[str]:
    pairs = [
        ("--warmup", config["warmup"]),
        ("--frame-skip", config["frame_skip"]),
        ("--frame-stack", config["frame_stack"]),
        ("--resize-width", config["resize_width"]),
        ("--resize-height", config["resize_height"]),
        ("--crop-top", config["crop_top"]),
        ("--crop-bottom", config["crop_bottom"]),
        ("--action", config["action"]),
        ("--pre-start-steps", config["pre_start_steps"]),
        ("--start-steps", config["start_steps"]),
        ("--post-start-steps", config["post_start_steps"]),
    ]
    result = [item for pair in pairs for item in (pair[0], str(pair[1]))]
    if config["rgb"]:
        result.append("--rgb")
    if config["include_info"]:
        result.append("--include-info")
    if config["terminate_on_flag"]:
        result.append("--terminate-on-flag")
    if config["no_start_game"]:
        result.append("--no-start-game")
    return result


def benchmark_make_vars(
    config: dict[str, Any],
    remote_rom: Path,
    remote_state_dir: Path,
    states: list[str],
    python_bin: str,
) -> list[str]:
    benchmark_args_list = [
        "--rom-path",
        str(remote_rom),
        "--json",
        "--state-dir",
        str(remote_state_dir),
        *benchmark_args(config),
    ]
    if tuple(states) != DEFAULT_STATES:
        benchmark_args_list.extend(["--states", ",".join(states)])
    return [
        f"PYTHON={python_bin}",
        f"BENCHMARK_NUM_ENVS={config['num_envs']}",
        f"BENCHMARK_STEPS={config['steps']}",
        "BENCHMARK_REPEATS=1",
        f"BENCHMARK_ARGS={shlex.join(benchmark_args_list)}",
    ]


def stat_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize empty values")
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def measured_pairs(replica: dict[str, Any]) -> list[dict[str, Any]]:
    return [pair for pair in replica["pairs"] if not pair.get("discarded_as_warmup")]


def sample_value(pair: dict[str, Any], label: str) -> float:
    return pair[label]["summary"]["env_steps_per_sec"]["mean"]


def replica_value_groups(
    replicas: list[dict[str, Any]],
    value_key: str,
) -> list[list[float]]:
    groups: list[list[float]] = []
    for replica in replicas:
        values: list[float] = []
        for pair in measured_pairs(replica):
            if value_key == "candidate_over_baseline":
                values.append(pair["candidate_over_baseline"])
            else:
                values.append(sample_value(pair, value_key))
        groups.append(values)
    return groups


def flattened(groups: list[list[float]]) -> list[float]:
    return [value for group in groups for value in group]


def variance_decomposition(groups: list[list[float]]) -> dict[str, Any]:
    non_empty = [group for group in groups if group]
    all_values = flattened(non_empty)
    replica_means = [statistics.fmean(group) for group in non_empty]
    within_denom = sum(len(group) - 1 for group in non_empty)
    if within_denom > 0:
        within_variance = sum(
            (len(group) - 1) * statistics.variance(group)
            for group in non_empty
            if len(group) > 1
        ) / within_denom
        within_stdev = within_variance**0.5
    else:
        within_stdev = 0.0
    return {
        "pooled": stat_summary(all_values),
        "replica_means": replica_means,
        "within_replica_stdev": within_stdev,
        "between_replica_mean_stdev": statistics.stdev(replica_means)
        if len(replica_means) > 1
        else 0.0,
    }


def bootstrap_median_ci(
    values: list[float],
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot bootstrap empty values")
    rng = random.Random(seed)
    n = len(values)
    samples = sorted(
        statistics.median(values[rng.randrange(n)] for _ in range(n))
        for _ in range(iterations)
    )
    lower_index = int(0.025 * (iterations - 1))
    upper_index = int(0.975 * (iterations - 1))
    return {
        "confidence": 0.95,
        "iterations": iterations,
        "seed": seed,
        "lower": samples[lower_index],
        "upper": samples[upper_index],
    }


def acceptance_decision(replica_medians: list[float], ci: dict[str, Any]) -> dict[str, Any]:
    median_ratio = statistics.median(replica_medians)
    faster_count = sum(1 for value in replica_medians if value > 1.0)
    replica_count = len(replica_medians)
    min_faster = min(ACCEPTANCE_MIN_FASTER_REPLICAS, replica_count)
    accepted = (
        median_ratio > ACCEPTANCE_MEDIAN_THRESHOLD
        and ci["lower"] > ACCEPTANCE_CI_LOWER_THRESHOLD
        and faster_count >= min_faster
    )
    if accepted:
        verdict = "accept"
    elif median_ratio <= 1.0 or faster_count < (replica_count + 1) // 2:
        verdict = "reject"
    else:
        verdict = "inconclusive"
    return {
        "verdict": verdict,
        "accepted": accepted,
        "median_threshold": ACCEPTANCE_MEDIAN_THRESHOLD,
        "ci_lower_threshold": ACCEPTANCE_CI_LOWER_THRESHOLD,
        "min_faster_replicas": min_faster,
        "candidate_faster_replica_medians": faster_count,
        "replica_count": replica_count,
    }


def aggregate_mixed_levels(result: dict[str, Any]) -> dict[str, Any]:
    values = [run["env_steps_per_sec"] for run in result["runs"]]
    return {
        "env_steps_per_sec": stat_summary(values),
        "run_count": len(values),
    }


def local_metadata(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    rom_path: Path,
    rom_bytes: bytes,
    state_files: dict[str, bytes],
) -> dict[str, Any]:
    return {
        "repo_root": str(REPO_ROOT),
        "git": {
            "status_short": git_text("status", "--short"),
            "branch": git_text("branch", "--show-current"),
        },
        "baseline": {
            "ref": baseline["ref"],
            "sha": baseline["sha"],
            "archive_sha256": baseline["archive_sha256"],
            "archive_bytes": baseline["archive_bytes_len"],
        },
        "candidate": {
            "ref": candidate["ref"],
            "sha": candidate["sha"],
            "archive_sha256": candidate["archive_sha256"],
            "archive_bytes": candidate["archive_bytes_len"],
        },
        "rom": {
            "local_path": str(rom_path),
            "bytes": len(rom_bytes),
            "sha256": sha256(rom_bytes),
        },
        "states": {
            name: {
                "bytes": len(state_bytes),
                "sha256": sha256(state_bytes),
            }
            for name, state_bytes in state_files.items()
        },
    }


def safe_extract_tar_gz(archive_bytes: bytes, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    raw = gzip.decompress(archive_bytes)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar:
        dst_resolved = dst.resolve()
        for member in tar.getmembers():
            target = (dst / member.name).resolve()
            if not str(target).startswith(str(dst_resolved) + os.sep):
                raise RuntimeError(f"refusing to extract unsafe archive path: {member.name}")
        tar.extractall(dst)


def remote_metadata(replica_index: int) -> dict[str, Any]:
    import platform
    import sys

    cpu_model = ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass

    return {
        "replica_index": replica_index,
        "cpu_request": CPU_REQUEST,
        "memory_mb": MEMORY_MB,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": cpu_model,
        "os_cpu_count": os.cpu_count(),
        "affinity_cpu_count": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "commands": {},
    }


def run_command(command: list[str], cwd: Path) -> str:
    proc = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed: {command!r}\n"
            f"cwd={cwd}\n"
            f"stdout={proc.stdout}\n"
            f"stderr={proc.stderr}"
        )
    return proc.stdout


def build_wheel(label: str, repo_dir: Path, replica_dir: Path) -> Path:
    wheel_dir = replica_dir / "wheels" / label
    wheel_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        ["python", "-m", "maturin", "build", "--release", "--out", str(wheel_dir)],
        repo_dir,
    )
    wheels = sorted(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one built wheel for {label}, got {wheels}")
    return wheels[0]


def create_sample_venv(label: str, wheel: Path, replica_dir: Path) -> Path:
    venv_dir = replica_dir / "venvs" / label
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    run_command(
        ["/.uv/uv", "venv", "--python", PYTHON_BIN, str(venv_dir)],
        Path("/tmp"),
    )
    python_bin = venv_dir / "bin" / "python"
    run_command(
        [
            "/.uv/uv",
            "pip",
            "install",
            "--python",
            str(python_bin),
            str(wheel),
        ],
        Path("/tmp"),
    )
    return python_bin


def run_one_sample(repo_dir: Path, make_vars: list[str]) -> dict[str, Any]:
    command = ["make", "--silent", "benchmark", *make_vars]
    output = run_command(command, repo_dir)
    result = json.loads(output)
    result["summary_flat"] = aggregate_mixed_levels(result)
    result["command"] = command
    return result


@app.function(image=image, cpu=CPU_REQUEST, memory=MEMORY_MB, timeout=3600)
def run_paired_replica(payload: dict[str, Any]) -> dict[str, Any]:
    replica_started = time.perf_counter()
    replica_index = payload["replica_index"]
    replica_dir = Path(REMOTE_COMPARE_ROOT) / f"replica-{replica_index}"
    baseline_dir = replica_dir / "baseline"
    candidate_dir = replica_dir / "candidate"
    safe_extract_tar_gz(payload["baseline_archive_bytes"], baseline_dir)
    safe_extract_tar_gz(payload["candidate_archive_bytes"], candidate_dir)

    remote_rom = Path(REMOTE_ROM)
    remote_rom.write_bytes(payload["rom_bytes"])
    remote_state_dir = Path(REMOTE_STATE_DIR)
    remote_state_dir.mkdir(parents=True, exist_ok=True)
    for state, state_bytes in payload["state_files"].items():
        (remote_state_dir / f"{state}.state").write_bytes(state_bytes)

    baseline_wheel = build_wheel("baseline", baseline_dir, replica_dir)
    candidate_wheel = build_wheel("candidate", candidate_dir, replica_dir)
    baseline_python = create_sample_venv("baseline", baseline_wheel, replica_dir)
    candidate_python = create_sample_venv("candidate", candidate_wheel, replica_dir)
    make_vars_by_label = {
        "baseline": benchmark_make_vars(
            payload["config"],
            remote_rom,
            remote_state_dir,
            payload["states"],
            str(baseline_python),
        ),
        "candidate": benchmark_make_vars(
            payload["config"],
            remote_rom,
            remote_state_dir,
            payload["states"],
            str(candidate_python),
        ),
    }

    modal_info = remote_metadata(replica_index)
    modal_info["commands"]["make_vars"] = make_vars_by_label
    modal_info["warmup_pairs_per_replica"] = payload["warmup_pairs_per_replica"]

    pairs = []
    for pair_index in range(payload["pairs_per_replica"]):
        baseline_first = pair_index % 2 == 0
        order = ["baseline", "candidate"] if baseline_first else ["candidate", "baseline"]
        pair: dict[str, Any] = {
            "pair_index": pair_index,
            "order": order,
            "discarded_as_warmup": pair_index < payload["warmup_pairs_per_replica"],
        }
        for label in order:
            repo_dir = baseline_dir if label == "baseline" else candidate_dir
            pair[label] = run_one_sample(repo_dir, make_vars_by_label[label])
        baseline_sps = pair["baseline"]["summary"]["env_steps_per_sec"]["mean"]
        candidate_sps = pair["candidate"]["summary"]["env_steps_per_sec"]["mean"]
        pair["candidate_over_baseline"] = candidate_sps / baseline_sps
        pairs.append(pair)

    measured = [pair for pair in pairs if not pair["discarded_as_warmup"]]
    ratios = [pair["candidate_over_baseline"] for pair in measured]
    baseline_values = [sample_value(pair, "baseline") for pair in measured]
    candidate_values = [sample_value(pair, "candidate") for pair in measured]
    modal_info["replica_wall_time_s"] = time.perf_counter() - replica_started
    return {
        "replica_index": replica_index,
        "modal": modal_info,
        "pairs": pairs,
        "summary": {
            "candidate_over_baseline": stat_summary(ratios),
            "baseline_env_steps_per_sec": stat_summary(baseline_values),
            "candidate_env_steps_per_sec": stat_summary(candidate_values),
            "measured_pair_count": len(measured),
            "warmup_pair_count": len(pairs) - len(measured),
        },
    }


def summarize_replicas(replicas: list[dict[str, Any]]) -> dict[str, Any]:
    ratio_groups = replica_value_groups(replicas, "candidate_over_baseline")
    baseline_groups = replica_value_groups(replicas, "baseline")
    candidate_groups = replica_value_groups(replicas, "candidate")
    ratios = flattened(ratio_groups)
    baseline_values = flattened(baseline_groups)
    candidate_values = flattened(candidate_groups)
    ratio_summary = stat_summary(ratios)
    baseline_summary = stat_summary(baseline_values)
    candidate_summary = stat_summary(candidate_values)
    replica_median_ratios = [statistics.median(group) for group in ratio_groups if group]
    replica_ratio_summary = stat_summary(replica_median_ratios)
    replica_ratio_ci = bootstrap_median_ci(replica_median_ratios)
    decision = acceptance_decision(replica_median_ratios, replica_ratio_ci)
    warmup_pair_count = sum(
        1
        for replica in replicas
        for pair in replica["pairs"]
        if pair.get("discarded_as_warmup")
    )
    total_pair_count = sum(len(replica["pairs"]) for replica in replicas)
    return {
        "candidate_over_baseline": ratio_summary,
        "replica_median_candidate_over_baseline": replica_ratio_summary,
        "replica_median_bootstrap_ci": replica_ratio_ci,
        "baseline_env_steps_per_sec": baseline_summary,
        "candidate_env_steps_per_sec": candidate_summary,
        "paired_speedup_pct_median": (ratio_summary["median"] - 1.0) * 100.0,
        "paired_speedup_pct_mean": (ratio_summary["mean"] - 1.0) * 100.0,
        "paired_speedup_pct_replica_median": (replica_ratio_summary["median"] - 1.0)
        * 100.0,
        "variance_decomposition": {
            "baseline_env_steps_per_sec": variance_decomposition(baseline_groups),
            "candidate_env_steps_per_sec": variance_decomposition(candidate_groups),
            "candidate_over_baseline": variance_decomposition(ratio_groups),
        },
        "decision": decision,
        "pair_count": len(ratios),
        "warmup_pair_count": warmup_pair_count,
        "total_pair_count": total_pair_count,
        "replica_count": len(replicas),
    }


def estimate_modal_cost(replicas: list[dict[str, Any]]) -> dict[str, Any]:
    total_wall_time_s = sum(
        replica["modal"].get("replica_wall_time_s", 0.0) for replica in replicas
    )
    memory_gib = MEMORY_MB / 1024
    cpu_core_seconds = total_wall_time_s * CPU_REQUEST
    memory_gib_seconds = total_wall_time_s * memory_gib
    cpu_cost_usd = cpu_core_seconds * MODAL_CPU_USD_PER_CORE_SEC
    memory_cost_usd = memory_gib_seconds * MODAL_MEMORY_USD_PER_GIB_SEC
    return {
        "kind": "requested_resource_wall_time_estimate",
        "source": MODAL_PRICING_SOURCE,
        "note": (
            "Estimated from requested Modal Function CPU/memory and measured "
            "replica wall time; excludes image-build, control-plane, discounts, "
            "credits, taxes, and account-specific adjustments."
        ),
        "replica_wall_time_s": total_wall_time_s,
        "cpu_request_cores": CPU_REQUEST,
        "memory_request_gib": memory_gib,
        "cpu_core_seconds": cpu_core_seconds,
        "memory_gib_seconds": memory_gib_seconds,
        "cpu_usd_per_core_sec": MODAL_CPU_USD_PER_CORE_SEC,
        "memory_usd_per_gib_sec": MODAL_MEMORY_USD_PER_GIB_SEC,
        "cpu_cost_usd": cpu_cost_usd,
        "memory_cost_usd": memory_cost_usd,
        "total_cost_usd": cpu_cost_usd + memory_cost_usd,
    }


def print_summary(result: dict[str, Any]) -> None:
    summary = result["summary"]
    ratio = summary["candidate_over_baseline"]
    replica_ratio = summary["replica_median_candidate_over_baseline"]
    replica_ci = summary["replica_median_bootstrap_ci"]
    decision = summary["decision"]
    baseline = summary["baseline_env_steps_per_sec"]
    candidate = summary["candidate_env_steps_per_sec"]
    print(
        "paired_compare="
        f"baseline_ref={result['refs']['baseline']['ref']} "
        f"baseline_sha={result['refs']['baseline']['sha'][:12]} "
        f"candidate_ref={result['refs']['candidate']['ref']} "
        f"candidate_sha={result['refs']['candidate']['sha'][:12]}"
    )
    print(
        "paired_speedup="
        f"median_ratio={ratio['median']:.6f} mean_ratio={ratio['mean']:.6f} "
        f"median_gain_pct={summary['paired_speedup_pct_median']:.2f} "
        f"mean_gain_pct={summary['paired_speedup_pct_mean']:.2f} "
        f"pairs={summary['pair_count']} warmup_pairs={summary['warmup_pair_count']} "
        f"total_pairs={summary['total_pair_count']} replicas={summary['replica_count']}"
    )
    print(
        "robust_paired_speedup="
        f"replica_median_ratio={replica_ratio['median']:.6f} "
        f"replica_median_mean={replica_ratio['mean']:.6f} "
        f"ci95=({replica_ci['lower']:.6f},{replica_ci['upper']:.6f}) "
        f"candidate_faster_replicas="
        f"{decision['candidate_faster_replica_medians']}/{decision['replica_count']} "
        f"verdict={decision['verdict']}"
    )
    print(
        "baseline="
        f"median={baseline['median']:.1f} mean={baseline['mean']:.1f} "
        f"stdev={baseline['stdev']:.1f} min={baseline['min']:.1f} max={baseline['max']:.1f}"
    )
    print(
        "candidate="
        f"median={candidate['median']:.1f} mean={candidate['mean']:.1f} "
        f"stdev={candidate['stdev']:.1f} min={candidate['min']:.1f} max={candidate['max']:.1f}"
    )
    for replica in result["replicas"]:
        replica_ratio = replica["summary"]["candidate_over_baseline"]
        modal_info = replica["modal"]
        print(
            "replica="
            f"{replica['replica_index']} "
            f"median_ratio={replica_ratio['median']:.6f} "
            f"mean_ratio={replica_ratio['mean']:.6f} "
            f"measured_pairs={replica['summary']['measured_pair_count']} "
            f"warmup_pairs={replica['summary']['warmup_pair_count']} "
            f"cpu_model={modal_info['cpu_model']!r} "
            f"affinity_cpu_count={modal_info['affinity_cpu_count']}"
        )
    cache = result.get("cache", {})
    if cache:
        entries = cache.get("entries", {})
        print(
            "stats_cache="
            f"path={cache.get('stats_cache_path')} "
            f"context_sha={cache.get('benchmark_context_sha256')} "
            f"baseline_hit={entries.get('baseline', {}).get('hit_before_run')} "
            f"candidate_hit={entries.get('candidate', {}).get('hit_before_run')}"
        )
    cost = result.get("cost_estimate")
    if cost:
        print(
            "estimated_modal_compute_cost="
            f"total_usd={cost['total_cost_usd']:.4f} "
            f"cpu_usd={cost['cpu_cost_usd']:.4f} "
            f"memory_usd={cost['memory_cost_usd']:.4f} "
            f"replica_wall_time_s={cost['replica_wall_time_s']:.1f} "
            f"pricing_source={cost['source']}"
        )


@app.local_entrypoint()
def main(
    candidate_ref: str,
    baseline_ref: str = "main",
    rom_path: str = str(DEFAULT_ROM),
    output_json: str = "",
    stats_cache_json: str = str(DEFAULT_STATS_CACHE),
    write_stats_cache: bool = True,
    print_json: bool = False,
    replicas: int = DEFAULT_REPLICAS,
    pairs_per_replica: int = DEFAULT_PAIRS_PER_REPLICA,
    warmup_pairs_per_replica: int = DEFAULT_WARMUP_PAIRS_PER_REPLICA,
    num_envs: int = 16,
    steps: int = DEFAULT_STEPS,
    warmup: int = 100,
    frame_skip: int = 4,
    frame_stack: int = 4,
    resize_width: int = 84,
    resize_height: int = 84,
    crop_top: int = 32,
    crop_bottom: int = 0,
    action: str = "noop",
    rgb: bool = False,
    include_info: bool = False,
    terminate_on_flag: bool = False,
    no_start_game: bool = False,
    pre_start_steps: int = 30,
    start_steps: int = 8,
    post_start_steps: int = 30,
    states: str = ",".join(DEFAULT_STATES),
    state_dir: str = "",
) -> None:
    if replicas <= 0:
        raise ValueError("--replicas must be positive")
    if pairs_per_replica <= 0:
        raise ValueError("--pairs-per-replica must be positive")
    if warmup_pairs_per_replica < 0:
        raise ValueError("--warmup-pairs-per-replica must be non-negative")
    if warmup_pairs_per_replica >= pairs_per_replica:
        raise ValueError("--warmup-pairs-per-replica must be less than --pairs-per-replica")

    state_names = parse_states(states)
    config = {
        "num_envs": num_envs,
        "steps": steps,
        "warmup": warmup,
        "frame_skip": frame_skip,
        "frame_stack": frame_stack,
        "resize_width": resize_width,
        "resize_height": resize_height,
        "crop_top": crop_top,
        "crop_bottom": crop_bottom,
        "action": action,
        "rgb": rgb,
        "include_info": include_info,
        "terminate_on_flag": terminate_on_flag,
        "no_start_game": no_start_game,
        "pre_start_steps": pre_start_steps,
        "start_steps": start_steps,
        "post_start_steps": post_start_steps,
        "replicas": replicas,
        "pairs_per_replica": pairs_per_replica,
        "warmup_pairs_per_replica": warmup_pairs_per_replica,
        "measured_pairs_per_replica": pairs_per_replica - warmup_pairs_per_replica,
    }
    local_rom_path = Path(rom_path).expanduser().resolve()
    if not local_rom_path.exists():
        raise FileNotFoundError(f"ROM not found: {local_rom_path}")

    baseline = archive_ref(baseline_ref)
    candidate = archive_ref(candidate_ref)
    rom_bytes = local_rom_path.read_bytes()
    state_files = load_state_files(state_names, state_dir)
    hashes = tool_hashes()
    context = cache_context(config, state_names, rom_bytes, state_files, hashes)
    context_sha256 = hash_json(context)
    cache_keys = {
        "baseline": cache_key(baseline, context_sha256),
        "candidate": cache_key(candidate, context_sha256),
    }
    stats_cache_path = (
        resolve_local_path(stats_cache_json)
        if stats_cache_json
        else resolve_local_path(DEFAULT_STATS_CACHE)
    )
    cache_hits: dict[str, bool] = {}
    if write_stats_cache:
        existing_cache = load_stats_cache(stats_cache_path)
        cache_hits = {
            label: key in existing_cache["entries"]
            for label, key in cache_keys.items()
        }

    base_payload = {
        "baseline_archive_bytes": baseline["archive_bytes"],
        "candidate_archive_bytes": candidate["archive_bytes"],
        "rom_bytes": rom_bytes,
        "state_files": state_files,
        "states": state_names,
        "config": config,
        "pairs_per_replica": pairs_per_replica,
        "warmup_pairs_per_replica": warmup_pairs_per_replica,
    }
    payloads = [
        {
            **base_payload,
            "replica_index": replica_index,
        }
        for replica_index in range(replicas)
    ]
    replica_results = list(run_paired_replica.map(payloads))
    result = {
        "kind": "paired_modal_compare",
        "refs": {
            "baseline": {
                key: baseline[key]
                for key in ("ref", "sha", "archive_sha256", "archive_bytes_len")
            },
            "candidate": {
                key: candidate[key]
                for key in ("ref", "sha", "archive_sha256", "archive_bytes_len")
            },
        },
        "config": {
            **config,
            "states": state_names,
            "repeats_per_sample": 1,
        },
        "local": local_metadata(baseline, candidate, local_rom_path, rom_bytes, state_files),
        "cache": {
            "stats_cache_enabled": write_stats_cache,
            "stats_cache_path": str(stats_cache_path),
            "schema_version": CACHE_SCHEMA_VERSION,
            "benchmark_context_sha256": context_sha256,
            "tool_hashes": hashes,
            "entries": {
                label: {
                    "cache_key": key,
                    "hit_before_run": cache_hits.get(label, False),
                }
                for label, key in cache_keys.items()
            },
        },
        "replicas": sorted(replica_results, key=lambda item: item["replica_index"]),
    }
    result["summary"] = summarize_replicas(result["replicas"])
    result["cost_estimate"] = estimate_modal_cost(result["replicas"])

    output_path = Path(output_json).expanduser() if output_json else None
    if output_path is not None and not output_path.is_absolute():
        output_path = (REPO_ROOT / output_path).resolve()

    if write_stats_cache:
        result["cache"]["write"] = update_stats_cache(
            stats_cache_path,
            result,
            context,
            context_sha256,
            cache_keys,
            output_path,
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n")

    if print_json:
        print(json.dumps(result, indent=2))
    else:
        print_summary(result)
        if output_path is not None:
            print(f"wrote_json={output_path}")
