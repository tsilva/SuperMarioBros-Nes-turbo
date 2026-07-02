#!/usr/bin/env python3
"""Copy fixed-host benchmark evidence locally, then optionally purge remote bulk."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


REMOTE_ROOT = PurePosixPath("/home/tsilva/SuperMarioBros-Nes-turbo-host-bench")
RUN_NAME_RE = re.compile(r"^host-(single|compare)-[A-Za-z0-9_.-]+$")


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_remote_run_dir(remote_run_dir: str) -> tuple[PurePosixPath, str]:
    remote = PurePosixPath(remote_run_dir)
    runs_root = REMOTE_ROOT / "runs"
    try:
        relative = remote.relative_to(runs_root)
    except ValueError as exc:
        raise SystemExit(f"Refusing remote path outside {runs_root}: {remote}") from exc
    if len(relative.parts) != 1:
        raise SystemExit(f"Refusing nested remote run path: {remote}")
    run_name = relative.parts[0]
    if not RUN_NAME_RE.match(run_name):
        raise SystemExit(f"Refusing unexpected run directory name: {run_name}")
    return remote, run_name


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


def remote_test_file(args: argparse.Namespace, remote_path: PurePosixPath) -> None:
    cmd = ssh_base(args) + [f"test -f {quote(str(remote_path))}"]
    run(cmd)


def quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def copy_results(args: argparse.Namespace, remote: PurePosixPath, run_name: str) -> tuple[Path, dict]:
    local_dir = args.local_results_root / run_name
    if local_dir.exists():
        shutil.rmtree(local_dir)
    raw_dir = local_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    remote_test_file(args, remote / "aggregate.json")

    remote_prefix = f"{args.ssh_target}:{remote}"
    rsh = rsync_rsh(args)
    run(["rsync", "-az", "-e", rsh, f"{remote_prefix}/aggregate.json", str(local_dir) + "/"])
    run(
        [
            "rsync",
            "-az",
            "-e",
            rsh,
            "--exclude=*.stdout.json",
            "--include=*.json",
            "--include=*.txt",
            "--exclude=*",
            f"{remote_prefix}/raw/",
            str(raw_dir) + "/",
        ]
    )

    aggregate = load_json(local_dir / "aggregate.json")
    if "mode" not in aggregate or "shas" not in aggregate:
        raise SystemExit("Copied aggregate.json is missing required mode/shas fields")

    copied_files = []
    for path in sorted(local_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            if path.suffix == ".json":
                load_json(path)
            copied_files.append(
                {
                    "path": str(path.relative_to(local_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )

    if not any(item["path"].startswith("raw/") for item in copied_files):
        raise SystemExit("No raw benchmark files were copied; refusing cleanup")

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "remote_run_dir": str(remote),
        "ssh_target": args.ssh_target,
        "host_key_alias": args.host_key_alias,
        "local_result_dir": str(local_dir),
        "copied_files": copied_files,
        "cleanup_policy": {
            "remote_bulk_purge_requested": args.purge_remote_bulk,
            "remote_bulk_removed": [],
            "remote_kept": ["aggregate.json", "raw/*.json except *.stdout.json", "raw/*.txt"],
            "shared_uv_cache_pruned": False,
            "local_archives_removed": [],
        },
    }
    (local_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return local_dir, aggregate


def append_index(args: argparse.Namespace, run_name: str, local_dir: Path, aggregate: dict) -> None:
    summary = {
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
        "tier": aggregate.get("tier"),
    }
    args.local_results_root.mkdir(parents=True, exist_ok=True)
    index_path = args.local_results_root / "index.jsonl"
    existing = []
    if index_path.exists():
        for line in index_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                existing.append(line)
                continue
            if record.get("run_name") != run_name:
                existing.append(json.dumps(record, sort_keys=True))
    with index_path.open("w") as handle:
        for line in existing:
            handle.write(line + "\n")
        handle.write(json.dumps(summary, sort_keys=True) + "\n")


def purge_remote_bulk(args: argparse.Namespace, remote: PurePosixPath, local_dir: Path) -> list[str]:
    manifest_path = local_dir / "manifest.json"
    manifest = load_json(manifest_path)
    if not manifest.get("copied_files"):
        raise SystemExit("Local manifest has no copied files; refusing remote cleanup")

    remote_cmd = (
        f"rm -rf -- {quote(str(remote / 'sources'))} {quote(str(remote / 'archives'))}; "
        f"if test -d {quote(str(remote / 'raw'))}; then "
        f"find {quote(str(remote / 'raw'))} -type f -name '*.stdout.json' -delete; "
        "fi"
    )
    run(ssh_base(args) + [remote_cmd])
    removed = ["sources/", "archives/", "raw/*.stdout.json"]
    manifest["cleanup_policy"]["remote_bulk_removed"] = removed
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return removed


def purge_local_archives(args: argparse.Namespace, aggregate: dict, local_dir: Path) -> list[str]:
    archives_dir = args.local_archives_dir
    if not archives_dir.exists():
        return []
    sha_values = set()
    for value in aggregate.get("shas", {}).values():
        if isinstance(value, str) and len(value) >= 12 and re.fullmatch(r"[0-9a-fA-F]+", value):
            sha_values.add(value[:12].lower())
    removed = []
    for path in sorted(archives_dir.glob("*.tar.gz")):
        lower_name = path.name.lower()
        if any(short in lower_name for short in sha_values):
            removed.append(str(path))
            path.unlink()
    if removed:
        manifest_path = local_dir / "manifest.json"
        manifest = load_json(manifest_path)
        manifest["cleanup_policy"]["local_archives_removed"] = removed
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return removed


def prune_uv_cache(args: argparse.Namespace, local_dir: Path) -> None:
    run(ssh_base(args) + ["uv cache prune"])
    manifest_path = local_dir / "manifest.json"
    manifest = load_json(manifest_path)
    manifest["cleanup_policy"]["shared_uv_cache_pruned"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-run-dir", required=True)
    parser.add_argument("--ssh-target", default="beast-3-local")
    parser.add_argument("--host-key-alias")
    parser.add_argument("--local-results-root", type=Path, default=Path("artifacts/benchmarks/host-results"))
    parser.add_argument("--local-archives-dir", type=Path, default=Path("artifacts/benchmarks/host-archives"))
    parser.add_argument("--purge-remote-bulk", action="store_true")
    parser.add_argument("--purge-local-archives", action="store_true")
    parser.add_argument("--prune-uv-cache", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if not shutil.which("rsync"):
        raise SystemExit("rsync is required")
    remote, run_name = validate_remote_run_dir(args.remote_run_dir)
    local_dir, aggregate = copy_results(args, remote, run_name)
    append_index(args, run_name, local_dir, aggregate)
    removed_remote = purge_remote_bulk(args, remote, local_dir) if args.purge_remote_bulk else []
    removed_local = purge_local_archives(args, aggregate, local_dir) if args.purge_local_archives else []
    if args.prune_uv_cache:
        prune_uv_cache(args, local_dir)
    print(
        json.dumps(
            {
                "local_result_dir": str(local_dir),
                "index": str(args.local_results_root / "index.jsonl"),
                "remote_bulk_removed": removed_remote,
                "local_archives_removed": removed_local,
                "uv_cache_pruned": args.prune_uv_cache,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
