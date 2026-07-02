#!/usr/bin/env python3
"""Deterministic helpers for SuperMarioBros-Nes-turbo release wheel builds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback.
    import tomli as tomllib  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[4]
PYPROJECT = REPO_ROOT / "pyproject.toml"
CARGO_TOML = REPO_ROOT / "Cargo.toml"
CARGO_LOCK = REPO_ROOT / "Cargo.lock"
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
PACKAGE_NAME = "supermariobrosnes-turbo"
IMPORT_NAME = "supermariobrosnes_turbo"
EXTENSION_NAME = "_supermariobrosnes_turbo"
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+|\.post\d+|\.dev\d+)?$")

IGNORED_DIR_NAMES_ANYWHERE = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".uv-cache",
    ".venv",
    "__pycache__",
    "target",
}
IGNORED_ROOT_DIR_NAMES = {
    "artifacts",
    "build",
    "dist",
}
IGNORED_FILE_SUFFIXES = {".o", ".a", ".so", ".dylib", ".d", ".pyc", ".pyo"}
ROM_SUFFIXES = {".nes", ".sfc", ".smc", ".gb", ".gbc", ".gen", ".sms", ".bin"}


def read_pyproject() -> dict[str, object]:
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)


def pyproject_name() -> str:
    return str(read_pyproject()["project"]["name"])  # type: ignore[index]


def pyproject_version() -> str:
    return str(read_pyproject()["project"]["version"])  # type: ignore[index]


def section_version(path: Path, section: str, *, package_name: str | None = None) -> str:
    current_section: str | None = None
    current_name: str | None = None
    in_matching_package = package_name is None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("[[") and stripped.endswith("]]"):
            current_section = stripped[2:-2].strip()
            current_name = None
            in_matching_package = package_name is None
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped[1:-1].strip()
            current_name = None
            in_matching_package = package_name is None
            continue
        if current_section != section:
            continue
        if stripped.startswith("name = "):
            current_name = stripped.split("=", 1)[1].strip().strip('"')
            if package_name is not None:
                in_matching_package = current_name == package_name
            continue
        if in_matching_package and stripped.startswith("version = "):
            return stripped.split("=", 1)[1].strip().strip('"')
    raise RuntimeError(f"could not find version in [{section}] of {path}")


def cargo_version() -> str:
    return section_version(CARGO_TOML, "package")


def cargo_lock_version() -> str:
    return section_version(CARGO_LOCK, "package", package_name=PACKAGE_NAME)


def validate_version(version: str) -> None:
    if VERSION_RE.match(version) is None:
        raise SystemExit(f"unsupported version format: {version!r}")


def split_release(version: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version)
    if match is None:
        raise SystemExit(
            f"cannot compute major/minor/patch bump from non-final version {version!r}; pass --to"
        )
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def next_version(version: str, part: str) -> str:
    major, minor, patch = split_release(version)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(part)


def replace_section_version(path: Path, section: str, version: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    current_section: str | None = None
    changed = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]") and not stripped.startswith("[["):
            current_section = stripped[1:-1].strip()
            continue
        if current_section == section and stripped.startswith("version = "):
            newline = "\n" if line.endswith("\n") else ""
            lines[index] = f'version = "{version}"{newline}'
            changed = True
            break
    if not changed:
        raise RuntimeError(f"could not replace version in [{section}] of {path}")
    path.write_text("".join(lines), encoding="utf-8")


def check_version(args: argparse.Namespace) -> None:
    versions = {
        "pyproject": pyproject_version(),
        "cargo_toml": cargo_version(),
        "cargo_lock": cargo_lock_version(),
    }
    expected = args.version
    failures = []
    if pyproject_name() != PACKAGE_NAME:
        failures.append(f"pyproject package name is {pyproject_name()!r}, expected {PACKAGE_NAME!r}")
    if len(set(versions.values())) != 1:
        failures.append(f"version mismatch: {versions}")
    if expected is not None and set(versions.values()) != {expected}:
        failures.append(f"expected version {expected!r}, saw {versions}")
    result = {"package": pyproject_name(), "versions": versions}
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit("; ".join(failures))


def run_capture(args_list: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            args_list,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError as exc:
        return 127, str(exc)
    return completed.returncode, completed.stdout.strip()


def check_tools(args: argparse.Namespace) -> None:
    checks = {
        "cargo": run_capture(["cargo", "--version"]),
        "docker": run_capture(["docker", "--version"]),
        "maturin": run_capture([str(PYTHON), "-m", "maturin", "--version"]),
        "cibuildwheel": run_capture(
            [
                str(PYTHON),
                "-c",
                "from importlib.metadata import version; print('cibuildwheel ' + version('cibuildwheel'))",
            ]
        ),
        "twine": run_capture([str(PYTHON), "-m", "twine", "--version"]),
    }
    result = {
        name: {"ok": code == 0, "output": output}
        for name, (code, output) in checks.items()
    }
    print(json.dumps(result, indent=2))
    missing = [name for name, item in result.items() if not item["ok"]]
    if missing:
        raise SystemExit(f"missing release tooling: {', '.join(missing)}")


def bump_version(args: argparse.Namespace) -> None:
    target = args.to or next_version(pyproject_version(), args.part)
    validate_version(target)
    if args.write:
        replace_section_version(PYPROJECT, "project", target)
        replace_section_version(CARGO_TOML, "package", target)
    print(target)


def check_pypi(args: argparse.Namespace) -> None:
    validate_version(args.version)
    url = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(json.dumps({"package": PACKAGE_NAME, "exists": False, "version_exists": False}, indent=2))
            return
        raise
    releases = data.get("releases", {})
    exists = args.version in releases and bool(releases[args.version])
    print(json.dumps({"package": PACKAGE_NAME, "version": args.version, "version_exists": exists}, indent=2))
    if exists:
        raise SystemExit(f"{PACKAGE_NAME} {args.version} already exists on PyPI")


def should_ignore(rel: Path) -> bool:
    if any(part in IGNORED_DIR_NAMES_ANYWHERE for part in rel.parts):
        return True
    if len(rel.parts) == 1 and rel.name in IGNORED_ROOT_DIR_NAMES:
        return True
    if rel.name.startswith("wheelhouse"):
        return True
    if rel.suffix in IGNORED_FILE_SUFFIXES:
        return True
    return False


def copy_clean_tree(destination: Path, *, force: bool = False) -> None:
    if destination.exists():
        if not force:
            raise FileExistsError(f"{destination} already exists; pass --force to replace it")
        shutil.rmtree(destination)

    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        base = Path(directory)
        for name in names:
            rel = (base / name).relative_to(REPO_ROOT)
            if should_ignore(rel):
                ignored.add(name)
        return ignored

    shutil.copytree(REPO_ROOT, destination, symlinks=True, ignore=ignore)


def find_contamination(root: Path) -> dict[str, list[str]]:
    compiled: list[str] = []
    roms: list[str] = []
    pycache: list[str] = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if "__pycache__" in rel.parts:
            pycache.append(rel.as_posix())
        if not path.is_file():
            continue
        if rel.suffix in IGNORED_FILE_SUFFIXES:
            compiled.append(rel.as_posix())
        if rel.suffix.lower() in ROM_SUFFIXES:
            roms.append(rel.as_posix())
    return {"compiled_artifacts": compiled, "rom_payloads": roms, "pycache": pycache}


def fail_on_contamination(root: Path) -> None:
    contamination = find_contamination(root)
    failures = {key: value for key, value in contamination.items() if value}
    if failures:
        print(json.dumps(failures, indent=2), file=sys.stderr)
        raise SystemExit(f"{root} is not a clean release source copy")


def wheelhouse(version: str, platform_name: str) -> Path:
    return REPO_ROOT / f"wheelhouse-v{version}-{platform_name}"


def prepare_sources(args: argparse.Namespace) -> None:
    version = args.version or pyproject_version()
    validate_version(version)
    root = args.root or Path(
        tempfile.mkdtemp(prefix=f"supermariobrosnes-turbo-v{version}-builds.", dir="/private/tmp")
    )
    root = root.resolve()
    macos_src = root / "macos-src"
    linux_src = root / "linux-src-clean"
    copy_clean_tree(macos_src, force=args.force)
    copy_clean_tree(linux_src, force=args.force)
    fail_on_contamination(macos_src)
    fail_on_contamination(linux_src)
    print(
        json.dumps(
            {
                "version": version,
                "root": str(root),
                "macos_src": str(macos_src),
                "linux_src_clean": str(linux_src),
                "macos_wheelhouse": str(wheelhouse(version, "macos")),
                "linux_wheelhouse": str(wheelhouse(version, "linux")),
            },
            indent=2,
        )
    )


def shell_quote(value: str | Path) -> str:
    import shlex

    return shlex.quote(str(value))


def build_commands(args: argparse.Namespace) -> None:
    version = args.version or pyproject_version()
    validate_version(version)
    macos_src = Path(args.macos_src) if args.macos_src else Path("<macos-src>")
    linux_src = Path(args.linux_src) if args.linux_src else Path("<linux-src-clean>")
    macos_out = wheelhouse(version, "macos")
    linux_out = wheelhouse(version, "linux")
    print("# macOS arm64")
    print(f"cd {shell_quote(macos_src)}")
    print(
        "MACOSX_DEPLOYMENT_TARGET=14.0 "
        "ARCHFLAGS='-arch arm64' "
        f"CARGO_TARGET_DIR={shell_quote(macos_src / 'target-release')} "
        f"{shell_quote(PYTHON)} -m maturin build --release --out {shell_quote(macos_out)}"
    )
    print()
    print("# Linux x86_64 manylinux")
    print(f"cd {shell_quote(linux_src)}")
    print(
        "CIBW_BUILD=cp39-manylinux_x86_64 "
        "CIBW_ARCHS_LINUX=x86_64 "
        "CIBW_SKIP='*-musllinux_*' "
        "CIBW_BEFORE_ALL_LINUX='curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal' "
        "CIBW_ENVIRONMENT_LINUX='PATH=\"$HOME/.cargo/bin:$PATH\" CARGO_NET_GIT_FETCH_WITH_CLI=true' "
        f"{shell_quote(PYTHON)} -m cibuildwheel --platform linux --output-dir {shell_quote(linux_out)}"
    )


def wheel_names(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as zf:
        return zf.namelist()


def audit_wheel(wheel: Path, version: str) -> dict[str, object]:
    names = wheel_names(wheel)
    extension_entries = [
        name
        for name in names
        if name.startswith(f"{IMPORT_NAME}/{EXTENSION_NAME}") and name.endswith((".so", ".pyd"))
    ]
    metadata_entries = [name for name in names if name.endswith(".dist-info/METADATA")]
    rom_payloads = [name for name in names if Path(name).suffix.lower() in ROM_SUFFIXES]
    pycache_entries = [name for name in names if "__pycache__" in Path(name).parts or name.endswith(".pyc")]
    checks = {
        "version_in_filename": version in wheel.name,
        "abi3_wheel": "abi3" in wheel.name,
        "has_package_init": f"{IMPORT_NAME}/__init__.py" in names,
        "has_env_source": f"{IMPORT_NAME}/env.py" in names,
        "has_extension": bool(extension_entries),
        "has_metadata": bool(metadata_entries),
        "no_rom_payloads": not rom_payloads,
        "no_pycache": not pycache_entries,
    }
    return {
        "wheel": str(wheel),
        "extension_entries": extension_entries,
        "metadata_entries": metadata_entries,
        "rom_payloads": rom_payloads,
        "pycache_entries": pycache_entries,
        "checks": checks,
    }


def assert_audit_passed(results: list[dict[str, object]]) -> None:
    failures: dict[str, list[str]] = {}
    for result in results:
        checks = result["checks"]
        assert isinstance(checks, dict)
        failed = [key for key, value in checks.items() if not value]
        if failed:
            failures[str(result["wheel"])] = failed
    if failures:
        print(json.dumps(results, indent=2), file=sys.stderr)
        raise SystemExit(f"wheel audit failed: {failures}")


def find_wheels(version: str) -> list[Path]:
    candidates = list(wheelhouse(version, "macos").glob(f"*{version}*.whl"))
    candidates.extend(wheelhouse(version, "linux").glob(f"*{version}*.whl"))
    return sorted(candidates)


def audit_wheels(args: argparse.Namespace) -> None:
    version = args.version or pyproject_version()
    wheels = [Path(wheel) for wheel in args.wheels] if args.wheels else find_wheels(version)
    if len(wheels) < 2:
        raise SystemExit(f"expected macOS and Linux wheels for {version}, found {wheels}")
    results = [audit_wheel(wheel, version) for wheel in wheels]
    assert_audit_passed(results)
    print(json.dumps(results, indent=2))


def run(args_list: list[str], **kwargs: object) -> None:
    print("+", " ".join(shell_quote(arg) for arg in args_list))
    subprocess.run(args_list, check=True, **kwargs)


def smoke_wheel(args: argparse.Namespace) -> None:
    wheel = args.wheel.resolve()
    python = args.python.resolve()
    with tempfile.TemporaryDirectory(prefix="supermariobrosnes-wheel-smoke.", dir="/private/tmp") as tmp:
        target = Path(tmp)
        run([str(python), "-m", "pip", "install", "--no-deps", "--target", str(target), str(wheel)])
        code = f"""
import {IMPORT_NAME}
from {IMPORT_NAME} import {EXTENSION_NAME}
print({IMPORT_NAME}.__file__)
print({EXTENSION_NAME}.__file__)
assert {IMPORT_NAME}.__file__.startswith({str(target)!r})
assert hasattr({IMPORT_NAME}, "SuperMarioBrosVecEnv")
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(target)
        run([str(python), "-c", code], cwd="/private/tmp", env=env)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_check(args: argparse.Namespace) -> None:
    version = args.version or pyproject_version()
    wheels = find_wheels(version)
    if len(wheels) < 2:
        raise SystemExit(f"expected macOS and Linux wheels for {version}, found {wheels}")
    results = [audit_wheel(wheel, version) for wheel in wheels]
    assert_audit_passed(results)
    run([str(PYTHON), "-m", "twine", "check", *[str(wheel) for wheel in wheels]])
    hashes = {str(wheel): sha256(wheel) for wheel in wheels}
    print(json.dumps({"audits": results, "sha256": hashes}, indent=2))
    print()
    print(f"{shell_quote(PYTHON)} -m twine upload --config-file .pypirc \\")
    for index, wheel in enumerate(wheels):
        suffix = " \\" if index < len(wheels) - 1 else ""
        print(f"  {shell_quote(wheel)}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check-version", help="Check package identity and version consistency")
    check.add_argument("--version")
    check.set_defaults(func=check_version)

    tools = subparsers.add_parser("check-tools", help="Check release build tooling")
    tools.set_defaults(func=check_tools)

    bump = subparsers.add_parser("bump-version", help="Print or write a target version")
    bump.add_argument("--to")
    bump.add_argument("--part", choices=("major", "minor", "patch"), default="patch")
    bump.add_argument("--write", action="store_true")
    bump.set_defaults(func=bump_version)

    pypi = subparsers.add_parser("check-pypi", help="Fail if the target version exists on PyPI")
    pypi.add_argument("--version", required=True)
    pypi.set_defaults(func=check_pypi)

    prepare = subparsers.add_parser("prepare-sources", help="Create clean macOS/Linux source copies")
    prepare.add_argument("--version")
    prepare.add_argument("--root", type=Path)
    prepare.add_argument("--force", action="store_true")
    prepare.set_defaults(func=prepare_sources)

    commands = subparsers.add_parser("build-commands", help="Print platform build commands")
    commands.add_argument("--version")
    commands.add_argument("--macos-src")
    commands.add_argument("--linux-src")
    commands.set_defaults(func=build_commands)

    audit = subparsers.add_parser("audit-wheels", help="Audit wheel contents")
    audit.add_argument("--version")
    audit.add_argument("wheels", nargs="*")
    audit.set_defaults(func=audit_wheels)

    smoke = subparsers.add_parser("smoke-wheel", help="Install and import-test a built wheel")
    smoke.add_argument("wheel", type=Path)
    smoke.add_argument("--python", type=Path, default=PYTHON)
    smoke.set_defaults(func=smoke_wheel)

    final = subparsers.add_parser("final-check", help="Audit wheels, run twine check, hash, and print upload command")
    final.add_argument("--version")
    final.set_defaults(func=final_check)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
