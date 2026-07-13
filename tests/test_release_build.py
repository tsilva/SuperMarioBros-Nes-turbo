import importlib.util
from pathlib import Path


def _release_build_module():
    root = Path(__file__).resolve().parents[1]
    path = root / ".codex" / "skills" / "build-release" / "scripts" / "release_build.py"
    spec = importlib.util.spec_from_file_location("release_build", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_version_file_is_the_single_source_of_truth():
    root = Path(__file__).resolve().parents[1]

    assert (root / "VERSION.txt").read_text(encoding="utf-8").strip()
    assert "VERSION.txt" in (
        root / ".codex" / "skills" / "build-release" / "scripts" / "release_build.py"
    ).read_text(encoding="utf-8")
    assert "VERSION.txt" in (root / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert '"VERSION.txt", "pyproject.toml", "Cargo.toml", "Cargo.lock", "uv.lock"' in (
        root / "scripts" / "release.py"
    ).read_text(encoding="utf-8")


def test_release_validates_python_314_with_stable_abi_wheels():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8",
    )
    cargo = (root / "Cargo.toml").read_text(encoding="utf-8")

    assert 'PYTHON_VERSION: "3.14"' in workflow
    assert 'features = ["abi3-py39", "extension-module"]' in cargo


def test_release_wheel_builds_use_platform_scoped_cargo_caches():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8",
    )
    release_build = _release_build_module()

    assert "actions/cache@v4" in workflow
    assert "path: ${{ matrix.cargo_target_dir }}" in workflow
    assert (
        "key: cargo-target-v1-${{ matrix.platform }}-${{ runner.arch }}-${{ steps.source.outputs.sha }}"
        in workflow
    )
    assert 'run: echo "sha=$(git rev-parse HEAD)" >> "$GITHUB_OUTPUT"' in workflow
    assert "cargo-target-v1-${{ matrix.platform }}-${{ runner.arch }}-" in workflow
    assert "cargo_target_dir: target-release" in workflow
    assert "cargo_target_dir: target-release-linux" in workflow

    assert release_build.cargo_target_dir("macos", root) == root / "target-release"
    assert release_build.cargo_target_dir("linux", root) == root / "target-release-linux"


def test_linux_release_cache_is_mounted_into_cibuildwheel(tmp_path):
    release_build = _release_build_module()
    env = release_build.linux_build_env(tmp_path)

    assert env["CIBW_CONTAINER_ENGINE"] == (
        f"docker; create_args: --volume={(tmp_path / 'target-release-linux').resolve()}:/cargo-target"
    )
    assert "CARGO_TARGET_DIR=/cargo-target" in env["CIBW_ENVIRONMENT_LINUX"]
    assert release_build.should_ignore(Path("target-release-linux"))
    assert not release_build.should_ignore(Path("nested/target-release-linux"))


def test_latest_non_yanked_pypi_version_ignores_fully_yanked_latest_release():
    release_build = _release_build_module()
    releases = {
        "0.2.3": [{"filename": "older.whl", "yanked": False}],
        "0.2.4": [{"filename": "current.whl", "yanked": False}],
        "0.3.0": [
            {"filename": "macos.whl", "yanked": True},
            {"filename": "linux.whl", "yanked": True},
        ],
    }

    assert release_build.latest_non_yanked_pypi_version(releases) == "0.2.4"


def test_latest_non_yanked_pypi_version_accepts_release_with_any_non_yanked_file():
    release_build = _release_build_module()
    releases = {
        "0.2.4": [{"filename": "current.whl", "yanked": False}],
        "0.2.5": [
            {"filename": "bad-platform.whl", "yanked": True},
            {"filename": "good-platform.whl", "yanked": False},
        ],
    }

    assert release_build.latest_non_yanked_pypi_version(releases) == "0.2.5"
