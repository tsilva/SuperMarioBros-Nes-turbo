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
