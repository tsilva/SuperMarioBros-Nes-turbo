from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_git_ref_benchmark import (
    BenchmarkPlan,
    BenchmarkRef,
    aggregate_single,
    cap_checkpoints,
    decide_mode,
    load_ok_for_validity,
    load_snapshot_shell,
    parse_args,
    parse_load1,
    uv_sync_command,
)


def write_raw(path: Path, values: list[float]) -> None:
    path.write_text(
        json.dumps({"runs": [{"env_steps_per_sec": value} for value in values]}) + "\n"
    )


def test_decide_mode_rejects_multi_ref_single() -> None:
    try:
        decide_mode(["a", "b"], single=True)
    except SystemExit as exc:
        assert "--single requires exactly one ref" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


def test_parse_args_reads_rom_path_from_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SMB_ROM_PATH", raising=False)
    (tmp_path / ".env").write_text("SMB_ROM_PATH=~/roms/SuperMarioBros.nes\n")

    args = parse_args(["--single", "HEAD", "--dry-run"])

    assert args.rom_path == str(Path("~/roms/SuperMarioBros.nes").expanduser())


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--steps", "0", "--steps must be positive"),
        ("--repeats", "0", "--repeats must be positive"),
        ("--warmups", "-1", "--warmups must be non-negative"),
        ("--max-measured-invocations", "0", "--max-measured-invocations must be positive"),
        ("--max-wall-clock-minutes", "0", "--max-wall-clock-minutes must be positive"),
    ],
)
def test_parse_args_rejects_invalid_benchmark_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    value: str,
    message: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SMB_ROM_PATH", raising=False)
    (tmp_path / ".env").write_text("SMB_ROM_PATH=/tmp/SuperMarioBros.nes\n")

    with pytest.raises(SystemExit, match=message):
        parse_args(["--single", "HEAD", "--dry-run", flag, value])


def test_parse_load1_extracts_unix_load_average() -> None:
    assert parse_load1("load average: 1.23, 4.56, 7.89") == 1.23
    assert parse_load1(" 15:41 up 10 days,  load average: 0.42, 0.50, 0.60") == 0.42
    assert parse_load1("no load here") is None


def test_load_snapshot_shell_closes_group_once() -> None:
    shell = load_snapshot_shell("/tmp/load.txt")

    assert "; } > /tmp/load.txt" in shell
    assert "}}" not in shell


def test_uv_sync_command_includes_common_user_tool_paths() -> None:
    command = uv_sync_command()

    assert "$HOME/.local/bin" in command
    assert "$HOME/.cargo/bin" in command
    assert command.endswith("uv sync --frozen --no-dev")


def test_cap_checkpoints_preserves_limit_as_final_checkpoint() -> None:
    assert cap_checkpoints((7, 11, 15, 21, 31), None) == (7, 11, 15, 21, 31)
    assert cap_checkpoints((7, 11, 15, 21, 31), 3) == (3,)
    assert cap_checkpoints((7, 11, 15, 21, 31), 11) == (7, 11)
    assert cap_checkpoints((7, 11, 15, 21, 31), 40) == (7, 11, 15, 21, 31, 40)


def test_force_busy_overrides_load_for_validity_only() -> None:
    assert not load_ok_for_validity(SimpleNamespace(force_busy=False, max_load=1.0), [2.0])
    assert load_ok_for_validity(SimpleNamespace(force_busy=True, max_load=1.0), [2.0])


def test_aggregate_single_uses_convergence_helper(tmp_path: Path) -> None:
    run_dir = tmp_path / "local-single-test"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
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
    for index, value in enumerate(medians):
        write_raw(raw_dir / f"measured-ref-{index:02d}.json", [value - 1.0, value, value + 1.0])

    plan = BenchmarkPlan(
        mode="single",
        run_name="local-single-test",
        run_dir=str(run_dir),
        refs=[
            BenchmarkRef(
                role="ref",
                ref="HEAD",
                sha="1" * 40,
                archive=tmp_path / "ref-111111111111.tar.gz",
            )
        ],
        rom_path=str(tmp_path / "SuperMarioBros.nes"),
        state_dir=str(tmp_path / "states"),
        checkpoints=(5, 8, 11, 15, 21, 31),
        warmups=2,
        measured_cap=31,
    )
    args = SimpleNamespace(
        max_load=99.0,
        force_busy=False,
        steps=50000,
        repeats=3,
        max_measured_invocations=None,
        max_wall_clock_minutes=None,
    )

    aggregate = aggregate_single(args, plan, measured_count=11, load_values=[0.5, 0.4])

    assert aggregate["mode"] == "single_ref_fixed_local"
    assert aggregate["decision"] == "converged"
    assert aggregate["official_median_sps"] == 1001.0
    assert aggregate["checkpoint_trace"][-1]["count"] == 11
    assert aggregate["load_gate_passed"]
    assert not aggregate["load_gate_ignored_for_validity"]
