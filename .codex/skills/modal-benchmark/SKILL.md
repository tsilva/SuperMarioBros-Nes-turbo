---
name: modal-benchmark
description: Run the canonical clean-machine Modal CPU benchmark for this SuperMarioBros-Nes-turbo repo with the first four Super Mario Bros NES Level1 states mixed across vector lanes, then report mixed-lane throughput. Use when the user invokes /modal-benchmark, asks to benchmark on Modal/modal.com, wants a clean CPU-only baseline, wants to compare an optimization on fresh compute, or asks for the current Modal benchmark result format.
---

# Modal Benchmark

## Workflow

Run the repo-local Modal launcher from the repository root:

```bash
modal run scripts/modal_benchmark_sps.py --output-json artifacts/benchmarks/modal-baseline-YYYY-MM-DD.json
```

Use the current date in the artifact name. If that path already exists, add a short time suffix, for example `modal-baseline-YYYY-MM-DD-HHMM.json`.

The command uploads the local repo snapshot to Modal and sends the local ROM bytes at runtime. On Modal, the launcher must run the benchmark through the repo Makefile with `make --silent benchmark`, not by calling `scripts/benchmark_sps.py` directly. Pass Modal-only details such as uploaded ROM/state paths and `--json` through `BENCHMARK_ARGS`, and pass the canonical count knobs through `BENCHMARK_NUM_ENVS`, `BENCHMARK_STEPS`, and `BENCHMARK_REPEATS`.

If the user explicitly invoked this skill or asked for a Modal benchmark, treat the invocation as full-access approval for Modal network/auth/upload. If Modal access is still unavailable in the execution environment, report that the benchmark could not run and name the blocker plainly.

Use the launcher defaults unless the user asks otherwise:

- `num_envs=16`
- `steps=500`
- `repeats=3`
- `frame_skip=4`
- `frame_stack=4`
- grayscale, crop top 32, resize 84x84
- action `noop`
- states `Level1-1,Level1-2,Level1-3,Level1-4`

The launcher runs one 16-env benchmark with those states repeated round-robin
across lanes by invoking `make --silent benchmark` inside the Modal container.
It uploads the local ROM bytes plus the four stable-retro `.state` files at
runtime. State files resolve from an explicit `--state-dir`, `SUPERMARIOBROSNES_FASTENV_STATE_DIR`,
an installed `stable_retro` package, or the sibling `stable-retro-turbo`
checkout.

## Reporting

After the Modal run completes, read the saved JSON artifact and report the result in this shape:

- Start with whether it worked and briefly say Modal built the image, uploaded the repo snapshot, built/installed the Rust extension, uploaded ROM bytes and state bytes at runtime, and ran `make --silent benchmark` for one 16-env mixed-lane benchmark across the first four Level1 states.
- Link the saved artifact with an absolute file link.
- Include a `Mixed-lane results` code block:

```text
states: Level1-1, Level1-2, Level1-3, Level1-4
lane_states: LANE_STATE_LIST
runs env_steps_per_sec: RUN1, RUN2, RUN3 | mean: MEAN | stdev: STDEV | best: BEST
obs_shape: (16, 4, 84, 84)
obs_dtype: uint8
```

- Include an `Average` code block:

```text
mean_of_all_runs: MEAN
stdev_of_all_runs: STDEV
best_run: BEST
```

- Include a `Modal machine metadata` code block:

```text
cpu_request: CPU_REQUEST
memory_mb: MEMORY_MB
os_cpu_count: OS_CPU_COUNT
affinity_cpu_count: AFFINITY_CPU_COUNT
```

- If the command output includes a Modal run URL, include it. If not, omit the URL rather than inventing one.
- Mention any launcher fixes made during the run. If no code changed, say so briefly.

## JSON Extraction

Use a short local read after the run to avoid hand-copying console output:

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path("artifacts/benchmarks/modal-baseline-YYYY-MM-DD.json")
data = json.loads(path.read_text())
print("path", path)
result = data["mixed_levels"]
summary = result["summary"]["env_steps_per_sec"]
print("states", result["config"]["states"])
print("lane_states", result["config"]["lane_states"])
print(
    "mixed_levels",
    "mean", summary["mean"],
    "stdev", summary["stdev"],
    "best", summary["max"],
    "runs", [round(r["env_steps_per_sec"], 1) for r in result["runs"]],
    "obs", result["observation"]["shape"], result["observation"]["dtype"],
)
all_runs = data["summary"]["all_runs_env_steps_per_sec"]
print("mean_of_all_runs", all_runs["mean"])
print("stdev_of_all_runs", all_runs["stdev"])
print("best_run", all_runs["max"])
print(
    "modal",
    data["modal"]["cpu_request"],
    data["modal"]["memory_mb"],
    data["modal"]["os_cpu_count"],
    data["modal"]["affinity_cpu_count"],
)
PY
```

Run `git status --short` before the final answer so changed launcher/docs files are not hidden.
