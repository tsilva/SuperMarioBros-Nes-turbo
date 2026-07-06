---
name: host-benchmark
description: Benchmark one SuperMarioBros-Nes-turbo commit or compare two commits with the fixed local-host protocol. Use when the user asks to benchmark, compare, confirm, or measure env_steps_per_sec on the dedicated local host, especially for exact-ref single-run baselines or deciding whether a candidate commit is faster than a baseline.
---

# Host Benchmark

## Use The Runner

For local git refs, do not retype the benchmark protocol as shell heredocs.
Run the checked-in deterministic runner from the repo root:

```bash
.venv/bin/python scripts/run_git_ref_host_benchmark.py --single REF \
  --rom-path /path/to/SuperMarioBros.nes

.venv/bin/python scripts/run_git_ref_host_benchmark.py BASELINE_REF CANDIDATE_REF \
  --rom-path /path/to/SuperMarioBros.nes

.venv/bin/python scripts/run_git_ref_host_benchmark.py CANDIDATE_REF \
  --rom-path /path/to/SuperMarioBros.nes
```

The one-ref comparison form uses local `main` as baseline. Use `--dry-run` to
inspect the exact run plan without creating archives or touching the benchmark
run root.

The runner owns deterministic mechanics:

- exact `git rev-parse` ref resolution
- reproducible `git archive` snapshots without switching branches
- exclusion of dirty/untracked local files from measured sources
- local run directory naming and isolation
- state-file setup from the sibling stable-retro checkout
- per-ref `uv sync --frozen --no-dev`
- smoke checks
- warmup discard
- sequential convergence checkpoints
- raw JSON naming
- `aggregate.json` fields
- local result retention/finalization

If the runner lacks a needed feature, patch the runner and add tests instead of
working around it in the skill text.

## Mode Rules

Input forms:

```text
single_ref only
candidate_ref
baseline_ref candidate_ref
pypi-stable-retro-turbo
pypi-supermariobrosnes-turbo
```

Use single-ref mode only when the user explicitly asks for one ref only. If the
user provides one ref without saying "only", "single", "no comparison", or
equivalent, compare latest local `main` against that candidate. If two refs are
provided, first is baseline and second is candidate. If ambiguous, ask one short
clarifying question before running.

Do not install system packages. If Rust/Cargo is truly absent, ask before
installing it.

## Git-Ref Protocol

The runner implements this workload:

- `RAYON_NUM_THREADS=12`
- `num_envs=16`
- `steps=50000`
- `repeats=3`
- single-ref warmups: `2`, discarded
- single-ref checkpoints: `5,8,11,15,21,31`
- comparison warmup pairs: `2`, discarded
- comparison checkpoints: `7,11,15,21,31`
- `frame_skip=4`
- `frame_stack=4`
- grayscale, crop top 32, resize 84x84
- action set `simple`, action `noop`
- states `Level1-1,Level1-2,Level1-3,Level1-4`
- observation contract `obs_shape=(16, 4, 84, 84)`, `obs_dtype=uint8`

Sequential convergence is preregistered: collect to the next checkpoint, compute
the official robust statistic, and stop only when stability, CI, CV, outlier,
decision, and load gates pass, or when the max checkpoint is reached. Do not
continue indefinitely and do not stop just because the observed result is
favorable.

For single-ref mode, official SPS is:

```text
median(measured invocation medians)
```

For comparison mode, official ratio is:

```text
median(candidate invocation median / baseline invocation median per pair)
```

The convergence math lives in:

```bash
.venv/bin/python scripts/host_benchmark_stats.py --help
```

## Load Gate

The runner records load snapshots in `raw/load-*.txt`. By default it blocks
when the initial 1-minute load is above roughly one-third of logical CPU count
unless `--force-busy` is passed.

Smoke checks are acceptable on a busy host, but official timing should use a
calm host. If the host is busy and the user did not ask to force it, stop and
report the blocker.

## Published PyPI Baselines

These modes are not local git-ref benchmarks. They measure exact installable
PyPI artifacts and cache results by package version and workload hash.

Stable Retro oracle:

```bash
.venv/bin/python scripts/run_pypi_stable_retro_turbo_host_benchmark.py \
  --rom-path /path/to/SuperMarioBros.nes
```

Published SuperMarioBros-Nes-turbo:

```bash
.venv/bin/python scripts/run_pypi_supermariobrosnes_turbo_host_benchmark.py \
  --rom-path /path/to/SuperMarioBros.nes
```

Use cached PyPI baselines until PyPI publishes a newer version or the workload
hash changes.

## Interpreting Results

For single-ref baselines:

- clean: checkpoint stability span below `0.25%`, run-median CV below `0.75%`,
  CI width below `0.5%`, no outliers, host load clean
- noisy/tentative: run-median CV above `1.5%`, flagged outliers, failed load
  gate, or `decision=max_samples_no_convergence`

For comparisons:

- `median_pair_ratio < 1.01`: no meaningful win
- `1.01 <= median_pair_ratio < 1.03`: accept only if CI lower bound is above
  `1.00` and pair direction is stable
- `1.03 <= median_pair_ratio < 1.05`: small fixed-host win if faster-pair
  count, CV, outlier, and load gates are clean
- `median_pair_ratio >= 1.05`: likely real if correctness checks pass and host
  stayed calm
- `median_pair_ratio >= 1.10`: strong throughput result if correctness checks
  pass

Runner decisions:

- `converged`: stable single-ref baseline
- `converged_candidate_win`: stable comparison win
- `converged_no_meaningful_win`: stable reject, do not keep sampling for luck
- `continue`: only appears in intermediate aggregate state
- `max_samples_no_convergence`: noisy/tentative; rerun on calmer host before
  using as a merge/release baseline

## Reporting

Before the final answer, run:

```bash
git status --short
```

Report:

- whether the fixed-host protocol completed
- execution target and absolute run directory
- local result bundle path, especially `aggregate.json`
- refs and SHAs
- dirty local files excluded
- measured count and warmup count
- for single-ref: official median SPS, mean SPS, CI, run-median CV,
  all-sample CV, decision, validity gates
- for comparison: median pair ratio, mean pair ratio, CI, faster-pair count,
  paired gain percent, decision, validity gates
- host load before/after and any obvious competing process
- whether cleanup/finalization ran

Keep smoke results clearly labeled as setup validation, not official throughput.
