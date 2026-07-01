---
name: host-benchmark
description: Compare two SuperMarioBros-Nes-turbo commits on the fixed beast-3-local CPU host for reliable throughput. Use when the user asks to benchmark, compare, confirm, or measure env_steps_per_sec on beast-3/local host instead of Modal, especially for deciding whether a candidate commit is faster than a baseline without Modal host variance.
---

# Host Benchmark

## Contract

This skill compares exact committed refs on `beast-3-local`. It is for fixed-host
CPU throughput, not clean-machine Modal validation.

Input forms:

```text
candidate_ref
baseline_ref candidate_ref
```

If one ref is provided, baseline is the latest local `main` commit and the
provided ref is the candidate. If two refs are provided, the first is baseline
and the second is candidate. If no candidate is provided, ask for it.

Do not switch local branches. Resolve refs locally with `git rev-parse`, create
exact `git archive` snapshots, and copy those snapshots to `beast-3-local`.
Local dirty or untracked files are not included unless the user commits them
and passes that commit/ref.

## Host And Isolation

Use this fixed host and persistent benchmark root:

```text
host: beast-3-local
root: /home/tsilva/SuperMarioBros-Nes-turbo-host-bench
rom:  /home/tsilva/roms/NES/mapper-000-NROM/SuperMarioBros-Nes-v0.nes
states: /home/tsilva/SuperMarioBros-Nes-turbo-host-bench/states/SuperMarioBros-Nes-v0
```

Each comparison must create a unique run directory:

```text
/home/tsilva/SuperMarioBros-Nes-turbo-host-bench/runs/host-compare-YYYY-MM-DD-HHMMSS-BBASE-CCAND
```

Inside that run directory, keep all extracted sources, virtualenvs, raw samples,
host metadata, and aggregate JSON. Never reuse a previous run directory. Never
edit another run's files. Do not delete old runs unless the user explicitly asks
for cleanup.

Use per-ref isolated source directories and venvs:

```text
sources/baseline
sources/candidate
sources/baseline/.venv
sources/candidate/.venv
```

Use `uv sync --frozen --no-dev` in each extracted source directory. This keeps
runtime dependencies minimal and avoids the heavy dev/Torch stack. Build from
the checked-in `pyproject.toml`/`uv.lock`; do not `apt install` packages or use
system Python packages. If Rust/Cargo is missing, use the existing user-local
rustup setup if present. If Rust is truly absent, ask before installing it.

## Workload

Default fixed-host comparison workload:

- `RAYON_NUM_THREADS=12`
- `num_envs=16`
- `steps=50000`
- `repeats=3`
- `warmup_pairs=2`, discarded
- `measured_pairs=11`
- `frame_skip=4`
- `frame_stack=4`
- grayscale, crop top 32, resize 84x84
- action `noop`
- states `Level1-1,Level1-2,Level1-3,Level1-4`
- observation contract `obs_shape=(16, 4, 84, 84)`, `obs_dtype=uint8`

This host protocol is intentionally stronger than a quick smoke benchmark: the
host is free and fixed, so spend time on longer same-machine paired sampling
rather than Modal-style host-lottery mitigation. Use `/modal-benchmark` when a
clean-machine external validation is required.

## Load Gate

Before benchmarking, record:

```bash
ssh beast-3-local 'hostname; uptime; nproc; lscpu | sed -n "1,40p"; ps -eo pid,pcpu,pmem,comm,args --sort=-pcpu | head -20'
```

If the 1-minute load is above about `4` on this 12-CPU host, or an obvious CPU
training/build job is active, report that the host is busy and defer unless the
user explicitly wants to run anyway. A quick smoke check is okay; final timing
should use a calm host.

## Setup

From the local repo root:

1. Run `git status --short`.
2. Resolve baseline and candidate SHAs with `git rev-parse --verify REF^{commit}`.
3. Verify SSH:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 beast-3-local 'hostname && whoami'
```

4. Ensure the benchmark root and state directory exist. If the four state files
are missing on the host, copy them from the sibling checkout:

```bash
ssh beast-3-local 'mkdir -p ~/SuperMarioBros-Nes-turbo-host-bench/states/SuperMarioBros-Nes-v0'
rsync -az /Users/tsilva/repos/tsilva/stable-retro-turbo/stable_retro/data/stable/SuperMarioBros-Nes-v0/Level1-{1,2,3,4}.state \
  beast-3-local:~/SuperMarioBros-Nes-turbo-host-bench/states/SuperMarioBros-Nes-v0/
```

5. Create exact archives without switching branches:

```bash
mkdir -p artifacts/benchmarks/host-archives
git archive --format=tar BASELINE_SHA | gzip -n > artifacts/benchmarks/host-archives/baseline-BASELINE_SHA12.tar.gz
git archive --format=tar CANDIDATE_SHA | gzip -n > artifacts/benchmarks/host-archives/candidate-CANDIDATE_SHA12.tar.gz
```

These local archive files are temporary convenience artifacts. Leave them unless
the user asks for cleanup.

## Remote Run Shape

On `beast-3-local`, create a new run directory, copy archives into it, extract
to `sources/baseline` and `sources/candidate`, run `uv sync --frozen --no-dev`
inside each source, and verify a smoke benchmark before measured runs.

Run warmup and measured invocations in alternating paired order:

```text
warmup pair 0: baseline then candidate
warmup pair 1: candidate then baseline
measured pair 0: baseline then candidate
measured pair 1: candidate then baseline
measured pair 2: baseline then candidate
...
```

For each invocation, save the raw JSON from `scripts/benchmark_sps.py`. Compute
both the invocation mean and median from the raw `runs[*].env_steps_per_sec`;
the paired decision statistic must use the invocation median so one bad repeat
inside a single invocation does not dominate. For each measured pair, compute:

```text
pair_ratio = candidate_median_env_steps_per_sec / baseline_median_env_steps_per_sec
```

The official comparison metric is:

```text
median_pair_ratio
```

Also compute a bootstrap 95% confidence interval over the measured pair ratios,
the count of candidate-faster pairs, mean pair ratio, paired gain percent,
baseline/candidate run-median summary, all-sample summary, and coefficient of
variation. Do not delete statistical outliers by default; flag possible outliers
with a median-absolute-deviation or IQR rule and optionally show a diagnostic
with/without flagged points, while keeping the preregistered official metric as
`median_pair_ratio`.

## Recommended Remote Command Skeleton

Prefer generating a small shell/Python heredoc for the exact refs and run path
rather than editing files on the host. The core benchmark command inside each
source is:

```bash
RAYON_NUM_THREADS=12 .venv/bin/python scripts/benchmark_sps.py \
  --rom-path /home/tsilva/roms/NES/mapper-000-NROM/SuperMarioBros-Nes-v0.nes \
  --state-dir /home/tsilva/SuperMarioBros-Nes-turbo-host-bench/states/SuperMarioBros-Nes-v0 \
  --num-envs 16 \
  --steps 50000 \
  --repeats 3 \
  --json
```

Use Python on the host to aggregate raw JSON into:

```text
aggregate.json
```

Minimum `aggregate.json` fields:

- baseline/candidate refs and SHAs
- source archive SHA-256 values
- ROM SHA-256 and state file SHA-256 values
- host metadata and load snapshots
- command and environment (`RAYON_NUM_THREADS`)
- warmup raw files
- per-invocation mean and median env steps/sec
- per-pair baseline/candidate medians and `pair_ratio`
- `median_pair_ratio`, `mean_pair_ratio`, paired gain percent
- `pair_ratio_bootstrap_ci95`, `candidate_faster_pairs`, `measured_pairs`
- baseline and candidate run-median summaries: mean, median, stdev, min, max, CV
- baseline and candidate all-sample summaries
- flagged outlier diagnostics, if any, without changing the official metric
- raw result file paths

## Interpreting Results

On this host, the validated long-sample single-ref protocol produced about:

```text
mean: 40695 env_steps/sec
run-mean CV: 0.88%
```

For commit comparisons, treat:

- `median_pair_ratio < 1.01`: no meaningful win
- `1.01 <= median_pair_ratio < 1.03`: accept only if the 95% bootstrap lower
  bound is above `1.00` and pair ratios mostly agree
- `1.03 <= median_pair_ratio < 1.05`: accept as a small fixed-host win if at
  least `8/11` measured pairs favor the candidate and CV is clean
- `median_pair_ratio >= 1.05`: likely real if correctness checks pass, the host
  stayed calm, and pair ratios mostly agree
- `median_pair_ratio >= 1.10`: strong throughput result if correctness checks
  pass

If either commit's run-median CV is above `2%`, the bootstrap CI crosses `1.0`
for a claimed win, fewer than `8/11` pairs favor the candidate, or pair ratios
disagree materially in sign, mark the result noisy or tentative and rerun when
host load is lower.

For a final high-confidence run before merging a subtle optimization, use:

- `steps=50000`
- `warmup_pairs=3`, discarded
- `measured_pairs=21`
- same median repeat, median pair ratio, bootstrap CI, and outlier-flagging
  rules
- require at least `15/21` measured pairs to favor the candidate for a positive
  call
- record host load snapshots before setup, before measured timing, mid-run, and
  after timing

## Reporting

Report:

- whether the fixed-host benchmark worked
- absolute remote run directory
- baseline/candidate refs and SHAs
- command shape and workload knobs
- pair count and warmup count
- median pair ratio, mean pair ratio, bootstrap CI, candidate-faster pair count,
  paired gain percent, and verdict
- baseline and candidate mean/median/stdev/min/max/CV env steps/sec from
  invocation medians
- all-sample CV for each ref
- flagged outlier diagnostics, if any, without changing the official metric
- host load before/after and any obvious competing processes
- whether dirty local files were excluded
- whether remote setup changed anything persistent

End with a compact table:

```text
| Metric | Baseline | Candidate | Paired / Notes |
| --- | ---: | ---: | --- |
| Ref | BASELINE_REF | CANDIDATE_REF | exact git archive snapshots |
| SHA | BASELINE_SHA12 | CANDIDATE_SHA12 | dirty local files excluded |
| Mean SPS | BASELINE_MEAN | CANDIDATE_MEAN | median pair ratio RATIO |
| Median SPS | BASELINE_MEDIAN | CANDIDATE_MEDIAN | mean pair ratio RATIO |
| Bootstrap CI |  |  | CI95 LOW-HIGH, faster FAST/PAIRS |
| Run-Median Stdev | BASELINE_STDEV | CANDIDATE_STDEV | gain GAIN% |
| Run-Median CV | BASELINE_CV% | CANDIDATE_CV% | PAIR_COUNT measured pairs |
| Range SPS | BASELINE_MIN-BASELINE_MAX | CANDIDATE_MIN-CANDIDATE_MAX | host load LOAD |
| Artifact |  |  | REMOTE_AGGREGATE_JSON |
```

Run `git status --short` before the final answer so local protocol or skill
changes are not hidden.
