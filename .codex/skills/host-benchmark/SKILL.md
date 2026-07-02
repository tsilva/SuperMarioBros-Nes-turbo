---
name: host-benchmark
description: Benchmark one SuperMarioBros-Nes-turbo commit or compare two commits on the fixed beast-3-local CPU host for reliable throughput. Use when the user asks to benchmark, compare, confirm, or measure env_steps_per_sec on beast-3/local host, especially for exact-ref single-run baselines or deciding whether a candidate commit is faster than a baseline.
---

# Host Benchmark

## Contract

This skill benchmarks exact committed refs on `beast-3-local` for fixed-host CPU
throughput.

Input forms:

```text
single_ref only
candidate_ref
baseline_ref candidate_ref
pypi-stable-retro-turbo
```

If the user explicitly asks to benchmark one ref only, run single-ref mode on
that ref. If the user provides one ref without saying "only", "single", "no
comparison", or equivalent, baseline is the latest local `main` commit and the
provided ref is the candidate. If two refs are provided, the first is baseline
and the second is candidate. If the mode is ambiguous, ask one short clarifying
question before running.

If the user asks for the latest published `stable-retro-turbo` PyPI baseline,
use the "Published stable-retro-turbo Oracle Baseline" mode below. That mode is
not a git-ref benchmark; it measures the latest exact PyPI version once per
workload hash and caches the result locally.

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

Each benchmark must create a unique run directory:

```text
/home/tsilva/SuperMarioBros-Nes-turbo-host-bench/runs/host-compare-YYYY-MM-DD-HHMMSS-BBASE-CCAND
/home/tsilva/SuperMarioBros-Nes-turbo-host-bench/runs/host-single-YYYY-MM-DD-HHMMSS-RREF
```

During the run, keep all extracted sources, virtualenvs, raw samples, host
metadata, and aggregate JSON inside that run directory. Never reuse a previous
run directory. Never edit another run's files. After the benchmark has a valid
`aggregate.json`, run the local finalizer in the "Result Retention And Cleanup"
section so the durable result is copied back to this checkout and remote build
bulk is removed.

Use per-ref isolated source directories and venvs. For single-ref mode, use only
`sources/ref/.venv`; for comparisons, use baseline/candidate trees:

```text
sources/baseline
sources/candidate
sources/ref
sources/baseline/.venv
sources/candidate/.venv
sources/ref/.venv
```

Use `uv sync --frozen --no-dev` in each extracted source directory. This keeps
runtime dependencies minimal and avoids the heavy dev/Torch stack. Build from
the checked-in `pyproject.toml`/`uv.lock`; do not `apt install` packages or use
system Python packages. If Rust/Cargo is missing, use the existing user-local
rustup setup if present. If Rust is truly absent, ask before installing it.

## Workload

Default fixed-host workload:

- `RAYON_NUM_THREADS=12`
- `num_envs=16`
- `steps=50000`
- `repeats=3`
- `warmup_pairs=2`, discarded
- `measured_pairs=11`
- single-ref quick official: `warmup_invocations=1`, `measured_invocations=5`
- single-ref full official: `warmup_invocations=2`, `measured_invocations=11`
- comparison quick official: `warmup_pairs=1`, `measured_pairs=7`
- comparison full official: `warmup_pairs=2`, `measured_pairs=11`
- `frame_skip=4`
- `frame_stack=4`
- grayscale, crop top 32, resize 84x84
- action `noop`
- states `Level1-1,Level1-2,Level1-3,Level1-4`
- observation contract `obs_shape=(16, 4, 84, 84)`, `obs_dtype=uint8`

This host protocol is intentionally stronger than a smoke benchmark but should
not waste wall-clock time on already-clean results. Use the quick official tier
first, then escalate to the full official tier only when the preregistered
validity gates below fail or the user explicitly asks for a final/high-confidence
run.

## Load Gate

Before benchmarking, record:

```bash
ssh beast-3-local 'hostname; uptime; nproc; lscpu | sed -n "1,40p"; ps -eo pid,pcpu,pmem,comm,args --sort=-pcpu | head -20'
```

If `beast-3-local` resolves to an unreachable LAN address, try the known tailnet
route before giving up:

```bash
ssh -o HostKeyAlias=beast-3-local beast-3.tail50040f.ts.net 'hostname && whoami'
```

Use that route only when the host key matches the existing `beast-3-local` key,
and record the route used in `aggregate.json` and the final report.

If the 1-minute load is above about `4` on this 12-CPU host, or an obvious CPU
training/build job is active, report that the host is busy and defer unless the
user explicitly wants to run anyway. A quick smoke check is okay; final timing
should use a calm host.

## Setup

From the local repo root:

1. Run `git status --short`.
2. Resolve the single ref or baseline/candidate SHAs with
   `git rev-parse --verify REF^{commit}`.
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
git archive --format=tar REF_SHA | gzip -n > artifacts/benchmarks/host-archives/ref-REF_SHA12.tar.gz
git archive --format=tar BASELINE_SHA | gzip -n > artifacts/benchmarks/host-archives/baseline-BASELINE_SHA12.tar.gz
git archive --format=tar CANDIDATE_SHA | gzip -n > artifacts/benchmarks/host-archives/candidate-CANDIDATE_SHA12.tar.gz
```

These local archive files are temporary convenience artifacts. After the final
result bundle is copied back, remove archive files for the benchmarked SHAs; the
exact archives are reproducible from Git commits.

## Remote Run Shape

On `beast-3-local`, create a new run directory, copy archives into it, extract
to isolated source trees, run `uv sync --frozen --no-dev` inside each source,
and verify a smoke benchmark before measured runs.

Single-ref mode:

- Extract to `sources/ref`.
- Run `uv sync --frozen --no-dev`.
- Run one smoke invocation with `steps=1000`, `repeats=1`.
- Run quick official: 1 discarded warmup invocation, then 5 measured
  invocations.
- Stop after quick official only if all validity gates pass:
  - invocation-median CV is below `0.75%`
  - all-sample CV is below `1.25%`
  - bootstrap CI width for median SPS is below `0.75%` of median SPS
  - no IQR/MAD outliers are flagged in invocation medians
  - host 1-minute load stayed below about `4` and no obvious competing CPU job
    appeared
- If any gate fails, escalate in the same run directory to full official by
  collecting enough additional measured invocations to reach 11 measured total
  and at least 2 discarded warmups total.

For single-ref mode, the official metric is the median of measured invocation
medians. Also report mean of invocation medians, bootstrap 95% CI over
invocation medians, all-sample summary, CVs, and outlier diagnostics.

Run warmup and measured invocations in alternating paired order:

```text
warmup pair 0: baseline then candidate
warmup pair 1: candidate then baseline
measured pair 0: baseline then candidate
measured pair 1: candidate then baseline
measured pair 2: baseline then candidate
...
```

For comparisons, run quick official first: 1 discarded warmup pair and 7
measured pairs. Stop after quick official only if the result is clearly decided:

- for a claimed win, `median_pair_ratio >= 1.03`
- bootstrap 95% lower bound is above `1.00`
- at least `6/7` pairs favor the candidate
- baseline and candidate run-median CVs are each below `1.5%`
- no IQR/MAD outliers are flagged in pair ratios
- host 1-minute load stayed below about `4` and no obvious competing CPU job
  appeared

Otherwise escalate in the same run directory to full official by collecting
enough additional measured pairs to reach 11 measured pairs and at least 2
discarded warmup pairs total. Do not change the official statistic based on the
observed data; the escalation rule is preregistered here.

For each invocation, save the raw JSON from `scripts/benchmark_sps.py`. Compute
both the invocation mean and median from the raw `runs[*].env_steps_per_sec`;
if stdout is also redirected to a JSON mirror, name it `*.stdout.json` and never
include those mirrors in aggregation or retention. Aggregate only the canonical
raw files named by the protocol, such as `measured-ref-00.json`,
`warmup-baseline-00.json`, or `measured-candidate-10.json`.
The paired decision statistic must use the invocation median so one bad repeat
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

Use a reusable Python heredoc or checked-in helper when available, not a
freshly-invented ad hoc aggregator. This avoids token-heavy prompt code and
reduces wrapper bugs. The helper should accept mode, refs, run directory, route,
warmup count, measured count, and escalation policy, then write raw files and
`aggregate.json`.

Use Python on the host to aggregate raw JSON into:

```text
aggregate.json
```

Minimum `aggregate.json` fields:

- mode (`single_ref_fixed_host` or `paired_compare_fixed_host`)
- refs and SHAs
- source archive SHA-256 values
- ROM SHA-256 and state file SHA-256 values
- host metadata and load snapshots
- SSH route used, including tailnet fallback when applicable
- command and environment (`RAYON_NUM_THREADS`)
- warmup raw files
- per-invocation mean and median env steps/sec
- single-ref summary fields when in single-ref mode: official median SPS, mean
  invocation-median SPS, bootstrap CI, run-median summary, all-sample summary,
  validity gates, and whether quick official stopped or escalated
- comparison summary fields when in comparison mode: per-pair baseline/candidate
  medians and `pair_ratio`, `median_pair_ratio`, `mean_pair_ratio`, paired gain
  percent, `pair_ratio_bootstrap_ci95`, `candidate_faster_pairs`,
  `measured_pairs`, baseline/candidate run-median summaries, and
  baseline/candidate all-sample summaries
- flagged outlier diagnostics, if any, without changing the official metric
- raw result file paths

## Result Retention And Cleanup

The host benchmark root must not grow without bound. The durable source of truth
after a completed benchmark is a local result bundle, not the remote extracted
source tree or copied tarballs.

After writing and validating `aggregate.json`, run the checked-in finalizer from
the local repo root:

```bash
python3 scripts/finalize_host_benchmark.py \
  --ssh-target beast-3-local \
  --remote-run-dir /home/tsilva/SuperMarioBros-Nes-turbo-host-bench/runs/RUN_NAME \
  --purge-remote-bulk \
  --purge-local-archives
```

If the benchmark used the tailnet fallback, pass the same route:

```bash
python3 scripts/finalize_host_benchmark.py \
  --ssh-target beast-3.tail50040f.ts.net \
  --host-key-alias beast-3-local \
  --remote-run-dir /home/tsilva/SuperMarioBros-Nes-turbo-host-bench/runs/RUN_NAME \
  --purge-remote-bulk \
  --purge-local-archives
```

The finalizer must:

- refuse to operate on paths outside
  `/home/tsilva/SuperMarioBros-Nes-turbo-host-bench/runs/host-*`
- copy `aggregate.json`, canonical `raw/*.json` files except
  `*.stdout.json`, and `raw/*.txt` load snapshots into
  `artifacts/benchmarks/host-results/RUN_NAME/`
- parse every copied JSON file and write
  `artifacts/benchmarks/host-results/RUN_NAME/manifest.json` with SHA-256
  checksums before deleting anything remote
- append a compact record to `artifacts/benchmarks/host-results/index.jsonl`
- remove remote `sources/`, remote `archives/`, and duplicate
  `raw/*.stdout.json` only after the local manifest exists
- remove local `artifacts/benchmarks/host-archives/*SHA12*.tar.gz` files for
  the benchmarked SHAs

Keep the remote `aggregate.json` and canonical `raw/` files as a tiny breadcrumb
unless the user explicitly asks to delete the entire run directory. Do not
delete the shared state directory or ROM. Do not `rm -rf ~/.cache/uv`; if UV
cache pressure becomes material, use `uv cache prune` explicitly and report it
as shared-cache cleanup.

Before the final report, verify the local bundle exists and point the user to
the local `aggregate.json` first. Report the remote run directory as cleaned if
`sources/` and `archives/` were removed.

## Published stable-retro-turbo Oracle Baseline

Use this special mode when the user wants to compare this specialized
`SuperMarioBros-Nes-turbo` environment against the latest published
`stable-retro-turbo` PyPI wheel on the same ROM, same state set, same
preprocessing, and same fixed host. The point is to show that this repo preserves
the stable-retro-turbo behavior contract for `SuperMarioBros-Nes-v0` while being
faster.

Run from the local repo root:

```bash
python3 scripts/run_pypi_stable_retro_turbo_host_benchmark.py \
  --ssh-target beast-3-local
```

If the direct LAN route is unavailable, use the tailnet route:

```bash
python3 scripts/run_pypi_stable_retro_turbo_host_benchmark.py \
  --ssh-target beast-3.tail50040f.ts.net \
  --host-key-alias beast-3-local
```

The runner must:

- query `https://pypi.org/pypi/stable-retro-turbo/json` and resolve the latest
  exact version at runtime
- use Python `3.14` on the host, because current published turbo wheels are
  Python-version-specific
- create an isolated remote venv and install exactly
  `stable-retro-turbo==VERSION` from PyPI
- run `scripts/benchmark_stable_retro_turbo_pypi.py` with the host workload:
  `num_envs=16`, `num_threads=12`, `steps=50000`, `repeats=3`,
  2 warmup invocations, 11 measured invocations, frame skip 4, frame stack 4,
  crop top 32, resize 84x84, grayscale, states
  `Level1-1,Level1-2,Level1-3,Level1-4`, action `noop`
- cache results under
  `artifacts/benchmarks/host-results/pypi-stable-retro-turbo/VERSION/WORKLOAD_HASH/`
- write `aggregate.json`, `manifest.json`, raw invocation JSON, load snapshots,
  and `index.jsonl`
- return a cache hit without touching `beast-3-local` when the exact
  version/workload hash already has an aggregate, unless `--force` is passed
- purge the remote PyPI run directory after copying and validating the local
  cache, unless `--keep-remote` is passed

The PyPI oracle benchmark intentionally uses a tiny local
`stable_baselines3.common.vec_env.VecEnv` shim if SB3 is absent, matching the
repo's parity-test approach. Do not install the heavy SB3/Torch stack just to
time `RetroVecEnv`.

When both local SuperMario and PyPI oracle aggregates exist, report:

```text
speedup = supermariobrosnes_turbo_official_median_sps / stable_retro_turbo_pypi_official_median_sps
```

Use cached PyPI baselines for future comparisons until PyPI publishes a newer
version or the workload hash changes.

## Interpreting Results

On this host, validated single-ref protocol examples have produced:

```text
older long sample: mean 40695 env_steps/sec, run-mean CV 0.88%
main 17c60e1eb88e quick/full-style sample: median 47553 env_steps/sec,
run-median CV 0.34%
```

For single-ref benchmark results, treat:

- run-median CV below `0.75%` and CI width below `0.75%` of the median as a
  clean official result
- run-median CV `0.75-1.5%` as usable but worth escalating if the result will
  drive a merge/release decision
- run-median CV above `1.5%`, flagged outliers, or host load above the gate as
  noisy/tentative; rerun on a calmer host before using it as a baseline

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

For a final high-confidence single-ref baseline, use:

- `steps=50000`
- `warmup_invocations=3`, discarded
- `measured_invocations=21`
- same median-of-invocation-medians, bootstrap CI, CV, and outlier-flagging
  rules
- record host load snapshots before setup, before measured timing, mid-run, and
  after timing

## Reporting

Report:

- whether the fixed-host benchmark worked
- absolute remote run directory
- mode and ref SHAs
- command shape and workload knobs
- measured count and warmup count
- for single-ref mode: official median SPS, mean SPS from invocation medians,
  bootstrap CI, invocation-median CV, all-sample CV, validity gates, and whether
  the quick tier stopped or escalated
- for comparison mode: median pair ratio, mean pair ratio, bootstrap CI,
  candidate-faster pair count, paired gain percent, and verdict
- mean/median/stdev/min/max/CV env steps/sec from invocation medians
- all-sample CV
- flagged outlier diagnostics, if any, without changing the official metric
- host load before/after and any obvious competing processes
- whether dirty local files were excluded
- whether remote setup changed anything persistent
- local result bundle path under `artifacts/benchmarks/host-results/`
- whether remote bulk cleanup ran and what remains on `beast-3-local`

End comparison reports with a compact table:

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
| Artifact |  |  | LOCAL_AGGREGATE_JSON |
```

End single-ref reports with a compact table:

```text
| Metric | Ref |
| --- | ---: |
| Ref | REF |
| SHA | SHA12 |
| Official Median SPS | MEDIAN |
| Mean SPS | MEAN |
| Bootstrap CI | CI95 LOW-HIGH |
| Run-Median Stdev | STDEV |
| Run-Median CV | CV% |
| All-Sample CV | CV% |
| Range SPS | MIN-MAX |
| Validity | quick stopped / escalated; gates |
| Host Load | LOAD |
| Artifact | LOCAL_AGGREGATE_JSON |
```

Run `git status --short` before the final answer so local protocol or skill
changes are not hidden.
