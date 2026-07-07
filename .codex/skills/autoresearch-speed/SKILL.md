---
name: autoresearch-speed
description: Profiler-first Super Mario Bros emulator speed-improvement loop for this repo. Use when optimizing, profiling, benchmarking, or autonomously iterating on Super Mario Bros NES throughput with fast local diagnosis, exact-ref triage, official acceptance gates, and compact experiment tracking.
---

# Autoresearch Speed

## Contract

Read `SPECS.md` in this skill directory before optimizing. Always obey its
`PURPOSE` and `REQUIREMENTS`; do not accept speedups that violate them. Never
modify `SPECS.md` during an autoresearch-speed run.

Optimize the live repo on the current branch. Use local benchmarks in this
checkout unless the user explicitly approves another target. Do not use SSH,
tailnet, cloud, Modal, or benchmark skills for autoresearch timing unless the
current turn explicitly authorizes that.

Maximize the benchmark output variable `env_steps_per_sec`: biggest valid gain,
least wall-clock time, fewest useful tokens. Spend official benchmark time only
on candidates with a concrete mechanism and cheap evidence that they can win.

Required path variables, from CLI flags, environment, or `.env`:

- `ROM_PATH`: SMB NES ROM.
- `AUTORESEARCH_ROOT_PATH`: existing directory for all mutable autoresearch
  state; scripts must fail if it is unset, missing, or not a directory.

Derived paths under `AUTORESEARCH_ROOT_PATH`:

- `benchmarks/`: benchmark run root, archives, result caches, profiles.
- `states/SuperMarioBros-Nes-v0/`: benchmark state cache.
- `candidates/`: compact patches/rejections.
- `current.json`: active campaign state.
- `results.tsv`: trial and decision ledger.
- `ideas.md`: durable ideas queue.
- `scratchpad.md`: unreviewed loop-improvement notes.

Use `--run-root`, `--state-dir`, or `--state-source` only for one-off overrides.
If states are missing, populate the derived state cache or pass `--state-source`.

## Loop

1. `orient_once`: verify git state/current branch, read ledger and ideas, inspect
   only files needed for the active hypothesis.
2. `diagnose`: use profiler output, smoke timings, source inspection, or narrow
   equivalence checks to kill weak ideas cheaply.
3. `prototype`: make one small edit, build only what is needed, revise or stop
   immediately when the mechanism is wrong.
4. `screen`: commit only plausible candidates, then run exact-ref
   `local_triage`; never triage speculative broad patches.
5. `accept`: run required checks and `local_acceptance` only after triage
   promotes a candidate or the user asks for official evidence.
6. `record`: write the ledger row or rejection before starting the next idea.
7. `retrospect`: append compact lessons to `AUTORESEARCH_ROOT_PATH/scratchpad.md`
   when a mistake, hindsight conclusion, friction, missing check, or heuristic
   would make the next run stronger. Do not update this skill or docs from
   scratchpad notes unless the user asks.

Fast local diagnosis is learning evidence only:

```bash
RAYON_NUM_THREADS=12 .venv/bin/python scripts/benchmark_sps.py \
  --num-envs 16 --steps 1000 --repeats 1 --warmup 20 \
  --frame-skip 4 --frame-stack 4 --crop-top 32 --crop-bottom 0 \
  --resize-width 84 --resize-height 84 \
  --states Level1-1,Level1-2,Level1-3,Level1-4 \
  --action-set simple --action noop --no-start-game \
  --json --output-json "$AUTORESEARCH_ROOT_PATH/benchmarks/local-diagnosis.json"

RAYON_NUM_THREADS=12 .venv/bin/python scripts/benchmark_sps.py \
  --num-envs 16 --steps 2000 --repeats 1 --warmup 100 \
  --frame-skip 4 --frame-stack 4 --crop-top 32 --crop-bottom 0 \
  --resize-width 84 --resize-height 84 \
  --states Level1-1,Level1-2,Level1-3,Level1-4 \
  --action-set simple --action noop --no-start-game \
  --profile-output "$AUTORESEARCH_ROOT_PATH/benchmarks/local-profile.json" \
  --json --output-json "$AUTORESEARCH_ROOT_PATH/benchmarks/local-profile-benchmark.json"
```

After native edits, run `.venv/bin/python -m maturin develop --release` before
Python diagnosis. Do not run `make benchmark` by reflex; use it only when it is
the cheapest targeted check for the active question.

## Benchmark Funnel

- `local_diagnosis`: uncommitted profiling, smoke tests, narrow checks. Stop
  when the profile or smoke result makes a win implausible.
- `local_triage`: exact committed refs, shorter paired runner settings, same
  ROM/state/workload/load policy. Screening only.
- `local_acceptance`: official exact-ref paired benchmark. Only this tier can
  justify `keep`, `keep_small_gain`, accepted speedups, or baseline updates.

Default triage:

```bash
.venv/bin/python scripts/run_git_ref_benchmark.py <baseline_ref> <candidate_ref> --steps 5000 --repeats 1 --warmups 0 --max-measured-invocations 3
```

Official acceptance:

```bash
.venv/bin/python scripts/run_git_ref_benchmark.py <baseline_ref> <candidate_ref> --steps 50000 --repeats 3
```

With explicit user limits:

```bash
.venv/bin/python scripts/run_git_ref_benchmark.py <baseline_ref> <candidate_ref> --steps 50000 --repeats 3 --max-measured-invocations <N> --max-wall-clock-minutes <minutes>
```

Use `--dry-run` before expensive acceptance to confirm tier, refs, checkpoint
ladder, limits, ROM/state paths, and load policy. Dry-run `workload_hash` is
only a `planned_workload_hash`; never compare it to aggregate hashes.

Triage rules:

- One default triage per candidate; a second run is only for noisy-but-promising
  evidence and usually uses `--steps 10000`.
- Below baseline, or below `+1%` with unstable samples: discard or revise.
- `+3%` or better: run checks and acceptance unless risky or contract-sensitive.
- `+1%` to `+3%`: escalate only for simple, low-risk, composable changes.
- Never accept from triage evidence alone.

Acceptance statuses:

- `keep`: required checks passed; aggregate has
  `decision=converged_candidate_win`, `validity_passed=true`, contract checks
  passed, and `load_gate_passed=true` or user-approved
  `load_gate_ignored_for_validity=true`.
- `keep_small_gain`: required checks passed; all `stability_gates` true;
  `median_pair_ratio > 1.0`; `pair_ratio_bootstrap_ci95[0] >= 1.0`;
  `candidate_faster_pairs >= candidate_faster_pairs_required_for_win`; change is
  simple, low-risk, simplifying, composable, or plausibly compounding.
- `discard`: equal/slower/no meaningful win/noisy/too complex/contract
  weakening.
- `inconclusive`: malformed metadata, load failure, missing ROM, incomparable
  outputs, skipped required coverage, limit stop without valid gates, or noise.

Run at most one benchmark at a time. Cheap screening may run on a busy machine;
official acceptance is blocked by load unless the user explicitly forces it. If
load fails after a checkpoint, stop before the next measured sample and treat
`limit_stop_reason=load_gate_failed` as inconclusive.

## State And Ledger

Work on the current branch. Do not create/switch branches, fork subagents,
create worker worktrees, merge, push, or delete branches unless the user
explicitly approves it in the current turn.

Before work:

1. Verify git state and current branch.
2. If resuming, read `AUTORESEARCH_ROOT_PATH/current.json` and
   `AUTORESEARCH_ROOT_PATH/results.tsv`.
3. Verify needed refs exist: root SHA, current baseline, accepted commits,
   candidate commits. If missing, ask whether to recover or migrate the ledger.
4. If unrelated dirty changes would be carried in, stop and ask.
5. Inspect hot-path files once per session, then read only hypothesis-relevant
   files. Defaults: `scripts/benchmark_sps.py`,
   `python/supermariobrosnes_turbo/env.py`, `src/py_api.rs`, `src/vec_env.rs`,
   `src/emulator.rs`, `Cargo.toml`, `pyproject.toml`, relevant docs.

Track every trial in `AUTORESEARCH_ROOT_PATH/current.json`,
`AUTORESEARCH_ROOT_PATH/results.tsv`, `AUTORESEARCH_ROOT_PATH/ideas.md`,
optional `AUTORESEARCH_ROOT_PATH/candidates/`, and accepted aggregate/commit
records. Keep mutable research metadata outside the repo unless the user asks
to migrate or commit a reviewed artifact.

Preferred `results.tsv` header for new rows:

```text
epoch	commit	baseline_commit	mode	benchmark_tier	workload_hash	measured_invocation_count	measured_pairs	official_median_sps	mean_invocation_median_sps	bootstrap_ci95_invocation_median_sps	median_pair_ratio	mean_pair_ratio	pair_ratio_bootstrap_ci95	candidate_faster_pairs	candidate_faster_pairs_required_for_win	validity_passed	load_gate_passed	load_gate_ignored_for_validity	limit_stop_reason	previous_limit_stop_reason	benchmark_limits	discarded_incomplete_pair_raw_files	expected_rom_sha256	rom_sha256	state_sha256	decision	status	description	artifact
```

Column names from benchmark output must match aggregate/index JSON field names
exactly. Do not split, rename, or alias emitted benchmark fields. The benchmark
runner's `RESULTS_TSV_COLUMNS` is the source of truth.

Statuses: `baseline`, `triage_discard`, `triage_promote`, `keep`,
`keep_small_gain`, `discard`, `skip`, `crash`, `regression_fixed_keep`,
`regression_unfixed_discard`, `inconclusive`.

If `AUTORESEARCH_ROOT_PATH/ideas.md` has no high- or medium-ROI ready idea, do
not spend official benchmark time on leftovers. Refresh profiler evidence, add
a targeted idea, or ask before continuing.

## Checks And Optimization Guidance

Before every `local_acceptance`:

```bash
cargo fmt --check
cargo check --release
.venv/bin/python -m maturin develop --release
make test
```

Before `local_triage`, run the cheapest checks matching the changed surface; for
Rust/PyO3 hot-path changes default to `cargo fmt --check`, `cargo check
--release`, and `.venv/bin/python -m maturin develop --release`.

Add targeted tests before triage when touching observations, rewards,
termination, reset behavior, action mapping, info fields, preprocessing bytes,
state loading, or Python API contracts. Treat failures as regressions unless
proven unrelated; fix first, then rerun the failing test and required gates. If
repair fails after focused attempts, log `regression_unfixed_discard`, reset the
trial away, and move on.

`ROM_PATH` is mandatory for ROM-dependent checks, smokes, and benchmarks. It
must resolve to SHA-256
`f61548fdf1670cffefcc4f0b7bdcdd9eaba0c226e3b74f8666071496988248de`.

Prefer Rust-side changes in `src/emulator.rs`, `src/vec_env.rs`, and
`src/py_api.rs`. Before coding, name the expected speed mechanism and falsifying
metric. Spend diagnosis on the highest-current-cost mechanism; kill candidates
when profile share, prior rejects, or smoke timings make the plausible gain
smaller than benchmark cost.

Accepted scope limits: SMB mapper 0 / NROM only, no audio requirement, no
general Gym Retro/arbitrary NES mapper compatibility, RGB/uncropped renderers as
compatibility paths. Document Mario/NES shortcut assumptions in the ledger and
accepted commit message.

Preserve or strengthen:

- identical lanes may share emulator state only while deterministic and uniform;
  mixed actions must materialize independent lanes
- cropped grayscale tile rendering must preserve SMB/NES background runs and
  sprite overlay semantics

## Reporting And Cleanup

Use compact phase reports:

```text
phase: orient_once | diagnose | prototype | screen | triage | acceptance | record | cleanup | pause
inputs: refs/artifacts read
outputs: commits/results/artifacts written
decision: keep | keep_small_gain | discard | inconclusive | defer | pause
next: one concrete next action
```

Append scratchpad lessons only when useful:

```text
- date=<YYYY-MM-DD> phase=<phase> lesson=<mistake|hindsight|friction|heuristic> note=<one concrete improvement> evidence=<artifact/ref/command>
```

Pause when access fails, user limits are exhausted, regressions persist,
metadata is untrustworthy, unrelated branch changes appear, refs are missing,
background jobs contaminate timing, low-ROI leftovers remain, two candidates die
for the same reason, or the user asks to stop.

Review with the user after one clear `keep`, two or three `keep_small_gain`
commits, six to ten rejects, three to four hours without progress, or before
changing benchmark semantics, public APIs, or major test contracts.

After each decision and at session end, remove deterministic temporary state:
scratch outputs, abandoned prototype artifacts, and temporary benchmark
byproducts. Preserve `current.json`, `results.tsv`, official aggregates, kept
SHAs, durable patches/bundles, and compact rejection summaries.

On pause/final report include branch/mode/epoch, baseline and latest aggregate
fields, accepted commits, discard count, checks, changed files, cleanup status,
benchmark limits, triage artifacts, and next experiment. Include paste-ready
`scripts/play.py` commands when reporting a result.
