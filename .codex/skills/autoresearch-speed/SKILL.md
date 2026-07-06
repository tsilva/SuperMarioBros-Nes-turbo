---
name: autoresearch-speed
description: Phased Super Mario Bros emulator speed-improvement loop for this repo. Use when optimizing, profiling, benchmarking, or autonomously iterating on Super Mario Bros NES throughput with fixed local experiments, make-test regression gating, commit/revert discipline, and experiment tracking.
---

# Autoresearch Speed

## Contract

Optimize the live repo only. Preserve the canonical workload unless the user
explicitly changes it:

- `obs_shape=(16, 4, 84, 84)`, `obs_dtype=uint8`
- `Level1-1` through `Level1-4` round-robin by default
- real SMB NES reset/step behavior
- correct frame skip, frame stack, grayscale/crop/resize, action mapping,
  rewards, dones/truncations, resets, and info scalar semantics

Do not fake speed by skipping emulator progression, weakening the workload,
returning stale observations, changing public commands, or loosening the
observed contract.

Use local benchmarks in this checkout only unless the user explicitly approves a
different target in the current turn. Do not use SSH, tailnet, cloud, Modal, or
benchmark skills for autoresearch timing.

Benchmark funnel:

- `local_diagnosis`: uncommitted profiling, smoke tests, and narrow checks.
  Helpful for edits only; cannot accept or reject a committed candidate.
- `local_triage`: coordinator-owned paired screening from exact committed refs,
  fresh baseline and candidate, identical shorter Make variables, ROM, state
  directory, workload, load policy, and output metadata. Screening only.
- `local_acceptance`: official fixed-ref paired benchmark. Only this tier can
  justify `keep`, `keep_small_gain`, accepted speedups, or baseline updates.

Official acceptance command:

```bash
.venv/bin/python scripts/run_git_ref_benchmark.py <baseline_ref> <candidate_ref> --steps 50000 --repeats 3
```

Default triage command shape for each side:

```bash
BENCHMARK_STEPS=5000 BENCHMARK_REPEATS=1 BENCHMARK_WARMUP=100 BENCHMARK_ARGS="--json --output-json artifacts/benchmarks/triage-<role>-<label>.json" make benchmark
```

Triage interpretation:

- Below paired baseline, or below `+1%` with unstable samples: discard or
  revise without full acceptance.
- `+3%` or better: run full checks and acceptance unless the change is risky or
  contract-sensitive.
- `+1%` to `+3%`: escalate only for simple, low-risk, compounding, or
  simplifying changes.
- Noisy triage: rerun once on a calmer machine or with `--steps 10000`; do not
  keep sampling until favorable.
- Never accept from triage evidence alone.

Acceptance decisions:

- `keep`: required checks passed, official aggregate has
  `decision=converged_candidate_win`, `validity_passed=true`, load gates passed,
  and contract checks passed.
- `keep_small_gain`: required checks passed, all official `stability_gates=true`,
  median pair ratio is above `1.0`, CI lower bound is not below `1.0`, enough
  faster pairs exist, and the change is simple, low-risk, simplifying,
  composable, or plausibly compounding.
- `discard`: equal/slower/no meaningful win/noisy/too complex/contract
  weakening.
- `inconclusive`: malformed metadata, load failure, missing ROM, incomparable
  outputs, skipped required contract coverage, or too much noise.

If the user provides benchmark run or wall-clock limits, record and obey them.
Otherwise leave limit fields `null`. Run at most one benchmark at a time. A busy
machine can be used for cheap screening, but blocks official acceptance unless
the user explicitly says to force through load.

## Branch And State

Work on the current branch by default. For an approved worker campaign, the
coordinator may create campaign-scoped worker worktrees, worker branches, replay
branches, and temporary triage worktrees from the recorded baseline ref. Do not
switch to `main`, merge into `main`, push, or delete non-campaign branches
unless the user explicitly approves that operation in the current turn.

Conventional campaign branch, if approved:

```text
codex/autoresearch-continuous
```

Before work:

1. Verify git state and current branch.
2. Stay on the current branch unless creating, switching to, or resuming a
   campaign branch was explicitly approved.
3. If creating an approved campaign branch, branch from local `main` unless the
   user approved starting from the current dirty tree.
4. If resuming, read `.codex/optimization_campaigns/current.json`,
   `.codex/optimization_campaigns/results.tsv`, and durable candidate manifests.
5. Verify needed refs exist locally: campaign branch, root SHA, current
   baseline, accepted commits, and candidate commits. If not, stop and ask
   whether to recover, recreate from `main`, or migrate the ledger.
6. If unrelated dirty changes would be carried in, stop and ask.
7. Inspect the hot path: `scripts/benchmark_sps.py`,
   `python/supermariobrosnes_turbo/env.py`, `src/py_api.rs`,
   `src/vec_env.rs`, `src/emulator.rs`, `Cargo.toml`, `pyproject.toml`, and
   relevant docs.

Track every trial, including crashes and rejects:

- `.codex/optimization_campaigns/current.json`
- `.codex/optimization_campaigns/results.tsv`
- `/Users/tsilva/.codex/autoresearch/SuperMarioBros-Nes-turbo/candidates/`
- accepted benchmark aggregates and kept commit SHAs

Keep campaign metadata and `results.tsv` uncommitted unless the user asks to
commit logs. Accepted source commits stay on the approved work branch; rejected
replay commits are reset away. Historical `ideas.md` files are not the active
workflow source during worker campaigns.

Preferred `results.tsv` header for new rows:

```text
epoch	commit	baseline_commit	official_median_sps	median_pair_ratio	ci95_low	ci95_high	candidate_faster_pairs	measured_pairs	status	description	artifact
```

Statuses: `baseline`, `triage_discard`, `triage_promote`, `keep`,
`keep_small_gain`, `discard`, `skip`, `crash`, `regression_fixed_keep`,
`regression_unfixed_discard`, `inconclusive`.

Campaign manifest state should include campaign id/mode, branch names, root SHA,
epoch, worker count, benchmark output root, optional benchmark limits, current
baseline, imported manifests, candidate states, triage fields, official
aggregate fields, accepted/rejected commits, cleanup state, and stop reason.

## Worker Campaign

Use phased worker batch mode for autoresearch by default. Workers generate and
implement candidates in parallel; the coordinator evaluates them serially. There
is no coordinator direct-implementation mode and no active idea queue mode.
Default `N=4` workers unless the user provides a different `N`.

Phases:

1. `launch_workers`: record `baseline_ref`; read campaign state, prior rejects,
   profiler evidence, and hot-path source; create `N` worker worktrees and
   branches from `baseline_ref`; give each worker one lane and the contract.
2. `freeze`: stop all workers and background activity before timing.
3. `filter`: import manifests and discard incomplete, uncommitted,
   unrecoverable, duplicate, contract-weakening, too broad, too risky, or
   unchecked candidates.
4. `replay`: rank survivors by expected speed mechanism, simplicity, risk,
   overlap, and check quality; replay the highest-ranked patch onto the current
   accepted baseline to create `candidate_ref`.
5. `triage`: run paired `local_triage` for `baseline_ref` and `candidate_ref`;
   discard, skip, or promote based only on those paired artifacts.
6. `acceptance`: for promoted candidates, run full required checks and official
   acceptance. If accepted, update `baseline_ref = candidate_ref`, then rerank
   and replay remaining candidates onto the new baseline.
7. `cleanup`: delete campaign-created worker worktrees and branches whose
   manifests and patch artifacts are durable and whose changes are kept,
   rejected, skipped, or inconclusive. Run `git worktree prune` at campaign end.

Workers must not run `make benchmark`, triage, official benchmarks, mutate
campaign ledgers, touch `main`, reset/switch the main thread branch, or continue
after freeze. Worker timings, if accidentally produced, are ignored.

Each worker returns one committed candidate, one manifest, one durable patch or
bundle artifact, and at most five short notes. Required manifest fields:

```json
{
  "schema_version": 1,
  "candidate_id": "CANDIDATE-YYYYMMDD-NNN",
  "worker_id": "worker-1",
  "repo_path": "/Users/tsilva/repos/tsilva/SuperMarioBros-Nes-turbo",
  "worktree_path": "/absolute/path/to/worktree",
  "branch": "codex/autoresearch-worker-...",
  "base_sha": "40-hex-sha",
  "candidate_sha": "40-hex-sha",
  "patch_id": "git patch-id --stable value if available",
  "patch_artifact": "/Users/tsilva/.codex/autoresearch/SuperMarioBros-Nes-turbo/candidates/CANDIDATE-YYYYMMDD-NNN.patch",
  "bundle_artifact": null,
  "changed_files": ["src/emulator.rs"],
  "checks_run": ["cargo fmt --check", "cargo check --release"],
  "risk_level": "low | medium | high",
  "candidate_summary": "short concrete idea and implementation",
  "expected_speed_mechanism": "short concrete mechanism",
  "worker_verdict": "ready | incomplete | discard",
  "notes": "short handoff notes"
}
```

Import manifests into
`/Users/tsilva/.codex/autoresearch/SuperMarioBros-Nes-turbo/candidates/`; do not
let workers append to shared repo-local candidate files. If a manifest lacks a
recoverable commit or durable patch/bundle artifact, mark it `incomplete` or
`discard`.

Coordinator rules:

- Never benchmark worker branches directly.
- Replay each candidate onto the current accepted baseline in a fresh campaign
  replay branch or worktree; skip or discard messy replays.
- After accepting candidate A, replay and recheck candidate B onto
  `baseline + A`; isolated worker gains never add automatically.
- Reject candidates that become redundant, conflict-heavy, contract-weakening,
  too complex for their measured gain, or no longer improve SPS.

## Required Checks

Before every `local_acceptance` benchmark:

```bash
cargo fmt --check
cargo check --release
.venv/bin/python -m maturin develop --release
make test
```

`make test` is mandatory for acceptance. Do not substitute `cargo test`,
`cargo check`, smoke scripts, local timings, or triage. Do not run
`make test-retro-oracle` in the normal loop unless the user explicitly asks.

Before `local_triage`, run the cheapest checks that match the changed surface.
For Rust/PyO3 hot-path changes default to:

```bash
cargo fmt --check
cargo check --release
.venv/bin/python -m maturin develop --release
```

Add targeted tests before triage when touching observations, rewards,
termination, reset behavior, action mapping, info fields, preprocessing bytes,
state loading, or Python API contracts. Contract-sensitive changes require
explicit evidence that relevant non-oracle parity or contract tests ran without
skips before acceptance.

The SMB ROM is mandatory for ROM-dependent checks, smoke runs, and benchmarks:
`SMB_ROM_PATH` must resolve from the environment or `.env` to an existing ROM
path. A missing ROM is a blocker.

If tests fail, treat it as a regression unless proven unrelated. Fix while
preserving the optimization if possible; rerun the failing test first, then
required checks. If repair fails after focused attempts, log
`regression_unfixed_discard`, reset the trial away, and move on.

## Optimization Guidance

Prefer simple, maintainable Rust-side changes in `src/emulator.rs`,
`src/vec_env.rs`, and `src/py_api.rs`. Separate Python boundary cost, Rust
vector scheduling, CPU emulation, PPU/rendering, resize/preprocessing, stack
movement, and output-buffer copying.

Mario/NES shortcuts are allowed only when they preserve observed SMB behavior.
Document shortcut assumptions in the manifest, ledger, and accepted commit
message. Removing code while preserving or improving speed is a strong keep
signal.

Accept documented scope limits: SMB mapper 0 / NROM only, no audio requirement,
no general Gym Retro/arbitrary NES mapper compatibility, and RGB/uncropped
renderers as compatibility paths rather than the optimized RL benchmark path.

Preserve or replace with stronger checks:

- identical lanes may share emulator state only while deterministic and uniform;
  mixed actions must materialize independent lanes
- cropped grayscale tile rendering must preserve SMB/NES background runs and
  sprite overlay semantics

## Token Discipline And Reports

Spend tokens on decisions and evidence, not narration. Treat `current.json`,
`results.tsv`, imported manifests, compact rejection evidence, and benchmark
aggregates as the source of truth. Use compact phase reports:

```text
phase: launch_workers | freeze | filter | replay | triage | acceptance | cleanup | pause
inputs: refs/artifacts read
outputs: manifests/commits/results written
decision: keep | keep_small_gain | discard | inconclusive | defer | pause
next: one concrete next action
```

Early-kill rules:

- Mark a worker `incomplete` when it cannot produce a small committed patch
  after focused attempts.
- Workers record only the final blocker, failed check, and next recommended
  action.
- Reject malformed or verbose handoffs unless the candidate is valuable enough
  to request a corrected manifest.

Pause cleanly when access fails, user limits are exhausted, the same regression
cannot be fixed, benchmark metadata is untrustworthy, unexpected unrelated
branch changes appear, required refs are missing, active agents/background jobs
would contaminate timing, or the user asks to stop.

Review with the user after one clear `keep`, two or three clean
`keep_small_gain` commits, six to ten consecutive rejects, three to four hours
without meaningful progress, or before changing benchmark semantics, public
APIs, or major test contracts.

Before merging accepted campaign work into `main`, the user must explicitly ask.
Ensure no active agents or background jobs are running, rerun the required
checks, then run the final paired gate:

```bash
.venv/bin/python scripts/run_git_ref_benchmark.py main <accepted_ref> --steps 50000 --repeats 3
```

Merge only when the accepted batch shows a real measured win versus current
`main`. Do not switch to `main`, merge, delete branches, push, or commit
experiment logs unless the user explicitly asks.

After each candidate decision and at campaign end, remove deterministic
temporary state: campaign worker worktrees and branches that are durable and no
longer needed, stale replay branches, scratch outputs, incomplete handoffs, and
temporary artifacts. Preserve `current.json`, `results.tsv`, official
aggregates, kept SHAs, final manifests, durable patches/bundles, and compact
rejection summaries. Never delete a worktree containing unrecovered useful
changes.

On pause or final report include branch/mode/epoch, baseline and latest official
aggregate fields, accepted commits, discard count, checks, changed files,
cleanup status, benchmark limits if any, triage artifacts used, next experiment,
and whether the branch appears fast-forwardable from `main`.

Include paste-ready playback commands when reporting a result:

```bash
.venv/bin/python scripts/play.py --mode external --view raw --state Level1-1 --scale 3
.venv/bin/python scripts/play.py --mode external --view preprocessed --state Level1-1 --frame-skip 4 --frame-stack 4 --crop-top 32 --crop-bottom 0 --resize-width 84 --resize-height 84 --scale 4
```
