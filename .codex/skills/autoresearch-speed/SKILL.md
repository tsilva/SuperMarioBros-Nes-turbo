---
name: autoresearch-speed
description: Single-threaded Super Mario Bros emulator speed-improvement loop for this repo. Use when optimizing, profiling, benchmarking, or autonomously iterating on Super Mario Bros NES throughput with fixed-host experiments, make-test regression gating, commit/revert discipline, and experiment tracking.
---

# Autoresearch Speed

## Contract

Optimize the live repo only. Preserve the canonical benchmark unless the user
explicitly changes it:

```bash
.venv/bin/python scripts/benchmark_sps.py --num-envs 16 --steps 500 --repeats 3
```

Benchmark contract:

- `obs_shape=(16, 4, 84, 84)`
- `obs_dtype=uint8`
- default lanes use `Level1-1`, `Level1-2`, `Level1-3`, `Level1-4` round-robin
- real SMB NES reset/step behavior
- correct frame skip, frame stack, grayscale/crop/resize, action mapping,
  rewards, dones/truncations, resets, and info scalar semantics

Do not fake speed by skipping emulator progression, weakening the workload,
returning stale observations, changing the public command, or loosening the
observed contract.

Throughput evidence must go through `/host-benchmark`. At the start of every
campaign turn that needs timing evidence, read
`.codex/skills/host-benchmark/SKILL.md` and follow that skill as the benchmark
subroutine. Invoke it with exact committed refs; local dirty files are never part
of benchmark acceptance. Local commands are for correctness, compilation,
formatting, profiling, and diagnosis only, never acceptance.

For comparisons, `/host-benchmark` acceptance must use its paired fixed-host
decision metric: median candidate/baseline ratio after warmup-pair discard, the
bootstrap confidence interval, faster-pair count, and validity gates. Treat
absolute SPS as useful context, not as the decision statistic for candidate
acceptance.

## Benchmark Access

Assume `/autoresearch-speed` may use the fixed `beast-3-local` benchmark host
through `/host-benchmark`. Do not ask for cloud spend, upload approval, or
external-service confirmation before benchmarking.

If the user provides host benchmark run or wall-clock limits, record and obey
them. If not, leave limit fields as `null` and continue until stopped or
blocked. Run at most one host benchmark at a time.

If `/host-benchmark` cannot produce a valid local `aggregate.json` because the
host is unreachable, the load gate is busy, setup fails, metadata is malformed,
or the result is too noisy, do not accept the candidate. Either mark the trial
`inconclusive` and reset it away, or stop with a clear `stop_reason` when another
attempt would just repeat the same blocker.

## Branch And State

Use one persistent campaign branch, normally:

```text
codex/autoresearch-continuous
```

Before work:

1. Verify git state and current branch.
2. Create or resume the campaign branch.
3. If creating it, branch from local `main` unless the user explicitly approved
   starting from the current dirty tree.
4. If resuming, read `.codex/optimization_campaigns/current.json` and
   `.codex/optimization_campaigns/results.tsv`.
5. Verify every manifest ref needed for resuming exists locally:
   `branch`, `root_sha`, current baseline commit, accepted commits, and any
   candidate commit being compared. If the branch or required commits are
   missing, stop and ask whether to recreate a fresh campaign from `main`,
   recover the missing refs, or deliberately migrate the ledger.
6. If unrelated dirty changes would be carried in, stop and ask.
7. Inspect the hot path: `scripts/benchmark_sps.py`,
   `python/supermariobrosnes_turbo/env.py`, `src/py_api.rs`,
   `src/vec_env.rs`, `src/emulator.rs`, `Cargo.toml`, `pyproject.toml`, and
   relevant docs.

Track every trial, including crashes and rejects:

- `.codex/optimization_campaigns/current.json` for resume state
- `.codex/optimization_campaigns/results.tsv` for human scanning
- `.codex/optimization_campaigns/ideas.md` for the live idea queue

Keep `results.tsv` and `ideas.md` uncommitted unless the user asks to commit
logs. Accepted source commits stay on the campaign branch; rejected commits are
reset away.

`results.tsv` should stay human-scannable. Older campaigns may already have the
legacy remote-benchmark header:

```text
epoch	commit	mean_env_steps_per_sec	stdev_env_steps_per_sec	best_env_steps_per_sec	gain_pct	status	description	artifact
```

For new host-benchmark rows, prefer the host decision fields and migrate the
header when practical:

```text
epoch	commit	baseline_commit	official_median_sps	median_pair_ratio	ci95_low	ci95_high	candidate_faster_pairs	measured_pairs	status	description	artifact
```

Statuses: `baseline`, `keep`, `keep_small_gain`, `discard`, `crash`,
`regression_fixed_keep`, `regression_unfixed_discard`, `inconclusive`.

Manifest fields should include campaign id/mode, branch names, root SHA, epoch,
allowed benchmark skill/output root, optional host benchmark limits, host
benchmarks used, current baseline artifact and official median SPS, latest host
comparison aggregate fields, accepted commits, discarded commits, current
experiment, and stop reason.

## Ideas Queue

Create `.codex/optimization_campaigns/ideas.md` when missing and keep it as the
authoritative optimization backlog. It is a Markdown document, not a table, so
ideas may include rich rationale, links, checklists, code snippets, profiling
notes, or benchmark hypotheses.

Use this shape:

```markdown
# Autoresearch Ideas Queue

## Ready

### IDEA-YYYYMMDD-NNN: Short Title

- Status: ready
- Perspective: emulator-core | ppu-render | vec-env | python-boundary | tests | cleanup | other
- Hypothesis: ...
- Target files: ...
- Prior evidence: ...
- Plan: ...
- Contract risks: ...
- Required checks: ...
- Expected benchmark signal: ...

## In Progress

## Done
```

Statuses are `ready`, `in_progress`, `keep`, `keep_small_gain`, `discard`,
`crash`, `regression_unfixed_discard`, and `inconclusive`. Keep completed ideas
in `Done` with the final result row status, benchmark artifact if any, commit if
kept, and a short reason. Do not delete rejected ideas; they are part of the
anti-repeat ledger.

Before selecting an experiment:

1. Read `ideas.md`, `results.tsv`, `.codex/optimization_campaigns/current.json`,
   `docs/PERFORMANCE_PLAN.md`, and the current hot-path source.
2. Prefer the first high-quality `ready` idea that is not a duplicate of prior
   rejected or accepted work.
3. Move or mark that idea as `in_progress`, record the epoch/pre-experiment SHA,
   and set it as the current experiment in `current.json`.
4. After the experiment, move the idea to `Done` with the decision, result row,
   artifact, and rationale before starting another idea.

Keep the queue topped up. If fewer than three non-duplicate `ready` ideas remain,
or if the remaining ideas all target the same subsystem, launch idea-generation
subagents before the next experiment. Use the user-provided `N` if present;
otherwise use `N=3`. Run them in parallel, each with a different perspective,
for example:

- emulator CPU/interpreter specialization
- PPU/render/preprocessing path
- vector environment scheduling and lane semantics
- Python/Rust boundary and buffer movement
- tests, instrumentation, simplification, or dead-code removal

Each idea subagent must return Markdown queue entries only. Instruct subagents
to read the current source, `docs/PERFORMANCE_PLAN.md`, `results.tsv`, and
`ideas.md`; avoid already-tried ideas; preserve the benchmark contract; include
contract risks and required checks; and prefer one concrete, implementable
experiment per entry. Merge their entries into `ideas.md`, deduplicate by
mechanism and target files, assign stable `IDEA-YYYYMMDD-NNN` IDs, and keep the
highest-signal ideas near the top of `Ready`.

Do not spend host benchmark runs merely to generate ideas. Idea generation is analysis
work; only concrete candidates selected from the queue go through the required
checks and `/host-benchmark`.

## Required Checks

Before every candidate host benchmark, run:

```bash
cargo fmt --check
cargo check --release
.venv/bin/python -m maturin develop --release
make test
```

`make test` is the mandatory regression gate. It runs the repo-approved Rust
unit tests plus the stable-retro-turbo oracle parity suite, including
observation/preprocessing checks for renderer, termination, reset, and info
surface regressions. Do not substitute `cargo test`, `cargo check`, smoke
scripts, or local throughput runs for `make test`.

Use narrower checks such as `scripts/check_vec_env_equivalence.py` or
`scripts/smoke_smb.py` only for diagnosis or rerunning the first failing surface.
After any fix, rerun `make test` before benchmarking. Add targeted tests when
touching observations, rewards, termination, reset behavior, noop stepping,
uniform/divergent lanes, action mapping, info fields, preprocessing bytes, or
benchmark parsing.

If tests fail, treat it as a regression unless proven unrelated. Fix while
preserving the optimization if possible. Rerun the failing test first, then the
required checks. If repair fails after a few focused attempts, log
`regression_unfixed_discard`, reset the trial away, and move on.

## Loop

Fresh campaign:

1. If `.codex/optimization_campaigns/current.json` says
   `requires_fresh_host_baseline`, satisfy that before selecting another idea.
2. Run the initial `/host-benchmark` single-ref baseline from the unmodified
   campaign branch exact `HEAD` commit.
3. Record the local result bundle, `aggregate.json`, official median SPS,
   bootstrap CI, CVs, validity gates, host route/load metadata, and baseline
   status. Do not use older remote/local mean-SPS artifacts as the active
   baseline for new candidates.

Each experiment:

1. Record pre-experiment SHA.
2. Choose one concrete optimization idea from
   `.codex/optimization_campaigns/ideas.md`, refilling the queue first if it is
   running low.
3. Edit directly on the campaign branch.
4. Run local diagnosis/build checks as needed.
5. Run required checks.
6. Commit the candidate before benchmarking.
7. Run exactly one `/host-benchmark` comparison from the current baseline commit
   to that candidate commit. The host-benchmark skill may escalate from quick to
   full official inside that single comparison according to its preregistered
   gates.
8. Parse the copied local `aggregate.json` and append a result row with the host
   decision fields.
9. Decide:
   - `keep`: required checks passed, the aggregate is a valid
     `paired_compare_fixed_host` result, host load/validity gates passed, the
     bootstrap lower bound is above `1.00`, candidate-faster pair count satisfies
     the `/host-benchmark` interpretation rules, and median paired speedup is
     clearly positive. Treat `median_pair_ratio >= 1.10` as a strong keep signal;
     smaller wins may still need the `keep_small_gain` rules.
   - `keep_small_gain`: allowed only when `/host-benchmark` would call the result
     a small fixed-host win, required checks passed, the change is simple,
     low-risk, simplifying, or plausibly compounding, the bootstrap lower bound
     is above `1.00`, and the candidate-faster pair count/CV/outlier/load gates
     are clean. A raw `median_pair_ratio > 1.00` by itself is never enough.
   - `discard`: equal/slower/noisy/too complex/contract weakening.
   - `inconclusive`: malformed, too noisy, or incomparable metadata.
10. If kept, update baseline fields to the candidate commit and its latest host
    comparison artifact/official metrics, then continue from the improved
    branch.
11. If rejected, reset back to pre-experiment SHA and continue.

Never assume independent gains add. Every accepted commit becomes the new source
baseline and later candidates are judged against a fresh host benchmark
comparison.

## Optimization Guidance

Prefer simple, maintainable Rust-side changes in `src/emulator.rs`,
`src/vec_env.rs`, and `src/py_api.rs`. Separate Python boundary cost, Rust
vector scheduling, CPU emulation, PPU/rendering, resize/preprocessing, stack
movement, and output-buffer copying.

Mario/NES-specific shortcuts are allowed only when they preserve observed SMB
behavior. Document important shortcut assumptions in `docs/PERFORMANCE_PLAN.md`.
Removing code while preserving or improving speed is a strong keep signal.

Accept documented scope limits: SMB mapper 0 / NROM only, no audio requirement,
no general Gym Retro/arbitrary NES mapper compatibility, and RGB/uncropped
renderers as compatibility paths rather than the optimized RL benchmark path.

Preserve or replace with stronger checks:

- identical lanes may share emulator state only while deterministic and uniform;
  mixed actions must materialize independent lanes
- cropped grayscale tile rendering must preserve SMB/NES background runs and
  sprite overlay semantics

## Stop And Report

Pause cleanly when access fails, user-provided run/spend limits are exhausted,
the same regression cannot be fixed, benchmark metadata is untrustworthy,
unexpected unrelated branch changes appear, or the user asks to stop.

For unattended goal execution, prefer blocking over guessing. A successful
overnight goal may leave accepted commits only when each accepted candidate has:
passing required checks, an exact-ref host comparison artifact copied locally,
valid `/host-benchmark` gates, and an updated campaign ledger. Host busy/unreachable
states, missing fresh baseline, dirty branch ambiguity, failing tests, malformed
aggregates, noisy CIs, missing campaign branches, or missing manifest refs must
end as `blocked`, `inconclusive`, `discard`, or a clear pause state, not as
`keep`.

On pause, leave accepted commits on the campaign branch, rejected commits out of
history, update campaign state, and report:

- branch, mode, epoch
- baseline/latest accepted samples, mean, stdev, best, gain, speedup
- accepted commits and discarded count
- checks run
- changed files
- host benchmarks/remaining limits if provided
- next plausible experiment
- whether the branch appears fast-forwardable from `main`

Do not switch to `main`, merge, delete the branch, push, or commit experiment
logs unless the user explicitly asks.

Include paste-ready playback commands when reporting a result:

```bash
.venv/bin/python scripts/play.py --mode external --view raw --state Level1-1 --scale 3
.venv/bin/python scripts/play.py --mode external --view preprocessed --state Level1-1 --frame-skip 4 --frame-stack 4 --crop-top 32 --crop-bottom 0 --resize-width 84 --resize-height 84 --scale 4
```
