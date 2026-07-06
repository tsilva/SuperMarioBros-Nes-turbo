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

Throughput acceptance evidence must go through `/host-benchmark`. At the start
of every campaign turn that needs timing evidence, read
`.codex/skills/host-benchmark/SKILL.md` and follow that skill as the benchmark
subroutine. Invoke it with exact committed refs; local dirty files are never part
of benchmark acceptance. Local commands are for correctness, compilation,
formatting, profiling, and diagnosis only, never acceptance.

For comparisons, `/host-benchmark` acceptance must use its paired fixed-host
decision metric: median candidate/baseline ratio after warmup-pair discard, the
bootstrap confidence interval, faster-pair count, and validity gates. Treat
absolute SPS as useful context, not as the decision statistic for candidate
acceptance.

Default campaign mode is dedicated-machine fast iteration: reject bad ideas as
soon as a cheap exact-ref screen shows they are not promising, and spend full
acceptance time only on candidates that look likely to improve results.

## Benchmark Access

Assume `/autoresearch-speed` uses the dedicated local benchmark host through
`/host-benchmark`. Do not use SSH, tailnet, cloud, Modal, or other non-local
benchmark variants for autoresearch timing.

If the user provides host benchmark run or wall-clock limits, record and obey
them. If not, leave limit fields as `null` and continue until stopped or
blocked. Run at most one host benchmark at a time.

If `/host-benchmark` cannot produce a valid local `aggregate.json` because the
load gate is busy, setup fails, metadata is malformed, or the result is too
noisy, do not accept the candidate. Either mark the trial `inconclusive` and
reset it away, or stop with a clear `stop_reason` when another attempt would
just repeat the same blocker.

Use the dedicated host aggressively for screening. A busy host should still
block official acceptance unless the user explicitly says to force through load,
but cheap triage runs may use the same exact-ref runner with shorter workload
settings to avoid wasting time on obvious losers.

## Benchmark Tiers

Use a funnel, not the full official protocol for every idea:

1. `local_diagnosis`: uncommitted local profiling, smoke tests, and narrow
   checks for fast feedback. These results can guide edits only; they cannot
   reject or accept a committed candidate by themselves.
2. `host_triage`: exact committed refs on the dedicated local host, using the same public
   benchmark contract but shorter runner settings. Default command shape:

   ```bash
   .venv/bin/python scripts/run_git_ref_host_benchmark.py BASELINE_REF CANDIDATE_REF \
     --rom-path /path/to/SuperMarioBros-Nes-v0.nes \
     --steps 5000 --repeats 1 --warmups 1
   ```

   The runner still performs smoke checks, alternating paired order, exact git
   archives, isolated source trees, state hashes, and local result retention.
   Treat `host_triage` as screening only. It can justify `triage_discard`,
   another focused edit, or escalation to acceptance. It cannot justify `keep`.
3. `host_acceptance`: the full `/host-benchmark` protocol with default official
   workload and convergence gates. Only this tier can justify `keep`,
   `keep_small_gain`, updating the active baseline, or reporting an accepted
   campaign speedup.

Triage interpretation:

- If median paired ratio is below `1.00`, or below `1.01` with unstable/noisy
  direction, discard or revise without running the full acceptance protocol.
- If median paired ratio is at least `1.03`, the candidate is worth full checks
  and `host_acceptance` unless the change is risky or contract-sensitive.
- If median paired ratio is `1.01` to `1.03`, escalate only for simple,
  low-risk, compounding, or simplifying changes. Otherwise prefer the next idea.
- If triage is noisy, rerun triage once on a calmer host or with `--steps 10000`;
  do not keep sampling until the result becomes favorable.
- Never accept a candidate whose full acceptance result fails validity gates or
  does not improve the paired decision statistic.

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
legacy benchmark header:

```text
epoch	commit	mean_env_steps_per_sec	stdev_env_steps_per_sec	best_env_steps_per_sec	gain_pct	status	description	artifact
```

For new host-benchmark rows, prefer the host decision fields and migrate the
header when practical:

```text
epoch	commit	baseline_commit	official_median_sps	median_pair_ratio	ci95_low	ci95_high	candidate_faster_pairs	measured_pairs	status	description	artifact
```

Statuses: `baseline`, `triage_discard`, `triage_promote`, `keep`,
`keep_small_gain`, `discard`, `crash`, `regression_fixed_keep`,
`regression_unfixed_discard`, `inconclusive`.

Manifest fields should include campaign id/mode, branch names, root SHA, epoch,
allowed benchmark skill/output root, optional host benchmark limits, host
benchmarks used, triage benchmarks used, current baseline artifact and official
median SPS, latest triage aggregate fields, latest host comparison aggregate
fields, accepted commits, discarded commits, current experiment, and stop
reason.

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
work; only concrete candidates selected from the queue go through diagnosis,
triage, and possible `/host-benchmark` acceptance.

## Required Checks

Before every `host_acceptance` benchmark, run:

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
scripts, local throughput runs, or `host_triage` for `make test` before
acceptance.

Before `host_triage`, run the cheapest checks that match the changed surface.
Default to `cargo fmt --check` and `cargo check --release` for Rust-only hot-path
changes. Add targeted tests before triage when touching observations, rewards,
termination, reset behavior, action mapping, info fields, preprocessing bytes,
state loading, or Python API contracts. Full `make test` may wait until after a
positive triage signal unless the change is contract-sensitive.

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
   bootstrap CI, CVs, validity gates, host load metadata, and baseline status.
   Do not use older mean-SPS artifacts as the active baseline for new candidates.

Each experiment:

1. Record pre-experiment SHA.
2. Choose one concrete optimization idea from
   `.codex/optimization_campaigns/ideas.md`, refilling the queue first if it is
   running low.
3. Edit directly on the campaign branch.
4. Run local diagnosis/build checks as needed.
5. Run pre-triage checks appropriate to the changed surface.
6. Commit the candidate before any host timing.
7. Run one `host_triage` comparison from the current baseline commit to that
   candidate commit using exact refs and shorter workload settings.
8. Parse the copied local `aggregate.json` and append a triage row. If triage is
   clearly slower, noisy without enough upside, contract-weakening, or below the
   escalation bar, mark `triage_discard`, reset to the pre-experiment SHA, and
   continue with the next idea.
9. For triage survivors, run the full required checks.
10. Run exactly one `host_acceptance` `/host-benchmark` comparison from the
    current baseline commit to that candidate commit. The host-benchmark skill
    may continue through its sequential convergence checkpoints inside that
    single comparison according to its preregistered gates.
11. Parse the copied local `aggregate.json` and append a result row with the host
    decision fields.
12. Decide:
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
13. If kept, update baseline fields to the candidate commit and its latest host
    comparison artifact/official metrics, then continue from the improved
    branch.
14. If rejected, reset back to pre-experiment SHA and continue.

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

Triage artifacts are useful evidence for why ideas were discarded, but they are
not accepted commits and must not remain in history after rejection. Reset
triage rejects away promptly so the dedicated machine keeps producing useful
experiments instead of polishing losing candidates.

On pause, leave accepted commits on the campaign branch, rejected commits out of
history, update campaign state, and report:

- branch, mode, epoch
- baseline/latest accepted samples, mean, stdev, best, gain, speedup
- accepted commits and discarded count
- checks run
- changed files
- host benchmarks/remaining limits if provided
- triage benchmarks used, accepted/escalated count, and triage rejects
- next plausible experiment
- whether the branch appears fast-forwardable from `main`

Do not switch to `main`, merge, delete the branch, push, or commit experiment
logs unless the user explicitly asks.

Include paste-ready playback commands when reporting a result:

```bash
.venv/bin/python scripts/play.py --mode external --view raw --state Level1-1 --scale 3
.venv/bin/python scripts/play.py --mode external --view preprocessed --state Level1-1 --frame-skip 4 --frame-stack 4 --crop-top 32 --crop-bottom 0 --resize-width 84 --resize-height 84 --scale 4
```
