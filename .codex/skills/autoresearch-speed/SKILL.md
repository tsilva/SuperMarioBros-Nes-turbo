---
name: autoresearch-speed
description: Phased Super Mario Bros emulator speed-improvement loop for this repo. Use when optimizing, profiling, benchmarking, or autonomously iterating on Super Mario Bros NES throughput with fixed local experiments, make-test regression gating, commit/revert discipline, and experiment tracking.
---

# Autoresearch Speed

## Contract

Optimize the live repo only. Preserve the canonical workload unless the user
explicitly changes it. The workload contract is:

- `obs_shape=(16, 4, 84, 84)`
- `obs_dtype=uint8`
- default lanes use `Level1-1`, `Level1-2`, `Level1-3`, `Level1-4` round-robin
- real SMB NES reset/step behavior
- correct frame skip, frame stack, grayscale/crop/resize, action mapping,
  rewards, dones/truncations, resets, and info scalar semantics

Do not fake speed by skipping emulator progression, weakening the workload,
returning stale observations, changing the public command, or loosening the
observed contract.

Use `make benchmark` for smoke and triage timing only. Official acceptance
timing must come from the fixed-ref paired local benchmark helper:

```bash
.venv/bin/python scripts/run_git_ref_benchmark.py <baseline_ref> <candidate_ref> --steps 50000 --repeats 3
```

Run official timing only after the candidate is committed and all required
checks for the changed surface have passed. Local commands are for correctness,
compilation, formatting, profiling, diagnosis, and triage; do not call a
benchmark skill.

Default campaign mode is phased dedicated-machine fast iteration: generate and
code a small batch of candidate proposals, stop all agent activity, then
benchmark and accept candidates serially. Reject bad ideas as soon as a cheap
same-tier `make benchmark` screen shows they are not promising, and spend full
acceptance time only on candidates that look likely to improve results.

## Benchmark Access

Assume `/autoresearch-speed` uses local benchmark commands in this checkout.
Use `make benchmark` for diagnosis or triage, and use
`scripts/run_git_ref_benchmark.py` for official acceptance. Do not use SSH,
tailnet, cloud, Modal, benchmark skills, or other non-local benchmark variants
for autoresearch timing unless the user explicitly approves a different
benchmark target in the current turn.

If the user provides benchmark run or wall-clock limits, record and obey
them. If not, leave limit fields as `null` and continue until stopped or
blocked. Run at most one benchmark at a time.

If a timing run cannot complete because the machine is busy, setup fails,
metadata is malformed, or the result is too noisy, do not accept the candidate.
Either mark the trial `inconclusive` and reset it away, or stop with a clear
`stop_reason` when another attempt would just repeat the same blocker.

Use the local machine aggressively for screening. A busy machine should still
block official acceptance unless the user explicitly says to force through load,
but cheap triage runs may use shorter Make benchmark variables to avoid wasting
time on obvious losers.

## Benchmark Tiers

Use a funnel, not full timing for every idea:

1. `local_diagnosis`: uncommitted local profiling, smoke tests, and narrow
   checks for fast feedback. These results can guide edits only; they cannot
   reject or accept a committed candidate by themselves.
2. `local_triage`: committed candidate on the local machine, using the public
   Make target with shorter Make variables and comparing only against a fresh
   same-tier baseline from the current accepted source baseline. Default command
   shape:

   ```bash
   BENCHMARK_STEPS=5000 BENCHMARK_REPEATS=1 BENCHMARK_WARMUP=100 BENCHMARK_ARGS="--json --output-json artifacts/benchmarks/triage-<label>.json" make benchmark
   ```

   Treat `local_triage` as screening only. It can justify `triage_discard`,
   another focused edit, or escalation to acceptance. It cannot justify `keep`.
3. `local_acceptance`: the fixed-ref paired benchmark helper. Only this tier can
   justify `keep`, `keep_small_gain`, updating the active baseline, or reporting
   an accepted campaign speedup.

   ```bash
   .venv/bin/python scripts/run_git_ref_benchmark.py <baseline_ref> <candidate_ref> --steps 50000 --repeats 3
   ```

Triage interpretation:

- If candidate triage SPS is below the same-tier baseline, or below `+1%` with
  unstable or noisy samples, discard or revise without running the full
  acceptance protocol.
- If candidate triage SPS is at least `+3%`, the candidate is worth full
  checks and `local_acceptance` unless the change is risky or contract-sensitive.
- If candidate triage SPS is `+1%` to `+3%`, escalate only for simple,
  low-risk, compounding, or simplifying changes. Otherwise prefer the next idea.
- If triage is noisy, rerun triage once on a calmer machine or with `--steps 10000`;
  do not keep sampling until the result becomes favorable.
- Never accept a candidate from triage evidence alone.
- For `keep`, require the official paired aggregate to have
  `decision=converged_candidate_win` and `validity_passed=true`.
- For `keep_small_gain`, require official paired evidence with all
  `stability_gates=true`, median pair ratio above `1.0`, pair-ratio confidence
  interval lower bound not below `1.0`, sufficient faster pairs, and a
  simple/low-risk/simplifying change. Do not require helper-level
  `validity_passed`, because the helper reserves that for `>=3%` wins or clear
  no-win decisions.
- Discard or mark `inconclusive` for noisy, invalid, load-failed, no-win,
  incomparable, or contract-risky official aggregates.

## Branch And State

Work on the current branch by default. Do not create branches, switch branches,
create worker worktrees, create replay branches, merge, or delete branches
unless the user explicitly approves that operation in the current turn. If the
user approves a campaign branch, a conventional name is:

```text
codex/autoresearch-continuous
```

Before work:

1. Verify git state and current branch.
2. Stay on the current branch unless the user explicitly approved creating,
   switching to, or resuming a campaign branch.
3. If creating an approved campaign branch, branch from local `main` unless the
   user explicitly approved starting from the current dirty tree.
4. If resuming an approved campaign, read `.codex/optimization_campaigns/current.json` and
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
- `/Users/tsilva/.codex/autoresearch/SuperMarioBros-Nes-turbo/ideas.md` for the
  branch-independent durable idea queue
- `/Users/tsilva/.codex/autoresearch/SuperMarioBros-Nes-turbo/candidates/` for
  coordinator-imported implementation candidate manifests

Keep `results.tsv` and the repo copy of `ideas.md` uncommitted unless the user
asks to commit logs. Accepted source commits stay on the current approved work
branch; rejected commits are reset away. The durable queue outside the repo must
survive checkout, reset, branch switching, and rejected-candidate rewinds.

`results.tsv` should stay human-scannable. Older campaigns may already have the
legacy benchmark header:

```text
epoch	commit	mean_env_steps_per_sec	stdev_env_steps_per_sec	best_env_steps_per_sec	gain_pct	status	description	artifact
```

For new benchmark rows, prefer paired fixed-ref benchmark fields and migrate the
header when practical:

```text
epoch	commit	baseline_commit	official_median_sps	median_pair_ratio	ci95_low	ci95_high	candidate_faster_pairs	measured_pairs	status	description	artifact
```

Statuses: `baseline`, `triage_discard`, `triage_promote`, `keep`,
`keep_small_gain`, `discard`, `crash`, `regression_fixed_keep`,
`regression_unfixed_discard`, `inconclusive`.

Manifest fields should include campaign id/mode, branch names, root SHA, epoch,
benchmark command/output root, optional benchmark limits, benchmark runs used,
triage benchmarks used, current baseline output and same-tier triage SPS, latest
triage benchmark fields, latest official paired aggregate fields, accepted
commits, discarded commits, current experiment, and stop reason.

## Ideas Queue

Use `/Users/tsilva/.codex/autoresearch/SuperMarioBros-Nes-turbo/ideas.md` as
the authoritative optimization backlog. Mirror it into
`.codex/optimization_campaigns/ideas.md` for repo-local context when needed, but
sync the durable copy before any reset, checkout, branch switch, or rejected
candidate rewind. If the two copies conflict, preserve `Done` and rejection
evidence from both, then use the durable queue's `Ready` ordering unless live
results prove it stale.

Create the durable queue and repo mirror when missing. The queue is a Markdown
document, not a table, so ideas may include rich rationale, links, checklists,
code snippets, profiling notes, or benchmark hypotheses.

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

1. Sync the durable idea queue into the repo mirror, then read both queue copies,
   `results.tsv`, `.codex/optimization_campaigns/current.json`,
   `docs/PERFORMANCE_PLAN.md`, and the current hot-path source.
2. Prefer the first high-quality `ready` idea that is not a duplicate of prior
   rejected or accepted work.
3. If selecting the last non-duplicate `ready` idea, immediately start the
   background refill below before implementing that idea. Do not wait for the
   queue to become empty if the refill can run while the last task is in
   diagnosis, implementation, testing, or benchmarking.
4. Move or mark the selected idea as `in_progress` in both durable and repo
   copies, record the epoch/pre-experiment SHA, and set it as the current
   experiment in `current.json`.
5. After the experiment, move the idea to `Done` with the decision, result row,
   artifact, and rationale in both queue copies before starting another idea.

Keep flushing the queue. When there are zero non-duplicate `ready` ideas, or
when the selected idea is the last one, use up to four idea-generation subagents
in parallel when tools and user approval allow it. Otherwise refill the queue
manually in the main thread. Give each idea lane a distinct perspective:

- emulator CPU/interpreter specialization
- PPU/render/preprocessing path
- vector environment scheduling and lane semantics
- Python/Rust boundary, buffer movement, tests, instrumentation, and cleanup

Each idea subagent must return Markdown queue entries only. Instruct subagents
to read the current source, `docs/PERFORMANCE_PLAN.md`, `results.tsv`, both
queue copies, and `.codex/optimization_campaigns/current.json`; avoid
already-tried ideas; preserve the benchmark contract; include expected ROI,
contract risks, required checks, and prior evidence; and prefer concrete,
implementable experiments over broad themes. Subagents must not edit files,
commit, benchmark, reset, or mutate campaign state.

When idea generation completes, aggregate entries, deduplicate by mechanism and
target files, reject ideas that duplicate `Done`/discarded work, assign stable
`IDEA-YYYYMMDD-NNN` IDs, rank by expected ROI, and write the ranked result to
the durable queue first. Then mirror it into the repo queue. Expected ROI means
biggest plausible benchmark impact for the least code/risk/checking cost, with
compounding/simplifying ideas preferred over isolated complexity.

Do not spend benchmark runs merely to generate ideas. Idea generation is
analysis work; only concrete candidates selected from the queue go through
diagnosis, triage, and possible fixed-ref paired acceptance.

## Trajectory Token Discipline

Spend tokens on decisions and evidence, not narration. Treat
`.codex/optimization_campaigns/current.json`, `results.tsv`, both ideas queues,
candidate manifests, and benchmark aggregates as the source of truth. Do not
restate campaign history in chat unless it changes the next decision.

Use compact phase reports:

```text
phase: generate | code | freeze | benchmark | merge | next_batch | pause
inputs: refs/artifacts read
outputs: manifests/commits/results written
decision: keep | keep_small_gain | discard | inconclusive | defer | pause
next: one concrete next action
```

Keep subagent prompts narrow:

- Idea agents get one perspective, the current baseline SHA, known
  Done/blocked mechanisms, max idea count, required queue fields, and
  "Markdown queue entries only".
- Implementation agents get one `idea_id`, target files, hypothesis, forbidden
  mechanisms, required checks, manifest schema, and stop conditions.
- The benchmark coordinator gets imported manifests, current baseline,
  evaluation order, and acceptance gates.

Output limits:

- Idea agents return Markdown queue entries only; no commentary.
- Implementation agents return one committed candidate, one manifest, and at
  most five short notes.
- Benchmark phase records one result row per candidate plus one decision:
  `keep`, `keep_small_gain`, `discard`, or `inconclusive`.
- Final campaign reports are bounded to branch, baseline, accepted/rejected
  counts, checks, artifacts, cleanup, and next action.

Early-kill rules:

- Mark an implementation `incomplete` when it cannot produce a small committed
  patch after focused attempts.
- Workers record only the final blocker, failed check, and next recommended
  action; do not narrate full debugging history.
- The coordinator rejects malformed or verbose handoffs unless the candidate is
  valuable enough to request a corrected manifest.

## Phased Batch Mode

Use phased batch mode when there are multiple independent, small, high-quality
ready ideas. Use the serial single-candidate loop instead when there is one
obvious best idea, when the change is contract-sensitive, or when benchmark
feedback is needed before coding the next candidate.

Phases:

1. `generate`: fork idea agents, deduplicate their entries, and select a small
   batch by expected SPS impact, simplicity, correctness risk, and mechanism
   diversity. Preserve one lane for structural simplification ideas when the
   profiler only points at local hotspots.
2. `code`: if the user explicitly approves worker branches/worktrees, fork
   implementation agents in isolated `git worktree` checkouts. Use two
   implementation agents by default. Raise to three only when the selected
   candidates are small, clearly independent, and unlikely to compete for the
   same hot path. Each agent implements one candidate, runs targeted
   non-official checks, creates a final commit, and returns a manifest. Without
   approval, use the serial single-candidate loop on the current branch.
3. `freeze`: stop all implementation and idea agents before any official
   timing. No agent builds, tests, profiling runs, local timings, edits, or
   background jobs may continue during `benchmark` or `merge`.
4. `benchmark`: the main thread imports candidate manifests, chooses evaluation
   order, and evaluates candidates one at a time on the quiet machine.
5. `merge`: if the batch has accepted commits and the user explicitly asks to
   merge, run merge verification against current `main` and merge only if the
   accepted batch still shows a real win. If the batch has no kept candidates,
   do not merge.
6. `next_batch`: update the idea queue, rejection evidence, candidate manifests,
   and campaign state, then start a new generate/code phase or pause.

Implementation agents must not run official benchmarks, mutate
`.codex/optimization_campaigns/current.json`, mutate `results.tsv`, edit either
ideas queue copy, touch `main`, reset the coordinator branch, switch the main
thread's branch, or continue after the freeze phase. Worker-side timings, if
any, are diagnostic only and cannot justify `keep` or `keep_small_gain`.

Candidate manifests are proposals, not acceptance evidence. Each worker must
commit its final patch before handoff and report a manifest with enough identity
to recover and replay it. Include durable patch evidence, not only a live
branch/worktree path:

```json
{
  "schema_version": 1,
  "idea_id": "IDEA-YYYYMMDD-NNN",
  "worker_id": "worker-1",
  "repo_path": "/Users/tsilva/repos/tsilva/SuperMarioBros-Nes-turbo",
  "worktree_path": "/absolute/path/to/worktree",
  "branch": "codex/autoresearch-worker-...",
  "base_sha": "40-hex-sha",
  "candidate_sha": "40-hex-sha",
  "patch_id": "git patch-id --stable value if available",
  "patch_artifact": "/Users/tsilva/.codex/autoresearch/SuperMarioBros-Nes-turbo/candidates/IDEA-YYYYMMDD-NNN.patch",
  "bundle_artifact": null,
  "changed_files": ["src/emulator.rs"],
  "checks_run": ["cargo fmt --check", "cargo check --release"],
  "risk_level": "low | medium | high",
  "expected_speed_mechanism": "short concrete mechanism",
  "worker_verdict": "ready | incomplete | discard",
  "notes": "short handoff notes"
}
```

The coordinator imports manifests into
`/Users/tsilva/.codex/autoresearch/SuperMarioBros-Nes-turbo/candidates/`. Do not
let workers append to a shared repo-local candidate file. If a manifest names an
uncommitted diff, a missing commit, or an unrecoverable worktree, mark it
`incomplete` or `discard` rather than reconstructing the candidate by hand.
An importable manifest must contain recoverable git identity: repo path,
worktree path, branch, `base_sha`, `candidate_sha`, patch-id if available,
at least one durable patch or bundle artifact, changed files, checks run, risk,
expected speed mechanism, verdict, and notes.

Coordinator candidate evaluation rules:

- Choose evaluation order by expected incremental value, simplicity, risk, and
  overlap; do not use FIFO by default.
- Treat every worker candidate as a proposal until it is replayed onto the
  current accepted campaign baseline and measured there.
- Prefer creating a fresh replay branch from the current baseline and
  cherry-picking the candidate commit(s) only when the user explicitly approved
  replay branches/worktrees. Otherwise apply the candidate serially on the
  current branch after recording the pre-experiment SHA. If replay or
  cherry-pick is messy, either discard the candidate or return it to a worker in
  a later code phase.
- After accepting candidate A, candidate B must be replayed onto
  `baseline + A`, checked again, and benchmarked again before it can be kept.
- Reject candidates that become redundant, conflict-heavy, contract-weakening,
  too complex for their measured gain, or no longer improve SPS on the updated
  baseline.
- Never assume isolated worker gains add.

## Required Checks

Before every `local_acceptance` benchmark, run:

```bash
cargo fmt --check
cargo check --release
.venv/bin/python -m maturin develop --release
make test
```

`make test` is the mandatory regression gate. It runs the repo-approved Rust and
Python tests, but ROM-dependent stable-retro-turbo oracle tests may skip when
the ROM or oracle package is unavailable. Do not substitute `cargo test`,
`cargo check`, smoke scripts, local throughput runs, or `local_triage` for
`make test` before acceptance. For contract-sensitive changes to observations,
rewards, termination, reset behavior, action mapping, info fields,
preprocessing bytes, or state loading, require explicit evidence that the
relevant oracle/parity tests actually ran without skips; otherwise pause or
mark the trial blocked/inconclusive instead of accepting it.

Before `local_triage`, run the cheapest checks that match the changed surface.
Default to `cargo fmt --check`, `cargo check --release`, and
`.venv/bin/python -m maturin develop --release` for Rust/PyO3 hot-path changes
so timing never measures a stale native extension. Add targeted tests before
triage when touching observations, rewards, termination, reset behavior, action
mapping, info fields, preprocessing bytes, state loading, or Python API
contracts. Full `make test` may wait until after a positive triage signal unless
the change is contract-sensitive.

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
   `requires_fresh_baseline`, satisfy that before selecting another idea.
2. Run a fresh same-tier triage baseline for screening, and run official
   fixed-ref paired acceptance only when a candidate survives triage.
3. Record benchmark output, paired aggregate fields when available, load
   metadata, baseline refs, and baseline status. Do not use older benchmark
   outputs as the active baseline for new candidates.

Each experiment:

1. Record pre-experiment SHA.
2. Choose one concrete optimization idea from
   the durable idea queue, refilling in advance when it is empty or when the
   selected item is the last ready idea.
3. Edit directly on the current approved work branch.
4. Run local diagnosis/build checks as needed.
5. Run pre-triage checks appropriate to the changed surface.
6. Commit the candidate before any benchmark timing.
7. Run one `local_triage` benchmark with shorter Make variables and compare the
   resulting SPS to a fresh same-tier baseline from the current accepted source
   baseline.
8. Parse the benchmark output and append a triage row. If triage is clearly
   slower, noisy without enough upside, contract-weakening, or below the
   escalation bar, mark `triage_discard`, reset to the pre-experiment SHA, and
   continue with the next idea.
9. For triage survivors, run the full required checks.
10. Run one `local_acceptance` fixed-ref paired timing pass:

    ```bash
    .venv/bin/python scripts/run_git_ref_benchmark.py <baseline_ref> <candidate_ref> --steps 50000 --repeats 3
    ```

11. Parse the aggregate output and append a result row with paired decision
    fields.
12. Decide:
   - `keep`: required checks passed, the official paired aggregate has
     `decision=converged_candidate_win` and `validity_passed=true`, load gates
     passed, and contract checks passed.
   - `keep_small_gain`: required checks passed, the official paired aggregate
     has all `stability_gates=true`, median pair ratio above `1.0`, pair-ratio
     confidence interval lower bound not below `1.0`, sufficient faster pairs,
     and the change is simple, low-risk, composable, simplifying, or plausibly
     compounding. A raw tiny speedup by itself is never enough.
   - `discard`: equal/slower/no meaningful win/noisy/too complex/contract
     weakening.
   - `inconclusive`: malformed, load-failed, too noisy, skipped required oracle
     coverage, or incomparable metadata.
13. If kept, update baseline fields to the candidate commit and its latest
    official paired aggregate metrics, then continue from the improved work
    branch.
14. If rejected, reset back to pre-experiment SHA and continue.

Never assume independent gains add. Every accepted commit becomes the new source
baseline and later candidates are judged against fresh same-tier triage
baselines and fresh official paired acceptance results.

For phased batches, this serial experiment loop runs inside the `benchmark`
phase for each imported candidate. A candidate accepted during the batch updates
the campaign baseline immediately. Every remaining candidate must then be
replayed onto that updated baseline before any checks or timing.

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
passing required checks, an official fixed-ref paired benchmark aggregate
recorded locally, valid benchmark gates for the decision tier, and an updated
campaign ledger. Machine busy/unreachable states, missing fresh baseline, dirty
branch ambiguity, failing tests, malformed aggregates, noisy CIs, skipped
required oracle coverage, missing approved campaign branches, or missing
manifest refs must end as `blocked`, `inconclusive`, `discard`, or a clear pause
state, not as `keep`.

Triage artifacts are useful evidence for why ideas were discarded, but they are
not accepted commits and must not remain in history after rejection. Reset
triage rejects away promptly so the local machine keeps producing useful
experiments instead of polishing losing candidates.

Use a review checkpoint after any of: one clear `keep`, two or three clean
`keep_small_gain` commits, six to ten consecutive rejects after the last keep,
three to four hours of active campaign time without meaningful progress, or
before touching benchmark semantics, public APIs, or major test contracts. If
accepted commits exist and the user explicitly asks to merge, the checkpoint may
become a merge verification. If no accepted commits exist, do not merge; update
rejection evidence, refresh or re-rank ideas, and continue with a new batch or
pause.

Before merging accepted campaign work into `main`, first confirm the user
explicitly asked for that merge. Ensure there are no active agents or background
builds/tests/profiling/timing jobs, then run:

```bash
cargo fmt --check
cargo check --release
.venv/bin/python -m maturin develop --release
make test
```

Then run the final paired gate against current `main`:

```bash
.venv/bin/python scripts/run_git_ref_benchmark.py main <accepted_ref> --steps 50000 --repeats 3
```

Merge only when the accepted batch shows a real measured win versus current
`main`.

After the merge phase completes, clean up deterministic temporary state:

- Delete campaign-created worker worktrees whose changes are merged, rejected,
  or explicitly deferred.
- Prune campaign-created temporary worker and replay branches. Use `git branch
  -D` only for branches created by this campaign and no longer needed.
- Purge temporary worker artifacts, scratch outputs, stale replay branches, and
  incomplete candidate handoff files.
- Preserve audit and anti-repeat evidence: `current.json`, `results.tsv`, the
  durable ideas queue, accepted benchmark aggregates, kept commit SHAs, and
  final candidate manifests or compact archived summaries.
- Never delete a worktree containing unrecovered useful changes.

On pause, leave accepted commits on the current approved work branch, rejected
commits out of history, update campaign state, and report:

- branch, mode, epoch
- baseline/latest official paired aggregate, median pair ratio, CI bounds,
  candidate-faster pairs, measured pairs, decision, and status
- accepted commits and discarded count
- checks run
- changed files
- cleanup completed or deferred
- benchmark runs/remaining limits if provided
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
