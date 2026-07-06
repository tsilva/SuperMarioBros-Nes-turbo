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

Default campaign mode is phased dedicated-machine fast iteration with `N`
parallel implementation workers. When the user does not provide `N`, use
`N=4`. Workers generate and implement candidate proposals only. The coordinator
stops all worker activity, filters and replays candidates, then benchmarks and
accepts candidates serially. Reject bad candidates as soon as coordinator-owned
same-tier triage shows they are not promising, and spend full acceptance time
only on candidates that look likely to improve results.

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

Use a funnel, not full timing for every candidate:

1. `local_diagnosis`: uncommitted local profiling, smoke tests, and narrow
   checks for fast feedback. These results can guide edits only; they cannot
   reject or accept a committed candidate by themselves.
2. `local_triage`: coordinator-owned paired screening for a replayed committed
   candidate. Run a fresh baseline triage and candidate triage back-to-back from
   exact committed refs using the public Make target with identical shorter Make
   variables, workload, ROM, state directory, and output metadata. Compare only
   those paired artifacts; never compare against an older loose `make benchmark`
   number. Run each side from a coordinator-controlled temporary worktree or
   replay branch checked to the exact ref being measured, not from a dirty
   worker branch. Default command shape for each side:

   ```bash
   BENCHMARK_STEPS=5000 BENCHMARK_REPEATS=1 BENCHMARK_WARMUP=100 BENCHMARK_ARGS="--json --output-json artifacts/benchmarks/triage-<role>-<label>.json" make benchmark
   ```

   Treat `local_triage` as screening only. It can justify `triage_discard`,
   `skip`, one focused worker revision in a future batch, or escalation to
   acceptance. It cannot justify `keep`.
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
  low-risk, compounding, or simplifying changes. Otherwise prefer the next
  candidate.
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

Work on the current branch by default outside `/autoresearch-speed` campaigns.
For an approved worker campaign, the coordinator may create campaign-scoped
worker worktrees, worker branches, replay branches, and temporary triage
worktrees from the recorded baseline ref. Do not switch to `main`, merge into
`main`, push, or delete non-campaign branches unless the user explicitly
approves that operation in the current turn. If the user approves a campaign
branch, a conventional name is:

```text
codex/autoresearch-continuous
```

Before work:

1. Verify git state and current branch.
2. Stay on the current branch unless the user explicitly approved creating,
   switching to, or resuming a campaign branch.
3. If creating an approved campaign branch, branch from local `main` unless the
   user explicitly approved starting from the current dirty tree.
4. If resuming an approved campaign, read `.codex/optimization_campaigns/current.json`,
   `.codex/optimization_campaigns/results.tsv`, and durable candidate manifests.
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
- `/Users/tsilva/.codex/autoresearch/SuperMarioBros-Nes-turbo/candidates/` for
  coordinator-imported implementation candidate manifests and compact rejection
  evidence
- accepted benchmark aggregates and kept commit SHAs

Keep `results.tsv` and campaign metadata uncommitted unless the user asks to
commit logs. Accepted source commits stay on the current approved work branch;
rejected replay commits are reset away. Historical `ideas.md` files may exist,
but they are not the active workflow source and must not be synced, refilled, or
ranked during worker campaigns.

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
`keep_small_gain`, `discard`, `skip`, `crash`, `regression_fixed_keep`,
`regression_unfixed_discard`, `inconclusive`.

Campaign manifest fields should include campaign id/mode, branch names, root
SHA, epoch, default worker count, benchmark command/output root, optional
benchmark limits, current baseline commit, worker manifest paths, candidate
states, triage benchmarks used, latest paired triage fields, latest official
paired aggregate fields, accepted commits, discarded/skipped commits, cleanup
state, and stop reason.

## Worker Candidate Generation

The active search unit is a worker-generated candidate, not an idea queue item.
Use `N=4` workers by default unless the user provides a different `N`.

Before launching workers:

1. Record `baseline_ref` as the current accepted source baseline commit.
2. Read `results.tsv`, `.codex/optimization_campaigns/current.json`, durable
   candidate manifests, prior rejection evidence, relevant profiler artifacts,
   and the current hot-path source.
3. Create `N` isolated campaign worker worktrees and branches from
   `baseline_ref`.
4. Give each worker one lane, the benchmark contract, prior accepted/rejected
   mechanisms, required non-benchmark checks, the manifest schema, and the rule
   that workers must not benchmark.

Each worker independently chooses one concrete optimization idea, implements it,
runs non-benchmark checks, commits the final patch, writes a durable patch or
bundle artifact, and returns one manifest. Workers must not run `make
benchmark`, local triage, official acceptance, or mutate
`.codex/optimization_campaigns/current.json`, `results.tsv`, candidate ledgers,
`main`, or the coordinator branch. Worker-side timings, if any exist from
accidental local diagnostics, are ignored and cannot justify `triage_promote`,
`keep`, or `keep_small_gain`.

The coordinator waits for all workers to finish or hit the user-provided limit,
then freezes all worker activity before timing. No worker builds, tests,
profiling runs, local timings, edits, or background jobs may continue during
coordinator triage, acceptance, merge verification, or cleanup.

## Trajectory Token Discipline

Spend tokens on decisions and evidence, not narration. Treat
`.codex/optimization_campaigns/current.json`, `results.tsv`, candidate
manifests, compact rejection evidence, and benchmark aggregates as the source of
truth. Do not restate campaign history in chat unless it changes the next
decision.

Use compact phase reports:

```text
phase: launch_workers | freeze | filter | replay | triage | acceptance | cleanup | pause
inputs: refs/artifacts read
outputs: manifests/commits/results written
decision: keep | keep_small_gain | discard | inconclusive | defer | pause
next: one concrete next action
```

Keep worker prompts narrow:

- Workers get one lane, `baseline_ref`, known accepted/rejected mechanisms,
  allowed target areas, required non-benchmark checks, the manifest schema,
  cleanup expectations, and stop conditions.
- The benchmark coordinator gets imported manifests, current baseline,
  evaluation order, and acceptance gates.

Output limits:

- Workers return one committed candidate, one manifest, one durable patch or
  bundle artifact, and at most five short notes.
- Coordinator benchmark phases record one result row per candidate plus one
  decision:
  `keep`, `keep_small_gain`, `discard`, or `inconclusive`.
- Final campaign reports are bounded to branch, baseline, accepted/rejected
  counts, checks, artifacts, cleanup, and next action.

Early-kill rules:

- Mark a worker candidate `incomplete` when it cannot produce a small committed
  patch after focused attempts.
- Workers record only the final blocker, failed check, and next recommended
  action; do not narrate full debugging history.
- The coordinator rejects malformed or verbose handoffs unless the candidate is
  valuable enough to request a corrected manifest.

## Phased Batch Mode

Use worker batch mode for autoresearch by default. Workers generate candidates
in parallel; the coordinator evaluates them serially. There is no coordinator
direct-implementation mode and no active idea queue mode.

Phases:

1. `launch_workers`: create `N` campaign worker worktrees from `baseline_ref`.
   Each worker proposes and implements one candidate in its own branch.
2. `freeze`: stop all workers and background activity before any coordinator
   timing.
3. `filter`: import manifests and discard unfit candidates before benchmarking:
   incomplete, uncommitted, unrecoverable, duplicate, contract-weakening, too
   broad, too risky, or missing required worker checks.
4. `replay`: rank surviving candidates by expected speed mechanism, simplicity,
   correctness risk, overlap, and check quality. Replay the highest-ranked
   worker patch onto the current accepted baseline to create `candidate_ref`.
5. `triage`: run fresh same-tier paired triage for `baseline_ref` and
   `candidate_ref`; discard, skip, or promote based only on those paired
   artifacts.
6. `acceptance`: for promoted candidates, run full required checks and the
   official fixed-ref paired benchmark. If accepted, update
   `baseline_ref = candidate_ref`; then rerank and replay remaining candidates
   onto the new baseline before evaluating them.
7. `cleanup`: after each candidate decision, delete campaign-created worker
   worktrees and branches whose manifests and patch artifacts are durable and
   whose changes are kept, rejected, skipped, or inconclusive. Run `git worktree
   prune` at campaign end.

Workers must not run `make benchmark`, local triage, official benchmarks, mutate
`.codex/optimization_campaigns/current.json`, mutate `results.tsv`, touch
`main`, reset the coordinator branch, switch the main thread's branch, or
continue after the freeze phase.

Candidate manifests are proposals, not acceptance evidence. Each worker must
commit its final patch before handoff and report a manifest with enough identity
to recover and replay it. Include durable patch evidence, not only a live
branch/worktree path:

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
- Create a fresh campaign replay branch or worktree from the current accepted
  baseline and cherry-pick or apply the worker patch there. If replay is messy,
  skip or discard the candidate rather than benchmarking the worker branch.
- Never benchmark worker branches directly.
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
Python tests; do not run `make test-retro-oracle` as part of the normal
autoresearch loop unless the user explicitly asks for oracle coverage. Do not
substitute `cargo test`, `cargo check`, smoke scripts, local throughput runs, or
`local_triage` for `make test` before acceptance. The SMB ROM is mandatory for
ROM-dependent checks, smoke runs, and benchmarks: `SMB_ROM_PATH` must resolve
from the environment or `.env` to an existing ROM path. A missing ROM is a
blocker, not a reason to skip relevant non-oracle tests, triage, or acceptance.
For contract-sensitive changes to observations, rewards, termination, reset
behavior, action mapping, info fields, preprocessing bytes, or state loading,
require explicit evidence that the relevant non-oracle parity or contract tests
ran without skips; otherwise pause or mark the trial blocked/inconclusive
instead of accepting it.

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

Fresh worker campaign:

1. If `.codex/optimization_campaigns/current.json` says
   `requires_fresh_baseline`, satisfy that before launching workers.
2. Record `baseline_ref` as the current accepted source baseline commit.
3. Launch `N` worker worktrees from `baseline_ref`, defaulting to `N=4`.
4. Wait for workers to return committed candidate manifests or hit the
   user-provided limit.
5. Freeze all worker and background activity before coordinator timing.
6. Filter imported manifests and skip unfit candidates before benchmarking:
   incomplete, uncommitted, unrecoverable, duplicate, contract-weakening, too
   broad, too risky, missing required worker checks, or missing durable
   patch/bundle artifact.
7. Rank surviving candidates by expected speed mechanism, simplicity, risk,
   overlap, and check quality.

Each coordinator evaluation:

1. Record `pre_replay_sha = baseline_ref`.
2. Replay the highest-ranked worker patch onto `baseline_ref` in a fresh
   campaign replay branch or worktree, producing `candidate_ref`.
3. If replay conflicts, becomes redundant, or no longer fits the updated
   baseline, mark `skip` or `discard`, clean up that worker worktree/branch when
   safe, and continue with the next candidate.
4. Run pre-triage checks appropriate to the changed surface.
5. Run paired `local_triage`: fresh baseline triage from `baseline_ref`, then
   fresh candidate triage from `candidate_ref`, with identical Make variables,
   workload, ROM, state directory, machine/load policy, and output metadata.
6. Parse the paired triage outputs and append a triage row. If triage is clearly
   slower, noisy without enough upside, contract-weakening, or below the
   escalation bar, mark `triage_discard`, reset the replay away, clean up that
   worker worktree/branch when safe, and continue with the next candidate.
7. For triage survivors, run the full required checks.
8. Run one `local_acceptance` fixed-ref paired timing pass:

   ```bash
   .venv/bin/python scripts/run_git_ref_benchmark.py <baseline_ref> <candidate_ref> --steps 50000 --repeats 3
   ```

9. Parse the aggregate output and append a result row with paired decision
   fields.
10. Decide:
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
   - `inconclusive`: malformed, load-failed, too noisy, skipped required
     non-oracle parity/contract coverage, missing ROM, or incomparable metadata.
11. If kept, update baseline fields to `candidate_ref`, set
    `baseline_ref = candidate_ref`, clean up the worker worktree/branch when
    safe, then rerank and replay remaining candidates onto the new baseline.
12. If rejected or inconclusive, reset the replay away, keep
    `baseline_ref = pre_replay_sha`, clean up the worker worktree/branch when
    safe, and continue.

Never assume independent gains add. Every accepted commit becomes the new source
baseline and every remaining worker candidate must be replayed onto that
baseline before checks or timing.

## Optimization Guidance

Prefer simple, maintainable Rust-side changes in `src/emulator.rs`,
`src/vec_env.rs`, and `src/py_api.rs`. Separate Python boundary cost, Rust
vector scheduling, CPU emulation, PPU/rendering, resize/preprocessing, stack
movement, and output-buffer copying.

Mario/NES-specific shortcuts are allowed only when they preserve observed SMB
behavior. Document important shortcut assumptions in the candidate manifest,
campaign ledger, and accepted commit message.
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
required non-oracle parity/contract coverage, missing approved campaign
branches, or missing manifest refs must end as `blocked`, `inconclusive`,
`discard`, or a clear pause state, not as `keep`.

Triage artifacts are useful evidence for why candidates were discarded, but they
are not accepted commits and must not remain in history after rejection. Reset
triage rejects away promptly so the local machine keeps producing useful
experiments instead of polishing losing candidates.

Use a review checkpoint after any of: one clear `keep`, two or three clean
`keep_small_gain` commits, six to ten consecutive rejects after the last keep,
three to four hours of active campaign time without meaningful progress, or
before touching benchmark semantics, public APIs, or major test contracts. If
accepted commits exist and the user explicitly asks to merge, the checkpoint may
become a merge verification. If no accepted commits exist, do not merge; update
rejection evidence, launch a new worker batch, or pause.

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

After each candidate decision and after the merge phase completes, clean up
deterministic temporary state:

- Delete campaign-created worker worktrees whose changes are kept, rejected,
  skipped, inconclusive, merged, or explicitly deferred once the manifest and
  patch/bundle artifact are durable.
- Prune campaign-created temporary worker and replay branches. Use `git branch
  -D` only for branches created by this campaign and no longer needed.
- Purge temporary worker artifacts, scratch outputs, stale replay branches, and
  incomplete candidate handoff files.
- Preserve audit and anti-repeat evidence: `current.json`, `results.tsv`,
  accepted benchmark aggregates, kept commit SHAs, final candidate manifests,
  durable patch/bundle artifacts, and compact rejection summaries.
- Run `git worktree prune` at campaign end after deleting campaign-created
  worktrees.
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
