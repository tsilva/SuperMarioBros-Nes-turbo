---
name: autoresearch-speed
description: Lightweight Super Mario Bros NES throughput optimization loop for this repo. Use when optimizing, profiling, benchmarking, or iterating on emulator speed around env_steps_per_sec without feature or accuracy regressions.
---

# Autoresearch Speed

## Contract

Read this skill's `SPECS.md` before optimizing. It is the invariant contract:
accuracy is non-negotiable, benchmark semantics stay canonical, and only valid
`env_steps_per_sec` gains count.

Optimize the live repo on the current branch. Do not create or switch branches,
fork workers, use SSH/tailnet/cloud/Modal, merge, push, or delete branches
unless the user explicitly approves that in the current turn.

Every candidate needs:

- A concrete speed mechanism.
- The cheapest falsification path before exact-ref benchmarking.
- No risk to public API shape, deterministic lane behavior, observation bytes,
  rewards, resets, terminations, infos, state loading, or benchmark workload.

Required paths, from CLI flags, environment, or `.env`:

- `ROM_PATH`: SMB NES ROM.
- `AUTORESEARCH_ROOT_PATH`: existing directory for mutable autoresearch state.

Keep mutable research artifacts out of the repo by default. The root owns
`benchmarks/`, `states/`, `candidates/`, `current.json`, `results.tsv`,
`ideas.md`, and `scratchpad.md`.

## Sources of Truth

Use `scripts/autoresearch.py` as the loop controller. It owns the normal command
shapes, benchmark sizes, exact-ref tiers, checks, recording, and next-step hints:

```bash
.venv/bin/python scripts/autoresearch.py init
.venv/bin/python scripts/autoresearch.py status
.venv/bin/python scripts/autoresearch.py next
.venv/bin/python scripts/autoresearch.py probe [--dry-run]
.venv/bin/python scripts/autoresearch.py diagnose [--quick] [--profile] [--dry-run]
.venv/bin/python scripts/autoresearch.py screen <baseline_ref> <candidate_ref> [--dry-run] [--no-record] [-- <extra runner args>]
.venv/bin/python scripts/autoresearch.py checks [--quick] [--dry-run] [--surface auto|native|python|benchmark]
.venv/bin/python scripts/autoresearch.py accept <baseline_ref> <candidate_ref> [--full] [--dry-run] [--no-record] [-- <extra runner args>]
.venv/bin/python scripts/autoresearch.py calibrate <ref> [--full] [--dry-run] [--no-record] [-- <extra runner args>]
.venv/bin/python scripts/autoresearch.py record <aggregate.json> [--status <status>] [--description "..."] [--artifact <path>]
```

Also defer to:

- `scripts/run_git_ref_benchmark.py` for official benchmark semantics.
- `RESULTS_TSV_COLUMNS` for the exact `results.tsv` schema.
- `scripts/benchmark_sps.py` for the live current-checkout SPS workload.
- Repo-root `SPECS.md` for durable package/API/release requirements.

Do not duplicate, rename, split, or alias benchmark fields in this skill or in
the ledger. If prose and code disagree, trust the code and fix the prose.

## Loop

1. `orient`: run `status`, verify git state/current branch, read both SPECS
   files, inspect `results.tsv` and `ideas.md`, then run `next` and choose the
   highest expected verified SPS gain per unit time.
2. `falsify`: use source inspection, prior rejects, `probe`, `diagnose --quick`,
   `diagnose --profile`, or a narrow equivalence check. Local probe/diagnosis is
   learning evidence only; it can justify discard or deeper work, not a keep.
3. `prototype`: make one small edit. After native edits, rebuild with
   `.venv/bin/python -m maturin develop --release` before Python timing.
4. `check`: before triage, run
   `scripts/autoresearch.py checks --quick --surface <surface>` matching the
   touched surface. Add targeted tests before triage when touching observations,
   rewards, termination, reset behavior, action mapping, info fields,
   preprocessing bytes, state loading, or Python API contracts.
5. `screen`: commit only plausible candidates, then run exact-ref
   `local_triage` with `scripts/autoresearch.py screen`. One default screen per
   candidate is enough unless the user asks otherwise.
6. `accept`: for clear wins, low-risk small wins, contract-sensitive changes, or
   direct user requests, run full controller checks and then
   `scripts/autoresearch.py accept`. Use `accept --full` for public claims,
   baseline resets, noisy contenders worth more samples, or user-requested full
   ladders.
7. `record`: `screen`, `accept`, and `calibrate` auto-record finalized
   aggregates by default. Use `record` only for manual/imported aggregates or
   status overrides.
8. `retrospect`: append compact lessons to `scratchpad.md` when a mistake,
   hindsight conclusion, friction, missing check, or heuristic would make the
   next run stronger. Do not update docs or skills from scratchpad notes unless
   asked.

## Evidence Ladder

- `probe`, `diagnose`, and `make benchmark`: learning evidence only.
- `local_triage`: exact-ref screening only.
- `local_acceptance`: the only tier that can justify `keep`,
  `keep_small_gain`, accepted speedups, or baseline updates.

Use `calibrate <ref>` periodically on the dedicated host to refresh single-ref
variance for the accepted baseline; calibration is context, not a substitute for
paired acceptance.

Discard quickly when profile share, prior rejects, smoke timings, tests, or
benchmark noise make the plausible gain smaller than the cost of proving it.

Treat failures as regressions unless proven unrelated. If repair does not
converge after focused attempts, record the rejection, reset the trial, and move
on. Do not accept busy-machine or load-failed official measurements unless the
user explicitly forces or waives that validity gate.

Prefer Rust hot-path changes in `src/emulator.rs`, `src/vec_env.rs`, and
`src/py_api.rs`. Scope limits are intentional: SMB mapper 0 / NROM only, no
audio requirement, no general Gym Retro/arbitrary NES mapper compatibility, and
RGB/uncropped renderers as compatibility paths.

Preserve or strengthen deterministic lane behavior, public Python API shape,
benchmark workload identity, observation/reward/termination semantics, state
handling, and cropped grayscale tile/sprite rendering semantics.

## Reporting

Use compact phase reports:

```text
phase: orient | diagnose | prototype | screen | acceptance | record | pause
inputs: refs/artifacts read
outputs: commits/results/artifacts written
decision: keep | keep_small_gain | discard | inconclusive | defer | pause
next: one concrete next action
```

Pause when access fails, user limits are exhausted, regressions persist, refs are
missing, metadata/timing is untrustworthy, unrelated branch changes would be
included, low-ROI leftovers remain, two candidates die for the same reason, or
the user asks to stop.

At session end, report branch, baseline/candidate refs, latest aggregate,
accepted commits, rejects, checks, changed files, cleanup status, benchmark
limits, artifacts, and the next experiment. Preserve reviewed ledgers and
official aggregates; remove abandoned scratch outputs and temporary benchmark
byproducts.
