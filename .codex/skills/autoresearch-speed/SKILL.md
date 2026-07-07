---
name: autoresearch-speed
description: Lightweight Super Mario Bros NES throughput optimization loop for this repo. Use when optimizing, profiling, benchmarking, or iterating on emulator speed around env_steps_per_sec without feature or accuracy regressions.
---

# Autoresearch Speed

## Contract

Read this skill's `SPECS.md` before optimizing. Always obey its `PURPOSE` and
`REQUIREMENTS`; never accept speedups that violate them, and never modify
`SPECS.md` during an autoresearch-speed run.

Optimize the live repo on the current branch. Do not create or switch branches,
fork workers, use SSH/tailnet/cloud/Modal, merge, push, or delete branches
unless the user explicitly approves that in the current turn.

Maximize the benchmark output variable `env_steps_per_sec`: largest valid gain,
least wall-clock time, fewest useful tokens. Every candidate needs a concrete
speed mechanism, cheap falsification path, and no environment-contract risk.

Required path variables, from CLI flags, environment, or `.env`:

- `ROM_PATH`: SMB NES ROM.
- `AUTORESEARCH_ROOT_PATH`: existing directory for all mutable autoresearch
  state. Tools must fail if it is unset, missing, or not a directory.

Keep mutable research artifacts outside the repo under `AUTORESEARCH_ROOT_PATH`:
`benchmarks/`, `states/`, `candidates/`, `current.json`, `results.tsv`,
`ideas.md`, and `scratchpad.md`.

## Controller

Use `scripts/autoresearch.py` as the loop controller. It owns the standard
diagnosis, screening, acceptance, check, and recording command shapes:

```bash
.venv/bin/python scripts/autoresearch.py diagnose [--profile]
.venv/bin/python scripts/autoresearch.py screen <baseline_ref> <candidate_ref> [--dry-run] [-- <extra runner args>]
.venv/bin/python scripts/autoresearch.py checks
.venv/bin/python scripts/autoresearch.py accept <baseline_ref> <candidate_ref> [--dry-run] [-- <extra runner args>]
.venv/bin/python scripts/autoresearch.py record <aggregate.json> [--description "..."] [--artifact <path>]
```

`scripts/run_git_ref_benchmark.py` owns official benchmark semantics.
`RESULTS_TSV_COLUMNS` is the source of truth for `results.tsv`; do not duplicate,
rename, split, or alias benchmark fields in the skill or ledger.

## Loop

1. `orient`: verify git state/current branch, read `SPECS.md`, inspect
   `AUTORESEARCH_ROOT_PATH/results.tsv` and `ideas.md`, and pick the highest-ROI
   live hypothesis.
2. `diagnose`: run the cheapest source inspection, profiler, smoke timing, or
   narrow equivalence check that can falsify the mechanism.
3. `prototype`: make one small edit; after native edits, rebuild with
   `.venv/bin/python -m maturin develop --release` before Python timings.
4. `screen`: only committed plausible candidates get exact-ref `local_triage`
   through `scripts/autoresearch.py screen`.
5. `accept`: after a promoted screen or explicit user request, run
   `scripts/autoresearch.py checks`, then `scripts/autoresearch.py accept`.
6. `record`: write every keep, discard, crash, skip, or inconclusive result with
   `scripts/autoresearch.py record` before starting the next idea.
7. `retrospect`: append compact lessons to
   `AUTORESEARCH_ROOT_PATH/scratchpad.md` when a mistake, hindsight conclusion,
   friction, missing check, or heuristic would make the next run stronger. The
   user reviews these later; do not update docs or skills from scratchpad notes
   unless asked.

## Decisions

Fast local diagnosis is learning evidence only. Exact-ref `local_triage` is
screening only. Only `local_acceptance` can justify `keep`, `keep_small_gain`,
accepted speedups, or baseline updates.

Use one default screen per candidate. Escalate to acceptance only for clear
wins, low-risk small wins, contract-sensitive questions that need official
evidence, or direct user requests. Discard ideas quickly when profile share,
prior rejects, smoke timings, tests, or benchmark noise make the plausible gain
smaller than the cost of proving it.

Before triage, run the cheapest checks matching the touched surface. Before
acceptance, run the controller checks. Add targeted tests before triage when
touching observations, rewards, termination, reset behavior, action mapping,
info fields, preprocessing bytes, state loading, or Python API contracts.

Treat failures as regressions unless proven unrelated. If repair does not
converge after focused attempts, record the rejection, reset the trial away, and
move on. Do not accept from busy-machine or load-failed official measurements
unless the user explicitly forced or waived that validity gate.

Prefer Rust-side hot-path changes in `src/emulator.rs`, `src/vec_env.rs`, and
`src/py_api.rs`. Accepted scope limits: SMB mapper 0 / NROM only, no audio
requirement, no general Gym Retro/arbitrary NES mapper compatibility, and
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

Pause when access fails, user limits are exhausted, regressions persist,
metadata is untrustworthy, unrelated branch changes would be included, refs are
missing, timing is contaminated, low-ROI leftovers remain, two candidates die
for the same reason, or the user asks to stop.

At session end, report branch, baseline/candidate refs, latest aggregate,
accepted commits, rejects, checks, changed files, cleanup status, benchmark
limits, artifacts, and the next experiment. Preserve reviewed ledgers and
official aggregates; remove abandoned scratch outputs and temporary benchmark
byproducts.
