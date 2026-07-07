---
name: autoresearch-speed
description: Lightweight Super Mario Bros NES throughput optimization loop for this repo. Use when optimizing, profiling, benchmarking, or iterating on emulator speed around env_steps_per_sec without feature or accuracy regressions.
---

# Autoresearch Speed

## Non-Negotiables

Before optimizing, read this skill's `SPECS.md` and repo-root `SPECS.md`.
Optimize the live repo on the current branch. Do not create or switch branches,
fork workers, use SSH/tailnet/cloud/Modal, merge, push, or delete branches
unless the user explicitly approves that in the current turn.

Only valid `env_steps_per_sec` gains count. Every candidate needs:
- A concrete speed mechanism.
- The cheapest falsification path before exact-ref benchmarking.
- No risk to public API shape, deterministic lane behavior, observation bytes,
  rewards, resets, terminations, infos, state loading, or benchmark workload.

Required paths come from CLI flags, environment, or `.env`: `ROM_PATH` for the
SMB ROM and `AUTORESEARCH_ROOT_PATH` for mutable research state. Keep mutable
artifacts out of the repo; the controller owns the root's `benchmarks/`,
`states/`, `candidates/`, `current.json`, `results.tsv`, `ideas.md`, and
`scratchpad.md`.

## Control Plane

Use `scripts/autoresearch.py` as the only routine interface:

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

The controller owns command shapes, benchmark sizes, exact-ref tiers, checks,
recording, and next-step hints. Use `scripts/run_git_ref_benchmark.py`,
`RESULTS_TSV_COLUMNS`, and `scripts/benchmark_sps.py` only as implementation
truth. Do not duplicate, rename, split, or alias benchmark fields. If prose and
code disagree, trust the code and fix the prose.

## Loop

1. `orient`: run `status`, read specs, inspect `ideas.md` and latest results,
   then run `next`.
2. `falsify`: use source inspection, prior rejects, `probe`,
   `diagnose --quick`, `diagnose --profile`, or a narrow equivalence check.
   These can justify discard or continued work, never a keep.
3. `prototype`: make one small edit. After native edits, rebuild with
   `.venv/bin/python -m maturin develop --release` before Python timing.
4. `check`: before exact-ref triage, run
   `scripts/autoresearch.py checks --quick --surface <surface>`. Add targeted
   tests first for observations, rewards, terminations, resets, actions, infos,
   preprocessing, state loading, or public Python contracts.
5. `screen`: commit only plausible candidates, then run
   `scripts/autoresearch.py screen <baseline_ref> <candidate_ref>`. One default
   screen per candidate is enough unless asked otherwise.
6. `accept`: for promoted candidates, contract-sensitive changes, public
   claims, baseline resets, or direct user requests, run full controller checks
   and `scripts/autoresearch.py accept <baseline_ref> <candidate_ref>`. Use
   `--full` only when the claim or user request needs the full ladder.
7. `record`: finalized `screen`, `accept`, and `calibrate` runs auto-record by
   default. Use `record` only for manual imports or status overrides.
8. `retrospect`: put compact lessons in `scratchpad.md`; do not update docs or
   skills from scratchpad notes unless asked.

`probe`, `diagnose`, and `make benchmark` are learning evidence only.
`local_triage` is screening only. `local_acceptance` is the only tier that can
justify `keep`, `keep_small_gain`, accepted speedups, or baseline updates.

Discard when the likely gain is smaller than the cost of proving it. Treat
failures as regressions unless proven unrelated. Do not accept busy-machine or
load-failed official measurements unless the user explicitly forces or waives
that gate. Use `calibrate <ref>` periodically for accepted-baseline variance;
calibration is context, not paired acceptance.

Prefer Rust hot-path changes in `src/emulator.rs`, `src/vec_env.rs`, and
`src/py_api.rs`. Preserve deterministic lanes, public Python API shape,
benchmark workload identity, observation/reward/termination semantics, state
handling, and cropped grayscale tile/sprite rendering semantics.

## Reporting

Report compactly:

```text
phase: orient | diagnose | prototype | screen | acceptance | record | pause
inputs: refs/artifacts read
outputs: commits/results/artifacts written
decision: keep | keep_small_gain | discard | inconclusive | defer | pause
next: one concrete next action
```

Pause when access fails, user limits are exhausted, regressions persist, refs are
missing, metadata or timing is untrustworthy, unrelated branch changes would be
included, two candidates die for the same reason, or the user asks to stop.

At session end, report branch, baseline/candidate refs, latest aggregate,
accepted commits, rejects, checks, changed files, cleanup status, benchmark
limits, artifacts, and the next experiment.
