---
name: autoresearch-speed
description: Lightweight Super Mario Bros NES throughput optimization loop for this repo. Use when optimizing, profiling, benchmarking, or iterating on emulator speed around env_steps_per_sec without feature or accuracy regressions, including repeated +10% cumulative improvement rounds resumed from AUTORESEARCH_ROOT_PATH state.
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

## Round Contract

Each time this skill is launched for open-ended throughput work, run one
improvement round. The round goal is always at least a 10% cumulative
`env_steps_per_sec` gain over the round baseline, unless the user explicitly
asks for a larger target in the current turn. Do not finish a round by merely
rediscovering that an older accepted commit is already fast enough.

At orientation, derive the round baseline from `AUTORESEARCH_ROOT_PATH`:

- Read `results.tsv`, `current.json`, `scratchpad.md`, and `status`.
- Select the latest recorded accepted result with status `keep`, `keep_stack`,
  or `keep_small_gain` as the baseline for the new round.
- If no accepted result exists, use the current checked-out `HEAD` as the first
  baseline after running a calibration or equivalent controller-supported
  baseline measurement.
- Treat the selected round baseline as fixed for that skill invocation. Small
  accepted gains can be stacked, but the round is complete only when the
  candidate is accepted against the fixed round baseline at ratio `>= 1.10`.
- If live `HEAD` already contains unaccepted work above the baseline, it can be
  part of the round only after controller evidence accepts it against the fixed
  baseline. Otherwise continue optimizing from the live tree and do not claim
  completion.

When the round reaches `>= 1.10` with accepted evidence, finish like any other
accepted stack: record the aggregate, write the lesson to `scratchpad.md`, leave
the work committed on the current branch, report the artifacts and next
baseline, and stop. On the next launch, use the latest accepted result stored in
`AUTORESEARCH_ROOT_PATH` as the new round baseline and repeat the same process.

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
.venv/bin/python scripts/autoresearch.py accept-stack <baseline_ref> <candidate_ref> [--dry-run] [--no-record] [-- <extra runner args>]
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

1. `orient`: run `status`, read specs, inspect `ideas.md`, `current.json`,
   `scratchpad.md`, and latest `results.tsv` entries. Lock the round baseline
   and target from the Round Contract, then run `next`.
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
   screen per candidate is enough unless asked otherwise. Cheap screens may
   promote clear, low-risk likely winners into a bundle instead of proving each
   patch individually.
6. `accept-stack`: for a bundle of screened likely winners, run correctness
   checks once, then run
   `scripts/autoresearch.py accept-stack <baseline_ref> <candidate_ref>`.
   Use this when the bundle plausibly has at least 5% upside, has one obvious
   monster candidate, reaches about three good candidates, or follows about
   eight cheap screens. For round completion, `<baseline_ref>` must be the fixed
   round baseline and the accepted median ratio must be `>= 1.10`. The proof
   unit is the whole stack; attribution cleanup is optional after acceptance.
7. `accept`: for strongest final evidence, contract-sensitive changes, public
   claims, baseline resets, or direct user requests, run full controller checks
   and `scripts/autoresearch.py accept <baseline_ref> <candidate_ref>`. Use
   `--full` only when the claim or user request needs the full ladder.
8. `record`: finalized `screen`, `accept-stack`, `accept`, and `calibrate`
   runs auto-record by default. Use `record` only for manual imports or status
   overrides.
9. `retrospect`: put compact lessons in `scratchpad.md`; do not update docs or
   skills from scratchpad notes unless asked.

`probe`, `diagnose`, and `make benchmark` are learning evidence only.
`local_triage` is screening only. `stack_acceptance` can justify `keep_stack`
for a bundled patch stack, but does not prove individual attribution.
`local_acceptance` remains the strongest tier for `keep`, `keep_small_gain`,
accepted speedups, or baseline updates that need gold-standard evidence. Do not
mark a round complete from `inconclusive`, load-failed, busy-machine, or
non-recorded evidence, even if the raw ratio is above 1.10.

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
decision: keep | keep_small_gain | keep_stack | discard | discard_stack | inconclusive | defer | pause
next: one concrete next action
```

Pause when access fails, user limits are exhausted, regressions persist, refs are
missing, metadata or timing is untrustworthy, unrelated branch changes would be
included, two candidates die for the same reason, or the user asks to stop.

At session end, report branch, baseline/candidate refs, latest aggregate,
accepted commits, rejects, checks, changed files, cleanup status, benchmark
limits, artifacts, round baseline, round target, cumulative ratio versus the
round baseline, and the next experiment.
