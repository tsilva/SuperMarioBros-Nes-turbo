---
name: autoresearch-speed
description: Single-threaded Super Mario Bros emulator speed-improvement loop for this repo. Use when optimizing, profiling, benchmarking, or autonomously iterating on Super Mario Bros NES throughput with Modal-judged experiments, make-test regression gating, commit/revert discipline, and experiment tracking.
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

Throughput evidence is Modal-only and must go through `/modal-benchmark`.
Local commands are for correctness, compilation, formatting, profiling, and
diagnosis only, never acceptance.

## Full Access

Assume `/autoresearch-speed` is invoked with full access. Do not ask for Modal
permission, upload approval, spend approval, or confirmation before benchmarking.
The invocation grants Modal network/auth/upload, repo snapshot upload, local ROM
byte upload, and local state byte upload.

If the user provides Modal run or spend limits, record and obey them. If not,
leave limit fields as `null` and continue until stopped or blocked. Run at most
one Modal benchmark at a time.

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
5. If unrelated dirty changes would be carried in, stop and ask.
6. Inspect the hot path: `scripts/benchmark_sps.py`,
   `scripts/modal_benchmark_sps.py`, `python/supermariobrosnes_turbo/env.py`,
   `src/py_api.rs`, `src/vec_env.rs`, `src/emulator.rs`, `Cargo.toml`,
   `pyproject.toml`, and relevant docs.

Track every trial, including crashes and rejects:

- `.codex/optimization_campaigns/current.json` for resume state
- `.codex/optimization_campaigns/results.tsv` for human scanning

Keep `results.tsv` uncommitted unless the user asks to commit logs. Accepted
source commits stay on the campaign branch; rejected commits are reset away.

`results.tsv` header:

```text
epoch	commit	mean_env_steps_per_sec	stdev_env_steps_per_sec	best_env_steps_per_sec	gain_pct	status	description	artifact
```

Statuses: `baseline`, `keep`, `keep_small_gain`, `discard`, `crash`,
`regression_fixed_keep`, `regression_unfixed_discard`, `inconclusive`.

Manifest fields should include campaign id/mode, branch names, root SHA, epoch,
allowed benchmark skill/output root, optional run/spend limits, Modal runs used,
current baseline artifact/mean, accepted commits, discarded commits, current
experiment, and stop reason.

## Required Checks

Before every candidate Modal benchmark, run:

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

1. Run the initial `/modal-benchmark` baseline from the unmodified campaign
   branch.
2. Record mean, stdev, best, samples, artifact, metadata, and baseline status.

Each experiment:

1. Record pre-experiment SHA.
2. Choose one concrete optimization idea.
3. Edit directly on the campaign branch.
4. Run local diagnosis/build checks as needed.
5. Run required checks.
6. Commit the candidate before benchmarking.
7. Run exactly one `/modal-benchmark` from that commit.
8. Append a result row.
9. Decide:
   - `keep`: reproduced mean gain `> 10%`, checks pass, complexity acceptable.
   - `keep_small_gain`: `0% < gain <= 10%` only if simple, low-risk,
     simplifying, or compounding.
   - `discard`: equal/slower/noisy/too complex/contract weakening.
   - `inconclusive`: malformed, too noisy, or incomparable metadata.
10. If kept, update baseline fields and continue from the improved branch.
11. If rejected, reset back to pre-experiment SHA and continue.

Never assume independent gains add. Every accepted commit becomes the new source
baseline and later candidates are judged against a fresh Modal benchmark.

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

On pause, leave accepted commits on the campaign branch, rejected commits out of
history, update campaign state, and report:

- branch, mode, epoch
- baseline/latest accepted samples, mean, stdev, best, gain, speedup
- accepted commits and discarded count
- checks run
- changed files
- Modal runs/remaining limits if provided
- next plausible experiment
- whether the branch appears fast-forwardable from `main`

Do not switch to `main`, merge, delete the branch, push, or commit experiment
logs unless the user explicitly asks.

Include paste-ready playback commands when reporting a result:

```bash
.venv/bin/python scripts/play.py --mode external --view raw --state Level1-1 --scale 3
.venv/bin/python scripts/play.py --mode external --view preprocessed --state Level1-1 --frame-skip 4 --frame-stack 4 --crop-top 32 --crop-bottom 0 --resize-width 84 --resize-height 84 --scale 4
```
