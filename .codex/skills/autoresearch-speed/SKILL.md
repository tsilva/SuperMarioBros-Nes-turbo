---
name: autoresearch-speed
description: Single-threaded Super Mario Bros emulator speed-improvement loop for this repo. Use when Codex is asked to optimize, profile, benchmark, or autonomously iterate on Super Mario Bros NES throughput with Modal-judged experiments, commit/revert discipline, regression testing, and experiment tracking.
---

# Autoresearch Speed

## Operating Contract

Optimize the live repo, not a remembered snapshot. Preserve the externally observed benchmark contract unless the user explicitly changes the goal:

```bash
.venv/bin/python scripts/benchmark_sps.py --num-envs 16 --steps 500 --repeats 3
```

Expected benchmark contract:

- `obs_shape=(16, 4, 84, 84)`
- `obs_dtype=uint8`
- lanes start from `Level1-1`, `Level1-2`, `Level1-3`, and `Level1-4` repeated round-robin
- real Super Mario Bros NES reset/step behavior
- correct frame skip, frame stack, grayscale/crop/resize, action mapping, rewards, dones/truncations, reset behavior, and info scalar semantics

Do not fake throughput by skipping required emulator progression, returning stale observations, changing the public command, or weakening the benchmark workload.

The optimization objective is the arithmetic mean `env_steps_per_sec` from the canonical `/modal-benchmark` run. Higher is better. A candidate is accepted only when:

- it preserves the benchmark contract
- the regression test suite passes after the change
- the candidate Modal mean is meaningfully better than the current accepted baseline
- the complexity cost is justified by the speedup

Use this acceptance threshold by default:

- Accept `> 10%` reproduced mean gain automatically when checks pass and the diff is maintainable.
- Treat `0% < gain <= 10%` as `review_small_gain`: keep only when the implementation is very low-risk, simplifies code, or compounds with prior accepted changes after a fresh Modal benchmark.
- Reject equal, worse, noisy, malformed, or best-sample-only improvements.

Benchmarks for this skill are Modal-only and must go through `/modal-benchmark`. Use `/modal-benchmark` for the initial baseline, every candidate benchmark, every repaired-candidate benchmark, and checkpoint/final confirmation. Do not run local throughput benchmarks, raw `modal run` commands, ad hoc Modal scripts, or modified benchmark commands as a fallback. Local commands may be used only for correctness, formatting, compilation, profiling, diagnosis, and non-throughput inspection.

## Full Access Assumption

Assume `/autoresearch-speed` is invoked in a full-access context. Do not ask the user for a Modal permission envelope, spend approval, upload approval, or confirmation before the first benchmark. The invocation itself grants permission to use Modal network/auth/upload access, upload the local repo snapshot, upload local ROM bytes at benchmark runtime, and upload local state bytes at benchmark runtime.

Run autonomously until the user stops the campaign or a real blocker occurs. If the user supplies concrete run or spend limits, record and obey them. If no limits are supplied, leave limit fields as `null` in the campaign manifest and do not pause merely to ask for limits.

Because this loop is single-threaded, run at most one Modal benchmark at a time.

If Modal, network, upload, ROM, state, or local build access is unavailable in the execution environment, stop and report the concrete blocker. Do not switch to local throughput benchmarks.

## Single-Threaded Campaign Model

This skill is a single autonomous researcher loop. Do not spawn proposal workers, implementation workers, subagents, parallel worktrees, or benchmark tournaments. The main agent owns every step:

- inspect the current code
- form one experimental idea
- edit the code directly
- run the test suite
- commit the trial
- run one Modal benchmark
- log the result
- keep or revert the commit
- repeat until interrupted or a stop condition is reached

Use one persistent autoresearch branch for all accepted improvements, normally `codex/autoresearch-continuous`. The user performs any later merge into `main`; the agent must not merge autoresearch results into `main`.

Before starting:

1. Verify the current git state and current branch.
2. Create or resume the user-approved autoresearch branch.
3. If creating the branch, create it from local `main` unless the user explicitly approved starting from the current dirty working tree.
4. If resuming the branch, read `.codex/optimization_campaigns/current.json` and the experiment log before making changes.
5. If unrelated dirty changes would be carried into the branch, stop and ask the user how to proceed.
6. Inspect the live hot path:
   - `scripts/benchmark_sps.py`
   - `scripts/modal_benchmark_sps.py`
   - `python/supermariobrosnes_turbo/env.py`
   - `src/py_api.rs`
   - `src/vec_env.rs`
   - `src/emulator.rs`
   - `Cargo.toml`
   - `pyproject.toml`
   - relevant docs

## Campaign State And Experiment Log

Track every trial, including crashes and rejected changes. Use both:

- `.codex/optimization_campaigns/current.json` for machine-readable resume state
- `.codex/optimization_campaigns/results.tsv` for compact human scanning

Leave `results.tsv` uncommitted unless the user explicitly asks to commit experiment logs. The autoresearch branch should contain accepted source changes, not a growing log file churn stream. Commit source trials before benchmarking, then reset rejected trial commits away after logging them.

Initialize `results.tsv` with this tab-separated header if it does not exist:

```text
epoch	commit	mean_env_steps_per_sec	stdev_env_steps_per_sec	best_env_steps_per_sec	gain_pct	status	description	artifact
```

Use these statuses:

- `baseline`
- `keep`
- `keep_small_gain`
- `discard`
- `crash`
- `regression_fixed_keep`
- `regression_unfixed_discard`
- `inconclusive`

The manifest should include at least:

```json
{
  "campaign_id": "sps-continuous",
  "mode": "single-thread-continuous",
  "full_access_invocation": true,
  "started_at": "ISO-8601",
  "root_branch": "main",
  "orchestration_branch": "codex/autoresearch-continuous",
  "epoch_index": 0,
  "created_from_sha": "main sha",
  "resumed_from_sha": null,
  "allowed_benchmark_skill": "/modal-benchmark",
  "allowed_output_root": "artifacts/benchmarks/",
  "max_total_modal_runs": null,
  "max_estimated_spend_usd": null,
  "modal_runs_used": 0,
  "current_baseline_artifact": null,
  "current_baseline_mean_env_steps_per_sec": null,
  "accepted_commits": [],
  "discarded_commits": [],
  "current_experiment": null,
  "stop_reason": null
}
```

## Required Checks

Run the regression suite before every candidate Modal benchmark. At minimum:

```bash
cargo fmt --check
cargo check --release
cargo test --release
.venv/bin/python -m maturin develop --release
.venv/bin/python scripts/check_vec_env_equivalence.py
.venv/bin/python scripts/smoke_smb.py
```

If the repo has additional tests relevant to the changed surface, run them too. Add or update targeted checks when changing observations, rewards, termination flags, reset behavior, noop stepping, uniform-action lanes, divergent-action lanes, action mapping, info fields, render/preprocessing bytes, or benchmark parsing.

If sandboxing blocks uv/pip/cache writes, rerun with the required approval and explain why.

If tests fail after an optimization:

1. Treat the failure as a regression unless proven unrelated.
2. Try to fix the regression while preserving the optimization.
3. Rerun the failing test first, then rerun the required checks.
4. If the fix works, commit the repaired candidate and benchmark it.
5. If the regression cannot be fixed after a few focused attempts, log `regression_unfixed_discard`, reset the trial away, and move on.

Do not benchmark a candidate on Modal until the required checks pass.

## Experiment Loop

The first Modal run on a fresh campaign is always the baseline from the unmodified campaign branch. Record it in the manifest and `results.tsv` with status `baseline`.

Then loop:

1. Look at the current git state and manifest.
2. Choose one concrete optimization idea.
3. Edit the code directly in the campaign branch.
4. Run formatting, build, and targeted/local checks as needed while developing.
5. Run the required regression checks.
6. If checks fail, attempt to repair the regression while preserving the optimization. If repair fails, log and reset the trial away.
7. Commit the candidate source change before benchmarking. Use a concise commit message that names the optimization.
8. Run one `/modal-benchmark` from that commit and save the JSON artifact under `artifacts/benchmarks/`.
9. Parse mean, stdev, best, per-level samples when available, run URL, and artifact path.
10. Append a row to `results.tsv`.
11. Decide:
    - `keep`: candidate beats the current baseline by more than 10%, checks passed, and complexity is acceptable.
    - `keep_small_gain`: candidate improves less than or equal to 10% but is simple, low-risk, or simplifying.
    - `discard`: candidate is equal, slower, too noisy, too complex for its gain, or weakens the contract.
    - `inconclusive`: benchmark metadata is malformed, unusually noisy, or not comparable.
12. If keeping, leave the commit on the campaign branch, update `current_baseline_*`, append to `accepted_commits`, increment `epoch_index`, and continue from the improved branch.
13. If discarding, record the candidate commit in `discarded_commits`, reset the branch back to the pre-experiment commit, increment `epoch_index`, and continue.

Never assume independent gains add. Every kept commit becomes the new source baseline and every later candidate is judged against a fresh Modal benchmark from the current campaign branch.

## Commit And Revert Discipline

Every candidate gets a commit before Modal benchmarking. Rejected commits must not remain in the branch history. Use non-interactive git commands:

- Record the pre-experiment SHA before editing.
- Commit the candidate after tests pass.
- If rejected, reset the branch back to the recorded pre-experiment SHA.
- If accepted, keep the commit and update the baseline.

Do not revert unrelated user changes. Do not reset past accepted autoresearch commits. If the branch contains unexpected unrelated changes, stop and report the blocker.

## Optimization Guidance

Optimize aggressively but honestly:

- Use local profiling and instrumentation only for diagnosis, never as throughput evidence.
- Separate Python boundary cost, Rust vector-env scheduling, CPU emulation, PPU/rendering, resize/preprocessing, stack movement, and output-buffer copying.
- Favor Rust-side changes in `src/emulator.rs`, `src/vec_env.rs`, and `src/py_api.rs` when they preserve behavior.
- Mario/NES-specific shortcuts are allowed only when they preserve observed SMB behavior for this repo's supported game.
- Document important shortcut assumptions in `docs/PERFORMANCE_PLAN.md`.
- Prefer simple improvements over complex low-gain tricks.
- Removing code while preserving or improving speed is a strong keep signal.

Unsupported or intentionally narrow cases are acceptable when documented:

- only Super Mario Bros mapper 0 / NROM is in scope
- no audio requirement
- no general Gym Retro or arbitrary NES mapper compatibility requirement
- RGB and uncropped renderers are compatibility paths, not the primary optimized RL benchmark path

Preserve or deliberately replace these fast-path assumptions with stronger checks:

- Deterministic synced lanes: after reset, identical lanes can share one emulator state while all actions are uniform; the state must materialize into independent lanes before mixed actions.
- Cropped grayscale tile rendering: the RL benchmark path emits SMB/NES background tile-row runs and then applies sprite overlay semantics.

## Stop And Pause Conditions

Once the loop begins, do not ask "should I continue?" after each experiment. Continue autonomously while access remains.

Pause cleanly, update the manifest, and report the exact resume state when any of these happen:

- user-provided Modal run or spend limit is exhausted
- required Modal, network, upload, ROM, state, or local build access becomes unavailable
- the same regression cannot be fixed after a few focused attempts
- benchmark metadata is too noisy or incomparable to trust
- the campaign branch has unexpected unrelated changes
- the user asks to pause or stop

When paused, leave all accepted improvements committed on the persistent branch, leave rejected or stale experiments out of branch history, update `.codex/optimization_campaigns/current.json`, and report:

- branch name
- current accepted baseline artifact and mean
- accepted commits
- discarded experiment count
- Modal runs used and remaining, if a limit was provided
- next plausible experiment idea
- whether the branch appears fast-forwardable from `main`

Do not switch to `main`, merge into `main`, delete the orchestration branch, or push unless the user explicitly gives a separate instruction for that specific action.

## Report Format

Checkpoint and pause reports should include:

- orchestration branch name
- campaign mode and epoch index
- remaining Modal run/spend budget, if a limit was provided
- baseline and latest accepted samples
- mean, stdev, best, gain percentage, and speedup multiplier
- checks run
- changed files
- accepted and discarded commit counts
- paste-ready manual playback commands:

```bash
.venv/bin/python scripts/play.py --mode external --view raw --state Level1-1 --scale 3
.venv/bin/python scripts/play.py --mode external --view preprocessed --state Level1-1 --frame-skip 4 --frame-stack 4 --crop-top 32 --crop-bottom 0 --resize-width 84 --resize-height 84 --scale 4
```

If state files require a non-default location, append `--state-dir <path>` to both commands.
