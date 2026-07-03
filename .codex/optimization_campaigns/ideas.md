# Autoresearch Ideas Queue

## Prerequisites

- Build a low-overhead hot-path profiler before selecting another speed
  candidate. This is infrastructure for choosing ideas, not itself an
  optimization idea. The profiler should be disabled by default and should not
  be judged by host benchmark speedup unless it accidentally changes the normal
  hot path.
- Profiler evidence now exists. Policy-completion profile:
  `artifacts/benchmarks/policy-profile-level1-1-native-maxpool-levelchange-strict-20260701-224421.json`.
  Canonical local validation profile:
  `artifacts/benchmarks/local-profile-validation-20260701-224610.json`.
  These are local diagnostic artifacts only; they rank candidate mechanisms but
  do not establish accepted speed wins.

## Ready

Curated order: highest estimated ROI first. ROI is judged from local profiler
evidence, expected host benchmark signal, implementation size, correctness risk,
and the existing reject/keep ledger. Local profiles rank candidates only; the
fixed host benchmark is the acceptance source of truth.

### IDEA-20260701-006: Tune Rayon Chunking For Group Leaders

- Status: ready
- Perspective: vec-env
- Estimated ROI: lowest among current ready ideas. Group sharing itself already
  won; local profiling shows copy cost is tiny and the mask-reuse follow-up was
  a large benchmark regression.
- Hypothesis: The accepted grouped-lane candidate wins by stepping repeated
  state group leaders in parallel, then copying peer outputs. The remaining
  overhead may be Rayon scheduling and cache behavior for a small number of
  leaders. A fixed chunking or small-group path could reduce scheduling cost
  without changing semantics.
- Target files: `src/vec_env.rs`
- Prior evidence: The single-thread grouped attempt was slower, while the
  parallel grouped-leader candidate was kept. The follow-up mask-reuse attempt
  was discarded, so avoid that mechanism. The 2026-07-01 canonical local
  profiler showed `group_hit_rate=1.0`, `6000` group leaders, `18000` peer
  copies, and about `460ns` grouped-copy time per env step. This makes the idea
  plausible but lower priority than CPU/render work; do not chase observation
  copy overhead first.
- Plan: Only try after the CPU/render candidates stall. Test one scheduling
  change at a time: fixed small-group leader arrays, explicit leader index list
  reuse, or a branch that uses sequential stepping when the leader count is
  below the measured break-even point.
- Contract risks: Accidentally serializing too much work, changing reset/group
  materialization semantics, or repeating the discarded mask-reuse mechanism.
- Required checks: `scripts/check_vec_env_equivalence.py`, full required checks,
  and the fixed-host paired benchmark.
- Expected benchmark signal: Rough expectation `+1%` to `+4%`; discard quickly
  if host variance or scheduling overhead hides the signal.

## Done

### IDEA-20260701-007: Hot Basic-Block Interpreter Prototype

- Status: keep-small-gain-combined
- Perspective: emulator-core
- Result: restored as commit `1c2178d` after combined validation versus `main`;
  standalone trial commit `a262398`, artifact
  `artifacts/benchmarks/host-results/host-compare-2026-07-02-220902-B702255342113-Ca262398ee7d6/aggregate.json`.
- Reason: The bounded prototype specialized SMB's `$8223` OAM clear helper
  rather than adding a broad block cache. It passed `cargo fmt --check`,
  `cargo check --release`, `.venv/bin/python -m maturin develop --release`,
  and `make test`, including an interpreted-routine equivalence test. Full
  official beast-3 comparison versus `7022553` measured median pair ratio
  `1.0352329816719152`, CI95 `[1.0306466330220425, 1.0369758374516294]`,
  and `11/11` faster pairs. This was a real positive signal but below the
  campaign's original per-candidate `>=5%` keep threshold. On 2026-07-03 the
  change was restored on top of the kept sprite-0 polling optimization and
  compared directly against `main` (`33c273d`) in a high-confidence beast-3
  21-pair run. The combined branch measured median pair ratio
  `1.0821469471532668`, CI95 `[1.080402979603871, 1.0859394974851821]`, and
  `21/21` faster pairs at artifact
  `artifacts/benchmarks/host-results/host-compare-2026-07-03-102429-B33c273d4ef30-C1c2178db0ca3/aggregate.json`.
  The strict IQR outlier gate still flagged pair ratios 1 and 4, but all pair
  ratios were positive (`1.069x` minimum), MAD outliers were empty, run-median
  CV was below 1%, and the bootstrap lower bound stayed above `+8%`, so the
  combined change was accepted for merge.

### IDEA-20260701-003: Profile-Guided Audio Routine Fast-Forward

- Status: discard
- Perspective: emulator-core
- Result: analysis-only discard, diagnostic artifact
  `artifacts/benchmarks/local-profile-audio-candidate-baseline-20260702.json`.
- Reason: A high-`top_n` local profiler pass on the accepted `7022553` baseline
  showed the hot `$F200-$F2FF` page is mostly sprite/object helper code
  (`$F1F6-$F2C9`) that writes gameplay-visible sprite RAM around `$0200`, not
  a side-effect-limited audio loop. The likely audio entry at `$F2D0` and
  immediate follow-up range `$F2D0-$F2EF` accounted for only about `310k` of
  `108.5M` CPU steps in the diagnostic run. A narrow audio skip cannot plausibly
  reach the `>=5%` keep threshold, while a broader `$F2xx` skip would be
  correctness-risky, so no beast-3 benchmark was spent.

### IDEA-20260701-005: Cache Sprite Overlay Background Priority Data

- Status: discard
- Perspective: ppu-render
- Result: commit `ef959e1`, artifact
  `artifacts/benchmarks/host-results/host-compare-2026-07-02-213419-B702255342113-Cef959e1f9b2d/aggregate.json`.
- Reason: A narrow merge of background opacity/color lookup for behind-background
  sprite priority paths passed the full local checks, including a targeted
  sprite-priority regression test, but full official beast-3 comparison versus
  `7022553` measured median pair ratio `0.993356136665428`, CI95
  `[0.9910008932257317, 0.9986968827492797]`, and only `1/11` faster pairs.
  The aggregate was valid and below both neutral and the `>=5%` keep threshold.

### IDEA-20260701-002: Fast-Forward SMB Sprite-0 Polling

- Status: keep
- Perspective: emulator-core
- Result: commit `7022553`, artifact
  `artifacts/benchmarks/host-results/host-compare-2026-07-02-202650-B86a3a5f4602f-C702255342113/aggregate.json`.
- Reason: Full official beast-3 paired comparison versus `86a3a5f` measured
  median pair ratio `1.0525804254262712`, CI95
  `[1.0487970667130555, 1.0600049333144261]`, and `11/11` faster pairs.
  Required checks passed before benchmarking. The load snapshot peak was caused
  by the benchmark itself on an otherwise calm 12-CPU host and is recorded in
  the aggregate.

### IDEA-20260701-004: Specialize Controller Polling Loop Safely

- Status: discard
- Perspective: emulator-core
- Result: commit `d546564`, artifact
  `artifacts/benchmarks/host-results/host-compare-2026-07-02-210407-B702255342113-Cd546564be8c5/aggregate.json`.
- Reason: Whole-routine specialization passed `make test`, including a
  controller routine equivalence test that caught and fixed the serial bit
  reversal before benchmarking, but full official beast-3 comparison versus
  `7022553` measured median pair ratio `0.9419859635865181`, CI95
  `[0.9392962361088418, 0.9432440777348456]`, and `0/11` faster pairs. Host
  load and run-median CV were clean; one pair-ratio IQR outlier was flagged but
  the conclusion is still a clear discard.

### IDEA-20260630-D01: Group Repeated Saved-State Lanes

- Status: discard
- Perspective: vec-env
- Result: commit `c75909e`, artifact
  `artifacts/benchmarks/grouped-synced-state-lanes-2026-06-30-2135.json`.
- Reason: Benchmark result was substantially slower than the baseline despite the
  plausible mechanism. Do not repeat a single-thread grouped-lane design.

### IDEA-20260630-D02: Parallelize Repeated Saved-State Group Leaders

- Status: keep
- Perspective: vec-env
- Result: commit `68bea25`, artifact
  `artifacts/benchmarks/grouped-synced-state-lanes-parallel-2026-06-30-2142.json`.
- Reason: Parallel group leaders preserved CPU parallelism while avoiding
  duplicate emulator work for repeated saved-state lanes.

### IDEA-20260630-D03: Reuse Persistent Grouped-Action Mask

- Status: discard
- Perspective: vec-env
- Result: commit `40089b0`, artifact
  `artifacts/benchmarks/reuse-synced-group-action-mask-2026-06-30-2147.json`.
- Reason: Clean benchmark run was substantially slower than the accepted parallel
  grouped-state baseline. Avoid this mask-reuse follow-up shape.
