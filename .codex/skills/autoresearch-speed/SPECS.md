## PURPOSE

Optimize Super Mario Bros NES environment throughput for real PPO training
without weakening correctness, comparability, or experiment validity.

## REQUIREMENTS

- Maximize verified environment SPS while minimizing wasted research time.
- Accept only changes that plausibly improve stochastic PPO rollout/training;
  reject benchmark-only shortcuts based on repeated states, uniform/no-op action
  batches, disabled PPO semantics, or other non-training conditions.
- Preserve accuracy, public behavior, and the full observed env contract:
  observation bytes, preprocessing, actions, rewards, resets, terminations,
  infos, state loading, and real emulator progression.
- Preserve the canonical benchmark workload and never fake speed by skipping
  work, returning stale data, changing commands, or loosening semantics.
- Treat the benchmarking procedure as immutable for autoresearch acceptance:
  do not change benchmark scripts, commands, workload parameters, sampling,
  load gates, statistical checks, comparison refs, reported metrics, or
  acceptance criteria to make a candidate pass.
- Use the user-approved default acceptance cap of three measured comparison
  pairs for `scripts/autoresearch.py accept`; reserve `accept --full` for the
  longer sequential stability ladder.
- Start every newly created autoresearch goal as a fresh round from the live
  `HEAD`, even when controller state contains a completed prior round. Resume a
  fixed baseline only when continuing that same active goal.
- Do not represent invalid, incomparable, or contract-weakening measurements as
  accepted throughput improvements.
- Keep mutable autoresearch state and artifacts out of the repo under
  `AUTORESEARCH_ROOT_PATH`.
