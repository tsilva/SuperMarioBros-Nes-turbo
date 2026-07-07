## PURPOSE

Guide optimization of the Super Mario Bros NES high-throughput reinforcement
learning environment. Research needs many accurate experiments in little time:
accuracy is non-negotiable, and with accuracy preserved, faster throughput is
always better.

## REQUIREMENTS

- Maximize the environment's maximum steps per second.
- Minimize time to a valid throughput improvement by staying on the critical
  path, prioritizing the highest expected verified SPS gain per unit time, and
  avoiding repeated work, unfocused exploration, or preventable mistakes.
- Never accept or encourage optimizations that reduce environment accuracy or
  introduce feature regressions.
- Preserve the canonical benchmark workload and observed environment contract,
  including observation shape and dtype, level-state mix, real emulator
  progression, preprocessing, action mapping, rewards, terminations, resets,
  and info semantics.
- Never fake throughput by skipping emulator work, returning stale data,
  weakening the workload, changing public commands, or loosening benchmark
  semantics.
- Never represent invalid, incomparable, or contract-weakening measurements as
  accepted throughput improvements.
- Keep mutable autoresearch state and benchmark artifacts out of the repository
  by default; ledgers, ideas, scratchpads, candidate bundles, benchmark run
  directories, source archives, result caches, and indexes must live under
  `AUTORESEARCH_ROOT_PATH`.
