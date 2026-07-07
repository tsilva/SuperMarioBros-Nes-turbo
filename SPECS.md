## PURPOSE

Provide a throughput-first Super Mario Bros NES RL environment specialized for the supported SMB/NROM workload while preserving Gymnasium VectorEnv contracts, stable-retro-turbo parity, playback, benchmarking, and release validation.

## REQUIREMENTS

- Support only the Super Mario Bros NES mapper 0/NROM fast path unless broader emulator scope is explicitly designed and benchmarked.
- Keep ROM files out of the repo; require ROM paths through flags, environment, `.env`, or constructors; validate canonical benchmark/smoke input against SHA-256 `f61548fdf1670cffefcc4f0b7bdcdd9eaba0c226e3b74f8666071496988248de`.
- Preserve public package/API shape: `supermariobrosnes_turbo`, `SuperMarioBrosNesTurboVecEnv`, `Actions`, action meanings/sets, state helpers, constructors, and manual/policy playback scripts.
- Expose environment truth through Gymnasium `VectorEnv` APIs only: `reset()` returns `(obs, infos)`, `step()` returns `(obs, rewards, terminations, truncations, infos)`, `metadata["autoreset_mode"]` is set, and SB3 `VecEnv` adaptation is intentionally downstream responsibility.
- Preserve the fast native Rust/Python path for batch stepping, rewards, termination checks, preprocessing, frame stacking, typed info extraction, and reusable `step_fast()` arrays without per-step Python wrapper chains.
- Preserve action modes `Actions.ALL`, `Actions.FILTERED`, and Stable Retro-compatible 36-way `Actions.DISCRETE`.
- Preserve preprocessing options: grayscale/RGB, frame skip, optional max-pooling, crop remove or mask mode, resize, frame stack, and CHW/HWC layouts.
- Preserve stable-retro-style state handling: packaged/named states, paths/bytes, per-lane states, weighted mappings, `set_state(...)`, active-state reporting, state sampling weights, and documented SMB states including `Level1-1` through `Level1-4`.
- Preserve native game-over and flag-completion termination plus additive `done_on`/`done_on_info` semantics, named events, `change`/`increase`/`decrease`, Gymnasium vector `infos` payloads, and same-step `final_obs`/`final_info` terminal data.
- Maintain deterministic seeding and lane behavior; grouped/synced lane optimizations may only preserve the public vector-env contract.
- Keep benchmarks centered on `scripts/benchmark_sps.py`, reporting `env_steps_per_sec`, workload metadata, ROM/state identity, observation shape/dtype, and comparable JSON.
- Use `stable-retro-turbo==1.0.1.post7` as the Stable Retro PyPI oracle unless intentionally updating the benchmark contract; rerun oracle baselines before quoting speedups with identical ROM, states, preprocessing, vector-env count, and host context.
- Keep official local benchmarks exact-ref based, load-gated, and statistically checked through repo helpers; ad hoc timings are not acceptance evidence.
- Keep mutable autoresearch state and benchmark artifacts out of the repo by default; ledgers, ideas, scratchpads, candidate bundles, run dirs, source archives, result caches, and indexes live under `AUTORESEARCH_ROOT_PATH`.
- Preserve parity/regression tests for Gymnasium VectorEnv behavior, native/stable-retro constructor compatibility, state sampling, terminal infos, sticky/no-op actions, benchmark stats, release flow, and benchmark metadata.
- Preserve release and supply-chain hardening: aligned Python/Rust versions, clean synced release branch, local gates before tags, validated wheels, `uv` lock state, `exclude-newer`, bad-package constraints, and Rust release-profile performance settings.
- Document intentional compatibility breaks, benchmark-contract changes, or public API behavior changes in README and tests before accepting them.
