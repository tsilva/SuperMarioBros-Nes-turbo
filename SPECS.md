## PURPOSE

Provide a throughput-first Super Mario Bros NES RL environment specialized for the supported SMB/NROM workload while preserving Gymnasium VectorEnv contracts, stable-retro-turbo parity, playback, benchmarking, and release validation.

## REQUIREMENTS

- Support only the Super Mario Bros NES mapper 0/NROM fast path unless broader emulator scope is explicitly designed and benchmarked.
- Keep ROM files out of the repo; require ROM paths through flags, environment, `.env`, or constructors; validate canonical benchmark/smoke input against SHA-256 `f61548fdf1670cffefcc4f0b7bdcdd9eaba0c226e3b74f8666071496988248de`.
- Preserve public package/API shape: `supermariobrosnes_turbo`, `SuperMarioBrosNesTurboVecEnv`, `Actions`, action meanings/sets, state helpers, constructors, and manual/policy playback scripts.
- Expose environment truth through Gymnasium `VectorEnv` APIs only: `reset()` returns `(obs, infos)`, `step()` returns `(obs, rewards, terminations, truncations, infos)`, and `metadata["autoreset_mode"]` is set. Keep same-step autoreset as the default and support opt-in disabled/manual autoreset with Gymnasium-compatible `options["reset_mask"]` selective resets.
- Keep `train.py` as a standalone plain-PyTorch PPO implementation with no Stable Baselines3 dependency. Preserve playback for its native `.pt` checkpoints and existing legacy PPO `.zip` artifacts without requiring Stable Baselines3.
- Preserve a fast native Rust/Python `step()` path for batch stepping, rewards, termination checks, preprocessing, frame stacking, typed info extraction, Gymnasium vector `infos`, same-step autoreset data, and manual terminal transitions without per-step Python wrapper chains. Throughput benchmarks and autoresearch optimization target `step()`, not an info-bypassing alternate step API.
- Preserve action modes `Actions.ALL`, `Actions.FILTERED`, and Stable Retro-compatible 36-way `Actions.DISCRETE`.
- Preserve preprocessing options: grayscale/RGB, frame skip, optional max-pooling, crop remove or mask mode, resize, frame stack, and CHW/HWC layouts.
- Preserve stable-retro-style state handling: packaged/named states, paths/bytes, per-lane states, weighted mappings, `set_state(...)`, active-state reporting, state sampling weights, and documented SMB states including `Level1-1` through `Level1-4`.
- Preserve native game-over and flag-completion termination plus additive `done_on`/`done_on_info` semantics, named events, `change`/`increase`/`decrease`, Gymnasium vector `infos` payloads, and same-step `final_obs`/`final_info` terminal data.
- Maintain deterministic seeding and lane behavior; every lane must execute its
  own emulator state during vector steps, without repeated-state or
  uniform-action leader/peer copy shortcuts that do not apply to stochastic PPO
  rollout collection. Selective reset must preserve every unselected lane's
  emulator state, RNG stream, observation/frame stack, sticky action, and counters.
- Keep benchmarks centered on `scripts/benchmark_sps.py`, reporting `env_steps_per_sec`, workload metadata, ROM/state identity, observation shape/dtype, deterministic sampled action workload, and comparable JSON.
- Use `stable-retro-turbo==1.0.1.post8` as the Stable Retro PyPI oracle unless intentionally updating the benchmark contract; rerun oracle baselines before quoting speedups with identical ROM, states, preprocessing, vector-env count, and host context.
- Keep official local benchmarks exact-ref based, load-gated, and statistically checked through repo helpers; default autoresearch acceptance uses three measured comparison pairs, while `--full` remains available for the longer sequential stability ladder. Ad hoc timings are not acceptance evidence.
- Treat every newly started autoresearch goal as a fresh improvement round from
  the live `HEAD`, regardless of completed rounds in external controller state;
  only continuations of the same active goal resume its already-fixed baseline.
- Keep mutable autoresearch state and benchmark artifacts out of the repo by default; ledgers, ideas, scratchpads, candidate bundles, run dirs, source archives, result caches, and indexes live under `AUTORESEARCH_ROOT_PATH`.
- Preserve parity/regression tests for Gymnasium VectorEnv behavior, native/stable-retro constructor compatibility, state sampling, terminal infos, sticky/no-op actions, benchmark stats, release flow, and benchmark metadata.
- Preserve release and supply-chain hardening: aligned Python/Rust versions, clean synced release branch, local gates before tags, validated wheels, `uv` lock state, `exclude-newer`, bad-package constraints, and Rust release-profile performance settings.
- Document intentional compatibility breaks, benchmark-contract changes, or public API behavior changes in README and tests before accepting them.
