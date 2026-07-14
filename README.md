<div align="center">
  <img src="https://raw.githubusercontent.com/tsilva/SuperMarioBros-Nes-turbo/main/logo.png" alt="SuperMarioBros-Nes-turbo logo" width="320" />

  **🚀 Blazing fast SuperMarioBros-Nes environment for Reinforcement Learning 🍄**

  <p>
    <a href="https://pypi.org/project/supermariobrosnes-turbo/"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fpypi.org%2Fpypi%2Fsupermariobrosnes-turbo%2Fjson&amp;query=%24.info.version&amp;label=pypi&amp;prefix=v&amp;cacheSeconds=300" alt="PyPI version" /></a>
  </p>
</div>

`SuperMarioBros-Nes-turbo` is a blazing-fast vectorized Super Mario Bros NES environment for reinforcement-learning research. It uses a custom Rust NES emulator specialized for SuperMarioBros-Nes mapper 0/NROM, with vectorized stepping on the Rust side so Python crosses into Rust once per batched step. Game-specific preprocessing, including frame skip, grayscale or RGB rendering, cropping, resizing, frame stacking, reward extraction, termination checks, and observation-buffer writes, happens before data returns to Python. It follows the same throughput-first direction as [stable-retro-turbo](https://github.com/tsilva/stable-retro-turbo), but drops broad stable-retro compatibility so the emulator and batch API can specialize on Super Mario Bros NES.

## Why it is fast

Compared with upstream Stable Retro, this package does not run many Python
`RetroEnv` instances through `SubprocVecEnv`, `DummyVecEnv`, or wrapper stacks
for frame skip, resize, grayscale, frame stack, reward, and termination logic.
Compared with `stable-retro-turbo`, it keeps the same native-vector philosophy
but gives up the general Stable Retro compatibility layer, arbitrary game/core
support, and generic emulator contracts. The speed comes from these current fast
paths:

- **SMB/NROM-only Rust emulator**: the core supports the Super Mario Bros NES
  mapper 0/NROM shape directly instead of routing every access through a
  general multi-console emulator interface.
- **Fixed cartridge memory paths**: PRG/CHR reads use precomputed power-of-two
  masks, direct PRG ROM instruction fetches, fixed nametable mirroring, and
  direct CPU memory paths for RAM, PPU registers, controllers, and PRG ROM.
- **One Python call per vector step**: `reset_into()`, `step_into()`, and
  `info_into()` mutate caller-owned NumPy arrays, release the GIL, and avoid
  creating new observations, rewards, done arrays, and scalar info arrays on
  every step.
- **Rust-side batch execution**: vector lanes step in Rust with Rayon when the
  batch is large enough, so the Python side only submits action arrays and reads
  already-filled result buffers.
- **Fast `step()` path**: training and benchmark loops exercise the same
  Gymnasium vector `step()` API users call, while native code keeps x-position,
  score, lives, level, timer, and scroll values in typed arrays before
  assembling vector `infos`.
- **Fused RL preprocessing**: frame skip, optional max-pool, reward accumulation,
  termination checks, grayscale/RGB rendering, crop, area resize, and frame-stack
  writes happen in the native step loop before data returns to Python.
- **Observation buffer as frame-stack state**: the returned observation buffer is
  also the persistent stack buffer; old frames shift in place and only the newest
  processed frame is written into the final stack slot.
- **Direct grayscale renderer**: the common pixel path renders SMB background
  tile rows and sprite overlays directly to grayscale from NES palette values,
  instead of first materializing RGB and then converting it in Python.
- **Precomputed area resize plan**: resize bins are built once per env
  configuration, then reused for every frame and every lane.
- **Deterministic lane sharing**: identical reset lanes, and repeated saved-state
  groups such as the default `Level1-1` through `Level1-4` round-robin benchmark,
  can share one emulator state while actions remain uniform; mixed actions
  materialize independent lane states before stepping, preserving the public
  vector-env contract.
- **SMB routine fast-forwards**: the emulator recognizes exact Super Mario Bros
  ROM byte signatures for the idle loop, sprite-0 polling loop, and OAM clear
  helper, then advances equivalent CPU/PPU cycles without interpreting every
  repeated 6502 instruction.
- **Provider-native rewards and termination**: x-position reward, native game-over,
  and flag completion stay in the emulator. Raw lives, level, score, time,
  position, and scrolling signals remain available to downstream task kernels.
- **Scoped compatibility paths**: RGB, uncropped rendering, Gymnasium vector
  `info` dictionaries, sticky actions, random no-op
  starts, and multi-state curricula are still available, but the benchmark path
  keeps them on explicit typed/native routes instead of paying broad Stable Retro
  overhead unconditionally.

## Install

```bash
git clone https://github.com/tsilva/SuperMarioBros-Nes-turbo.git
cd SuperMarioBros-Nes-turbo
uv sync --extra dev
uv run maturin develop --release
```

ROM files are not included in this repository. Pass `--rom-path` to scripts, set `ROM_PATH` in the environment or `.env`, or provide `rom_path=` when constructing environments. Expected SHA-256 for the supported Super Mario Bros NES ROM:

```text
f61548fdf1670cffefcc4f0b7bdcdd9eaba0c226e3b74f8666071496988248de
```

## Train Level 1-1

`train.py` is a standalone PyTorch PPO implementation; it does not use Stable
Baselines3. On Apple Silicon it automatically selects MPS with the `64 x 128`
profile, while other machines use the original `16 x 512` B55 rollout shape.

```bash
.venv/bin/python train.py
```

Training terminates on life loss and level change, logs the rolling
`completion_rate`, stops at `100/100`, and writes `final_model.pt`, periodic
`.pt` checkpoints, and `level_completion.jsonl` under `runs/level1-1-b55/`.
Play a produced checkpoint with the matching preprocessing contract:

```bash
uv run python scripts/play_policy.py runs/level1-1-b55/final_model.pt --device auto --deterministic
```

Import the package as `supermariobrosnes_turbo`:

```python
import numpy as np

from supermariobrosnes_turbo import Actions, SuperMarioBrosNesTurboVecEnv

env = SuperMarioBrosNesTurboVecEnv(
    "SuperMarioBros-Nes-v0",
    rom_path="/path/to/SuperMarioBros.nes",
    num_envs=64,
    use_restricted_actions=Actions.ALL,
    frame_skip=4,
    obs_grayscale=True,
    frame_stack=4,
    obs_crop=(32, 0, 0, 0),
    obs_resize=(84, 84),
    obs_layout="chw",
)

obs, infos = env.reset(seed=123)
actions = np.zeros((env.num_envs, env.num_buttons), dtype=np.uint8)
obs, rewards, terminations, truncations, infos = env.step(actions)
```

`obs_crop` removes source pixels by default before resizing, matching Stable Retro's crop behavior and keeping the fastest training path for HUD removal. All crop, mask-fill, grayscale, resize, and frame-stack preprocessing happens in the native Rust vector env. For finetuning-friendly crop masking, pass `obs_crop_mode="mask"` to keep the full source canvas geometry while filling the cropped regions with `obs_crop_fill` before resize, layout conversion, and frame stacking:

```python
env = SuperMarioBrosNesTurboVecEnv(
    "SuperMarioBros-Nes-v0",
    obs_crop=(32, 0, 0, 0),
    obs_crop_mode="mask",
    obs_crop_fill=0,
    obs_resize=(84, 84),
    obs_grayscale=True,
)
```

Mask crop is useful for hiding HUD or other static regions during initial training while preserving spatial compatibility for later finetuning on full observations.

`SuperMarioBrosNesTurboVecEnv` follows the Gymnasium `VectorEnv` contract directly. `reset()` returns `(obs, infos)`, and `step(actions)` returns `(obs, rewards, terminations, truncations, infos)` with separate termination and truncation arrays. Autoreset is permanently disabled: terminal observations are returned directly and another step is rejected until all done lanes are reset with `reset(options={"reset_mask": mask})`. A masked reset changes only selected lanes and returns reset info only for those lanes.

```python
import numpy as np
env = SuperMarioBrosNesTurboVecEnv(
    "SuperMarioBros-Nes-v0",
    rom_path="/path/to/SuperMarioBros.nes",
    num_envs=16,
)
obs, infos = env.reset(seed=123)
obs, rewards, terminated, truncated, infos = env.step(actions)
done = terminated | truncated
if done.any():
    obs, reset_infos = env.reset(options={"reset_mask": done.copy()})
```

`reset_mask` must be a NumPy `bool` array with shape `(num_envs,)` and at least
one `True`. A scalar reset seed expands deterministically to `seed + lane_index`;
a sequence must contain exactly one integer or `None` per lane. Unselected seed
entries do not advance those lanes' RNG streams. To choose starts explicitly,
pass an `int32` `start_indices` array of the same shape:

```python
mask = np.array([False, True, False, True], dtype=np.bool_)
starts = np.array([-1, 1, -1, 0], dtype=np.int32)
obs, reset_infos = env.reset(
    seed=[None, 101, None, 303],
    options={"reset_mask": mask, "start_indices": starts},
)
```

Nonnegative values select entries from the configured start catalog; `-1`
retains its fixed or weighted sampling policy. Unselected lanes preserve emulator
and RNG state, frame stacks, observations, sticky actions, counters, and active
start identity. `active_state_indices()` and `active_states()` update only for
selected lanes.

Initial states can be a single stable-retro state, one state per env slot, or a weighted mapping sampled independently for each lane on reset:

```python
env = SuperMarioBrosNesTurboVecEnv(
    "SuperMarioBros-Nes-v0",
    rom_path="/path/to/SuperMarioBros.nes",
    num_envs=16,
    state={"Level1-1": 0.5, "Level1-4": 0.5},
)
env.seed(123)

obs, infos = env.reset()
sampled_states = env.active_states()
```

## Commands

```bash
uv sync --extra dev                 # install Python dev dependencies
uv run maturin develop --release    # build and install the Rust extension

make test                           # Rust tests + default Python regression suite
make test-retro-oracle              # manual HF policy/parity oracle acceptance tests

uv run python scripts/smoke_smb.py --rom-path /path/to/SuperMarioBros.nes  # quick ROM/emulator smoke check
uv run python scripts/benchmark_sps.py --rom-path /path/to/SuperMarioBros.nes --num-envs 16 --steps 500 --repeats 3

make play PLAY_ARGS="--rom-path /path/to/SuperMarioBros.nes"                              # SDL2 RGB + frame-stack play view
uv run python scripts/play.py --rom-path /path/to/SuperMarioBros.nes --mode external
uv run python scripts/play_policy.py https://huggingface.co/tsilva/SuperMarioBros-NES_Level1 --rom-path /path/to/SuperMarioBros.nes
```

## Release

Release tags drive the GitHub Actions wheel build. From a clean, synced branch
with the release environment installed, create the next patch release with:

```bash
uv sync --extra dev --group dev
make release
```

Use `scripts/release.py --part minor`, `--part major`, or `--to 0.3.0` for
other release shapes. The script refuses to run unless the current branch is
clean and synced with its upstream. It verifies the target version is not already
on PyPI, bumps `pyproject.toml` and `Cargo.toml`, refreshes lockfiles, runs local
gates, commits `Release v<version>`, creates the matching tag, and pushes the
branch plus tag. The pushed tag triggers the release workflow, which builds,
audits, and publishes the wheels to PyPI via trusted publishing.

Release validation runs on Python 3.14. The native extension remains a
`cp39-abi3` build, so each platform publishes one stable-ABI wheel that supports
Python 3.9 through 3.14 without multiplying artifacts per interpreter.

## Local benchmark target

Use `stable-retro-turbo==1.0.1.post30` as the Stable Retro PyPI oracle for new benchmarks and comparisons. Both environments use manual reset and provider-native termination.

For autoresearch throughput work, use the lightweight path first:

```bash
make benchmark
.venv/bin/python scripts/autoresearch.py diagnose
```

These benchmark the current checkout and are intended for direction, profiling,
and quick rejection. Promising committed candidates move to exact-ref paired
screening:

```bash
.venv/bin/python scripts/autoresearch.py screen <baseline_ref> <candidate_ref>
```

Acceptance uses the same official exact-ref runner, but the controller defaults
to the dedicated-host cap of `3` measured pairs. This keeps the normal
50,000-step, 3-repeat workload and accepts decisive paired evidence without the
long sequential stability ladder. Use the full ladder only when the extra time
is warranted:

```bash
.venv/bin/python scripts/autoresearch.py accept <baseline_ref> <candidate_ref>
.venv/bin/python scripts/autoresearch.py accept <baseline_ref> <candidate_ref> --full
.venv/bin/python scripts/autoresearch.py calibrate <baseline_ref>
```

Historical local benchmark results:

| Environment | Version / Ref | Official median env steps/sec | Mean invocation-median env steps/sec | Run-median CV | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `SuperMarioBros-Nes-turbo` | `main` | `47,611.14` | `47,605.89` | `0.28%` | Full official local benchmark run; all validity gates passed. |
| `stable-retro-turbo` PyPI oracle | `1.0.0.post23` | `7,437.65` | `7,440.04` | `0.44%` | Historical only; superseded by `1.0.1.post30` for new comparisons. |

New local benchmark runs, result caches, indexes, and matching source archives
default under `AUTORESEARCH_ROOT_PATH/benchmarks/` so benchmark state stays
outside the repository and can move cleanly across commits.

The native benchmark samples one precomputed action batch per vector step from
`noop`, `right`, `right_b`, and `right_a` with `--action-seed 0` by default.
Use `--actions ...` to change the sampled set or `--action noop` for a legacy
single-action loop.

## Notes

- Python `>=3.9` and a Rust toolchain are required to build the Maturin extension.
- The current emulator scope is SuperMarioBros-Nes mapper 0 NROM.
- The Python package exposes `SuperMarioBrosNesTurboVecEnv`, `ACTION_MEANINGS`, `CORE_ACTION_MEANINGS`, and `ACTION_SETS`. `SuperMarioBrosNesTurboVecEnv` subclasses Gymnasium `VectorEnv` and does not subclass or require Stable Baselines3.
- `use_restricted_actions=Actions.ALL` and `Actions.FILTERED` consume per-button `MultiBinary` masks; `Actions.DISCRETE` consumes Stable Retro's 36-way discrete action encoding.
- `scripts/play_policy.py` loads plain PyTorch PPO `.pt` checkpoints and legacy SB3 `.zip` checkpoints without depending on Stable Baselines3. Sources may be local files, Hugging Face repo ids, or `https://huggingface.co/...` URLs. Auto mode selects the exact native mask-crop/no-max-pool contract for plain `.pt` policies and the Stable Retro contract for legacy `.zip` policies; pass `--view preprocessed` to inspect model input.
- By default, `scripts/benchmark_sps.py` starts lanes from `Level1-1` through `Level1-4` round-robin, uses mask cropping, samples a deterministic action sequence, and explicitly resets provider-native terminal lanes. Constructor `state=` accepts a single state, one state per env, or a weighted mapping. `active_state_indices()` and `active_states()` report each lane's current start.
- Stable Retro oracle/playback tooling targets `stable-retro-turbo==1.0.1.post30` and compares raw life/level values as information signals, not provider-owned episode boundaries.
- Benchmark JSON can be written with `scripts/benchmark_sps.py --output-json ...`.
- Play mode uses the native SDL2 library and opens RGB gameplay plus the grayscale frame stack in separate windows. If SDL2 is not installed or discoverable, `scripts/play.py` exits with an SDL backend error.
- ROM files are not included in the repository; use the SHA-256 digest above to confirm test inputs when needed.

## Architecture

![SuperMarioBros-Nes-turbo architecture diagram](https://raw.githubusercontent.com/tsilva/SuperMarioBros-Nes-turbo/main/architecture.png)

## License

MIT, as declared in [pyproject.toml](https://github.com/tsilva/SuperMarioBros-Nes-turbo/blob/main/pyproject.toml) and [Cargo.toml](https://github.com/tsilva/SuperMarioBros-Nes-turbo/blob/main/Cargo.toml).
