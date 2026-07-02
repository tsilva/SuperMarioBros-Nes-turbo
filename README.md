<div align="center">
  <img src="./logo.png" alt="SuperMarioBros-Nes-turbo logo" width="320" />

  **🚀 Blazing fast SuperMarioBros-Nes environment for RL research 🍄**
</div>

`SuperMarioBros-Nes-turbo` is a blazing-fast vectorized Super Mario Bros NES environment for reinforcement-learning research. It uses a custom Rust NES emulator specialized for SuperMarioBros-Nes mapper 0/NROM, with vectorized stepping on the Rust side so Python crosses into Rust once per batched step. Game-specific preprocessing, including frame skip, grayscale or RGB rendering, cropping, resizing, frame stacking, reward extraction, termination checks, and observation-buffer writes, happens before data returns to Python. It follows the same throughput-first direction as [stable-retro-turbo](https://github.com/tsilva/stable-retro-turbo), but drops broad stable-retro compatibility so the emulator and batch API can specialize on Super Mario Bros NES.

## Install

```bash
git clone https://github.com/tsilva/SuperMarioBros-Nes-turbo.git
cd SuperMarioBros-Nes-turbo
uv sync --extra dev
uv run maturin develop --release
```

Point the scripts at a local SuperMarioBros-Nes ROM. The ROM is not included in this repository. The default script path is:

```bash
~/Desktop/roms/NES/mapper-000-NROM/SuperMarioBros-Nes-v0.nes
```

Pass `--rom-path` to use a different file. Verify the expected ROM with:

```bash
shasum -a 256 ~/Desktop/roms/NES/mapper-000-NROM/SuperMarioBros-Nes-v0.nes
```

Expected SHA-256:

```text
f61548fdf1670cffefcc4f0b7bdcdd9eaba0c226e3b74f8666071496988248de
```

Import the package as `supermariobrosnes_turbo`:

```python
import numpy as np

from supermariobrosnes_turbo import SuperMarioBrosVecEnv

env = SuperMarioBrosVecEnv(
    rom_path="~/Desktop/roms/NES/mapper-000-NROM/SuperMarioBros-Nes-v0.nes",
    num_envs=64,
    frame_skip=4,
    grayscale=True,
    frame_stack=4,
    crop_top=32,
    resize_width=84,
    resize_height=84,
)

obs = env.reset()
actions = np.zeros((env.num_envs,), dtype=np.uint8)
env.step_async(actions)
obs, rewards, dones, infos = env.step_wait()
```

`step_wait()` follows the Stable Baselines3 `VecEnv` contract: it calls the Rust `FastMarioVecEnv` once for the whole batch and returns `(obs, rewards, dones, infos)` from reusable NumPy arrays. Use `step_fast()` when you do not need per-env `info` dictionaries, or `step_wait_gymnasium()` when you need separate `terminated` and `truncated` arrays.

Initial states can be a single stable-retro state, one state per env slot, or a weighted mapping sampled independently for each lane on reset:

```python
env = SuperMarioBrosVecEnv(
    rom_path="~/Desktop/roms/NES/mapper-000-NROM/SuperMarioBros-Nes-v0.nes",
    num_envs=16,
    state={"Level1-1": 0.5, "Level1-4": 0.5},
    done_on_info={
        "life_loss": ("lives", "decrease"),
        "level_change": (("levelHi", "levelLo"), "change"),
    },
    seed=123,
)

obs = env.reset()
sampled_states = env.active_states()
```

## Commands

```bash
uv sync --extra dev                 # install Python dev dependencies
uv run maturin develop --release    # build and install the Rust extension

make test                           # Rust tests + HF policy completion/parity oracle

uv run python scripts/smoke_smb.py  # quick ROM/emulator smoke check
uv run python scripts/benchmark_vec_env.py --num-envs 8 --frame-skip 4 --frame-stack 4
uv run python scripts/benchmark_sps.py --num-envs 16 --steps 500 --repeats 3

uv run python scripts/play.py --mode external      # raw SDL2 play view
uv run python scripts/play.py --mode external --view preprocessed --scale 4
uv run python scripts/play_policy.py https://huggingface.co/tsilva/SuperMarioBros-NES_Level1
```

## Fixed-host benchmark

On the fixed `beast-3` CPU host, using the same `SuperMarioBros-Nes-v0` ROM, saved-state set, frame skip, frame stack, grayscale/crop/resize preprocessing, and `16` vector envs, this specialized environment is about `6.40x` faster than the latest published `stable-retro-turbo` PyPI oracle baseline.

| Environment | Version / Ref | Official median env steps/sec | Mean invocation-median env steps/sec | Run-median CV | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `SuperMarioBros-Nes-turbo` | `main` | `47,611.14` | `47,605.89` | `0.28%` | Full official fixed-host run; all validity gates passed. |
| `stable-retro-turbo` PyPI oracle | `1.0.0.post23` | `7,437.65` | `7,440.04` | `0.44%` | Full official fixed-host run; statistical gates passed, but the post-run host-load gate failed because the 1-minute load was sampled immediately after the benchmark's own CPU-heavy timing. |

Artifacts:

- [`SuperMarioBros-Nes-turbo` fixed-host aggregate](./artifacts/benchmarks/host-results/host-single-2026-07-02-123806-R17c60e1eb88e/aggregate.json)
- [`stable-retro-turbo==1.0.0.post23` PyPI oracle aggregate](./artifacts/benchmarks/host-results/pypi-stable-retro-turbo/1.0.0.post23/0bcebd32669e8e46/aggregate.json)

## Notes

- Python `>=3.9` and a Rust toolchain are required to build the Maturin extension.
- The current emulator scope is SuperMarioBros-Nes mapper 0 NROM.
- The Python package exposes `SuperMarioBrosVecEnv`, `ACTION_MEANINGS`, `CORE_ACTION_MEANINGS`, and `ACTION_SETS`. `SuperMarioBrosVecEnv` subclasses Stable Baselines3 `VecEnv` when SB3 is installed; `action_space` is the per-lane `Discrete` action space, while `vector_action_space` describes the batched action array.
- The default `simple` action set matches the Stable Retro Mario training mapper: `noop`, `right`, `right_b`, `right_a`, `right_a_b`, `a`, and `left`. Use `action_set="full"` when a tool needs the `start` button.
- `scripts/play_policy.py` loads Stable Baselines3 PPO checkpoints from a local `.zip`, a Hugging Face repo id, or a `https://huggingface.co/...` URL and displays raw RGB gameplay in the SDL2 GUI while feeding the model its preprocessed observation stack. It defaults to a Stable Retro playback backend so public SB3/Hugging Face checkpoints use the preprocessing they were trained with; pass `--view preprocessed` to inspect the model input or `--backend native` when checking this repo's fast-env parity. The SB3, PyTorch, and Hugging Face Hub dependencies are included in the repo's `uv` dev environment.
- By default, `scripts/benchmark_sps.py` starts lanes from `Level1-1`, `Level1-2`, `Level1-3`, and `Level1-4` repeated round-robin. Use `--state Level1-1` or another stable-retro state to start every lane from one saved level state. Use `--states ...` to choose a different round-robin state list. In Python, `state=` accepts a single state name/path/bytes value, a sequence with exactly one state per env, or a weighted mapping such as `{"Level1-1": 0.5, "Level1-4": 0.5}`. After reset, `active_state_indices()` and `active_states()` report the sampled state for each lane. If needed, pass `--state-dir` or set `SUPERMARIOBROSNES_FASTENV_STATE_DIR`.
- For `SuperMarioBrosVecEnv`, `done_on_info` accepts named terminal rules like `{"life_loss": ("lives", "decrease")}`. Supported ops are `change`, `increase`, and `decrease`; keys are drawn from `INFO_KEYS`. Fired rules are reported in `info["done_on_info"]` with `op`, `keys`, `prev`, and `next`.
- Stable Retro oracle/playback tooling targets `stable-retro-turbo==1.0.0.post23` and constructs `RetroVecEnv` with the current flat keyword names: `maxpool_last_two`, `noop_reset_max`, `sticky_action_prob`, `info_filter`, `obs_copy`, and `done_on`. Runtime fired terminal rules are still read from `info["done_on_info"]`.
- Benchmark JSON can be written with `scripts/benchmark_sps.py --output-json ...`.
- Play mode uses the native SDL2 library. If SDL2 is not installed or discoverable, `scripts/play.py` exits with an SDL backend error.
- ROM files are not included in the repository; use the SHA-256 digest above to confirm you are testing with the expected ROM.

## Architecture

![SuperMarioBros-Nes-turbo architecture diagram](./architecture.png)

## License

MIT, as declared in [pyproject.toml](./pyproject.toml) and [Cargo.toml](./Cargo.toml).
