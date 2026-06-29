<div align="center">
  <img src="./logo.png" alt="SuperMarioBros-Nes-turbo logo" width="320" />

  **Blazing fast SuperMarioBros-Nes environment for RL research**
</div>

`SuperMarioBros-Nes-turbo` exposes a Python API backed by a Rust NES emulator and vectorized environment. Its speed comes from a NES emulator written in Rust and optimized specifically for SuperMarioBros-Nes, not from a general-purpose NES compatibility layer. The hot path crosses from Python to Rust once per batched step, while frame skip, reward extraction, termination checks, preprocessing, frame stacking, and observation-buffer writes stay on the Rust side.

The objective is to push beyond the already juiced [stable-retro-turbo](https://github.com/tsilva/stable-retro-turbo) stack we were using for RL experimentation. These RL loops are mostly bottlenecked by CPU-based rollouts, so the goal is to make environment stepping as fast as possible: run the maximum number of experiments with the minimum hardware resources and wall-clock time.

Use it when you need a local, ROM-backed Super Mario Bros environment with a Gymnasium-style single-env wrapper, a vectorized batch API, stable-retro state loading, local throughput benchmarks, and a clean Modal CPU benchmark path.

## Install

```bash
git clone https://github.com/tsilva/SuperMarioBros-Nes-turbo.git
cd SuperMarioBros-Nes-turbo
uv sync --extra dev
uv run maturin develop --release
```

Point the scripts at a local SuperMarioBros-Nes ROM. The ROM is not included in this repository. The default script path is `~/Desktop/roms/NES/mapper-000-NROM/SuperMarioBros-Nes-v0.nes`; pass `--rom-path` to use a different file.

Verify the ROM before running benchmarks:

```bash
shasum -a 256 ~/Desktop/roms/NES/mapper-000-NROM/SuperMarioBros-Nes-v0.nes
```

The expected SHA-256 digest is:

```text
f61548fdf1670cffefcc4f0b7bdcdd9eaba0c226e3b74f8666071496988248de
```

Import the Python package as `supermarioemu`, or run one of the scripts below from the repo root.

## Usage

```python
import numpy as np

from supermarioemu import SuperMarioBrosVecEnv

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
obs, rewards, terminated, truncated, infos = env.step_wait()
```

`step_wait()` calls the Rust `FastMarioVecEnv` once for the whole batch and fills reusable NumPy arrays in place.

## Commands

```bash
uv sync --extra dev                 # install Python dev dependencies
uv run maturin develop --release    # build and install the Rust extension

uv run python scripts/smoke_smb.py  # quick ROM/emulator smoke check
uv run python scripts/benchmark_vec_env.py --num-envs 8 --frame-skip 4 --frame-stack 4
uv run python scripts/benchmark_sps.py --state Level1-1 --num-envs 16 --steps 500 --repeats 3

uv run python scripts/play.py --mode external      # raw SDL2 play view
uv run python scripts/play.py --mode external --view preprocessed --scale 4

modal run scripts/modal_benchmark_sps.py --output-json artifacts/benchmarks/modal-baseline.json
```

## Notes

- The current emulator scope is SuperMarioBros-Nes mapper 0 NROM. It is intentionally optimized for this game, with a 6502 CPU interpreter, no-audio PPU timing, VRAM, OAM DMA, controller input, vblank/NMI, grayscale/RGB output, and Rust-side vector stepping.
- The Python package exposes `SuperMarioBrosEnv`, `SuperMarioBrosVecEnv`, and `ACTION_MEANINGS`.
- The default action set is `noop`, `right`, `right_b`, `right_a`, `right_a_b`, `a`, `left`, and `start`.
- Use `--state Level1-1` or another stable-retro state to start from a saved level state. If needed, pass `--state-dir` or set `SUPERMARIOEMU_STATE_DIR`.
- Benchmark JSON can be written with `scripts/benchmark_sps.py --output-json ...`; Modal benchmark artifacts include per-level run config, timings, summary stats, Git metadata, state SHA-256 values, Modal CPU metadata, and the ROM SHA-256.
- The Modal benchmark path expects an authenticated Modal CLI, sends the local ROM and state bytes to the remote container at run time, and defaults to `Level1-1`, `Level1-2`, `Level1-3`, and `Level1-4`.
- Play mode uses the native SDL2 library. If SDL2 is not installed or discoverable, `scripts/play.py` exits with an SDL backend error.
- ROM files are not included in the repository; use the SHA-256 digest above to confirm you are testing with the expected ROM.

## Architecture

![SuperMarioBros-Nes-turbo architecture diagram](./architecture.png)

## License

MIT, as declared in [pyproject.toml](./pyproject.toml) and [Cargo.toml](./Cargo.toml).
