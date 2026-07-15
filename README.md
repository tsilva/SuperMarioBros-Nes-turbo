<div align="center">
  <img src="https://raw.githubusercontent.com/tsilva/SuperMarioBros-Nes-turbo/main/logo.png" alt="SuperMarioBros-Nes-turbo logo" width="320" />

  **🚀 Blazing fast SuperMarioBros-Nes environment for Reinforcement Learning 🍄**
</div>

[![CI](https://github.com/tsilva/SuperMarioBros-Nes-turbo/actions/workflows/ci.yml/badge.svg)](https://github.com/tsilva/SuperMarioBros-Nes-turbo/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/supermariobrosnes-turbo.svg)](https://pypi.org/project/supermariobrosnes-turbo/)
[![Python](https://img.shields.io/pypi/pyversions/supermariobrosnes-turbo.svg)](https://pypi.org/project/supermariobrosnes-turbo/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/tsilva/SuperMarioBros-Nes-turbo/blob/main/LICENSE)

SuperMarioBros-Nes-turbo is a Python library for reinforcement-learning researchers who need a fast, vectorized Super Mario Bros NES environment. It provides a Gymnasium `VectorEnv` backed by a Rust emulator specialized for the game's mapper 0/NROM cartridge, keeping batched emulation and observation preprocessing off the Python hot path. Bring a compatible ROM, construct `SuperMarioBrosNesTurboVecEnv`, then train, play, or benchmark with the included scripts.

The project intentionally supports one game and emulator path. Each vector lane runs its own emulator state, while frame skip, rewards, termination checks, cropping, resizing, grayscale or RGB conversion, and frame stacking run in native code.

## Install

```bash
python -m pip install supermariobrosnes-turbo
```

The trainer and local policy playback use the base dependencies. The optional
Hugging Face client adds authenticated downloads and standard cache handling;
without it, public policy files use the direct-download fallback:

```bash
python -m pip install "supermariobrosnes-turbo[playback]"
```

Release wheels use the CPython 3.9 stable ABI and support Python 3.9 through
3.14. The release workflow builds these artifacts:

| Platform | Architecture | Artifact |
| --- | --- | --- |
| macOS 14+ | Apple Silicon | wheel |
| macOS 13+ | Intel x86-64 | wheel |
| Linux with glibc 2.17+ | x86-64 and ARM64 | wheels |
| Windows | x86-64 | wheel |
| Other supported source-build systems | platform toolchain | source distribution |

Older PyPI releases may have a smaller wheel set. A source build requires
Python, `uv`, and Rust:

```bash
git clone https://github.com/tsilva/SuperMarioBros-Nes-turbo.git
cd SuperMarioBros-Nes-turbo
uv sync --frozen --extra dev
uv run maturin develop --release
```

ROM files are not included. Pass `--rom-path` to scripts, set `ROM_PATH` in the environment or a repo-root `.env`, or provide `rom_path=` to the constructor. The supported ROM has SHA-256:

```text
f61548fdf1670cffefcc4f0b7bdcdd9eaba0c226e3b74f8666071496988248de
```

## Use

```python
from supermariobrosnes_turbo import (
    Actions,
    SuperMarioBrosNesTurboVecEnv,
    action_batch,
)

env = SuperMarioBrosNesTurboVecEnv(
    "SuperMarioBros-Nes-v0",
    rom_path="/path/to/SuperMarioBros.nes",
    state="Level1-1",
    num_envs=16,
    use_restricted_actions=Actions.ALL,
    frame_skip=4,
    obs_grayscale=True,
    obs_crop=(32, 0, 0, 0),
    obs_resize=(84, 84),
    obs_layout="chw",
    frame_stack=4,
)

observations, infos = env.reset(seed=123)
actions = action_batch("right", env.num_envs)
observations, rewards, terminated, truncated, infos = env.step(actions)

done = terminated | truncated
if done.any():
    observations, reset_infos = env.reset(
        options={"reset_mask": done.copy()},
    )

env.close()
```

`reset()` returns `(observations, infos)`. `step()` returns `(observations, rewards, terminations, truncations, infos)`. Autoreset is disabled, so terminal lanes must be selectively reset before the next step.

## Train and play

Train the included observation-free JERK action-sequence policy from Level 1-1:

```bash
uv run python train.py --rom /path/to/SuperMarioBros.nes
```

The default run writes checkpoints, episode metrics, and `final_policy.json` under `runs/level1-1-jerk/`. Episodes end when Mario loses a life or reaches the step limit; changing levels does not end an episode. Play the retained sequence with:

```bash
uv run python scripts/play_policy.py runs/level1-1-jerk/final_policy.json \
  --rom-path /path/to/SuperMarioBros.nes \
  --backend native
```

## Commands

```bash
uv sync --frozen --extra dev                 # install the development environment
uv run maturin develop --release             # build the optimized Rust extension
make test                                    # run Rust and Python regression tests
make test-retro-oracle                       # run ROM-backed parity and policy tests

uv run python scripts/smoke_smb.py --rom-path /path/to/SuperMarioBros.nes
uv run python scripts/play.py --rom-path /path/to/SuperMarioBros.nes --mode external
uv run python scripts/benchmark_sps.py --rom-path /path/to/SuperMarioBros.nes --num-envs 16 --steps 500 --repeats 3
uv run python scripts/benchmark_sps.py --stable-retro-baseline --rom-path /path/to/SuperMarioBros.nes --num-envs 16 --steps 500 --repeats 3
```

## Notes

- Python `>=3.9`, `uv`, and a Rust toolchain are required for a source build.
- The emulator supports only `SuperMarioBros-Nes-v0` on mapper 0/NROM. It is not a general NES or Stable Retro replacement.
- Named saved states are packaged from `Level1-1` through `Level8-4`, with additional variants. `state=` also accepts a path, bytes, one state per lane, or a weighted mapping.
- `Actions.ALL` and `Actions.FILTERED` accept per-button masks. `Actions.DISCRETE` accepts Stable Retro-compatible 36-way discrete actions.
- `train.py` implements JERK (Just Enough Retained Knowledge): it explores scripted action sequences, retains the best reward-reaching prefix, and increasingly replays it. It does not use observations, PyTorch, or Stable Baselines3.
- `scripts/benchmark_sps.py` benchmarks this package by default or upstream `stable-retro==1.0.1` with `--stable-retro-baseline`. Both use frame skip 4, four grayscale frames, a zeroed 32-row HUD, integer area resize to `84x84`, CHW output, deterministic sampled actions, and manual terminal-lane resets. Stable Retro mode requires Python `>=3.10`.
- The play scripts require a discoverable native SDL2 library and open local gameplay windows.
- This is an unofficial research project and is not affiliated with or endorsed by Nintendo. See [NOTICE.md](https://github.com/tsilva/SuperMarioBros-Nes-turbo/blob/main/NOTICE.md).

## Community

- Read [CONTRIBUTING.md](https://github.com/tsilva/SuperMarioBros-Nes-turbo/blob/main/CONTRIBUTING.md) before opening a pull request.
- Use [GitHub Discussions](https://github.com/tsilva/SuperMarioBros-Nes-turbo/discussions) for usage questions and [GitHub Issues](https://github.com/tsilva/SuperMarioBros-Nes-turbo/issues) for reproducible bugs and scoped feature requests.
- Report vulnerabilities privately according to [SECURITY.md](https://github.com/tsilva/SuperMarioBros-Nes-turbo/blob/main/SECURITY.md).
- Release history and compatibility changes are recorded in [CHANGES.md](https://github.com/tsilva/SuperMarioBros-Nes-turbo/blob/main/CHANGES.md).
- Maintainer responsibilities and decision-making are described in [GOVERNANCE.md](https://github.com/tsilva/SuperMarioBros-Nes-turbo/blob/main/GOVERNANCE.md).

## Architecture

![SuperMarioBros-Nes-turbo architecture diagram](https://raw.githubusercontent.com/tsilva/SuperMarioBros-Nes-turbo/main/architecture.png)

## License

The project code is licensed under the [MIT License](https://github.com/tsilva/SuperMarioBros-Nes-turbo/blob/main/LICENSE). Third-party game names, marks, and user-supplied content are not covered by that license; see [NOTICE.md](https://github.com/tsilva/SuperMarioBros-Nes-turbo/blob/main/NOTICE.md).
