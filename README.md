<div align="center">
  <img src="./logo.png" alt="SuperMarioBros-Nes-turbo logo" width="320" />

  **🚀 Blazing fast SuperMarioBros-Nes environment for Reinforcement Learning 🍄**
</div>

SuperMarioBros-Nes-turbo is a Python library for reinforcement-learning researchers who need a fast, vectorized Super Mario Bros NES environment. It provides a Gymnasium `VectorEnv` backed by a Rust emulator specialized for the game's mapper 0/NROM cartridge, keeping batched emulation and observation preprocessing off the Python hot path. Bring a compatible ROM, construct `SuperMarioBrosNesTurboVecEnv`, then train, play, or benchmark with the included scripts.

The project intentionally supports one game and emulator path. Each vector lane runs its own emulator state, while frame skip, rewards, termination checks, cropping, resizing, grayscale or RGB conversion, and frame stacking run in native code.

## Install

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

Train the included standalone PyTorch PPO policy on Level 1-1:

```bash
uv run python train.py --rom /path/to/SuperMarioBros.nes
```

The default run writes checkpoints, completion events, and `final_model.pt` under `runs/level1-1-b55/`. Play the final checkpoint with its matching preprocessing contract:

```bash
uv run python scripts/play_policy.py runs/level1-1-b55/final_model.pt \
  --rom-path /path/to/SuperMarioBros.nes \
  --device auto \
  --deterministic
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
```

## Notes

- Python `>=3.9`, `uv`, and a Rust toolchain are required for a source build.
- The emulator supports only `SuperMarioBros-Nes-v0` on mapper 0/NROM. It is not a general NES or Stable Retro replacement.
- Named saved states are packaged from `Level1-1` through `Level8-4`, with additional variants. `state=` also accepts a path, bytes, one state per lane, or a weighted mapping.
- `Actions.ALL` and `Actions.FILTERED` accept per-button masks. `Actions.DISCRETE` accepts Stable Retro-compatible 36-way discrete actions.
- `train.py` is a plain-PyTorch PPO implementation with no Stable Baselines3 dependency. `scripts/play_policy.py` supports its `.pt` checkpoints and legacy PPO `.zip` artifacts.
- `scripts/benchmark_sps.py` uses deterministic sampled actions, manually resets terminal lanes, prints workload metadata, and can write JSON with `--output-json`.
- The play scripts require a discoverable native SDL2 library and open local gameplay windows.

## Architecture

![SuperMarioBros-Nes-turbo architecture diagram](./architecture.png)

## License

MIT, as declared in [pyproject.toml](./pyproject.toml) and [Cargo.toml](./Cargo.toml).
