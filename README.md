# supermarioemu

Fast Rust-first Super Mario Bros NES environment for RL.

The main performance rule is simple: Python should call Rust once per vectorized
batch step. Frame skip, grayscale conversion, frame stacking, reward extraction,
termination checks, and observation-buffer writes happen on the Rust side.

Current emulator scope:

- Super Mario Bros / mapper 0 NROM fast path
- 6502 CPU interpreter
- no-audio PPU timing, VRAM, OAM DMA, controller input, vblank/NMI
- grayscale and RGB frame output
- Rust-side vector stepping with lane parallelism for larger batches

## Intended hot path

```python
env = SuperMarioBrosVecEnv(
    rom_path="~/Desktop/roms/SuperMarioBros.nes",
    num_envs=64,
    frame_skip=4,
    grayscale=True,
    frame_stack=4,
)

obs = env.reset()
env.step_async(actions)
obs, rewards, terminated, truncated, infos = env.step_wait()
```

Under the hood `step_wait()` calls a single Rust `step_into(...)` method that
fills preallocated NumPy arrays in place.

## Build

```bash
uv sync --extra dev
uv run maturin develop --release
```

## Smoke And Benchmark

```bash
uv run python scripts/smoke_smb.py
uv run python scripts/benchmark_vec_env.py --num-envs 8 --frame-skip 4 --frame-stack 4
```

The `start` action is included so the raw ROM can leave the title screen without
special Python-side reset logic.

## Play

```bash
uv run python scripts/play.py --mode external
```

`external` mode reads keyboard input in Python, maps it to the discrete
Gymnasium-style action IDs, and calls `SuperMarioBrosEnv.step(action)` each
frame. It disables RL preprocessing for play: no frame stack, no grayscale, and
no frame skip. It also disables RL flagpole termination so the game can continue
through SMB's own end-of-level sequence. Play mode uses the native SDL2 backend
for fast scaled display.
Controls: arrows or A/D move, X/J/Space jump, Z/K/Shift run, Enter start,
Esc quit.
