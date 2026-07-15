# Contributing

Contributions are welcome when they preserve the project's deliberately narrow
Super Mario Bros NES mapper 0/NROM scope and its Gymnasium vector-environment
contract. Please read the [Code of Conduct](CODE_OF_CONDUCT.md) before
participating.

## Before opening a change

- Use an issue to discuss large API, emulator, observation, reward, termination,
  or performance changes before implementation.
- Keep ROMs, extracted game assets, credentials, generated binaries, benchmark
  output, and training runs out of commits.
- Submit only work you have the right to license under this repository's MIT
  license.
- Treat performance claims as experimental results: use matched workloads,
  include the exact refs and environment, and report statistical evidence.

## Development setup

Install Python 3.9 or newer, [uv](https://docs.astral.sh/uv/), and a Rust
toolchain. Then run:

```bash
git clone https://github.com/tsilva/SuperMarioBros-Nes-turbo.git
cd SuperMarioBros-Nes-turbo
uv sync --frozen --extra dev --group dev
uv run maturin develop --release
make test
```

The default test suite does not require a ROM. If you lawfully have the
supported ROM, run the manual compatibility checks separately:

```bash
ROM_PATH=/path/to/SuperMarioBros.nes make test-retro-oracle
```

## Pull requests

Before requesting review:

1. Run `cargo fmt --check`, `cargo check --release`, and `make test`.
2. Add regression coverage for observable behavior changes.
3. Update `README.md` for public API or user-workflow changes.
4. Add a concise entry under `Unreleased` in `CHANGES.md`.
5. Confirm no ROM or compiled payload is present in the diff.

Maintainers may ask to split unrelated changes. Merging a pull request does not
guarantee that a release will be cut immediately.
