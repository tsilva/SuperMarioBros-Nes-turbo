# SuperMarioBros-Nes-turbo Codex Notes

## Autoresearch Speed Skill

Use `/autoresearch-speed` for future throughput optimization rounds in this repo, especially work involving `scripts/benchmark_sps.py`, Super Mario Bros NES emulator hot paths, `env_steps_per_sec` targets, single-agent optimization tracks, or multi-agent fixed-host research campaigns. The skill lives at `.codex/skills/autoresearch-speed/SKILL.md`.

## Host Benchmark Skill

Use `/host-benchmark` when the user wants a reliable fixed-host CPU throughput benchmark or comparison on `beast-3-local`. The skill compares exact local git archives without switching branches, can benchmark cached latest PyPI release baselines, builds isolated per-run source/venv directories under `/home/tsilva/SuperMarioBros-Nes-turbo-host-bench`, runs the beast-3-local protocol, and reports fixed-host variance. The skill lives at `.codex/skills/host-benchmark/SKILL.md`.

## Build Release Skill

Use `/build-release` when the user wants to tag a SuperMarioBros-Nes-turbo version and build validated macOS arm64 plus Linux x86_64 PyPI wheels without uploading them. The skill uses this repo's owned version schema from `pyproject.toml` and `Cargo.toml`, creates clean source copies, and prints the final twine upload command after validation. The skill lives at `.codex/skills/build-release/SKILL.md`.
