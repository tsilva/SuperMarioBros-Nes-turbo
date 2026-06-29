# SuperMarioBros-Nes-turbo Codex Notes

## Autoresearch Speed Improvements Skill

Use `/autoresearch-speed-improvements` for future throughput optimization rounds in this repo, especially work involving `scripts/benchmark_sps.py`, Super Mario Bros NES emulator hot paths, `env_steps_per_sec` targets, single-agent optimization tracks, or multi-agent Modal-judged research campaigns. The skill lives at `.codex/skills/autoresearch-speed-improvements/SKILL.md`.

## Modal Benchmark Skill

Use `/modal-benchmark` when the user wants the canonical clean-machine Modal CPU benchmark or a fresh 16-env baseline/comparison run. The skill runs `scripts/modal_benchmark_sps.py`, saves a JSON artifact under `artifacts/benchmarks/`, and reports the same compact throughput summary each time. The skill lives at `.codex/skills/modal-benchmark/SKILL.md`.
