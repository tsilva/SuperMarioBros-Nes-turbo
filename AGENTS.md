# SuperMarioBros-Nes-turbo Codex Notes

## Autoresearch Speed Skill

Use `/autoresearch-speed` for future throughput optimization rounds in this repo, especially work involving `scripts/benchmark_sps.py`, Super Mario Bros NES emulator hot paths, `env_steps_per_sec` targets, single-agent optimization tracks, or multi-agent Modal-judged research campaigns. The skill lives at `.codex/skills/autoresearch-speed/SKILL.md`.

## Modal Benchmark Skill

Use `/modal-benchmark` when the user wants the canonical clean-machine Modal CPU comparison. The skill requires a candidate ref; if the user omits the baseline, use the latest local `main` commit. When two refs are provided, baseline is first and candidate is second. It runs `scripts/modal_compare_sps.py` without switching local branches, saves a paired JSON artifact under `artifacts/benchmarks/`, and reports robust paired speedup. The skill lives at `.codex/skills/modal-benchmark/SKILL.md`.

## Host Benchmark Skill

Use `/host-benchmark` when the user wants a reliable fixed-host CPU throughput comparison on `beast-3-local` instead of Modal. The skill compares exact local git archives for two refs without switching branches, builds isolated per-run source/venv directories under `/home/tsilva/SuperMarioBros-Nes-turbo-host-bench`, runs the lighter beast-3-local paired protocol, and reports fixed-host variance. The skill lives at `.codex/skills/host-benchmark/SKILL.md`.
