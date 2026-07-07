# SuperMarioBros-Nes-turbo Codex Notes

- Always obey repo-root `SPECS.md` as this checkout's durable acceptance
  contract. If it is missing a requirement, mismatched with purpose, ambiguous,
  contradictory, or incongruent with the live project, ask the user for
  clarification, then update `SPECS.md` from that feedback before dependent
  work continues.
- Use `/autoresearch-speed` for throughput optimization involving
  `scripts/benchmark_sps.py`, emulator hot paths, `env_steps_per_sec`, or
  autoresearch campaigns. Skill: `.codex/skills/autoresearch-speed/SKILL.md`.
- Use `/build-release` to tag a version and build validated macOS arm64 plus
  Linux x86_64 PyPI wheels without upload. Skill:
  `.codex/skills/build-release/SKILL.md`.
