# Changelog

## Unreleased

- Fix Gymnasium autoreset to `AutoresetMode.DISABLED` and remove the constructor option.
- Remove provider-owned life-loss/level-change rules, terminal payload synthesis,
  and dynamic reset-policy mutation APIs.
- Add lane-local masked reset through `options["reset_mask"]`, including active
  lanes selected by an external task kernel.
- Add deterministic scalar/per-lane reset seeds and explicit catalog selection
  through `options["start_indices"]` without mutating unselected lanes.
- Return terminal observations and raw infos directly, and reject stepping
  pending-reset lanes until they are reset explicitly.
