# Changelog

## Unreleased

- Keep Gymnasium `AutoresetMode.SAME_STEP` as the default and add opt-in
  `AutoresetMode.DISABLED` native vector lifecycle control.
- Add lane-local masked reset through `options["reset_mask"]`, including active
  lanes selected by an external task kernel.
- Add deterministic scalar/per-lane reset seeds and explicit catalog selection
  through `options["start_indices"]` without mutating unselected lanes.
- Return terminal observations and raw infos directly in disabled mode, omit
  same-step `final_obs`/`final_info`, and reject stepping pending-reset lanes.
