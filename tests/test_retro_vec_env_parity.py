from __future__ import annotations

import importlib.metadata

import pytest

from scripts import compare_retro_vec_env as compare


def require_stable_retro_oracle() -> None:
    rom_path = compare.DEFAULT_ROM.expanduser()
    if not rom_path.exists():
        pytest.skip(f"local SuperMarioBros-Nes ROM is missing: {rom_path}")
    try:
        version = importlib.metadata.version("stable-retro-turbo")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip(
            "stable-retro-turbo oracle is not installed; run `uv sync --extra dev` "
            "under Python 3.14",
        )
    assert version == compare.EXPECTED_STABLE_RETRO_VERSION


@pytest.mark.retro_oracle
def test_stable_retro_vec_env_constructs_with_oracle_keyword_surface() -> None:
    require_stable_retro_oracle()
    import stable_retro

    rom_path = compare.DEFAULT_ROM.expanduser()
    env = stable_retro.RetroVecEnv(
        compare.DEFAULT_STABLE_RETRO_GAME,
        state="Level1-1",
        num_envs=1,
        num_threads=1,
        rom_path=str(rom_path),
        render_mode="rgb_array",
        use_restricted_actions=stable_retro.Actions.ALL,
        obs_crop=(32, 0, 0, 0),
        obs_resize=(84, 84),
        obs_grayscale=True,
        obs_resize_algorithm="area",
        obs_layout="chw",
        obs_copy="safe_view",
        frame_skip=4,
        frame_stack=4,
        maxpool_last_two=True,
        noop_reset_max=0,
        sticky_action_prob=0.0,
        reward_clip=False,
        info_filter="all",
        done_on={
            "life_loss": ("lives", "decrease"),
            "level_change": (("levelHi", "levelLo"), "change"),
        },
    )
    try:
        assert env.num_envs == 1
        assert getattr(env, "obs_copy", None) == "safe_view"
    finally:
        env.close()
