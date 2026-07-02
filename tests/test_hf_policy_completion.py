from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import play_policy  # noqa: E402
import compare_supermariobrosnes_turbo_vec_env as compare  # noqa: E402
from rom_helpers import require_rom  # noqa: E402


HF_LEVEL1_POLICY = "https://huggingface.co/tsilva/SuperMarioBros-NES_Level1"
MAX_EPISODES = 10
MAX_STEPS_PER_EPISODE = 3_000
EXPECTED_STABLE_RETRO_VERSION = "1.0.0.post23"


def require_policy_prerequisites() -> None:
    require_rom()

    for package in ("stable_baselines3", "stable-retro-turbo"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pytest.skip(f"{package} is not installed")
        if package == "stable-retro-turbo":
            assert version == EXPECTED_STABLE_RETRO_VERSION


def level_was_cleared(info: dict[str, object]) -> bool:
    if bool(info.get("level_complete")) or bool(info.get("completion_event")):
        return True

    done_on_info = info.get("done_on_info")
    if isinstance(done_on_info, dict) and "level_change" in done_on_info:
        return True

    info_events = info.get("info_events")
    if isinstance(info_events, dict) and "level_change" in info_events:
        return True

    return int(info.get("levelLo", 0)) > 0


def level1_policy_config(*, steps: int = MAX_STEPS_PER_EPISODE) -> compare.ComparisonConfig:
    return compare.ComparisonConfig(
        rom_path=require_rom(),
        stable_retro_path=None,
        game=compare.DEFAULT_STABLE_RETRO_GAME,
        state="Level1-1",
        num_envs=1,
        env_threads=1,
        steps=steps,
        seed=10007,
        frame_skip=4,
        frame_stack=4,
        grayscale=True,
        crop_top=32,
        crop_bottom=0,
        resize_width=84,
        resize_height=84,
        action_set="simple",
        frame_maxpool=True,
        noop_reset_max=0,
        sticky_action_prob=0.0,
        obs_copy="safe_view",
        terminate_on_flag=False,
        terminate_on_life_loss=True,
        terminate_on_level_change=True,
        include_obs=True,
        include_rewards=False,
        include_dones=False,
        include_infos=False,
        stop_on_done=True,
        output_json=None,
        allow_version_mismatch=False,
        preprocessing_matrix=False,
    )


def load_level1_policy():
    from stable_baselines3 import PPO

    model_path = play_policy.resolve_model_path(
        HF_LEVEL1_POLICY,
        filename=None,
        cache_dir=Path("artifacts/hf_cache"),
    )
    model = PPO.load(model_path, device="cpu")
    action_names = play_policy.ACTION_SETS["simple"]
    assert getattr(model.action_space, "n", None) == len(action_names)
    return model


@pytest.mark.retro_oracle
def test_huggingface_level1_policy_completes_with_full_fast_env_parity() -> None:
    require_policy_prerequisites()
    compare.check_stable_retro_version(path=None, allow_mismatch=False)

    import stable_retro

    config = level1_policy_config()
    model = load_level1_policy()
    action_meanings = compare.ACTION_SETS[config.action_set]
    buttons = compare.retro_button_names(stable_retro, config.rom_path)
    retro_masks_by_action = compare.stable_action_masks(action_meanings, buttons)
    fast_env = compare.make_fast_env(config)
    retro_env = compare.make_retro_env(config)

    episode_summaries: list[dict[str, object]] = []
    try:
        for episode in range(1, MAX_EPISODES + 1):
            fast_obs = fast_env.reset()
            retro_obs = retro_env.reset()
            compare.require_array_equal(
                phase="policy_reset",
                step=None,
                field="obs",
                fast=fast_obs,
                retro=retro_obs,
            )
            final_info: dict[str, object] = {}
            for step in range(1, MAX_STEPS_PER_EPISODE + 1):
                action, _ = model.predict(retro_obs, deterministic=True)
                fast_actions = np.asarray(action, dtype=np.uint8).reshape(config.num_envs)
                action_names = [action_meanings[int(action_id)] for action_id in fast_actions]
                retro_actions = retro_masks_by_action[fast_actions]

                fast_obs, fast_rewards, fast_terminated, fast_truncated, fast_infos = fast_env.step_gymnasium(
                    fast_actions,
                )
                retro_obs, retro_rewards, retro_dones, retro_infos = retro_env.step(retro_actions)
                fast_dones = np.asarray(fast_terminated | fast_truncated, dtype=np.bool_)
                retro_dones = np.asarray(retro_dones, dtype=np.bool_)
                final_info = dict(retro_infos[0])

                compare.require_array_equal(
                    phase="policy_step",
                    step=step,
                    field="obs",
                    fast=fast_obs,
                    retro=retro_obs,
                    action_names=action_names,
                )
                compare.require_array_equal(
                    phase="policy_step",
                    step=step,
                    field="rewards",
                    fast=np.asarray(fast_rewards, dtype=np.float32),
                    retro=np.asarray(retro_rewards, dtype=np.float32),
                    action_names=action_names,
                )
                compare.require_array_equal(
                    phase="policy_step",
                    step=step,
                    field="dones",
                    fast=fast_dones,
                    retro=retro_dones,
                    action_names=action_names,
                )
                compare.compare_infos(
                    phase="policy_step",
                    step=step,
                    fast_infos=fast_infos,
                    retro_infos=retro_infos,
                    action_names=action_names,
                )
                if level_was_cleared(final_info):
                    return
                if bool(retro_dones[0]):
                    break
            episode_summaries.append(
                {
                    "episode": episode,
                    "steps": step,
                    "reward": float(retro_rewards[0]),
                    "final_info": final_info,
                },
            )
    finally:
        fast_env.close()
        retro_env.close()

    pytest.fail(
        f"HF Level1 policy did not complete with full parity within {MAX_EPISODES} episodes: "
        f"{episode_summaries!r}",
    )
