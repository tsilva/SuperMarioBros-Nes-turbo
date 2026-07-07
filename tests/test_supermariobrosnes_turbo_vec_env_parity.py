from __future__ import annotations

import importlib.metadata
import inspect

import pytest
import numpy as np
from gymnasium import spaces

from scripts import compare_supermariobrosnes_turbo_vec_env as compare
from supermariobrosnes_turbo import env as env_module
from supermariobrosnes_turbo import _supermariobrosnes_turbo as native
from supermariobrosnes_turbo import (
    Actions,
    Integrations,
    Observations,
    State,
    SuperMarioBrosNesTurboVecEnv,
)
from rom_helpers import require_rom

NES_BUTTONS = ("B", None, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A")
BUTTON_TO_INDEX = {name: index for index, name in enumerate(NES_BUTTONS) if name is not None}


def resize_chw_area_native_reference(src: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = src.shape[-2:]
    out = np.empty((*src.shape[:-2], height, width), dtype=np.uint8)
    for out_y in range(height):
        y0 = (out_y * src_h) // height
        y1 = min(max(((out_y + 1) * src_h) // height, y0 + 1), src_h)
        for out_x in range(width):
            x0 = (out_x * src_w) // width
            x1 = min(max(((out_x + 1) * src_w) // width, x0 + 1), src_w)
            patch = src[:, :, y0:y1, x0:x1].astype(np.uint32)
            out[:, :, out_y, out_x] = patch.mean(axis=(-2, -1)).astype(np.uint8)
    return out


def require_stable_retro_oracle() -> None:
    require_rom()
    try:
        version = importlib.metadata.version("stable-retro-turbo")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip(
            "stable-retro-turbo oracle is not installed; run `uv sync --extra dev` "
            "under Python 3.14",
        )
    assert version == compare.EXPECTED_STABLE_RETRO_VERSION


def make_level1_1_noop_probe(done_on) -> SuperMarioBrosNesTurboVecEnv:
    try:
        return SuperMarioBrosNesTurboVecEnv(
            compare.DEFAULT_STABLE_RETRO_GAME,
            state="Level1-1",
            num_envs=1,
            num_threads=1,
            rom_path=str(require_rom()),
            render_mode="rgb_array",
            use_restricted_actions=Actions.ALL,
            obs_crop=(32, 0, 0, 0),
            obs_resize=(84, 84),
            obs_grayscale=True,
            obs_resize_algorithm="area",
            obs_layout="chw",
            frame_skip=4,
            frame_stack=4,
            maxpool_last_two=False,
            noop_reset_max=0,
            sticky_action_prob=0.0,
            reward_clip=False,
            info_filter="all",
            done_on=done_on,
        )
    except ValueError as exc:
        if "ROM path required" in str(exc):
            pytest.skip(str(exc))
        raise


def test_native_vec_env_binding_is_private_retro_vec_env() -> None:
    assert SuperMarioBrosNesTurboVecEnv.__name__ == "SuperMarioBrosNesTurboVecEnv"
    assert not hasattr(native, "SuperMarioBrosNesTurboVecEnv")
    assert hasattr(native, "_RetroVecEnv")
    assert native._RetroVecEnv.__name__ == "_RetroVecEnv"
    assert hasattr(native._RetroVecEnv, "set_initial_states")
    assert not hasattr(native, "NativeVectorEnv")
    assert not hasattr(native, "FastMarioVecEnv")


def test_native_turbo_vec_env_defaults_match_stable_retro_turbo_signature() -> None:
    native_signature = inspect.signature(SuperMarioBrosNesTurboVecEnv)
    assert list(native_signature.parameters) == [
        "game",
        "state",
        "scenario",
        "info",
        "use_restricted_actions",
        "record",
        "players",
        "inttype",
        "obs_type",
        "render_mode",
        "num_envs",
        "num_threads",
        "rom_path",
        "obs_copy",
        "obs_resize",
        "obs_crop",
        "obs_crop_mode",
        "obs_crop_fill",
        "obs_grayscale",
        "obs_resize_algorithm",
        "obs_layout",
        "frame_skip",
        "frame_stack",
        "maxpool_last_two",
        "noop_reset_max",
        "sticky_action_prob",
        "reward_clip",
        "info_filter",
        "done_on",
    ]

    native_defaults = {
        name: parameter.default
        for name, parameter in native_signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }

    assert native_defaults["scenario"] is None
    assert native_defaults["info"] is None
    assert native_defaults["record"] is False
    assert native_defaults["players"] == 1
    assert native_defaults["render_mode"] == "human"
    assert native_defaults["num_envs"] == 1
    assert native_defaults["num_threads"] is None
    assert native_defaults["rom_path"] is None
    assert native_defaults["obs_copy"] == "copy"
    assert native_defaults["obs_resize"] is None
    assert native_defaults["obs_crop"] is None
    assert native_defaults["obs_crop_mode"] == "remove"
    assert native_defaults["obs_crop_fill"] == 0
    assert native_defaults["obs_grayscale"] is False
    assert native_defaults["obs_resize_algorithm"] == "nearest"
    assert native_defaults["obs_layout"] == "hwc"
    assert native_defaults["frame_skip"] == 1
    assert native_defaults["frame_stack"] == 1
    assert native_defaults["maxpool_last_two"] is False
    assert native_defaults["noop_reset_max"] == 0
    assert native_defaults["sticky_action_prob"] == 0.0
    assert native_defaults["reward_clip"] is False
    assert native_defaults["info_filter"] == "all"
    assert native_defaults["done_on"] is None

    assert native_defaults["state"] is State.DEFAULT
    assert native_defaults["state"].name == "DEFAULT"
    assert native_defaults["state"].value == -1
    assert native_defaults["use_restricted_actions"] is Actions.FILTERED
    assert native_defaults["use_restricted_actions"].name == "FILTERED"
    assert native_defaults["use_restricted_actions"].value == 1
    assert native_defaults["inttype"] is Integrations.STABLE
    assert native_defaults["inttype"].name == "STABLE"
    assert native_defaults["inttype"].value == 1
    assert native_defaults["obs_type"] is Observations.IMAGE
    assert native_defaults["obs_type"].name == "IMAGE"
    assert native_defaults["obs_type"].value == 0

    for name in (
        "copy_observations",
        "info_mode",
        "info_keys",
        "done_on_info",
        "unsafe_zero_copy",
    ):
        assert name not in native_signature.parameters


def test_native_turbo_vec_env_rejects_non_stable_retro_alias_keywords() -> None:
    with pytest.raises(TypeError, match="frame_maxpool"):
        SuperMarioBrosNesTurboVecEnv("SuperMarioBros-Nes-v0", frame_maxpool=False)
    with pytest.raises(TypeError, match="reset_noops"):
        SuperMarioBrosNesTurboVecEnv("SuperMarioBros-Nes-v0", reset_noops=0)
    with pytest.raises(TypeError, match="action_sticky_prob"):
        SuperMarioBrosNesTurboVecEnv("SuperMarioBros-Nes-v0", action_sticky_prob=0.0)


def test_native_turbo_vec_env_rejects_invalid_crop_mode_and_fill() -> None:
    with pytest.raises(ValueError, match="obs_crop_mode must be 'remove' or 'mask'"):
        SuperMarioBrosNesTurboVecEnv("SuperMarioBros-Nes-v0", obs_crop_mode="zero")
    with pytest.raises(ValueError, match=r"obs_crop_fill must be in \[0, 255\]"):
        SuperMarioBrosNesTurboVecEnv("SuperMarioBros-Nes-v0", obs_crop_fill=-1)
    with pytest.raises(ValueError, match=r"obs_crop_fill must be in \[0, 255\]"):
        SuperMarioBrosNesTurboVecEnv("SuperMarioBros-Nes-v0", obs_crop_fill=256)


def test_native_turbo_done_on_normalization_matches_stable_retro_shape() -> None:
    assert env_module._normalize_done_on_info(
        {"level_change": [["levelHi", "levelLo"], "change"]},
        False,
        False,
    ) == (("level_change", "default", ("levelHi", "levelLo"), "change", "reset"),)

    assert env_module._normalize_done_on_info(
        {"life_loss": ("lives", "decrease")},
        False,
        False,
    ) == (("life_loss", "default", ("lives",), "decrease", "reset"),)

    assert env_module._normalize_done_on_info(
        {
            "life_loss": {
                "triggers": [
                    {"id": "lives_decrease", "variables": "lives", "op": "decrease"},
                    {"id": "lives_change", "variables": "lives", "op": "change"},
                ],
            },
        },
        False,
        False,
    ) == (
        ("life_loss", "lives_decrease", ("lives",), "decrease", "reset"),
        ("life_loss", "lives_change", ("lives",), "change", "reset"),
    )


def test_native_turbo_done_on_info_writer_groups_triggers_without_rom() -> None:
    rules = env_module._normalize_done_on_info(
        {
            "life_loss": {
                "triggers": [
                    {"id": "lives_decrease", "variables": "lives", "op": "decrease"},
                    {"id": "lives_change", "variables": "lives", "op": "change"},
                ],
            },
        },
        False,
        False,
    )
    native_rules, metadata = env_module._native_done_on_info_rules(rules)

    class FakeCore:
        def done_on_info(self):
            return [
                [
                    (
                        native_rules[0][0],
                        ["lives"],
                        "decrease",
                        [2],
                        [1],
                    ),
                    (
                        native_rules[1][0],
                        ["lives"],
                        "change",
                        [2],
                        [1],
                    ),
                ],
            ]

    env = SuperMarioBrosNesTurboVecEnv.__new__(SuperMarioBrosNesTurboVecEnv)
    env._core = FakeCore()
    env._done_on_info_metadata = metadata

    env._write_done_on_info()

    payload = env._done_on_info[0]["life_loss"]
    assert payload["trigger"] == "lives_decrease"
    assert payload["op"] == "decrease"
    assert payload["compare"] == "reset"
    assert payload["keys"] == ["lives"]
    assert payload["variables"] == ["lives"]
    assert payload["prev"] == [2]
    assert payload["next"] == [1]
    assert [trigger["trigger"] for trigger in payload["triggers"]] == [
        "lives_decrease",
        "lives_change",
    ]
    assert [trigger["op"] for trigger in payload["triggers"]] == ["decrease", "change"]


def test_native_turbo_vec_env_crop_modes_configure_geometry_without_rom(monkeypatch) -> None:
    core_calls = []

    class FakeCore:
        def __init__(
            self,
            _rom_path,
            num_envs,
            frame_skip,
            grayscale,
            frame_stack,
            _terminate_on_flag,
            crop_top,
            crop_bottom,
            resize_width,
            resize_height,
            initial_states,
            initial_state_names,
            initial_state_weights,
            seed,
            terminate_on_life_loss,
            terminate_on_level_change,
            done_on_info,
            frame_maxpool,
            noop_reset_max,
            sticky_action_prob,
            crop_left,
            crop_right,
            crop_mode,
            crop_fill,
            resize_algorithm,
        ):
            core_calls.append(
                {
                    "crop_top": crop_top,
                    "crop_bottom": crop_bottom,
                    "crop_left": crop_left,
                    "crop_right": crop_right,
                    "crop_mode": crop_mode,
                    "crop_fill": crop_fill,
                    "resize_width": resize_width,
                    "resize_height": resize_height,
                    "resize_algorithm": resize_algorithm,
                }
            )
            self.num_envs = num_envs
            self.frame_skip = frame_skip
            self.grayscale = grayscale
            self.frame_stack = frame_stack
            self.frame_maxpool = frame_maxpool
            self.noop_reset_max = noop_reset_max
            self.sticky_action_prob = sticky_action_prob
            self.crop_top = crop_top
            self.crop_bottom = crop_bottom
            self.crop_left = crop_left
            self.crop_right = crop_right
            self.crop_mode = crop_mode
            self.crop_fill = crop_fill
            self.resize_width = resize_width
            self.resize_height = resize_height
            self.resize_algorithm = resize_algorithm
            self.initial_state_names = ()

        def initial_state_policy_names(self):
            return ()

        def initial_state_weights(self):
            return ()

        def obs_shape(self):
            channels = self.frame_stack if self.grayscale else self.frame_stack * 3
            return self.num_envs, channels, self.resize_height, self.resize_width

        def active_state_indices(self):
            return [-1] * self.num_envs

    monkeypatch.setattr(env_module, "_resolve_rom_path", lambda _game, _rom_path: "/tmp/fake.nes")
    monkeypatch.setattr(env_module, "_CoreRetroVecEnv", FakeCore)

    remove_env = SuperMarioBrosNesTurboVecEnv(
        compare.DEFAULT_STABLE_RETRO_GAME,
        obs_crop=(32, 0, 10, 0),
        obs_crop_mode="remove",
        obs_resize=None,
        obs_grayscale=True,
        obs_layout="chw",
    )
    mask_env = SuperMarioBrosNesTurboVecEnv(
        compare.DEFAULT_STABLE_RETRO_GAME,
        obs_crop=(32, 0, 10, 0),
        obs_crop_mode="mask",
        obs_crop_fill=7,
        obs_resize=None,
        obs_grayscale=True,
        obs_layout="chw",
    )
    try:
        assert core_calls == [
            {
                "crop_top": 32,
                "crop_bottom": 0,
                "crop_left": 10,
                "crop_right": 0,
                "crop_mode": "remove",
                "crop_fill": 0,
                "resize_width": 230,
                "resize_height": 192,
                "resize_algorithm": "nearest",
            },
            {
                "crop_top": 32,
                "crop_bottom": 0,
                "crop_left": 10,
                "crop_right": 0,
                "crop_mode": "mask",
                "crop_fill": 7,
                "resize_width": 240,
                "resize_height": 224,
                "resize_algorithm": "nearest",
            },
        ]
        assert remove_env.observation_space.shape == (1, 192, 230)
        assert mask_env.observation_space.shape == (1, 224, 240)
        assert not remove_env._needs_python_postprocess
        assert not mask_env._needs_python_postprocess

        remove_env._obs = np.arange(np.prod(remove_env._obs.shape), dtype=np.uint8).reshape(remove_env._obs.shape)
        mask_env._obs = np.arange(np.prod(mask_env._obs.shape), dtype=np.uint8).reshape(mask_env._obs.shape)

        remove_obs = remove_env._return_obs()
        mask_obs = mask_env._return_obs()

        np.testing.assert_array_equal(remove_obs, remove_env._obs)
        np.testing.assert_array_equal(mask_obs, mask_env._obs)
    finally:
        remove_env.close()
        mask_env.close()


@pytest.mark.retro_oracle
def test_stable_retro_vector_env_constructs_with_oracle_keyword_surface() -> None:
    require_stable_retro_oracle()
    import stable_retro

    rom_path = require_rom()
    env_class = getattr(stable_retro, "Retro" "Vec" "Env")
    env = env_class(
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


def test_native_turbo_vec_env_accepts_smb_keyword_surface() -> None:
    rom_path = require_rom()

    env = SuperMarioBrosNesTurboVecEnv(
        compare.DEFAULT_STABLE_RETRO_GAME,
        state="Level1-1",
        num_envs=1,
        num_threads=1,
        rom_path=str(rom_path),
        render_mode="rgb_array",
        use_restricted_actions=Actions.ALL,
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
        info_filter="terminal",
        done_on=["life_loss"],
    )
    try:
        assert env.num_envs == 1
        assert env.num_threads == 1
        assert env.obs_copy == "safe_view"
        assert not hasattr(env, "copy_observations")
        assert not hasattr(env, "unsafe_zero_copy")
        assert isinstance(env.action_space, spaces.MultiBinary)
        obs = env.reset()
        assert obs.shape == (1, 4, 84, 84)
        masks = np.zeros((1, env.num_buttons), dtype=np.uint8)
        obs, rewards, dones, infos = env.step(masks)
        assert obs.shape == (1, 4, 84, 84)
        assert rewards.shape == (1,)
        assert dones.shape == (1,)
        assert infos == [{}]
    finally:
        env.close()


def _make_level1_1_obs_env(rom_path, **kwargs) -> SuperMarioBrosNesTurboVecEnv:
    options = {
        "state": "Level1-1",
        "num_envs": 1,
        "num_threads": 1,
        "rom_path": str(rom_path),
        "render_mode": "rgb_array",
        "use_restricted_actions": Actions.ALL,
        "obs_grayscale": True,
        "obs_resize_algorithm": "area",
        "obs_layout": "chw",
        "obs_copy": "copy",
        "frame_skip": 1,
        "frame_stack": 1,
        "info_filter": "none",
    }
    options.update(kwargs)
    return SuperMarioBrosNesTurboVecEnv(compare.DEFAULT_STABLE_RETRO_GAME, **options)


def test_native_turbo_vec_env_mask_crop_preserves_full_canvas_shape_and_masks_before_resize() -> None:
    rom_path = require_rom()
    crop = (32, 0, 0, 0)

    full_env = _make_level1_1_obs_env(
        rom_path,
        obs_crop=None,
        obs_resize=(84, 84),
    )
    full_source_env = _make_level1_1_obs_env(
        rom_path,
        obs_crop=None,
        obs_resize=None,
    )
    mask_env = _make_level1_1_obs_env(
        rom_path,
        obs_crop=crop,
        obs_crop_mode="mask",
        obs_crop_fill=0,
        obs_resize=(84, 84),
    )
    try:
        full_obs = full_env.reset()
        full_source = full_source_env.reset()
        mask_obs = mask_env.reset()

        assert mask_env.observation_space.shape == full_env.observation_space.shape
        assert mask_obs.shape == full_obs.shape == (1, 1, 84, 84)
        assert full_source.shape == (1, 1, 224, 240)

        expected_source = full_source.copy()
        expected_source[:, :, : crop[0], :] = 0
        expected_mask = resize_chw_area_native_reference(expected_source, 84, 84)
        np.testing.assert_array_equal(mask_obs, expected_mask)
        assert np.any(mask_obs != full_obs)
    finally:
        full_env.close()
        full_source_env.close()
        mask_env.close()


def test_native_turbo_vec_env_remove_crop_matches_default_cropped_behavior() -> None:
    rom_path = require_rom()
    crop = (32, 0, 0, 0)

    default_env = _make_level1_1_obs_env(
        rom_path,
        obs_crop=crop,
        obs_resize=(84, 84),
    )
    remove_env = _make_level1_1_obs_env(
        rom_path,
        obs_crop=crop,
        obs_crop_mode="remove",
        obs_resize=(84, 84),
    )
    full_env = _make_level1_1_obs_env(
        rom_path,
        obs_crop=None,
        obs_resize=(84, 84),
    )
    try:
        default_obs = default_env.reset()
        remove_obs = remove_env.reset()
        full_obs = full_env.reset()

        assert remove_env.observation_space.shape == default_env.observation_space.shape
        assert remove_obs.shape == default_obs.shape == (1, 1, 84, 84)
        np.testing.assert_array_equal(remove_obs, default_obs)
        assert np.any(remove_obs != full_obs)
    finally:
        default_env.close()
        remove_env.close()
        full_env.close()


@pytest.mark.parametrize("obs_copy", ["copy", "safe_view", "unsafe_view"])
@pytest.mark.parametrize("obs_layout", ["chw", "hwc"])
def test_native_turbo_vec_env_mask_crop_terminal_observation_matches_public_layout(
    obs_copy: str,
    obs_layout: str,
) -> None:
    rom_path = require_rom()
    env = _make_level1_1_obs_env(
        rom_path,
        obs_crop=(32, 0, 0, 0),
        obs_crop_mode="mask",
        obs_resize=(84, 84),
        obs_layout=obs_layout,
        obs_copy=obs_copy,
        frame_skip=4,
        frame_stack=4,
        done_on={"time_tick": ("time", "decrease")},
        info_filter="all",
    )
    try:
        obs = env.reset()
        masks = np.zeros((1, env.num_buttons), dtype=np.uint8)
        for _step in range(1, 20):
            obs, _rewards, dones, infos = env.step(masks)
            if bool(dones[0]):
                terminal_observation = infos[0]["terminal_observation"]
                assert terminal_observation.shape == obs.shape[1:]
                assert terminal_observation.dtype == obs.dtype
                assert terminal_observation.shape == env.observation_space.shape
                assert infos[0]["done_on_info"]["time_tick"] == {
                    "trigger": "default",
                    "op": "decrease",
                    "compare": "reset",
                    "keys": ["time"],
                    "variables": ["time"],
                    "prev": [400],
                    "next": [399],
                }
                break
        else:
            pytest.fail("time_tick done_on rule did not fire before step 20")
    finally:
        env.close()


def test_native_turbo_vec_env_empty_done_on_keeps_native_game_over_done() -> None:
    env = make_level1_1_noop_probe(done_on=[])
    try:
        env.reset()
        masks = np.zeros((1, env.num_buttons), dtype=np.uint8)

        first_life_loss = second_life_loss = None
        for step in range(1, 7600):
            _obs, _rewards, dones, infos = env.step(masks)
            done = bool(dones[0])
            lives = int(infos[0]["lives"])
            if lives == 1 and first_life_loss is None:
                first_life_loss = (step, done)
            if lives == 0 and second_life_loss is None:
                second_life_loss = (step, done)
            if done:
                assert first_life_loss == (2456, False)
                assert second_life_loss == (4991, False)
                assert step == 7527
                assert lives == 2
                assert "done_on_info" not in infos[0]
                break
        else:
            pytest.fail("native game-over scenario done did not fire by step 7599")
    finally:
        env.close()


def test_native_turbo_vec_env_life_loss_done_on_is_additive_and_earlier() -> None:
    env = make_level1_1_noop_probe(done_on=["life_loss"])
    try:
        env.reset()
        masks = np.zeros((1, env.num_buttons), dtype=np.uint8)

        for step in range(1, 3000):
            _obs, _rewards, dones, infos = env.step(masks)
            if not bool(dones[0]):
                continue
            assert step == 2456
            assert infos[0]["done_on_info"] == {
                "life_loss": {
                    "trigger": "lives_decrease",
                    "op": "decrease",
                    "compare": "reset",
                    "keys": ["lives"],
                    "variables": ["lives"],
                    "prev": [2],
                    "next": [1],
                },
            }
            break
        else:
            pytest.fail("life_loss done_on rule did not fire before game-over")
    finally:
        env.close()


def test_native_turbo_vec_env_reports_multiple_done_on_triggers_same_event() -> None:
    env = make_level1_1_noop_probe(
        done_on={
            "life_loss": {
                "triggers": [
                    {"id": "lives_decrease", "variables": "lives", "op": "decrease"},
                    {"id": "lives_change", "variables": "lives", "op": "change"},
                ],
            },
        },
    )
    try:
        env.reset()
        masks = np.zeros((1, env.num_buttons), dtype=np.uint8)

        for _step in range(1, 3000):
            _obs, _rewards, dones, infos = env.step(masks)
            if not bool(dones[0]):
                continue
            payload = infos[0]["done_on_info"]["life_loss"]
            assert payload["trigger"] == "lives_decrease"
            assert payload["op"] == "decrease"
            assert payload["compare"] == "reset"
            assert payload["variables"] == ["lives"]
            assert [trigger["trigger"] for trigger in payload["triggers"]] == [
                "lives_decrease",
                "lives_change",
            ]
            assert [trigger["op"] for trigger in payload["triggers"]] == [
                "decrease",
                "change",
            ]
            break
        else:
            pytest.fail("life_loss done_on rule did not fire before game-over")
    finally:
        env.close()


def test_native_turbo_vec_env_hwc_layout_and_safe_view_survives_next_step() -> None:
    rom_path = require_rom()

    env = SuperMarioBrosNesTurboVecEnv(
        compare.DEFAULT_STABLE_RETRO_GAME,
        state="Level1-1",
        num_envs=1,
        rom_path=str(rom_path),
        render_mode="rgb_array",
        use_restricted_actions=Actions.ALL,
        obs_crop=(32, 0, 0, 0),
        obs_resize=(84, 84),
        obs_grayscale=True,
        obs_resize_algorithm="nearest",
        obs_layout="hwc",
        obs_copy="safe_view",
        frame_skip=1,
        frame_stack=1,
        info_filter="none",
    )
    try:
        first = env.reset()
        first_snapshot = first.copy()
        masks = np.zeros((1, env.num_buttons), dtype=np.uint8)
        masks[0, BUTTON_TO_INDEX["RIGHT"]] = 1
        second, _rewards, _dones, infos = env.step(masks)
        assert first.shape == (1, 84, 84, 1)
        assert second.shape == (1, 84, 84, 1)
        np.testing.assert_array_equal(first, first_snapshot)
        assert infos == [{}]
    finally:
        env.close()


def test_native_turbo_vec_env_reward_clip_and_info_filter_all() -> None:
    rom_path = require_rom()

    env = SuperMarioBrosNesTurboVecEnv(
        compare.DEFAULT_STABLE_RETRO_GAME,
        state="Level1-1",
        num_envs=1,
        rom_path=str(rom_path),
        render_mode="rgb_array",
        use_restricted_actions=Actions.ALL,
        obs_crop=(32, 0, 0, 0),
        obs_resize=(84, 84),
        obs_grayscale=True,
        obs_layout="chw",
        reward_clip=(0.0, 0.0),
        info_filter={"mode": "all", "keys": ("lives", "xscrollHi")},
        frame_skip=4,
    )
    try:
        env.reset()
        masks = np.zeros((1, env.num_buttons), dtype=np.uint8)
        masks[0, BUTTON_TO_INDEX["RIGHT"]] = 1
        _obs, rewards, _dones, infos = env.step(masks)
        assert rewards.tolist() == [0.0]
        assert set(infos[0]) <= {"lives", "xscrollHi"}
        assert "lives" in infos[0]
    finally:
        env.close()


def test_native_turbo_vec_env_actions_all_mask_matches_discrete_fast_env() -> None:
    rom_path = require_rom()

    retro_env = SuperMarioBrosNesTurboVecEnv(
        compare.DEFAULT_STABLE_RETRO_GAME,
        state="Level1-1",
        num_envs=1,
        rom_path=str(rom_path),
        render_mode="rgb_array",
        use_restricted_actions=Actions.ALL,
        obs_crop=(32, 0, 0, 0),
        obs_resize=(84, 84),
        obs_grayscale=True,
        obs_resize_algorithm="area",
        obs_layout="chw",
        frame_skip=1,
        frame_stack=1,
        info_filter="none",
    )
    discrete_env = SuperMarioBrosNesTurboVecEnv(
        compare.DEFAULT_STABLE_RETRO_GAME,
        state="Level1-1",
        num_envs=1,
        rom_path=str(rom_path),
        render_mode="rgb_array",
        use_restricted_actions=Actions.DISCRETE,
        obs_crop=(32, 0, 0, 0),
        obs_resize=(84, 84),
        obs_grayscale=True,
        obs_resize_algorithm="area",
        obs_layout="chw",
        frame_skip=1,
        frame_stack=1,
        info_filter="none",
    )
    try:
        np.testing.assert_array_equal(retro_env.reset(), discrete_env.reset())
        masks = np.zeros((1, retro_env.num_buttons), dtype=np.uint8)
        masks[0, BUTTON_TO_INDEX["RIGHT"]] = 1
        masks[0, BUTTON_TO_INDEX["A"]] = 1
        retro_obs, retro_rewards, retro_dones, _infos = retro_env.step(masks)
        discrete_obs, discrete_rewards, discrete_dones, _infos = discrete_env.step(np.asarray([24], dtype=np.uint8))
        np.testing.assert_array_equal(retro_obs, discrete_obs)
        np.testing.assert_array_equal(retro_rewards, discrete_rewards)
        np.testing.assert_array_equal(retro_dones, discrete_dones)
    finally:
        retro_env.close()
        discrete_env.close()
