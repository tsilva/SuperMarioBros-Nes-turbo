#!/usr/bin/env python3
"""Train the rlab B55 PPO recipe on Super Mario Bros. Level 1-1.

The environment stays a native Gymnasium VectorEnv.  ``MarioSb3VecEnv`` is the
small downstream adapter that supplies SB3's VecEnv contract, the rlab score
reward, the seven-action ``simple`` action set, and the 4,500-step time limit.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

# Avoid CPU oversubscription between PyTorch and the native vector emulator.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# Keep Apple Silicon training running if an occasional PyTorch op lacks MPS support.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecEnv

from supermariobrosnes_turbo import (
    ACTION_SETS,
    SuperMarioBrosNesTurboVecEnv,
    action_mask,
)


# B55 low-KL late-decay winner from ../rlab.
N_ENVS = 16
ENV_THREADS = 4
N_STEPS = 512
BATCH_SIZE = 512
N_EPOCHS = 10
LEARNING_RATE = 1.5e-4
LEARNING_RATE_FINAL = 1.0e-4
ENT_COEF = 0.01
ENT_COEF_FINAL = 1.0e-4
SCHEDULE_STEPS = 4_000_000
TOTAL_TIMESTEPS = 5_000_000
MAX_EPISODE_STEPS = 4_500
CHECKPOINT_FREQ = 500_000
SUCCESS_WINDOW = 100


def linear_schedule(initial: float, final: float, duration: int, total: int):
    """Return an SB3 schedule that reaches ``final`` after ``duration`` steps."""

    def schedule(progress_remaining: float) -> float:
        elapsed = (1.0 - progress_remaining) * total
        fraction = min(max(elapsed / duration, 0.0), 1.0)
        return initial + fraction * (final - initial)

    return schedule


class MarioSb3VecEnv(VecEnv):
    """Fast SB3 facade with the exact rlab Level1-1 task semantics."""

    def __init__(
        self,
        *,
        rom_path: str | Path | None,
        num_envs: int = N_ENVS,
        num_threads: int = ENV_THREADS,
        seed: int = 0,
        max_episode_steps: int = MAX_EPISODE_STEPS,
    ) -> None:
        self.native = SuperMarioBrosNesTurboVecEnv(
            "SuperMarioBros-Nes-v0",
            state="Level1-1",
            num_envs=num_envs,
            num_threads=num_threads,
            rom_path=rom_path,
            render_mode="rgb_array",
            obs_copy="safe_view",
            obs_resize=(84, 84),
            obs_crop=(32, 0, 0, 0),
            obs_crop_mode="mask",
            obs_crop_fill=0,
            obs_grayscale=True,
            obs_resize_algorithm="area",
            obs_layout="chw",
            frame_skip=4,
            frame_stack=4,
            maxpool_last_two=False,
            sticky_action_prob=0.0,
            info_filter="all",
            done_on=("life_loss", "level_change"),
            autoreset_mode="Disabled",
        )
        self._seed = int(seed)
        self._actions: np.ndarray | None = None
        self._max_episode_steps = int(max_episode_steps)
        self._action_masks = np.stack(
            [action_mask(name) for name in ACTION_SETS["simple"]]
        ).astype(np.uint8)
        self._native_actions = np.empty((num_envs, self.native.num_buttons), dtype=np.uint8)
        self._episode_steps = np.zeros(num_envs, dtype=np.int64)
        self._episode_returns = np.zeros(num_envs, dtype=np.float64)
        self._max_x = np.zeros(num_envs, dtype=np.int64)
        self._last_score = np.zeros(num_envs, dtype=np.int64)
        self._last_lives = np.zeros(num_envs, dtype=np.int64)
        self._last_level_hi = np.zeros(num_envs, dtype=np.int64)
        self._last_level_lo = np.zeros(num_envs, dtype=np.int64)
        super().__init__(
            num_envs,
            self.native.single_observation_space,
            spaces.Discrete(len(self._action_masks)),
        )

    def _signals(self) -> tuple[np.ndarray, ...]:
        return (
            self.native.xscroll_hi.astype(np.int64, copy=True),
            self.native.xscroll_lo.astype(np.int64, copy=True),
            self.native.score.astype(np.int64, copy=True),
            self.native.lives.astype(np.int64, copy=True),
            self.native.level_hi.astype(np.int64, copy=True),
            self.native.level_lo.astype(np.int64, copy=True),
        )

    def _initialize_lanes(self, mask: np.ndarray | None = None) -> None:
        x_hi, x_lo, score, lives, level_hi, level_lo = self._signals()
        if mask is None:
            mask = np.ones(self.num_envs, dtype=bool)
        x = x_hi * 256 + x_lo
        self._max_x[mask] = x[mask]
        self._last_score[mask] = score[mask]
        self._last_lives[mask] = lives[mask]
        self._last_level_hi[mask] = level_hi[mask]
        self._last_level_lo[mask] = level_lo[mask]
        self._episode_steps[mask] = 0
        self._episode_returns[mask] = 0.0

    def reset(self) -> np.ndarray:
        seeds = list(self._seeds)
        if all(seed is None for seed in seeds):
            seeds = [self._seed + lane for lane in range(self.num_envs)]
        obs, infos = self.native.reset(seed=seeds)
        self.reset_infos = self._lane_infos(infos)
        self._reset_seeds()
        self._reset_options()
        self._initialize_lanes()
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        values = np.asarray(actions, dtype=np.int64).reshape(-1)
        if values.shape != (self.num_envs,):
            raise ValueError(f"expected {self.num_envs} actions, got {values.shape}")
        if np.any(values < 0) or np.any(values >= len(self._action_masks)):
            raise ValueError("simple action ids are out of range")
        self._actions = values

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        if self._actions is None:
            raise RuntimeError("step_wait() called before step_async()")
        self._native_actions[...] = self._action_masks[self._actions]
        self._actions = None
        obs, _native_rewards, terminated, truncated, native_infos = self.native.step(
            self._native_actions
        )

        x_hi, x_lo, score, lives, level_hi, level_lo = self._signals()
        x = x_hi * 256 + x_lo
        progress = np.maximum(x - self._max_x, 0)
        self._max_x = np.maximum(self._max_x, x)
        score_delta = np.maximum(score - self._last_score, 0)
        life_lost = lives < self._last_lives
        level_changed = (level_hi != self._last_level_hi) | (level_lo != self._last_level_lo)
        completed = level_changed & ~life_lost

        # rlab B55 reward_mode=score, score_progress_clipped=false.
        rewards = progress.astype(np.float32)
        rewards += 0.01 * score_delta.astype(np.float32)
        rewards -= 25.0 * life_lost.astype(np.float32)

        self._episode_steps += 1
        timed_out = self._episode_steps >= self._max_episode_steps
        timed_out &= ~(terminated | truncated)
        dones = terminated | truncated | timed_out
        self._episode_returns += rewards
        lane_infos = self._lane_infos(native_infos)

        for lane in np.flatnonzero(dones):
            index = int(lane)
            lane_infos[index]["terminal_observation"] = obs[index].copy()
            lane_infos[index]["TimeLimit.truncated"] = bool(timed_out[index] or truncated[index])
            lane_infos[index]["level_complete"] = bool(completed[index])
            lane_infos[index]["episode"] = {
                "r": float(self._episode_returns[index]),
                "l": int(self._episode_steps[index]),
            }

        self._last_score[:] = score
        self._last_lives[:] = lives
        self._last_level_hi[:] = level_hi
        self._last_level_lo[:] = level_lo

        if np.any(dones):
            # The native env is in manual-reset mode so SB3 always receives the
            # true terminal frame and the reset frame in the expected places.
            reset_obs, reset_infos = self.native.reset(options={"reset_mask": dones})
            obs = obs.copy()
            obs[dones] = reset_obs[dones]
            reset_lanes = self._lane_infos(reset_infos)
            for lane in np.flatnonzero(dones):
                self.reset_infos[int(lane)] = reset_lanes[int(lane)]
            self._initialize_lanes(dones)

        return obs, rewards, dones, lane_infos

    def _lane_infos(self, columns: dict[str, Any]) -> list[dict[str, Any]]:
        result = [{} for _ in range(self.num_envs)]
        for key, values in columns.items():
            if key.startswith("_"):
                continue
            mask = columns.get(f"_{key}")
            for lane in range(self.num_envs):
                if mask is not None and not bool(np.asarray(mask)[lane]):
                    continue
                if isinstance(values, np.ndarray) and values.shape[:1] == (self.num_envs,):
                    value = values[lane]
                elif isinstance(values, (list, tuple)) and len(values) == self.num_envs:
                    value = values[lane]
                else:
                    value = values
                result[lane][key] = value.item() if isinstance(value, np.generic) else value
        return result

    def close(self) -> None:
        self.native.close()

    def get_images(self) -> list[np.ndarray | None]:
        return list(self.native.get_images())

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        value = getattr(self.native, attr_name)
        return [value for _ in self._get_indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        del indices
        setattr(self.native, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        method = getattr(self.native, method_name)
        return [method(*method_args, **method_kwargs) for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        del wrapper_class
        return [False for _ in self._get_indices(indices)]


class WinnerScheduleAndStop(BaseCallback):
    """Decay entropy and stop once the latest 100 episodes are all clears."""

    def __init__(self, *, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.outcomes: deque[bool] = deque(maxlen=SUCCESS_WINDOW)
        self.started_at = 0.0

    def _on_training_start(self) -> None:
        self.started_at = time.perf_counter()

    def _on_rollout_start(self) -> None:
        fraction = min(self.num_timesteps / SCHEDULE_STEPS, 1.0)
        self.model.ent_coef = ENT_COEF + fraction * (ENT_COEF_FINAL - ENT_COEF)

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
            if done:
                self.outcomes.append(bool(info.get("level_complete", False)))
        if len(self.outcomes) == SUCCESS_WINDOW and all(self.outcomes):
            elapsed = time.perf_counter() - self.started_at
            print(
                f"Reached strict {SUCCESS_WINDOW}/{SUCCESS_WINDOW} clears at "
                f"{self.num_timesteps:,} steps in {elapsed:.1f}s; stopping.",
                flush=True,
            )
            return False
        return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the rlab B55 PPO winner to clear Super Mario Bros Level1-1."
    )
    parser.add_argument("--rom", type=Path, help="ROM path; defaults to ROM_PATH or .env")
    parser.add_argument("--output", type=Path, default=Path("runs/level1-1-b55"))
    parser.add_argument("--seed", type=int, default=108, help="B55's first winning seed")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Training device; auto prefers CUDA, then Apple MPS, then CPU",
    )
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    parser.add_argument("--env-threads", type=int, default=ENV_THREADS)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--checkpoint-freq", type=int, default=CHECKPOINT_FREQ)
    return parser


def optimize_torch(num_threads: int) -> None:
    torch.set_num_threads(num_threads)
    torch.set_float32_matmul_precision("high")
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def resolve_device(device: str) -> str:
    """Resolve SB3's auto device with explicit Apple Silicon support."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(args.timesteps, args.n_envs, args.env_threads, args.torch_threads) <= 0:
        raise SystemExit("timesteps, n-envs, env-threads, and torch-threads must be positive")
    optimize_torch(args.torch_threads)
    set_random_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    env = MarioSb3VecEnv(
        rom_path=args.rom,
        num_envs=args.n_envs,
        num_threads=args.env_threads,
        seed=args.seed,
    )
    lr = linear_schedule(
        LEARNING_RATE,
        LEARNING_RATE_FINAL,
        SCHEDULE_STEPS,
        args.timesteps,
    )
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=lr,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=0.9,
        gae_lambda=1.0,
        ent_coef=ENT_COEF,
        vf_coef=1.0,
        clip_range=0.15,
        normalize_advantage=False,
        target_kl=0.16,
        policy_kwargs={"optimizer_kwargs": {"eps": 1e-8}},
        device=device,
        seed=args.seed,
        verbose=1,
    )

    callbacks: list[BaseCallback] = [
        WinnerScheduleAndStop(),
    ]
    if args.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(args.checkpoint_freq // args.n_envs, 1),
                save_path=str(args.output / "checkpoints"),
                name_prefix="ppo_level1-1_b55",
            )
        )

    print(
        f"B55: envs={args.n_envs} env_threads={args.env_threads} "
        f"rollout={N_STEPS} batch={BATCH_SIZE} device={model.device}",
        flush=True,
    )
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=CallbackList(callbacks),
            progress_bar=True,
        )
        model.save(args.output / "final_model.zip")
    finally:
        env.close()
    print(f"Saved {args.output / 'final_model.zip'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
