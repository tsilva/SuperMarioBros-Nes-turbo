#!/usr/bin/env python3
"""Standalone PyTorch PPO trainer for Super Mario Bros NES Level 1-1."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import platform
import time
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from torch import nn

from supermariobrosnes_turbo import ACTION_SETS, SuperMarioBrosNesTurboVecEnv, action_mask
from supermariobrosnes_turbo.ppo import PlainPPOPolicy, save_policy_checkpoint


LOGGER = logging.getLogger("ppo_train")
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
TRAIN_DONE_ON = ("life_loss", "level_change")
COMPLETION_RATE_METRIC = "completion_rate"
GAMMA = 0.9
GAE_LAMBDA = 1.0
CLIP_RANGE = 0.15
VF_COEF = 1.0
TARGET_KL = 0.16
MAX_GRAD_NORM = 0.5


@dataclass(frozen=True)
class TrainingProfile:
    name: str
    n_envs: int
    n_steps: int
    env_threads: int


PROFILES = {
    "b55": TrainingProfile("b55", n_envs=16, n_steps=512, env_threads=4),
    "m1": TrainingProfile("m1", n_envs=64, n_steps=128, env_threads=6),
}


def scheduled_value(initial: float, final: float, step: int, duration: int) -> float:
    fraction = min(max(step / duration, 0.0), 1.0)
    return initial + fraction * (final - initial)


class MarioVectorTask:
    """Native vector environment plus the rlab Level1-1 reward/task contract."""

    def __init__(
        self,
        *,
        rom_path: str | Path | None,
        num_envs: int,
        num_threads: int,
        seed: int,
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
            info_filter="terminal",
            done_on=TRAIN_DONE_ON,
            autoreset_mode="Disabled",
        )
        self.num_envs = int(num_envs)
        self.seed = int(seed)
        self.max_episode_steps = int(max_episode_steps)
        self.action_masks = np.stack(
            [action_mask(name) for name in ACTION_SETS["simple"]]
        ).astype(np.uint8)
        self.native_actions = np.empty((num_envs, self.native.num_buttons), dtype=np.uint8)
        self.episode_steps = np.zeros(num_envs, dtype=np.int64)
        self.episode_returns = np.zeros(num_envs, dtype=np.float64)
        self.max_x = np.zeros(num_envs, dtype=np.int64)
        self.last_score = np.zeros(num_envs, dtype=np.int64)
        self.last_lives = np.zeros(num_envs, dtype=np.int64)
        self.last_level_hi = np.zeros(num_envs, dtype=np.int64)
        self.last_level_lo = np.zeros(num_envs, dtype=np.int64)

    def _signals(self) -> tuple[np.ndarray, ...]:
        return tuple(
            values.astype(np.int64, copy=True)
            for values in (
                self.native.xscroll_hi,
                self.native.xscroll_lo,
                self.native.score,
                self.native.lives,
                self.native.level_hi,
                self.native.level_lo,
            )
        )

    def _initialize_lanes(self, mask: np.ndarray | None = None) -> None:
        x_hi, x_lo, score, lives, level_hi, level_lo = self._signals()
        mask = np.ones(self.num_envs, dtype=bool) if mask is None else mask
        x = x_hi * 256 + x_lo
        self.max_x[mask] = x[mask]
        self.last_score[mask] = score[mask]
        self.last_lives[mask] = lives[mask]
        self.last_level_hi[mask] = level_hi[mask]
        self.last_level_lo[mask] = level_lo[mask]
        self.episode_steps[mask] = 0
        self.episode_returns[mask] = 0.0

    def reset(self) -> np.ndarray:
        observations, _infos = self.native.reset(
            seed=[self.seed + lane for lane in range(self.num_envs)]
        )
        self._initialize_lanes()
        return observations

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        action_ids = np.asarray(actions, dtype=np.int64).reshape(-1)
        if action_ids.shape != (self.num_envs,):
            raise ValueError(f"expected {self.num_envs} actions, got {action_ids.shape}")
        self.native_actions[...] = self.action_masks[action_ids]
        obs, _native_rewards, terminated, truncated, native_infos = self.native.step(
            self.native_actions
        )
        x_hi, x_lo, score, lives, level_hi, level_lo = self._signals()
        x = x_hi * 256 + x_lo
        progress = np.maximum(x - self.max_x, 0)
        self.max_x = np.maximum(self.max_x, x)
        score_delta = np.maximum(score - self.last_score, 0)
        life_lost = lives < self.last_lives
        level_changed = (level_hi != self.last_level_hi) | (level_lo != self.last_level_lo)
        completed = level_changed & ~life_lost
        rewards = progress.astype(np.float32)
        rewards += 0.01 * score_delta.astype(np.float32)
        rewards -= 25.0 * life_lost.astype(np.float32)
        self.episode_steps += 1
        timed_out = (self.episode_steps >= self.max_episode_steps) & ~(
            terminated | truncated
        )
        dones = terminated | truncated | timed_out
        self.episode_returns += rewards
        infos = self._lane_infos(native_infos)
        for lane in np.flatnonzero(dones):
            index = int(lane)
            infos[index].update(
                {
                    "terminal_observation": obs[index].copy(),
                    "TimeLimit.truncated": bool(timed_out[index] or truncated[index]),
                    "level_complete": bool(completed[index]),
                    "life_loss": bool(life_lost[index]),
                    "level_change": bool(level_changed[index]),
                    "episode": {
                        "r": float(self.episode_returns[index]),
                        "l": int(self.episode_steps[index]),
                    },
                }
            )
        self.last_score[:] = score
        self.last_lives[:] = lives
        self.last_level_hi[:] = level_hi
        self.last_level_lo[:] = level_lo
        if np.any(dones):
            reset_obs, _reset_infos = self.native.reset(options={"reset_mask": dones})
            obs = obs.copy()
            obs[dones] = reset_obs[dones]
            self._initialize_lanes(dones)
        return obs, rewards, dones, infos

    def _lane_infos(self, columns: dict[str, Any]) -> list[dict[str, Any]]:
        return [self._lane_info(columns, lane) for lane in range(self.num_envs)]

    def _lane_info(self, columns: dict[str, Any], lane: int) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, values in columns.items():
            if key.startswith("_"):
                continue
            mask = columns.get(f"_{key}")
            if mask is not None and not bool(np.asarray(mask)[lane]):
                continue
            if isinstance(values, dict):
                value: Any = self._lane_info(values, lane)
            elif isinstance(values, np.ndarray) and values.shape[:1] == (self.num_envs,):
                value = values[lane]
            elif isinstance(values, (list, tuple)) and len(values) == self.num_envs:
                value = values[lane]
            else:
                value = values
            result[key] = value.item() if isinstance(value, np.generic) else value
        return result

    def close(self) -> None:
        self.native.close()


class CompletionTracker:
    def __init__(self, metrics_path: Path) -> None:
        self.outcomes: deque[bool] = deque(maxlen=SUCCESS_WINDOW)
        self.total_attempts = 0
        self.total_completions = 0
        self.metrics_path = metrics_path
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_path.write_text("", encoding="utf-8")

    @property
    def rate(self) -> float | None:
        return sum(self.outcomes) / len(self.outcomes) if self.outcomes else None

    @property
    def solved(self) -> bool:
        return len(self.outcomes) == SUCCESS_WINDOW and all(self.outcomes)

    def record(self, dones: np.ndarray, infos: list[dict[str, Any]]) -> None:
        for done, info in zip(dones, infos, strict=True):
            if not done:
                continue
            completed = bool(info.get("level_complete", False))
            self.outcomes.append(completed)
            self.total_attempts += 1
            self.total_completions += int(completed)

    def log(self, timesteps: int, *, stopped: bool = False) -> None:
        if self.rate is None:
            return
        LOGGER.info(
            "timesteps=%d completion_rate=%.6f window=%d/%d completions=%d/%d",
            timesteps,
            self.rate,
            len(self.outcomes),
            SUCCESS_WINDOW,
            self.total_completions,
            self.total_attempts,
        )
        row = {
            "timesteps": timesteps,
            COMPLETION_RATE_METRIC: self.rate,
            "window_size": len(self.outcomes),
            "window_completions": sum(self.outcomes),
            "total_attempts": self.total_attempts,
            "total_completions": self.total_completions,
            "stopped": stopped,
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


@dataclass
class Rollout:
    observations: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    old_values: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray


def collect_rollout(
    policy: PlainPPOPolicy,
    env: MarioVectorTask,
    observations: np.ndarray,
    *,
    n_steps: int,
    device: torch.device,
    tracker: CompletionTracker,
    timesteps: int,
) -> tuple[Rollout | None, np.ndarray, int]:
    obs_buffer = np.empty((n_steps, *observations.shape), dtype=np.uint8)
    actions = np.empty((n_steps, env.num_envs), dtype=np.int64)
    rewards = np.empty((n_steps, env.num_envs), dtype=np.float32)
    dones = np.empty((n_steps, env.num_envs), dtype=bool)
    values = np.empty((n_steps, env.num_envs), dtype=np.float32)
    log_probs = np.empty((n_steps, env.num_envs), dtype=np.float32)
    collected = 0
    for step in range(n_steps):
        obs_buffer[step] = observations
        with torch.no_grad():
            distribution, value_tensor = policy.distribution_and_value(
                torch.as_tensor(observations, device=device)
            )
            action_tensor = distribution.sample()
            log_prob_tensor = distribution.log_prob(action_tensor)
        action_array = action_tensor.cpu().numpy()
        next_observations, step_rewards, step_dones, infos = env.step(action_array)
        timeout_lanes = [
            lane
            for lane, info in enumerate(infos)
            if step_dones[lane] and info.get("TimeLimit.truncated")
        ]
        if timeout_lanes:
            terminal_obs = np.stack(
                [infos[lane]["terminal_observation"] for lane in timeout_lanes]
            )
            with torch.no_grad():
                _distribution, terminal_values = policy.distribution_and_value(
                    torch.as_tensor(terminal_obs, device=device)
                )
            step_rewards[timeout_lanes] += GAMMA * terminal_values.cpu().numpy()
        actions[step] = action_array
        rewards[step] = step_rewards
        dones[step] = step_dones
        values[step] = value_tensor.cpu().numpy()
        log_probs[step] = log_prob_tensor.cpu().numpy()
        observations = next_observations
        collected = step + 1
        timesteps += env.num_envs
        tracker.record(step_dones, infos)
        if tracker.solved:
            return None, observations, timesteps
    with torch.no_grad():
        _distribution, last_values_tensor = policy.distribution_and_value(
            torch.as_tensor(observations, device=device)
        )
    last_values = last_values_tensor.cpu().numpy()
    advantages = np.zeros_like(rewards)
    last_gae = np.zeros(env.num_envs, dtype=np.float32)
    for step in reversed(range(collected)):
        next_values = last_values if step == collected - 1 else values[step + 1]
        next_nonterminal = 1.0 - dones[step].astype(np.float32)
        delta = rewards[step] + GAMMA * next_values * next_nonterminal - values[step]
        last_gae = delta + GAMMA * GAE_LAMBDA * next_nonterminal * last_gae
        advantages[step] = last_gae
    returns = advantages + values
    rollout = Rollout(
        observations=obs_buffer.reshape((-1, *observations.shape[1:])),
        actions=actions.reshape(-1),
        old_log_probs=log_probs.reshape(-1),
        old_values=values.reshape(-1),
        advantages=advantages.reshape(-1),
        returns=returns.reshape(-1),
    )
    return rollout, observations, timesteps


def ppo_update(
    policy: PlainPPOPolicy,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    *,
    device: torch.device,
    ent_coef: float,
) -> dict[str, float]:
    sample_count = len(rollout.actions)
    indices = np.arange(sample_count)
    metrics: dict[str, list[float]] = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clip_fraction": [],
    }
    stop_early = False
    for _epoch in range(N_EPOCHS):
        np.random.shuffle(indices)
        epoch_kls: list[float] = []
        for start in range(0, sample_count, BATCH_SIZE):
            batch = indices[start : start + BATCH_SIZE]
            obs = torch.as_tensor(rollout.observations[batch], device=device)
            batch_actions = torch.as_tensor(rollout.actions[batch], device=device)
            old_log_probs = torch.as_tensor(rollout.old_log_probs[batch], device=device)
            advantages = torch.as_tensor(rollout.advantages[batch], device=device)
            returns = torch.as_tensor(rollout.returns[batch], device=device)
            new_log_probs, entropy, new_values = policy.evaluate_actions(obs, batch_actions)
            log_ratio = new_log_probs - old_log_probs
            ratio = torch.exp(log_ratio)
            policy_loss = -torch.min(
                advantages * ratio,
                advantages * torch.clamp(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE),
            ).mean()
            value_loss = torch.nn.functional.mse_loss(new_values, returns)
            loss = policy_loss + VF_COEF * value_loss - ent_coef * entropy.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            with torch.no_grad():
                approx_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()
                clip_fraction = (torch.abs(ratio - 1.0) > CLIP_RANGE).float().mean()
            epoch_kls.append(float(approx_kl.cpu()))
            metrics["policy_loss"].append(float(policy_loss.detach().cpu()))
            metrics["value_loss"].append(float(value_loss.detach().cpu()))
            metrics["entropy"].append(float(entropy.mean().detach().cpu()))
            metrics["approx_kl"].append(epoch_kls[-1])
            metrics["clip_fraction"].append(float(clip_fraction.cpu()))
        if epoch_kls and np.mean(epoch_kls) > 1.5 * TARGET_KL:
            stop_early = True
            break
    result = {name: float(np.mean(values)) for name, values in metrics.items()}
    variance = np.var(rollout.returns)
    result["explained_variance"] = (
        float(1.0 - np.var(rollout.returns - rollout.old_values) / variance)
        if variance > 0.0
        else float("nan")
    )
    result["early_kl_stop"] = float(stop_early)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", type=Path, help="ROM path; defaults to ROM_PATH or .env")
    parser.add_argument("--output", type=Path, default=Path("runs/level1-1-b55"))
    parser.add_argument("--seed", type=int, default=108)
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--profile", choices=("auto", "b55", "m1"), default="auto")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--env-threads", type=int)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--checkpoint-freq", type=int, default=CHECKPOINT_FREQ)
    return parser


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_training_profile(args: argparse.Namespace, device: torch.device) -> TrainingProfile:
    name = args.profile
    if name == "auto":
        apple_mps = (
            device.type == "mps"
            and platform.system() == "Darwin"
            and platform.machine() == "arm64"
        )
        name = "m1" if apple_mps else "b55"
    base = PROFILES[name]
    return TrainingProfile(
        name,
        base.n_envs if args.n_envs is None else args.n_envs,
        base.n_steps if args.n_steps is None else args.n_steps,
        base.env_threads if args.env_threads is None else args.env_threads,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    profile = resolve_training_profile(args, device)
    if min(args.timesteps, profile.n_envs, profile.n_steps, profile.env_threads) <= 0:
        raise SystemExit("training sizes must be positive")
    os.environ["RAYON_NUM_THREADS"] = str(profile.env_threads)
    torch.set_num_threads(args.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    env = MarioVectorTask(
        rom_path=args.rom,
        num_envs=profile.n_envs,
        num_threads=profile.env_threads,
        seed=args.seed,
    )
    policy = PlainPPOPolicy(input_channels=4, action_count=len(ACTION_SETS["simple"])).to(device)
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE, eps=1e-8)
    tracker = CompletionTracker(args.output / "level_completion.jsonl")
    observations = env.reset()
    timesteps = 0
    next_checkpoint = args.checkpoint_freq
    started_at = time.perf_counter()
    LOGGER.info(
        "plain_ppo profile=%s device=%s envs=%d rollout=%d samples_per_update=%d",
        profile.name,
        device,
        profile.n_envs,
        profile.n_steps,
        profile.n_envs * profile.n_steps,
    )
    try:
        while timesteps < args.timesteps and not tracker.solved:
            learning_rate = scheduled_value(
                LEARNING_RATE, LEARNING_RATE_FINAL, timesteps, SCHEDULE_STEPS
            )
            ent_coef = scheduled_value(ENT_COEF, ENT_COEF_FINAL, timesteps, SCHEDULE_STEPS)
            optimizer.param_groups[0]["lr"] = learning_rate
            rollout, observations, timesteps = collect_rollout(
                policy,
                env,
                observations,
                n_steps=profile.n_steps,
                device=device,
                tracker=tracker,
                timesteps=timesteps,
            )
            tracker.log(timesteps, stopped=tracker.solved)
            if tracker.solved or rollout is None:
                break
            metrics = ppo_update(
                policy, optimizer, rollout, device=device, ent_coef=ent_coef
            )
            elapsed = time.perf_counter() - started_at
            LOGGER.info(
                "timesteps=%d fps=%.1f lr=%.6g ent_coef=%.6g "
                "policy_loss=%.6g value_loss=%.6g entropy=%.6g "
                "approx_kl=%.6g clip_fraction=%.6g explained_variance=%.6g",
                timesteps,
                timesteps / elapsed,
                learning_rate,
                ent_coef,
                metrics["policy_loss"],
                metrics["value_loss"],
                metrics["entropy"],
                metrics["approx_kl"],
                metrics["clip_fraction"],
                metrics["explained_variance"],
            )
            if args.checkpoint_freq > 0 and timesteps >= next_checkpoint:
                save_policy_checkpoint(
                    args.output / "checkpoints" / f"ppo_level1-1_{timesteps}.pt",
                    policy,
                    timesteps=timesteps,
                )
                while next_checkpoint <= timesteps:
                    next_checkpoint += args.checkpoint_freq
        save_policy_checkpoint(
            args.output / "final_model.pt",
            policy,
            timesteps=timesteps,
            metadata={"profile": profile.name, "completion_rate": tracker.rate},
        )
    finally:
        env.close()
    LOGGER.info("saved=%s timesteps=%d", args.output / "final_model.pt", timesteps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
