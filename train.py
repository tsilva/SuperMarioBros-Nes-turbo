#!/usr/bin/env python3
"""Train a JERK action sequence from the Super Mario Bros NES Level 1-1 state."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np

from supermariobrosnes_turbo import (
    ACTION_SETS,
    Actions,
    SuperMarioBrosNesTurboVecEnv,
    action_mask,
)
from supermariobrosnes_turbo.jerk import save_jerk_checkpoint


LOGGER = logging.getLogger("jerk_train")
ACTION_SET = "simple"
TOTAL_TIMESTEPS = 1_000_000
MAX_EPISODE_STEPS = 4_500
CHECKPOINT_FREQ = 50_000
EXPLOIT_BIAS = 0.25
RIGHT_STEPS = 100
BACKTRACK_STEPS = 70
JUMP_PROBABILITY = 0.1
JUMP_REPEAT = 4


@dataclass(frozen=True)
class EpisodeBoundary:
    done: bool
    life_loss: bool
    truncated: bool
    level_changed: bool


def episode_boundary(
    *,
    previous_lives: int,
    current_lives: int,
    previous_level: tuple[int, int],
    current_level: tuple[int, int],
    episode_steps: int,
    max_episode_steps: int,
    native_truncated: bool = False,
) -> EpisodeBoundary:
    """End on life loss or truncation; a level transition is never terminal."""
    life_loss = current_lives < previous_lives
    truncated = native_truncated or episode_steps >= max_episode_steps
    return EpisodeBoundary(
        done=life_loss or truncated,
        life_loss=life_loss,
        truncated=truncated,
        level_changed=current_level != previous_level,
    )


@dataclass
class EpisodeTrace:
    actions: list[int] = field(default_factory=list)
    cumulative_rewards: list[float] = field(default_factory=list)
    total_reward: float = 0.0

    def record(self, action: int, reward: float) -> None:
        self.actions.append(int(action))
        self.total_reward += float(reward)
        self.cumulative_rewards.append(self.total_reward)

    @property
    def best_reward(self) -> float:
        return max(self.cumulative_rewards, default=0.0)

    def best_sequence(self) -> tuple[int, ...]:
        if not self.cumulative_rewards:
            return ()
        end = self.cumulative_rewards.index(self.best_reward) + 1
        return tuple(self.actions[:end])


@dataclass
class Solution:
    action_sequence: tuple[int, ...]
    rewards: list[float]

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self.rewards))


@dataclass(frozen=True)
class EpisodeResult:
    trace: EpisodeTrace
    life_loss: bool
    truncated: bool
    level_changes: int


class MarioJerkTask:
    """Single-lane Level1-1 task with the JERK episode boundary contract."""

    def __init__(
        self,
        *,
        rom_path: str | Path | None,
        seed: int,
        max_episode_steps: int,
    ) -> None:
        self.native = SuperMarioBrosNesTurboVecEnv(
            "SuperMarioBros-Nes-v0",
            state="Level1-1",
            num_envs=1,
            num_threads=1,
            rom_path=rom_path,
            render_mode="rgb_array",
            use_restricted_actions=Actions.ALL,
            obs_copy="safe_view",
            obs_resize=(1, 1),
            obs_grayscale=True,
            obs_layout="chw",
            frame_skip=4,
            frame_stack=1,
            maxpool_last_two=False,
            sticky_action_prob=0.0,
            info_filter="all",
        )
        self.seed = int(seed)
        self.max_episode_steps = int(max_episode_steps)
        self.action_masks = np.stack(
            [action_mask(name) for name in ACTION_SETS[ACTION_SET]]
        ).astype(np.uint8)
        self.episode_steps = 0
        self.previous_lives = 0
        self.previous_level = (0, 0)

    def reset(self, episode: int) -> None:
        self.native.reset(seed=self.seed + episode)
        self.episode_steps = 0
        self.previous_lives = int(self.native.lives[0])
        self.previous_level = (
            int(self.native.level_hi[0]),
            int(self.native.level_lo[0]),
        )

    def step(self, action: int) -> tuple[float, EpisodeBoundary]:
        _obs, rewards, terminated, truncated, _infos = self.native.step(
            self.action_masks[[int(action)]]
        )
        self.episode_steps += 1
        current_lives = int(self.native.lives[0])
        current_level = (int(self.native.level_hi[0]), int(self.native.level_lo[0]))
        boundary = episode_boundary(
            previous_lives=self.previous_lives,
            current_lives=current_lives,
            previous_level=self.previous_level,
            current_level=current_level,
            episode_steps=self.episode_steps,
            max_episode_steps=self.max_episode_steps,
            native_truncated=bool(truncated[0]),
        )
        if bool(terminated[0]) and not boundary.life_loss:
            raise RuntimeError(
                "native environment terminated without a life loss; "
                "JERK training requires flag/level termination to stay disabled"
            )
        self.previous_lives = current_lives
        self.previous_level = current_level
        return float(rewards[0]), boundary

    def close(self) -> None:
        self.native.close()


def exploit_probability(total_steps: int, total_timesteps: int) -> float:
    return min(1.0, EXPLOIT_BIAS + total_steps / total_timesteps)


def move(
    task: MarioJerkTask,
    trace: EpisodeTrace,
    *,
    steps: int,
    left: bool,
    rng: random.Random,
    remaining_steps: int,
) -> tuple[float, EpisodeBoundary | None, int, int]:
    total_reward = 0.0
    jump_steps_left = 0
    taken = 0
    level_changes = 0
    for _ in range(min(steps, remaining_steps)):
        if jump_steps_left > 0:
            action_name = "a" if left else "right_a"
            jump_steps_left -= 1
        elif rng.random() < JUMP_PROBABILITY:
            action_name = "a" if left else "right_a"
            jump_steps_left = JUMP_REPEAT - 1
        else:
            action_name = "left" if left else "right"
        action = ACTION_SETS[ACTION_SET].index(action_name)
        reward, boundary = task.step(action)
        trace.record(action, reward)
        total_reward += reward
        taken += 1
        level_changes += int(boundary.level_changed)
        if boundary.done:
            return total_reward, boundary, taken, level_changes
    return total_reward, None, taken, level_changes


def explore_episode(
    task: MarioJerkTask,
    *,
    episode: int,
    rng: random.Random,
    step_budget: int,
) -> EpisodeResult:
    task.reset(episode)
    trace = EpisodeTrace()
    level_changes = 0
    boundary: EpisodeBoundary | None = None
    while len(trace.actions) < step_budget and boundary is None:
        reward, boundary, _taken, changes = move(
            task,
            trace,
            steps=RIGHT_STEPS,
            left=False,
            rng=rng,
            remaining_steps=step_budget - len(trace.actions),
        )
        level_changes += changes
        if boundary is not None:
            break
        if reward <= 0.0 and len(trace.actions) < step_budget:
            _reward, boundary, _taken, changes = move(
                task,
                trace,
                steps=BACKTRACK_STEPS,
                left=True,
                rng=rng,
                remaining_steps=step_budget - len(trace.actions),
            )
            level_changes += changes
    return EpisodeResult(
        trace,
        life_loss=bool(boundary and boundary.life_loss),
        truncated=boundary is None or bool(boundary.truncated),
        level_changes=level_changes,
    )


def replay_episode(
    task: MarioJerkTask,
    action_sequence: Sequence[int],
    *,
    episode: int,
    step_budget: int,
) -> EpisodeResult:
    task.reset(episode)
    trace = EpisodeTrace()
    noop = ACTION_SETS[ACTION_SET].index("noop")
    boundary: EpisodeBoundary | None = None
    level_changes = 0
    for step in range(step_budget):
        action = int(action_sequence[step]) if step < len(action_sequence) else noop
        reward, boundary = task.step(action)
        trace.record(action, reward)
        level_changes += int(boundary.level_changed)
        if boundary.done:
            break
    return EpisodeResult(
        trace,
        life_loss=bool(boundary and boundary.life_loss),
        truncated=boundary is None or bool(boundary.truncated),
        level_changes=level_changes,
    )


def best_solution(solutions: Sequence[Solution]) -> Solution:
    if not solutions:
        return Solution((), [0.0])
    return max(solutions, key=lambda solution: solution.mean_reward)


def action_names(sequence: Sequence[int]) -> tuple[str, ...]:
    names = ACTION_SETS[ACTION_SET]
    return tuple(names[int(action)] for action in sequence)


def save_training_checkpoint(
    path: Path,
    solutions: Sequence[Solution],
    *,
    timesteps: int,
    episodes: int,
    seed: int,
    max_episode_steps: int,
) -> Path:
    best = best_solution(solutions)
    return save_jerk_checkpoint(
        path,
        action_names(best.action_sequence),
        timesteps=timesteps,
        episodes=episodes,
        best_reward=best.mean_reward,
        action_set=ACTION_SET,
        metadata={
            "algorithm": "JERK (Just Enough Retained Knowledge)",
            "seed": seed,
            "environment": {
                "state": "Level1-1",
                "frame_skip": 4,
                "terminate_on_life_loss": True,
                "terminate_on_level_change": False,
                "max_episode_steps": max_episode_steps,
            },
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", type=Path, help="ROM path; defaults to ROM_PATH or .env")
    parser.add_argument("--output", type=Path, default=Path("runs/level1-1-jerk"))
    parser.add_argument("--seed", type=int, default=108)
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--max-episode-steps", type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument("--checkpoint-freq", type=int, default=CHECKPOINT_FREQ)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if min(args.timesteps, args.max_episode_steps) <= 0:
        raise SystemExit("training sizes must be positive")
    if args.checkpoint_freq < 0:
        raise SystemExit("--checkpoint-freq must be non-negative")
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "episodes.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    rng = random.Random(args.seed)
    task = MarioJerkTask(
        rom_path=args.rom,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
    )
    solutions: list[Solution] = []
    timesteps = 0
    episodes = 0
    next_checkpoint = args.checkpoint_freq
    try:
        while timesteps < args.timesteps:
            episodes += 1
            remaining = args.timesteps - timesteps
            replay = bool(solutions) and rng.random() < exploit_probability(
                timesteps, args.timesteps
            )
            if replay:
                selected = best_solution(solutions)
                result = replay_episode(
                    task,
                    selected.action_sequence,
                    episode=episodes,
                    step_budget=remaining,
                )
                selected.rewards.append(result.trace.total_reward)
                mode = "replay"
            else:
                result = explore_episode(
                    task,
                    episode=episodes,
                    rng=rng,
                    step_budget=remaining,
                )
                solutions.append(
                    Solution(result.trace.best_sequence(), [result.trace.best_reward])
                )
                mode = "explore"
            timesteps += len(result.trace.actions)
            best = best_solution(solutions)
            row: dict[str, Any] = {
                "episode": episodes,
                "timesteps": timesteps,
                "mode": mode,
                "episode_reward": result.trace.total_reward,
                "episode_best_reward": result.trace.best_reward,
                "best_mean_reward": best.mean_reward,
                "best_sequence_steps": len(best.action_sequence),
                "life_loss": result.life_loss,
                "truncated": result.truncated,
                "level_changes": result.level_changes,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            LOGGER.info(
                "episode=%d timesteps=%d mode=%s reward=%.1f best=%.1f sequence=%d "
                "life_loss=%s level_changes=%d",
                episodes,
                timesteps,
                mode,
                result.trace.total_reward,
                best.mean_reward,
                len(best.action_sequence),
                result.life_loss,
                result.level_changes,
            )
            if args.checkpoint_freq > 0 and timesteps >= next_checkpoint:
                save_training_checkpoint(
                    args.output / "checkpoints" / f"jerk_level1-1_{timesteps}.json",
                    solutions,
                    timesteps=timesteps,
                    episodes=episodes,
                    seed=args.seed,
                    max_episode_steps=args.max_episode_steps,
                )
                while next_checkpoint <= timesteps:
                    next_checkpoint += args.checkpoint_freq
        final_path = save_training_checkpoint(
            args.output / "final_policy.json",
            solutions,
            timesteps=timesteps,
            episodes=episodes,
            seed=args.seed,
            max_episode_steps=args.max_episode_steps,
        )
    finally:
        task.close()
    LOGGER.info("saved=%s timesteps=%d episodes=%d", final_path, timesteps, episodes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
