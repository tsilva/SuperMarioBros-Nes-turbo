#!/usr/bin/env python3
"""Train rlab-compatible JERK on a named Super Mario Bros NES level."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any
import uuid

import numpy as np

from supermariobrosnes_turbo import (
    ACTION_SETS,
    Actions,
    SuperMarioBrosNesTurboVecEnv,
    action_mask,
    list_available_states,
)
from supermariobrosnes_turbo.jerk import (
    JerkPolicy,
    JerkSearch,
    JerkSequenceMinimizer,
    RetainedSequence,
    load_jerk_checkpoint,
    normalize_level_name,
    policy_path_for_level,
    run_directory_for_level,
)


LOGGER = logging.getLogger("jerk_train")
ACTION_SET = "simple"
TOTAL_TIMESTEPS = 10_000_000
N_ENVS = 16
MAX_EPISODE_STEPS = 4_500
STALL_STEPS = 300
CHECKPOINT_FREQ = 0
LOG_INTERVAL_STEPS = 10_000
ARCHIVE_REPLAY_PROBABILITY_INITIAL = 0.25
ARCHIVE_REPLAY_PROBABILITY_MAX = 0.9
PROTECTED_PREFIX_STEPS = 128
MAX_PREFIX_SHORTEN_STEPS = 128
RETAINED_LIMIT = 256
FALLBACK_ACTION = "noop"
MINIMIZE_TIMESTEPS = 1_000_000
MINIMIZE_MAX_CHUNK_STEPS = 128
MINIMIZE_PATIENCE = 4
MINIMIZE_COMPLETION_GRACE_STEPS = 8


@dataclass(frozen=True)
class EpisodeBoundary:
    done: bool
    life_loss: bool
    truncated: bool
    level_changed: bool
    stalled: bool = False


def episode_boundary(
    *,
    previous_lives: int,
    current_lives: int,
    previous_level: tuple[int, int],
    current_level: tuple[int, int],
    episode_steps: int,
    max_episode_steps: int,
    native_truncated: bool = False,
    stalled: bool = False,
) -> EpisodeBoundary:
    """End failed attempts on life loss, stall, or timeout, never level change."""
    life_loss = current_lives < previous_lives
    truncated = native_truncated or episode_steps >= max_episode_steps
    return EpisodeBoundary(
        done=life_loss or stalled or truncated,
        life_loss=life_loss,
        truncated=truncated,
        level_changed=current_level != previous_level,
        stalled=stalled,
    )


@dataclass(frozen=True)
class EpisodeRecord:
    lane: int
    completed: bool
    progress: float
    episode_return: float
    episode_length: int
    life_loss: bool
    stalled: bool
    truncated: bool


class MarioJerkTask:
    """Vectorized level task matching rlab's JERK reward and failure contract."""

    def __init__(
        self,
        *,
        level: str,
        rom_path: str | Path | None,
        seed: int,
        n_envs: int,
        max_episode_steps: int,
        stall_steps: int,
    ) -> None:
        self.n_envs = int(n_envs)
        self.level = normalize_level_name(level)
        self.seed = int(seed)
        self.max_episode_steps = int(max_episode_steps)
        self.stall_steps = int(stall_steps)
        self.native = SuperMarioBrosNesTurboVecEnv(
            "SuperMarioBros-Nes-v0",
            state=self.level,
            num_envs=self.n_envs,
            num_threads=self.n_envs,
            rom_path=rom_path,
            render_mode="rgb_array",
            use_restricted_actions=Actions.ALL,
            obs_copy="unsafe_view",
            obs_resize=(1, 1),
            obs_grayscale=True,
            obs_layout="chw",
            frame_skip=4,
            frame_stack=1,
            maxpool_last_two=False,
            sticky_action_prob=0.0,
            reward_clip=False,
            info_filter="all",
        )
        self.action_masks = np.stack(
            [action_mask(name) for name in ACTION_SETS[ACTION_SET]]
        ).astype(np.uint8)
        self.episode_steps = np.zeros(self.n_envs, dtype=np.int64)
        self.last_progress_step = np.zeros(self.n_envs, dtype=np.int64)
        self.episode_returns = np.zeros(self.n_envs, dtype=np.float64)
        self.previous_lives = np.zeros(self.n_envs, dtype=np.int16)
        self.previous_level_hi = np.zeros(self.n_envs, dtype=np.int16)
        self.previous_level_lo = np.zeros(self.n_envs, dtype=np.int16)
        self.previous_score = np.zeros(self.n_envs, dtype=np.int64)
        self.level_max_x = np.zeros(self.n_envs, dtype=np.int64)
        self.completed_base = np.zeros(self.n_envs, dtype=np.int64)
        self.max_global_x = np.zeros(self.n_envs, dtype=np.int64)

    def _initialize_lanes(self, mask: np.ndarray) -> None:
        current_x = self.native.xscroll_hi.astype(
            np.int64, copy=False
        ) * 256 + self.native.xscroll_lo.astype(np.int64, copy=False)
        self.episode_steps[mask] = 0
        self.last_progress_step[mask] = 0
        self.episode_returns[mask] = 0.0
        self.previous_lives[mask] = self.native.lives[mask]
        self.previous_level_hi[mask] = self.native.level_hi[mask]
        self.previous_level_lo[mask] = self.native.level_lo[mask]
        self.previous_score[mask] = self.native.score[mask]
        self.level_max_x[mask] = current_x[mask]
        self.completed_base[mask] = 0
        self.max_global_x[mask] = current_x[mask]

    def reset(self) -> np.ndarray:
        observations, _infos = self.native.reset(seed=self.seed)
        self._initialize_lanes(np.ones(self.n_envs, dtype=np.bool_))
        return observations

    def reset_lanes(self, mask: np.ndarray) -> None:
        reset_mask = np.asarray(mask, dtype=np.bool_)
        if not np.any(reset_mask):
            return
        self.native.reset(options={"reset_mask": reset_mask})
        self._initialize_lanes(reset_mask)

    def step(
        self, actions: np.ndarray
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, dict[int, EpisodeRecord], np.ndarray
    ]:
        observations, _native_rewards, native_terminated, native_truncated, _infos = (
            self.native.step(self.action_masks[np.asarray(actions, dtype=np.int64)])
        )
        current_lives = self.native.lives.astype(np.int64, copy=False)
        current_level_hi = self.native.level_hi.astype(np.int64, copy=False)
        current_level_lo = self.native.level_lo.astype(np.int64, copy=False)
        current_score = self.native.score.astype(np.int64, copy=False)
        current_x = self.native.xscroll_hi.astype(
            np.int64, copy=False
        ) * 256 + self.native.xscroll_lo.astype(np.int64, copy=False)

        life_loss = current_lives < self.previous_lives
        level_changed = (current_level_hi != self.previous_level_hi) | (
            current_level_lo != self.previous_level_lo
        )
        completed = level_changed & ~life_loss

        self.completed_base[completed] += self.level_max_x[completed]
        self.level_max_x[completed] = 0
        effective_x = np.where(level_changed, 0, current_x)
        self.level_max_x = np.maximum(self.level_max_x, effective_x)
        global_max = self.completed_base + self.level_max_x
        progress_delta = np.maximum(global_max - self.max_global_x, 0)
        self.max_global_x = np.maximum(self.max_global_x, global_max)
        score_delta = np.maximum(current_score - self.previous_score, 0)

        progressed = progress_delta > 0
        self.last_progress_step[progressed] = self.episode_steps[progressed]
        self.episode_steps += 1
        stalled = (self.stall_steps > 0) & (
            self.episode_steps - self.last_progress_step >= self.stall_steps
        )
        timed_out = self.episode_steps >= self.max_episode_steps

        shaped_rewards = (
            progress_delta.astype(np.float64)
            + 0.01 * score_delta.astype(np.float64)
            - 25.0 * life_loss.astype(np.float64)
        )
        self.episode_returns += shaped_rewards

        failures = life_loss | stalled | timed_out | native_truncated
        unexpected_native_terminal = native_terminated & ~completed & ~life_loss
        failures |= unexpected_native_terminal
        records: dict[int, EpisodeRecord] = {}
        search_dones = failures | completed
        for lane in np.flatnonzero(search_dones):
            index = int(lane)
            records[index] = EpisodeRecord(
                lane=index,
                completed=bool(completed[index]),
                progress=float(self.max_global_x[index]),
                episode_return=float(self.episode_returns[index]),
                episode_length=int(self.episode_steps[index]),
                life_loss=bool(life_loss[index]),
                stalled=bool(stalled[index]),
                truncated=bool(timed_out[index] or native_truncated[index]),
            )

        self.previous_lives[:] = current_lives
        self.previous_level_hi[:] = current_level_hi
        self.previous_level_lo[:] = current_level_lo
        self.previous_score[:] = current_score
        return observations, shaped_rewards, failures, records, completed

    def close(self) -> None:
        self.native.close()


def exploit_probability(total_steps: int, total_timesteps: int) -> float:
    return min(
        ARCHIVE_REPLAY_PROBABILITY_MAX,
        ARCHIVE_REPLAY_PROBABILITY_INITIAL + total_steps / max(total_timesteps, 1),
    )


def evaluate_action_sequences(
    task: MarioJerkTask,
    sequences: list[tuple[int, ...]],
    *,
    fallback_action: int,
    completion_grace_steps: int,
) -> tuple[list[RetainedSequence], int]:
    """Replay up to one candidate per lane and return only completed candidates."""
    if not sequences:
        return [], 0
    if len(sequences) > task.n_envs:
        raise ValueError("candidate batch exceeds the task environment count")
    if completion_grace_steps < 0:
        raise ValueError("completion_grace_steps must be non-negative")

    task.reset()
    active = np.zeros(task.n_envs, dtype=np.bool_)
    active[: len(sequences)] = True
    completed: list[RetainedSequence] = []
    transitions = 0
    max_steps = max(len(sequence) for sequence in sequences) + completion_grace_steps
    for step in range(max_steps):
        actions = np.full(task.n_envs, fallback_action, dtype=np.int64)
        for lane, sequence in enumerate(sequences):
            if active[lane] and step < len(sequence):
                actions[lane] = sequence[step]

        _observations, _rewards, failures, records, successes = task.step(actions)
        transitions += task.n_envs
        terminal = failures | successes
        for lane, sequence in enumerate(sequences):
            if not active[lane]:
                continue
            if successes[lane]:
                record = records[lane]
                retained_actions = sequence[: min(step + 1, len(sequence))]
                completed.append(
                    RetainedSequence(
                        actions=retained_actions,
                        return_sum=record.episode_return,
                        return_count=1,
                        completed=True,
                        progress=record.progress,
                    )
                )
                active[lane] = False
            elif failures[lane] or step + 1 >= len(sequence) + completion_grace_steps:
                active[lane] = False

        if not np.any(active):
            break
        if np.any(terminal):
            task.reset_lanes(terminal)
    return completed, transitions


def minimize_completed_sequence(
    task: MarioJerkTask,
    initial: RetainedSequence,
    *,
    seed: int,
    transition_budget: int,
    max_chunk_steps: int,
    patience: int,
    fallback_action: int,
    completion_grace_steps: int,
    metrics_path: Path,
) -> tuple[RetainedSequence, int, int]:
    minimizer = JerkSequenceMinimizer(
        initial=initial,
        n_envs=task.n_envs,
        seed=seed,
        max_chunk_steps=max_chunk_steps,
        patience=patience,
    )
    transitions = 0
    while not minimizer.done and transitions < transition_budget:
        before_steps = len(minimizer.incumbent.actions)
        chunk_steps = minimizer.chunk_steps
        mutations = minimizer.propose()
        completed, used = evaluate_action_sequences(
            task,
            [mutation.actions for mutation in mutations],
            fallback_action=fallback_action,
            completion_grace_steps=completion_grace_steps,
        )
        transitions += used
        improved = minimizer.observe(completed)
        row = {
            "phase": "minimize",
            "minimization_transitions": transitions,
            "round": minimizer.rounds,
            "attempts": minimizer.attempts,
            "chunk_steps": chunk_steps,
            "completed_mutations": len(completed),
            "improved": improved,
            "previous_sequence_steps": before_steps,
            "best_sequence_steps": len(minimizer.incumbent.actions),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        if improved:
            LOGGER.info(
                "minimized sequence=%d->%d removed=%d transitions=%d",
                before_steps,
                len(minimizer.incumbent.actions),
                before_steps - len(minimizer.incumbent.actions),
                transitions,
            )
    return minimizer.incumbent, transitions, minimizer.improvements


def _save_policy(policy: JerkPolicy, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.stem}.{uuid.uuid4().hex}.zip"
    policy.save(temporary)
    temporary.replace(path)
    return path


def _metric_row(
    search: JerkSearch, *, elapsed: float, accepted: bool
) -> dict[str, Any]:
    candidate = search.best_candidate()
    return {
        "timesteps": search.global_step,
        "episodes": search.completed_episodes,
        "retained_count": search.retained_count,
        "archive_replay_probability": search.archive_replay_probability,
        "archive_selected_prefix_return_mean": (
            search.archive_selected_prefix_return_mean
        ),
        "best_sequence_steps": len(candidate.actions) if candidate else 0,
        "best_mean_reward": candidate.mean_return if candidate else 0.0,
        "best_progress": candidate.progress if candidate else 0.0,
        "best_completed": candidate.completed if candidate else False,
        "accepted": accepted,
        "loop_fps": search.global_step / max(elapsed, 1e-9),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("level", help="Packaged level state, for example Level1-1")
    parser.add_argument(
        "--rom", type=Path, help="ROM path; defaults to ROM_PATH or .env"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Run directory; defaults to runs/<Level>-jerk",
    )
    parser.add_argument("--seed", type=int, default=108)
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    parser.add_argument("--max-episode-steps", type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument("--stall-steps", type=int, default=STALL_STEPS)
    parser.add_argument("--checkpoint-freq", type=int, default=CHECKPOINT_FREQ)
    parser.add_argument("--log-interval-steps", type=int, default=LOG_INTERVAL_STEPS)
    parser.add_argument(
        "--archive-replay-probability-initial",
        type=float,
        default=ARCHIVE_REPLAY_PROBABILITY_INITIAL,
    )
    parser.add_argument(
        "--archive-replay-probability-max",
        type=float,
        default=ARCHIVE_REPLAY_PROBABILITY_MAX,
    )
    parser.add_argument(
        "--protected-prefix-steps", type=int, default=PROTECTED_PREFIX_STEPS
    )
    parser.add_argument(
        "--max-prefix-shorten-steps", type=int, default=MAX_PREFIX_SHORTEN_STEPS
    )
    parser.add_argument("--retained-limit", type=int, default=RETAINED_LIMIT)
    parser.add_argument("--fallback-action", default=FALLBACK_ACTION)
    parser.add_argument("--minimize-timesteps", type=int, default=MINIMIZE_TIMESTEPS)
    parser.add_argument(
        "--minimize-max-chunk-steps",
        type=int,
        default=MINIMIZE_MAX_CHUNK_STEPS,
    )
    parser.add_argument("--minimize-patience", type=int, default=MINIMIZE_PATIENCE)
    parser.add_argument(
        "--minimize-completion-grace-steps",
        type=int,
        default=MINIMIZE_COMPLETION_GRACE_STEPS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = build_parser().parse_args(argv)
    try:
        args.level = normalize_level_name(args.level)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    available_states = set(list_available_states())
    if args.level not in available_states:
        choices = ", ".join(sorted(available_states))
        raise SystemExit(
            f"unknown packaged level {args.level!r}; available states: {choices}"
        )
    if (
        min(
            args.timesteps,
            args.n_envs,
            args.max_episode_steps,
            args.max_prefix_shorten_steps,
            args.retained_limit,
            args.log_interval_steps,
            args.minimize_max_chunk_steps,
            args.minimize_patience,
        )
        <= 0
    ):
        raise SystemExit("JERK training sizes must be positive")
    if args.timesteps % args.n_envs:
        raise SystemExit("--timesteps must be divisible by --n-envs")
    if (
        args.stall_steps < 0
        or args.checkpoint_freq < 0
        or args.protected_prefix_steps < 0
        or args.minimize_timesteps < 0
        or args.minimize_completion_grace_steps < 0
    ):
        raise SystemExit("JERK non-negative sizes must not be negative")

    run_dir = args.output or run_directory_for_level(args.level)
    target_policy_path = (
        policy_path_for_level(args.level)
        if args.output is None
        else run_dir / f"{args.level}.zip"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "episodes.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    (run_dir / "run_config.json").write_text(
        json.dumps(vars(args), default=str, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    search = JerkSearch(
        n_envs=args.n_envs,
        seed=args.seed,
        total_timesteps=args.timesteps,
        action_names=ACTION_SETS[ACTION_SET],
        fallback_action=args.fallback_action,
        archive_replay_probability_initial=args.archive_replay_probability_initial,
        archive_replay_probability_max=args.archive_replay_probability_max,
        protected_prefix_steps=args.protected_prefix_steps,
        max_prefix_shorten_steps=args.max_prefix_shorten_steps,
        retained_limit=args.retained_limit,
    )
    task = MarioJerkTask(
        level=args.level,
        rom_path=args.rom,
        seed=args.seed,
        n_envs=args.n_envs,
        max_episode_steps=args.max_episode_steps,
        stall_steps=args.stall_steps,
    )
    task.reset()
    started_at = time.perf_counter()
    next_log = args.log_interval_steps
    next_checkpoint = args.checkpoint_freq if args.checkpoint_freq > 0 else None
    accepted = False
    accepted_lane: int | None = None
    try:
        while search.global_step < args.timesteps:
            actions = search.next_actions()
            _observations, rewards, failure_dones, records, successes = task.step(
                actions
            )
            search_dones = failure_dones | successes
            search.observe(rewards, search_dones, records)
            step = search.global_step

            if np.any(successes):
                accepted = True
                accepted_lane = int(np.flatnonzero(successes)[0])
                accepted_path = run_dir / "checkpoints" / f"{args.level}-{step}.zip"
                _save_policy(search.policy(), accepted_path)
                LOGGER.info(
                    "accepted first %s success step=%d lane=%d path=%s",
                    args.level,
                    step,
                    accepted_lane,
                    accepted_path,
                )
                break

            task.reset_lanes(failure_dones)

            if step >= next_log:
                row = _metric_row(
                    search, elapsed=time.perf_counter() - started_at, accepted=False
                )
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                LOGGER.info(
                    "timesteps=%d episodes=%d retained=%d replay=%.3f best=%.1f "
                    "progress=%.0f sequence=%d fps=%.0f",
                    step,
                    search.completed_episodes,
                    search.retained_count,
                    search.archive_replay_probability,
                    row["best_mean_reward"],
                    row["best_progress"],
                    row["best_sequence_steps"],
                    row["loop_fps"],
                )
                while next_log <= step:
                    next_log += args.log_interval_steps

            while next_checkpoint is not None and step >= next_checkpoint:
                _save_policy(
                    search.policy(),
                    run_dir / "checkpoints" / f"{args.level}-{step}.zip",
                )
                next_checkpoint += args.checkpoint_freq

        discovery_candidate = search.best_candidate()
        existing_candidate: RetainedSequence | None = None
        if target_policy_path.is_file():
            try:
                existing_policy = load_jerk_checkpoint(target_policy_path)
                if existing_policy.action_names != search.action_names:
                    raise ValueError(
                        "existing policy action names do not match training"
                    )
                existing_completed, _existing_validation_transitions = (
                    evaluate_action_sequences(
                        task,
                        [existing_policy.action_sequence],
                        fallback_action=existing_policy.fallback_action,
                        completion_grace_steps=args.minimize_completion_grace_steps,
                    )
                )
                if existing_completed:
                    existing_candidate = existing_completed[0]
                    LOGGER.info(
                        "validated existing %s policy sequence=%d",
                        args.level,
                        len(existing_candidate.actions),
                    )
                else:
                    LOGGER.warning(
                        "ignoring existing %s policy because replay did not complete",
                        target_policy_path,
                    )
            except (OSError, ValueError) as exc:
                LOGGER.warning(
                    "ignoring unreadable existing policy %s: %s",
                    target_policy_path,
                    exc,
                )

        completed_candidates = [
            candidate
            for candidate in (discovery_candidate, existing_candidate)
            if candidate is not None and candidate.completed
        ]
        final_candidate = (
            max(completed_candidates, key=lambda candidate: candidate.rank)
            if completed_candidates
            else discovery_candidate
        )
        minimization_start_steps = (
            len(final_candidate.actions) if final_candidate is not None else 0
        )
        minimization_transitions = 0
        minimization_improvements = 0
        if accepted and final_candidate is not None and args.minimize_timesteps > 0:
            minimization_initial = final_candidate
            final_candidate, minimization_transitions, minimization_improvements = (
                minimize_completed_sequence(
                    task,
                    minimization_initial,
                    seed=args.seed,
                    transition_budget=args.minimize_timesteps,
                    max_chunk_steps=args.minimize_max_chunk_steps,
                    patience=args.minimize_patience,
                    fallback_action=search.fallback_action,
                    completion_grace_steps=args.minimize_completion_grace_steps,
                    metrics_path=metrics_path,
                )
            )
            LOGGER.info(
                "minimization finished sequence=%d->%d transitions=%d improvements=%d",
                len(minimization_initial.actions),
                len(final_candidate.actions),
                minimization_transitions,
                minimization_improvements,
            )
        final_policy = JerkPolicy(
            action_names=search.action_names,
            action_sequence=() if final_candidate is None else final_candidate.actions,
            fallback_action=search.fallback_action,
        )
        final_path = _save_policy(
            final_policy,
            target_policy_path,
        )
        final_row = _metric_row(
            search, elapsed=time.perf_counter() - started_at, accepted=accepted
        )
        final_row["accepted_lane"] = accepted_lane
        final_row["phase"] = "final"
        final_row["discovery_sequence_steps"] = (
            len(discovery_candidate.actions) if discovery_candidate is not None else 0
        )
        final_row["existing_sequence_steps"] = (
            len(existing_candidate.actions) if existing_candidate is not None else 0
        )
        final_row["minimization_start_sequence_steps"] = minimization_start_steps
        final_row["best_sequence_steps"] = len(final_policy.action_sequence)
        final_row["minimization_transitions"] = minimization_transitions
        final_row["minimization_improvements"] = minimization_improvements
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(final_row, sort_keys=True) + "\n")
    finally:
        task.close()

    LOGGER.info(
        "saved=%s timesteps=%d episodes=%d retained=%d accepted=%s",
        final_path,
        search.global_step,
        search.completed_episodes,
        search.retained_count,
        accepted,
    )
    if not accepted and search.global_step >= args.timesteps:
        raise RuntimeError(
            f"JERK exhausted {args.timesteps} transitions without a {args.level} success event"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
