from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from supermariobrosnes_turbo.jerk import (
    JerkPolicy,
    JerkSearch,
    JerkSequenceMinimizer,
    RetainedSequence,
    load_jerk_checkpoint,
    save_jerk_checkpoint,
)
from train import episode_boundary, exploit_probability


ACTIONS = ("noop", "right", "right_b", "right_a", "right_a_b", "a", "left")


def _search(
    *,
    seed: int = 7,
    n_envs: int = 1,
    initial_probability: float = 0.25,
    max_probability: float = 0.9,
    protected_prefix_steps: int = 128,
    max_prefix_shorten_steps: int = 128,
    retained_limit: int = 8,
) -> JerkSearch:
    return JerkSearch(
        n_envs=n_envs,
        seed=seed,
        total_timesteps=10,
        action_names=ACTIONS,
        fallback_action="noop",
        archive_replay_probability_initial=initial_probability,
        archive_replay_probability_max=max_probability,
        protected_prefix_steps=protected_prefix_steps,
        max_prefix_shorten_steps=max_prefix_shorten_steps,
        retained_limit=retained_limit,
    )


def _candidate(
    actions: tuple[int, ...],
    mean_return: float,
    *,
    completed: bool = False,
    progress: float = 0.0,
) -> RetainedSequence:
    return RetainedSequence(
        actions=actions,
        return_sum=mean_return,
        return_count=1,
        completed=completed,
        progress=progress,
    )


def test_level_change_does_not_end_episode() -> None:
    boundary = episode_boundary(
        previous_lives=2,
        current_lives=2,
        previous_level=(0, 0),
        current_level=(0, 1),
        episode_steps=1_000,
        max_episode_steps=4_500,
    )

    assert boundary.level_changed
    assert not boundary.life_loss
    assert not boundary.done


def test_life_loss_and_stall_end_failed_attempts() -> None:
    life_loss = episode_boundary(
        previous_lives=2,
        current_lives=1,
        previous_level=(0, 0),
        current_level=(0, 0),
        episode_steps=100,
        max_episode_steps=4_500,
    )
    stalled = episode_boundary(
        previous_lives=2,
        current_lives=2,
        previous_level=(0, 0),
        current_level=(0, 0),
        episode_steps=300,
        max_episode_steps=4_500,
        stalled=True,
    )

    assert life_loss.done and life_loss.life_loss
    assert stalled.done and stalled.stalled


def test_jerk_search_retains_successful_full_sequence() -> None:
    search = _search()
    first = search.next_actions()
    search.observe([1.0], [False])
    second = search.next_actions()
    record = SimpleNamespace(completed=True, progress=3161.0)
    search.observe([2.0], [True], {0: record})

    candidate = search.best_candidate()
    assert candidate is not None
    assert candidate.completed is True
    assert candidate.progress == 3161
    assert candidate.actions == (int(first[0]), int(second[0]))
    assert candidate.mean_return == 3.0


def test_jerk_uniform_sampling_is_seeded_and_covers_action_set() -> None:
    first = _search(initial_probability=0.0, max_probability=0.0)
    second = _search(initial_probability=0.0, max_probability=0.0)

    first_actions = [int(first.next_actions()[0]) for _ in range(256)]
    second_actions = [int(second.next_actions()[0]) for _ in range(256)]

    assert first_actions == second_actions
    assert set(first_actions) == set(range(len(ACTIONS)))


def test_jerk_replays_short_prefix_then_samples_uniform_suffix() -> None:
    search = _search(initial_probability=1.0, max_probability=1.0)
    prefix = (1, 2, 3)
    search._retained[prefix] = _candidate(prefix, 1.0, progress=100.0)
    search._start_lane(0)

    replayed = [int(search.next_actions()[0]) for _ in prefix]
    sampled_suffix = int(search.next_actions()[0])

    assert replayed == list(prefix)
    assert sampled_suffix in range(len(ACTIONS))
    assert search._lanes[0].mode == "explore"


def test_jerk_long_retained_path_shortens_within_protected_bounds() -> None:
    search = _search(
        initial_probability=1.0,
        max_probability=1.0,
        protected_prefix_steps=128,
        max_prefix_shorten_steps=128,
    )
    path = tuple(index % len(ACTIONS) for index in range(300))
    search._retained[path] = _candidate(path, 1.0, progress=100.0)
    search._start_lane(0)

    replay_limit = search._lanes[0].replay_limit
    assert 172 <= replay_limit <= 299
    assert replay_limit >= search.protected_prefix_steps
    assert len(path) - replay_limit <= search.max_prefix_shorten_steps


def test_jerk_archive_distribution_is_return_weighted() -> None:
    search = _search()
    search._retained[(2,)] = _candidate((2,), 5.0)
    search._retained[(0,)] = _candidate((0,), -5.0)
    search._retained[(1,)] = _candidate((1,), 0.0)

    candidates, probabilities = search._retained_distribution()

    assert [candidate.actions for candidate in candidates] == [(0,), (1,), (2,)]
    assert np.all(probabilities > 0.0)
    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities[1] == pytest.approx(1.0 / 3.0)
    assert probabilities[2] == pytest.approx(2.0 / 3.0)


def test_jerk_duplicate_updates_stats_and_unique_insert_evicts_worst() -> None:
    search = _search(retained_limit=2)
    search._upsert_retained((0,), score_return=1.0, completed=False, progress=10.0)
    search._upsert_retained((1,), score_return=2.0, completed=False, progress=20.0)
    search._upsert_retained((1,), score_return=4.0, completed=True, progress=20.0)
    search._upsert_retained((2,), score_return=3.0, completed=False, progress=30.0)

    assert search.retained_count == 2
    assert set(search._retained) == {(1,), (2,)}
    assert search._retained[(1,)].mean_return == 3.0
    assert search._retained[(1,)].completed is True


def test_completed_candidates_rank_shortest_before_reward_or_progress() -> None:
    short = _candidate((1, 2), 1.0, completed=True, progress=10.0)
    long = _candidate((1, 2, 3), 1000.0, completed=True, progress=9999.0)

    assert short.rank > long.rank


def test_incomplete_candidates_still_rank_progress_first() -> None:
    farther = _candidate((1, 2, 3), 1.0, progress=100.0)
    richer = _candidate((1,), 1000.0, progress=99.0)

    assert farther.rank > richer.rank


def test_sequence_minimizer_accepts_only_shorter_completed_mutations() -> None:
    initial = _candidate(tuple(range(10)), 10.0, completed=True, progress=100.0)
    minimizer = JerkSequenceMinimizer(
        initial=initial,
        n_envs=4,
        seed=7,
        max_chunk_steps=4,
        patience=2,
    )
    mutations = minimizer.propose()

    assert len(mutations) == 4
    assert all(len(mutation.actions) == 6 for mutation in mutations)
    failed = _candidate(mutations[0].actions, 100.0, completed=False, progress=999.0)
    successful = _candidate(mutations[1].actions, 1.0, completed=True, progress=100.0)

    assert minimizer.observe([failed, successful])
    assert minimizer.incumbent.actions == successful.actions
    assert minimizer.improvements == 1


def test_sequence_minimizer_reduces_chunk_after_misses() -> None:
    initial = _candidate(tuple(range(10)), 10.0, completed=True, progress=100.0)
    minimizer = JerkSequenceMinimizer(
        initial=initial,
        n_envs=2,
        seed=7,
        max_chunk_steps=4,
        patience=2,
    )

    minimizer.propose()
    assert not minimizer.observe([])
    minimizer.propose()
    assert not minimizer.observe([])
    assert minimizer.chunk_steps == 2


def test_jerk_policy_zip_round_trip_and_lane_resets(tmp_path) -> None:
    path = tmp_path / "model.zip"
    JerkPolicy(
        action_names=ACTIONS,
        action_sequence=(2, 4),
        fallback_action=0,
    ).save(path)
    loaded = JerkPolicy.load(path)

    obs = np.zeros((2, 1), dtype=np.float32)
    assert loaded.predict(obs)[0].tolist() == [2, 2]
    assert loaded.predict(obs)[0].tolist() == [4, 4]
    loaded.reset_lanes([True, False])
    assert loaded.predict(obs)[0].tolist() == [2, 0]


def test_legacy_json_checkpoint_stays_loadable(tmp_path) -> None:
    path = save_jerk_checkpoint(
        tmp_path / "policy.json",
        ("right", "right_a"),
        timesteps=123,
        episodes=4,
        best_reward=99.0,
        metadata={"terminate_on_level_change": False},
    )

    policy = load_jerk_checkpoint(path)
    observations = np.zeros((1, 1, 1, 1), dtype=np.uint8)
    actions = [int(policy.predict(observations)[0][0]) for _ in range(3)]

    assert actions == [1, 3, 0]
    assert policy.timesteps == 123
    assert policy.metadata["terminate_on_level_change"] is False


def test_exploit_probability_matches_rlab_schedule() -> None:
    assert exploit_probability(0, 1_000) == pytest.approx(0.25)
    assert exploit_probability(500, 1_000) == pytest.approx(0.75)
    assert exploit_probability(1_000, 1_000) == pytest.approx(0.9)
