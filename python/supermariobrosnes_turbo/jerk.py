from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .env import ACTION_SETS


CHECKPOINT_FORMAT = "jerk-v1"


@dataclass
class JerkPolicy:
    """Observation-free action sequence retained by JERK training."""

    action_sequence: tuple[str, ...]
    action_set: str = "simple"
    timesteps: int = 0
    episodes: int = 0
    best_reward: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    _cursor: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.action_set not in ACTION_SETS:
            raise ValueError(f"unknown action set {self.action_set!r}")
        valid_actions = set(ACTION_SETS[self.action_set])
        unknown = sorted(set(self.action_sequence) - valid_actions)
        if unknown:
            raise ValueError(
                f"checkpoint actions {unknown!r} are not in action_set={self.action_set!r}"
            )

    @property
    def action_count(self) -> int:
        return len(ACTION_SETS[self.action_set])

    def reset(self) -> None:
        self._cursor = 0

    def action_at(self, step: int) -> str:
        if 0 <= step < len(self.action_sequence):
            return self.action_sequence[step]
        return "noop"

    def predict(
        self,
        observations: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
        del deterministic
        action_name = self.action_at(self._cursor)
        self._cursor += 1
        action_id = ACTION_SETS[self.action_set].index(action_name)
        batch_size = int(np.asarray(observations).shape[0])
        return np.full(batch_size, action_id, dtype=np.int64), None


def save_jerk_checkpoint(
    path: str | Path,
    action_sequence: Sequence[str],
    *,
    timesteps: int,
    episodes: int,
    best_reward: float,
    action_set: str = "simple",
    metadata: dict[str, Any] | None = None,
) -> Path:
    policy = JerkPolicy(
        tuple(action_sequence),
        action_set=action_set,
        timesteps=int(timesteps),
        episodes=int(episodes),
        best_reward=float(best_reward),
        metadata=dict(metadata or {}),
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "format": CHECKPOINT_FORMAT,
                "action_set": policy.action_set,
                "action_sequence": list(policy.action_sequence),
                "timesteps": policy.timesteps,
                "episodes": policy.episodes,
                "best_reward": policy.best_reward,
                "metadata": policy.metadata,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def load_jerk_checkpoint(path: str | Path) -> JerkPolicy:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{source} is not a readable JERK checkpoint") from exc
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{source} is not a {CHECKPOINT_FORMAT} checkpoint")
    sequence = payload.get("action_sequence")
    if not isinstance(sequence, list) or not all(isinstance(action, str) for action in sequence):
        raise ValueError(f"{source} has an invalid JERK action sequence")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{source} has invalid JERK metadata")
    return JerkPolicy(
        tuple(sequence),
        action_set=str(payload.get("action_set", "simple")),
        timesteps=int(payload.get("timesteps", 0)),
        episodes=int(payload.get("episodes", 0)),
        best_reward=float(payload.get("best_reward", 0.0)),
        metadata=metadata,
    )
