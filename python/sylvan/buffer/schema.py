"""Strict transition schema shared by Godot and Python."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..constants import (
    DEFAULT_ACTION_DIM,
    DEFAULT_PROPRIO_DIM,
    LOCOMOTION_METRIC_KEYS,
    SCHEMA_VERSION,
)


def _ensure_float_list(values: list[Any], field_name: str) -> list[float]:
    if not isinstance(values, list):
        raise TypeError(f"{field_name} must be a list")
    return [float(value) for value in values]


def _ensure_float_mapping(
    values: dict[str, Any] | None,
    field_name: str,
) -> dict[str, float]:
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise TypeError(f"{field_name} must be a dict")
    return {str(key): float(value) for key, value in values.items()}


@dataclass(slots=True)
class Observation:
    proprio: list[float]
    vision: list[float]
    energy: float
    health: float
    metrics: dict[str, float] = field(default_factory=dict)

    def validate(self, *, proprio_dim: int = DEFAULT_PROPRIO_DIM) -> None:
        self.proprio = _ensure_float_list(self.proprio, "proprio")
        self.vision = _ensure_float_list(self.vision, "vision")
        self.energy = float(self.energy)
        self.health = float(self.health)
        self.metrics = _ensure_float_mapping(self.metrics, "metrics")
        self.metrics = {
            key: float(self.metrics.get(key, 0.0)) for key in LOCOMOTION_METRIC_KEYS
        }
        if len(self.proprio) != proprio_dim:
            raise ValueError(
                f"Expected proprio_dim={proprio_dim}, got {len(self.proprio)}"
            )


@dataclass(slots=True)
class TransitionInfo:
    episode_id: str
    step_id: int
    seed: int
    scene_version: str
    agent_version: str
    timestamp: str
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id is required")
        if self.step_id < 0:
            raise ValueError("step_id must be >= 0")
        if not self.timestamp:
            raise ValueError("timestamp is required")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version={self.schema_version}, expected {SCHEMA_VERSION}"
            )


@dataclass(slots=True)
class Transition:
    obs: Observation
    action: list[float]
    reward: float
    next_obs: Observation
    done: bool
    truncated: bool
    info: TransitionInfo

    def validate(
        self,
        *,
        proprio_dim: int = DEFAULT_PROPRIO_DIM,
        action_dim: int = DEFAULT_ACTION_DIM,
    ) -> None:
        self.obs.validate(proprio_dim=proprio_dim)
        self.next_obs.validate(proprio_dim=proprio_dim)
        self.action = _ensure_float_list(self.action, "action")
        if len(self.action) != action_dim:
            raise ValueError(f"Expected action_dim={action_dim}, got {len(self.action)}")
        self.reward = float(self.reward)
        self.done = bool(self.done)
        self.truncated = bool(self.truncated)
        self.info.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Transition":
        transition = cls(
            obs=Observation(
                proprio=payload["obs"]["proprio"],
                vision=payload["obs"].get("vision", []),
                energy=payload["obs"]["energy"],
                health=payload["obs"]["health"],
                metrics=payload["obs"].get("metrics", {}),
            ),
            action=list(payload["action"]),
            reward=float(payload["reward"]),
            next_obs=Observation(
                proprio=payload["next_obs"]["proprio"],
                vision=payload["next_obs"].get("vision", []),
                energy=payload["next_obs"]["energy"],
                health=payload["next_obs"]["health"],
                metrics=payload["next_obs"].get("metrics", {}),
            ),
            done=bool(payload["done"]),
            truncated=bool(payload["truncated"]),
            info=TransitionInfo(**payload["info"]),
        )
        transition.validate()
        return transition


def validate_episode_contiguity(transitions: list[Transition]) -> None:
    expected_step_id = 0
    episode_id: str | None = None
    for transition in transitions:
        transition.validate()
        if episode_id is None:
            episode_id = transition.info.episode_id
        elif transition.info.episode_id != episode_id:
            raise ValueError("Mixed episode_id values in a single episode shard")
        if transition.info.step_id != expected_step_id:
            raise ValueError(
                f"Non contiguous step_id in episode {episode_id}: "
                f"expected {expected_step_id}, got {transition.info.step_id}"
            )
        expected_step_id += 1
