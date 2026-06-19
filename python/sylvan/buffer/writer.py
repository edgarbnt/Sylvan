"""Replay buffer writer for Phase 1."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .schema import Transition, validate_episode_contiguity
from ..constants import EPISODE_FILE_SUFFIX, RUN_METADATA_FILENAME


class EpisodeWriter:
    """Writes one JSONL file per episode.

    JSONL is intentionally used in Phase 1 because it is readable and easy to
    inspect during debugging. The writer is encapsulated so the storage backend
    can later move to a binary format without changing the rest of the stack.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_run_metadata(self, metadata: dict[str, object]) -> Path:
        destination = self.run_dir / RUN_METADATA_FILENAME
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return destination

    def write_episode(self, episode_id: str, transitions: list[Transition]) -> Path:
        validate_episode_contiguity(transitions)
        destination = self.run_dir / f"{episode_id}{EPISODE_FILE_SUFFIX}"
        with destination.open("w", encoding="utf-8") as handle:
            for transition in transitions:
                handle.write(json.dumps(asdict(transition), separators=(",", ":")))
                handle.write("\n")
        return destination
