"""Replay buffer reader with strict integrity checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .schema import Transition, validate_episode_contiguity
from ..constants import EPISODE_FILE_SUFFIX


def load_episode(path: Path) -> list[Transition]:
    transitions: list[Transition] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                transitions.append(Transition.from_dict(json.loads(line)))
            except Exception as exc:
                # A single corrupt line is almost always a TRUNCATED TAIL — the last
                # transition was only partially flushed when the Godot worker exited
                # (observed repeatedly: w0/episode_NNNN.jsonl cut off mid-JSON). Raising
                # here crashed the ENTIRE training run, discarding every iteration since
                # the last checkpoint (cost us multiple long runs). The JSONL is written
                # append-only and in order, so the valid PREFIX is intact and contiguous:
                # keep it, drop the rest of this episode, and carry on. Logged, never
                # silent (blueprint §9.5: no silent truncation).
                print(
                    f"[buffer] WARNING: corrupt line {path}:{line_number} "
                    f"({type(exc).__name__}: {exc}) — keeping the {len(transitions)} "
                    f"valid transitions before it, dropping the rest of this episode.",
                    file=sys.stderr,
                    flush=True,
                )
                break
    validate_episode_contiguity(transitions)
    return transitions


def iter_episodes(run_dir: Path) -> list[list[Transition]]:
    # RECURSIVE (rglob): a flat run_dir (single Godot) still works, AND parallel
    # collection that writes each worker's episodes into run_dir/wK/ subdirs is read
    # transparently. Episode files are named episode_NNNN.jsonl per worker dir, so
    # there is no cross-worker name collision (distinct directories).
    episodes: list[list[Transition]] = []
    for path in sorted(run_dir.rglob(f"*{EPISODE_FILE_SUFFIX}")):
        episodes.append(load_episode(path))
    return episodes
