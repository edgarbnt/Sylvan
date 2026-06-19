"""Helpers to build replay windows across multiple day runs."""

from __future__ import annotations

from pathlib import Path


def _is_validation_run(path: Path) -> bool:
    return path.name.endswith("_validation")


def list_replay_runs(
    replay_buffer_dir: Path,
    *,
    include_validation_runs: bool = False,
) -> list[Path]:
    if not replay_buffer_dir.exists():
        return []
    runs = [
        path
        for path in replay_buffer_dir.iterdir()
        if path.is_dir() and (include_validation_runs or not _is_validation_run(path))
    ]
    runs.sort(key=lambda path: path.stat().st_mtime)
    return runs


def select_replay_window(
    replay_buffer_dir: Path,
    *,
    current_run_dir: Path,
    window_size: int,
    include_validation_runs: bool = False,
) -> list[Path]:
    if window_size <= 0:
        raise ValueError("window_size must be > 0")

    ordered_runs = list_replay_runs(
        replay_buffer_dir,
        include_validation_runs=include_validation_runs,
    )

    if current_run_dir.exists() and (
        include_validation_runs or not _is_validation_run(current_run_dir)
    ):
        already_present = any(path.resolve() == current_run_dir.resolve() for path in ordered_runs)
        if not already_present:
            ordered_runs.append(current_run_dir)

    if not ordered_runs:
        return []

    return ordered_runs[-window_size:]
