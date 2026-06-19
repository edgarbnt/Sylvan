"""Phase 2 locomotion report generation."""

from __future__ import annotations

from pathlib import Path

from .metrics import summarize_run
from .prediction_report import write_prediction_report


def write_locomotion_report(run_dir: Path, destination: Path) -> Path:
    payload = {"run_dir": str(run_dir), "summary": summarize_run(run_dir)}
    return write_prediction_report(destination, payload)
