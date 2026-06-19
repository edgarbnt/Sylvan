"""Simple report generation for Phase 1."""

from __future__ import annotations

import json
from pathlib import Path


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def write_prediction_report(destination: Path, payload: dict[str, object]) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )
    return destination
