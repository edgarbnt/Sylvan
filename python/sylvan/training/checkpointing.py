"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(
    *,
    destination: Path,
    model,
    optimizer,
    epoch: int,
    metrics: dict[str, float],
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        },
        destination,
    )
    return destination


def _grow_input_layers(state: dict, model) -> dict:
    """Zero-pad the INPUT columns of the first Linear layers when a checkpoint was
    trained on a smaller proprio vector than the current model (e.g. 120 -> 122 after
    appending the gait phase clock). The new observation dims are APPENDED last, so the
    old weights map onto the leading columns and the padded columns start at 0 → the
    warm-started policy behaves identically until it learns to use the new inputs. No-op
    when the dims already match (backward compatible)."""
    model_state = model.state_dict()
    grown = dict(state)
    for key, w_new in model_state.items():
        # Only the leading Linear weights take the observation as input columns.
        if not key.endswith("weight") or w_new.dim() != 2:
            continue
        if key not in grown:
            continue
        w_old = grown[key]
        if w_old.shape == w_new.shape:
            continue
        # Same outputs, fewer input columns → pad the extra columns with zeros.
        if w_old.shape[0] == w_new.shape[0] and w_old.shape[1] < w_new.shape[1]:
            padded = torch.zeros_like(w_new)
            padded[:, : w_old.shape[1]] = w_old
            grown[key] = padded
    return grown


def load_checkpoint(path: Path, model, optimizer=None) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = _grow_input_layers(payload["model_state_dict"], model)
    model.load_state_dict(state)
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload
