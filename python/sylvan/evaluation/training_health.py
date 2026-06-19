"""Training-health signals — the "is the gradient flowing?" light (BLUEPRINT §9.1).

The controller's actor loss equals minus the imagined return, which sits near a
large constant (~-53 = the imagined upright-return cap). An absolute threshold is
therefore meaningless; what matters is whether it MOVES across epochs. A flat
actor loss = no balance gradient is flowing = the policy is frozen, the failure
mode that wasted days this project because nothing flagged it automatically.
"""

from __future__ import annotations

from collections.abc import Sequence

# A relative span below this over >= MIN_EPOCHS epochs means "not moving".
DEFAULT_REL_SPAN_THRESHOLD = 1e-3
MIN_EPOCHS = 5


def actor_frozen_signal(
    history: Sequence[dict],
    *,
    rel_span_threshold: float = DEFAULT_REL_SPAN_THRESHOLD,
    min_epochs: int = MIN_EPOCHS,
) -> dict[str, object]:
    """Detect a frozen actor from the per-epoch controller history.

    `history` is the list of {"epoch", "actor_loss", "critic_loss"} dicts returned
    by ControllerTrainer.train(). Uses the RELATIVE span of actor_loss across
    epochs (span / max|loss|) so the near-(-53) absolute value doesn't break it.
    """
    losses = [float(h["actor_loss"]) for h in history if "actor_loss" in h]
    if len(losses) < min_epochs:
        return {
            "actor_frozen": False,
            "reason": f"insufficient epochs ({len(losses)} < {min_epochs})",
            "n_epochs": len(losses),
        }
    lo, hi = min(losses), max(losses)
    scale = max(abs(lo), abs(hi), 1e-6)
    abs_span = hi - lo
    rel_span = abs_span / scale
    frozen = rel_span < rel_span_threshold
    return {
        "actor_frozen": bool(frozen),
        "rel_span": rel_span,
        "abs_span": abs_span,
        "first": losses[0],
        "last": losses[-1],
        "n_epochs": len(losses),
        "rel_span_threshold": rel_span_threshold,
        "reason": (
            "actor loss is flat across epochs — no balance gradient flowing"
            if frozen
            else "actor loss is moving"
        ),
    }
