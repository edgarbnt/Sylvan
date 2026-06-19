"""Loss functions for the Phase 1 world model."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_world_model_losses(
    outputs: dict[str, torch.Tensor],
    *,
    next_obs: torch.Tensor,
    next_metrics: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    model: torch.nn.Module,
    proprio_dim: int,
    vision_dim: int,
    done_weight: float = 1.0,
    metrics_weight: float = 1.0,
    vision_weight: float = 5.0,
    energy_weight: float = 20.0,
    energy_sample_weight: torch.Tensor | None = None,
    done_pos_weight: float | None = None,
) -> dict[str, torch.Tensor]:
    """The WM now predicts the FULL next observation [proprio ++ vision ++ energy].

    The next-obs reconstruction is split into proprio / vision(food radar) / energy so
    each can be weighted: energy and the radar are 1 and 12 dims out of 107, so without
    up-weighting their gradient is swamped by the 94 proprio dims and the latent never
    learns the food/hunger dynamics the planner depends on. `energy_weight` is high because
    predicted energy IS the planner's intrinsic cost.

    Falls are rare, so `done_pos_weight` upweights the rare done=1 class in the BCE;
    `done_weight`/`metrics_weight` raise the pull on termination + height signals."""
    with torch.no_grad():
        encoded_target = model.encoder(next_obs)
    # 1. Prediction error in latent space (smooth transition)
    latent_loss = F.mse_loss(outputs["predicted_next_encoded"], encoded_target)
    # 2. Prediction error on the full physical+sensory+energy state (fixed target → no collapse)
    pred_obs = outputs["predicted_next_obs"]
    v_end = proprio_dim + vision_dim
    proprio_loss = F.mse_loss(pred_obs[..., :proprio_dim], next_obs[..., :proprio_dim])
    vision_loss = F.mse_loss(pred_obs[..., proprio_dim:v_end], next_obs[..., proprio_dim:v_end])
    # Energy: importance-weight the rare EAT steps (energy jumps) so the WM learns the
    # food→energy-up dynamics instead of only the ~constant decay that dominates the data.
    energy_se = (pred_obs[..., v_end:] - next_obs[..., v_end:]) ** 2
    if energy_sample_weight is not None:
        w = energy_sample_weight.unsqueeze(-1)
        energy_loss = (w * energy_se).sum() / w.sum().clamp_min(1e-6)
    else:
        energy_loss = energy_se.mean()

    metrics_loss = F.mse_loss(outputs["predicted_next_metrics"], next_metrics)
    reward_loss = F.mse_loss(outputs["predicted_reward"], reward)
    pos_weight = None
    if done_pos_weight is not None:
        pos_weight = torch.tensor(float(done_pos_weight), device=done.device, dtype=done.dtype)
    done_loss = F.binary_cross_entropy_with_logits(
        outputs["predicted_done_logits"], done, pos_weight=pos_weight
    )

    # Balance losses to avoid one component dominating the gradient
    total_loss = (
        latent_loss
        + (10.0 * proprio_loss)
        + (vision_weight * vision_loss)
        + (energy_weight * energy_loss)
        + (metrics_weight * metrics_loss)
        + reward_loss
        + (done_weight * done_loss)
    )
    return {
        "loss": total_loss,
        "latent_loss": latent_loss,
        "proprio_loss": proprio_loss,
        "vision_loss": vision_loss,
        "energy_loss": energy_loss,
        "metrics_loss": metrics_loss,
        "reward_loss": reward_loss,
        "done_loss": done_loss,
    }
