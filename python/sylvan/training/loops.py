"""Train and validation loops."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from .losses import compute_world_model_losses
from ..models.obs_assembly import assemble_wm_obs
from ..constants import DEFAULT_MAX_ENERGY

# Rare eat-steps (energy jump) are importance-weighted in the energy loss so the WM
# learns the food→energy-up dynamics the planner needs, not just the constant decay.
EAT_DELTA_THRESHOLD = 5.0 / DEFAULT_MAX_ENERGY   # normalised ΔE marking an eat
EAT_SAMPLE_WEIGHT = 30.0                         # weight multiplier on eat-steps (100 over-corrected:
                                                 # it predicted energy-up everywhere; 30 keeps normal≈0
                                                 # while still learning the eat-jump → sharper discrimination)


def _batch_obs(batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor]:
    """Assemble (obs, next_obs) = [proprio ++ vision ++ energy] on `device`, the proprio/
    vision dims the loss slices, and a per-step energy importance weight (eat-steps upweighted)."""
    obs = assemble_wm_obs(
        batch.proprio.to(device), batch.vision.to(device), batch.energy.to(device)
    )
    next_obs = assemble_wm_obs(
        batch.next_proprio.to(device), batch.next_vision.to(device), batch.next_energy.to(device)
    )
    eat_mask = (next_obs[..., -1] - obs[..., -1]) > EAT_DELTA_THRESHOLD
    energy_weight = 1.0 + EAT_SAMPLE_WEIGHT * eat_mask.float()
    return obs, next_obs, batch.proprio.shape[-1], batch.vision.shape[-1], energy_weight


def _reduce_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {
            "loss": 0.0,
            "latent_loss": 0.0,
            "proprio_loss": 0.0,
            "vision_loss": 0.0,
            "energy_loss": 0.0,
            "metrics_loss": 0.0,
            "reward_loss": 0.0,
            "done_loss": 0.0,
        }
    keys = metrics[0].keys()
    return {
        key: sum(metric[key] for metric in metrics) / len(metrics)
        for key in keys
    }


def train_epoch(model, dataloader: Iterable, optimizer, device: torch.device,
                *, loss_kwargs: dict | None = None) -> dict[str, float]:
    loss_kwargs = loss_kwargs or {}
    model.train()
    epoch_metrics: list[dict[str, float]] = []
    batch_count = 0
    for batch in dataloader:
        if batch_count >= 200:  # Allow enough batches for the World Model to converge properly
            break
        optimizer.zero_grad(set_to_none=True)
        obs, next_obs, proprio_dim, vision_dim, energy_weight = _batch_obs(batch, device)
        outputs = model(
            obs,
            batch.action.to(device),
            # Fraction of steps fed the REAL observation (teacher forcing).
            # Lowered 0.9->0.5: the controller imagines 100% open-loop for 30
            # steps, but at 0.9 TF the WM barely learned to predict from its OWN
            # rollouts, so in imagination it regressed to a "centred" world and
            # never showed the COM drift (J1: imagined 48 vs real 36 at H=30,
            # reward MAE 0.40). More free-running here trains the WM the way
            # imagination actually uses it, so drift appears in the dream.
            scheduled_sampling_prob=0.5,
        )
        losses = compute_world_model_losses(
            outputs,
            next_obs=next_obs,
            next_metrics=batch.next_metrics.to(device),
            reward=batch.reward.to(device),
            done=batch.done.to(device),
            model=model,
            proprio_dim=proprio_dim,
            vision_dim=vision_dim,
            energy_sample_weight=energy_weight,
            **loss_kwargs,
        )
        losses["loss"].backward()
        optimizer.step()
        epoch_metrics.append({key: float(value.detach().cpu()) for key, value in losses.items()})
        batch_count += 1
    return _reduce_metrics(epoch_metrics)


@torch.no_grad()
def validate_epoch(model, dataloader: Iterable, device: torch.device,
                   *, loss_kwargs: dict | None = None) -> dict[str, float]:
    loss_kwargs = loss_kwargs or {}
    model.eval()
    epoch_metrics: list[dict[str, float]] = []
    batch_count = 0
    for batch in dataloader:
        if batch_count >= 50:  # Validation limit
            break
        obs, next_obs, proprio_dim, vision_dim, energy_weight = _batch_obs(batch, device)
        outputs = model(
            obs,
            batch.action.to(device),
        )
        losses = compute_world_model_losses(
            outputs,
            next_obs=next_obs,
            next_metrics=batch.next_metrics.to(device),
            reward=batch.reward.to(device),
            done=batch.done.to(device),
            model=model,
            proprio_dim=proprio_dim,
            vision_dim=vision_dim,
            energy_sample_weight=energy_weight,
            **loss_kwargs,
        )
        epoch_metrics.append({key: float(value.detach().cpu()) for key, value in losses.items()})
        batch_count += 1
    return _reduce_metrics(epoch_metrics)
