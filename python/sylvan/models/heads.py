"""Prediction heads for the Phase 1 world model."""

from __future__ import annotations

import torch
from torch import nn


class RewardHead(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
        )

    def forward(self, latents):
        return self.network(latents).squeeze(-1)


class DoneHead(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
        )

    def forward(self, latents):
        return self.network(latents).squeeze(-1)


class ProprioPredictionHead(nn.Module):
    def __init__(self, latent_dim: int, proprio_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, proprio_dim),
        )

    def forward(self, latents):
        return self.network(latents)


class MetricsPredictionHead(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, metrics_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim + action_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, metrics_dim),
        )

    def forward(self, latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([latents, actions], dim=-1)
        return self.network(combined)
