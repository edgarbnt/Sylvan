"""Encoders for proprioception and optional vision."""

from __future__ import annotations

import torch
from torch import nn


class ProprioEncoder(nn.Module):
    def __init__(self, proprio_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        return self.net(proprio)


class VisualEncoder(nn.Module):
    """
    Placeholder for the Visual Encoder (CNN or ViT).
    Currently returns a zero tensor to maintain the V-M-C structure
    until vision is fully integrated.
    """
    def __init__(self, vision_shape: tuple[int, ...], latent_dim: int) -> None:
        super().__init__()
        self.vision_shape = vision_shape
        self.latent_dim = latent_dim
        # TODO: Implement CNN/ViT architecture here

    def forward(self, vision: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = vision.shape[:2]
        # Return empty latent vector for now
        return torch.zeros(batch_size, seq_len, self.latent_dim, device=vision.device, dtype=vision.dtype)

