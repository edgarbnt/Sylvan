"""Top-level controller container."""

from __future__ import annotations

import torch
from torch import nn

from .actor import Actor
from .critic import Critic


class LocomotionController(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.actor = Actor(input_dim, hidden_dim, action_dim)
        self.critic = Critic(input_dim, hidden_dim)

    def act(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor(state)

    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state)
