"""Continuous actor used by the locomotion controller."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.fc1(state))
        hidden = torch.tanh(self.fc2(hidden))
        return torch.tanh(self.fc3(hidden))

    def export_json(self, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "layers": [
                {
                    "weight": self.fc1.weight.detach().cpu().tolist(),
                    "bias": self.fc1.bias.detach().cpu().tolist(),
                    "activation": "tanh",
                },
                {
                    "weight": self.fc2.weight.detach().cpu().tolist(),
                    "bias": self.fc2.bias.detach().cpu().tolist(),
                    "activation": "tanh",
                },
                {
                    "weight": self.fc3.weight.detach().cpu().tolist(),
                    "bias": self.fc3.bias.detach().cpu().tolist(),
                    "activation": "tanh",
                },
            ]
        }
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return destination
