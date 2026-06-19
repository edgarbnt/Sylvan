"""SAC networks: a squashed-Gaussian actor and twin Q-critics.

The actor mirrors the locomotion `Actor` backbone (two tanh layers) but adds a
state-dependent log_std head and a tanh squash with the change-of-variables log-prob
correction (Haarnoja et al. 2018). The squashed action lives in (-1, 1) — the SAME range
the Godot servo expects, so the served action needs no rescaling and matches the stored
action exactly (off-policy correctness).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal

LOG_STD_FLOOR = math.log(0.02)
# Ceil kept MODEST: the default SAC ceil (std 2.0) let the entropy term blow the exploration
# std up to ~1.2 → near-random sampled actions → noisy collection that looked like a gait
# collapse (while the deterministic mean was fine). 0.45 keeps exploration a moderate jitter
# around the warm-started gait — enough to DISCOVER a qualitatively different turn (lateral
# foot placement) without the near-random collection the default ceil (std 2.0) caused.
LOG_STD_CEIL = math.log(0.45)
# Squash correction adds -sum log(1 - tanh(u)^2); clamp the input to keep it finite.
_EPS = 1e-6


class SacActor(nn.Module):
    def __init__(self, *, obs_dim: int, hidden_dim: int = 128, action_dim: int = 12) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc1(obs))
        return torch.tanh(self.fc2(h))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self._features(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_FLOOR, LOG_STD_CEIL)
        return mean, log_std

    def sample(
        self,
        obs: torch.Tensor,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterised squashed sample. Returns (action in (-1,1), log_prob[N])."""
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        if deterministic:
            action = torch.tanh(mean)
            # log_prob is unused on the deterministic path; return a finite placeholder.
            return action, torch.zeros(action.shape[0], device=action.device)
        eps = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
        u = mean + std * eps
        action = torch.tanh(u)
        # log N(u) - sum log(1 - tanh(u)^2)
        log_prob = Normal(mean, std).log_prob(u).sum(-1)
        log_prob = log_prob - torch.log(1.0 - action.pow(2) + _EPS).sum(-1)
        return action, log_prob

    def log_prob_of(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Log-prob of a STORED squashed action (for diagnostics / importance checks)."""
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        a = action.clamp(-1.0 + _EPS, 1.0 - _EPS)
        u = torch.atanh(a)
        log_prob = Normal(mean, std).log_prob(u).sum(-1)
        log_prob = log_prob - torch.log(1.0 - a.pow(2) + _EPS).sum(-1)
        return log_prob


class _QNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


class TwinQ(nn.Module):
    """Two independent Q-nets; SAC uses their min to fight overestimation bias."""

    def __init__(self, *, obs_dim: int, hidden_dim: int = 128, action_dim: int = 12) -> None:
        super().__init__()
        self.q1 = _QNet(obs_dim, action_dim, hidden_dim)
        self.q2 = _QNet(obs_dim, action_dim, hidden_dim)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs, action), self.q2(obs, action)

    def q_min(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)
