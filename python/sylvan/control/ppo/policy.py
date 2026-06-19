"""Gaussian stochastic actor-critic for continuous-action PPO (J0).

Reuses the existing `Actor` (its final tanh keeps the mean in (-1,1)) and `Critic`
MLPs, operating DIRECTLY on the proprioception (no world-model encoder). A
state-independent learnable `log_std` parameterises the Gaussian.

Squashing decision = "Option B": sample `a ~ Normal(mean, std)` then clamp to
[-1,1]; log-prob is the plain Normal density evaluated on the (clamped) STORED
action. The server's `sample` and the updater's `evaluate_actions` share the exact
same math, so the PPO ratio is exactly 1.0 at the start of each iteration. A
`log_std` floor + entropy bonus (in the updater) prevent exploration collapse.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal

from ..actor import Actor
from ..critic import Critic

# Keep std in a sane band: floor stops exploration collapse (the "frozen agent"
# failure mode); ceiling stops runaway noise early in training.
LOG_STD_FLOOR = math.log(0.05)
LOG_STD_CEIL = math.log(2.0)


class GaussianActorCritic(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int = 106,   # QUAD default (proprio 94 + vision 12); overridden by config.env.policy_input_dim
        hidden_dim: int = 128,
        action_dim: int = 12,  # QUAD default (12 DOF); overridden by config.env.action_dim
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()
        self.actor = Actor(obs_dim, hidden_dim, action_dim)   # mean, tanh-bounded to (-1,1)
        self.critic = Critic(obs_dim, hidden_dim)             # scalar value
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

    def _std(self) -> torch.Tensor:
        return self.log_std.clamp(LOG_STD_FLOOR, LOG_STD_CEIL).exp()

    def mean(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    @torch.no_grad()
    def sample(
        self, obs: torch.Tensor, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action and return (action, log_prob). Used by the server."""
        mean = self.actor(obs)
        std = self._std()
        eps = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
        action = (mean + std * eps).clamp(-1.0, 1.0)
        log_prob = Normal(mean, std).log_prob(action).sum(-1)
        return action, log_prob

    def evaluate_actions(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Log-prob, entropy, value of `action` under the CURRENT policy. Used by
        both the rollout (behavior snapshot → old_log_prob) and the PPO update."""
        mean = self.actor(obs)
        std = self._std()
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs)
        return log_prob, entropy, value

    def mean_std(self) -> float:
        return float(self._std().mean().item())
