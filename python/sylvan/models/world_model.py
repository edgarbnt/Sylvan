"""Top-level Phase 1 world model."""

from __future__ import annotations

import torch
from torch import nn

from .encoders import ProprioEncoder
from .heads import DoneHead, MetricsPredictionHead, RewardHead, ProprioPredictionHead
from .rssm import SimpleRSSM


class WorldModelV0(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        proprio_dim: int,
        action_dim: int,
        metrics_dim: int,
        hidden_dim: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        # obs_dim = proprio ++ vision(food radar) ++ energy(1). The encoder ingests the FULL
        # observation so the latent state "knows" where food is and how hungry the agent is —
        # the prerequisite for the planner to navigate to food by minimising predicted hunger.
        self.obs_dim = obs_dim
        self.proprio_dim = proprio_dim
        self.encoder = ProprioEncoder(obs_dim, hidden_dim, latent_dim)
        self.rssm = SimpleRSSM(latent_dim, action_dim, hidden_dim)
        self.reward_head = RewardHead(latent_dim)
        self.done_head = DoneHead(latent_dim)
        # Predicts the FULL next observation (proprio ++ vision ++ energy), not just proprio,
        # so the dream rollout carries food/energy dynamics and the planner can read predicted
        # energy off it. (ProprioPredictionHead is a generic latent->vec MLP.)
        self.obs_head = ProprioPredictionHead(latent_dim, obs_dim)
        self.encoded_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.metrics_head = MetricsPredictionHead(latent_dim, action_dim, metrics_dim)

    def forward(
        self,
        obs: torch.Tensor | None,
        actions: torch.Tensor,
        *,
        initial_hidden: torch.Tensor | None = None,
        encoded_obs: torch.Tensor | None = None,
        scheduled_sampling_prob: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        if encoded_obs is None:
            if obs is None:
                raise ValueError("Must provide either obs or encoded_obs")
            encoded_obs = self.encoder(obs)
        latents, hidden = self.rssm(
            encoded_obs,
            actions,
            initial_hidden=initial_hidden,
            encoded_predictor=self.encoded_predictor,
            scheduled_sampling_prob=scheduled_sampling_prob,
        )
        predicted_next_obs = self.obs_head(latents)
        return {
            "latents": latents,
            "hidden": hidden,
            "predicted_reward": self.reward_head(latents),
            "predicted_done_logits": self.done_head(latents),
            "predicted_next_encoded": self.encoded_predictor(latents),
            "predicted_next_obs": predicted_next_obs,
            # Compat slice for eval/report code that reads the proprio prediction directly.
            "predicted_next_proprio": predicted_next_obs[..., : self.proprio_dim],
            "predicted_next_metrics": self.metrics_head(latents, actions),
        }
