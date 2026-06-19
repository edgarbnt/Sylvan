"""A deliberately simple RSSM-style recurrent dynamics model.

Phase 1 starts with a deterministic recurrent latent state to keep the walking
skeletal stable. When a stochastic posterior is introduced later, detach
boundaries between rollout segments must remain explicit to avoid exploding
gradients and NaNs.

TODO Phase 3 ALife: Replace `to_latent` with a SparseMoE (defined in layers.py)
to scale to multiple physical regimes (contact / fall / recovery / gait).
Activation criteria: wm/proprio_loss plateaus above 0.01 once agent reliably
stands for 100+ steps AND buffer has 100k+ diverse transitions. Adding MoE
earlier hurts (overfit on tiny dataset, no load-balancing loss implemented).
"""

from __future__ import annotations

import torch
from torch import nn

from .layers import SparseMoE

class SimpleRSSM(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRUCell(latent_dim + action_dim, hidden_dim)
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(
        self,
        encoded_obs: torch.Tensor,
        actions: torch.Tensor,
        *,
        initial_hidden: torch.Tensor | None = None,
        detach_state: bool = False,
        encoded_predictor: nn.Module | None = None,
        scheduled_sampling_prob: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = encoded_obs.shape
        if initial_hidden is None:
            hidden = torch.zeros(
                batch_size,
                self.gru.hidden_size,
                device=encoded_obs.device,
                dtype=encoded_obs.dtype,
            )
        else:
            hidden = initial_hidden
            
        latents = []
        for step in range(sequence_length):
            if detach_state:
                hidden = hidden.detach()
            
            # Scheduled sampling: decide whether to use real observation or predicted observation
            if step > 0 and self.training and encoded_predictor is not None and scheduled_sampling_prob < 1.0:
                r = torch.rand((), device=encoded_obs.device)
                if r > scheduled_sampling_prob:
                    # Feed the predicted next encoded observation from the previous step
                    prev_hidden = latents[-1]
                    prev_latent = self.to_latent(prev_hidden)
                    obs_input = encoded_predictor(prev_latent)
                else:
                    obs_input = encoded_obs[:, step]
            else:
                obs_input = encoded_obs[:, step]

            gru_input = torch.cat((obs_input, actions[:, step]), dim=-1)
            hidden = self.gru(gru_input, hidden)
            latents.append(hidden)
        
        hidden_seq = torch.stack(latents, dim=1) # [B, T, H]
        latents_seq = self.to_latent(hidden_seq) # [B, T, L]
        return latents_seq, hidden
