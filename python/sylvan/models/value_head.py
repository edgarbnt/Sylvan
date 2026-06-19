"""Tête de VALEUR apprise sur le latent du WM (🅑-pur, 2026-06-19).

Le planner JEPA-pur note des états LATENTS via cette tête APPRISE (jamais de coordonnées). Cible = « va manger
sous K pas » (valeur de proximité-au-repas) — signal NET, ≠ l'énergie-readout mou qui a fait échouer le coût
pur-énergie. V(latent) ∈ [0,1] : haut = état proche d'un repas. Le planner maximise V sur le rollout.

Mesuré (diag_eat_value_probe.py) : latent gelé teacher-forced → AUC 0.78 (vs énergie 0.67) → le signal EST dans
le latent sans ré-entraîner le WM. mu/sd stockés en buffers → chargement autonome.
"""
from __future__ import annotations

import torch
from torch import nn


class ValueHead(nn.Module):
    def __init__(self, latent_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("mu", torch.zeros(latent_dim))
        self.register_buffer("sd", torch.ones(latent_dim))

    def logit(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net((latent - self.mu) / self.sd).squeeze(-1)

    def value(self, latent: torch.Tensor) -> torch.Tensor:
        """V ∈ [0,1] (proba 'repas imminent') — la quantité que le planner MAXIMISE sur le rollout."""
        return torch.sigmoid(self.logit(latent))


def load_value_head(path, map_location="cpu") -> ValueHead:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    head = ValueHead(ck["latent_dim"], ck.get("hidden", 256))
    head.load_state_dict(ck["state_dict"])
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head
