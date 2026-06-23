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


class OrientHead(nn.Module):
    """Tête d'ORIENTATION apprise sur le latent du WM (🅑-pur, 2026-06-21) — l'analogue LATENT du terme de cap.

    Lit le BEARING ÉGOCENTRIQUE de la cible depuis le latent (où est la ressource perçue PAR RAPPORT À MOI),
    sortie (cos, sin) du bearing. Ce n'est PAS une coordonnée-monde : c'est de la perception (la rétine est
    égocentrique). `ahead(latent)` ∈ [-1,1] = cos(bearing) = à quel point la cible est DEVANT. Le planner s'en
    sert pour récompenser « s'orienter vers la cible » quand elle n'est pas atteignable dans le rêve (combler le
    trou de credit-assignment que la value-de-proximité seule ne couvre pas pour les cibles arrière). Readability
    mesurée : devant/derrière 84% sur latent gelé → le signal EST dans le latent. mu/sd en buffers (autonome)."""

    def __init__(self, latent_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),                # (cos, sin) du bearing
        )
        self.register_buffer("mu", torch.zeros(latent_dim))
        self.register_buffer("sd", torch.ones(latent_dim))

    def cos_sin(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net((latent - self.mu) / self.sd)

    def ahead(self, latent: torch.Tensor) -> torch.Tensor:
        """∈ [-1,1] ≈ cos(bearing) : +1 = cible droit devant, -1 = droit derrière. Le 'cap' que le planner monte."""
        cs = self.cos_sin(latent)
        return cs[..., 0] / (cs.norm(dim=-1) + 1e-6)


def load_orient_head(path, map_location="cpu") -> OrientHead:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    head = OrientHead(ck["latent_dim"], ck.get("hidden", 256))
    head.load_state_dict(ck["state_dict"])
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head
