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


class RetinaAttentionEncoder(nn.Module):
    """Encodeur qui lit la rétine RAYON PAR RAYON, puis agrège par ATTENTION.

    POURQUOI IL EXISTE — VERROU A1, tranché par la mesure (2026-07-25). Le `ProprioEncoder`
    ci-dessus est un MLP à UNE couche cachée sur l'observation ENTIÈRE aplatie. Mesuré, à tâche
    ISOLÉE (aucune perte WM en concurrence, même corpus, même largeur, même graine, même budget) :

        MLP dense (l'encodeur servi) ....  41,5 % de lecture du type  (104 320 paramètres)
        attention par rayon ............  99,0 %                      ( 92 545 paramètres)

    Avec MOINS de paramètres : ce n'est donc pas une histoire de capacité, mais de STRUCTURE. Lire
    « la couleur de la proie la plus proche » demande de tester chaque rayon PUIS d'en sélectionner
    un ; sur un vecteur aplati de 278 nombres, un MLP peu profond ne le fait pas. C'est ce qui
    explique d'un coup toutes les mesures précédentes : l'apparence illisible dans le latent (33 %
    contre 27 % de majorité) et une pression de décodage explicite qui ne rachète que +4 points
    (33,3 → 37,3 % à poids x10), parce que l'architecture plafonne vers 41 %.

    Le slot, lui, y arrive sur LA MÊME rétine — parce qu'il la lit par attention géométrique. On
    donne ici la même structure à l'encodeur, sans lui donner la réponse : les requêtes ne sont pas
    codées, c'est un score APPRIS par rayon.

    Contrat préservé : même signature, même largeur de sortie. La proprioception et l'énergie (tout
    ce qui n'est pas la rétine) traversent inchangées, concaténées au résumé rétinien.
    """

    def __init__(self, obs_dim: int, hidden_dim: int, latent_dim: int,
                 retina_at: int, n_rays: int = 36, ray_ch: int = 4, d_token: int = 64) -> None:
        super().__init__()
        self.retina_at = retina_at
        self.n_rays, self.ray_ch = n_rays, ray_ch
        self.retina_dim = n_rays * ray_ch
        n_other = obs_dim - self.retina_dim
        if n_other < 0:
            raise ValueError(f"obs_dim {obs_dim} < rétine {self.retina_dim}")
        self.token = nn.Sequential(nn.Linear(ray_ch, d_token), nn.SiLU(),
                                   nn.Linear(d_token, d_token))
        self.score = nn.Sequential(nn.Linear(d_token, d_token), nn.SiLU(),
                                   nn.Linear(d_token, 1))
        self.net = nn.Sequential(nn.Linear(d_token + n_other, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, latent_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        lead = obs[..., :self.retina_at]
        ret = obs[..., self.retina_at:self.retina_at + self.retina_dim]
        tail = obs[..., self.retina_at + self.retina_dim:]
        rays = ret.reshape(*ret.shape[:-1], self.n_rays, self.ray_ch)
        tok = self.token(rays)                                        # [..., 36, d]
        w = torch.softmax(self.score(tok).squeeze(-1), dim=-1)        # [..., 36]
        pooled = (tok * w.unsqueeze(-1)).sum(dim=-2)                  # [..., d]
        return self.net(torch.cat([lead, pooled, tail], dim=-1))


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

