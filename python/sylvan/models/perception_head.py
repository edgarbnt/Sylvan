"""Tête de perception APPRISE (🅐) — RÉTINE → position estimée de la ressource.

Remplace l'oracle géométrique `food_xz_from_radar`. Entrée = les rayons couleur BRUTS (depth+RGB par
rayon) ; sortie = la position (x_right, z_fwd) en frame agent de la ressource la plus proche + une présence.
C'est ICI que « rouge = bouffe / bleu = eau » doit ÉMERGER de l'apprentissage (la rétine ne pré-associe
rien). Entraînée hors-ligne (label = vraie position du simulateur) ; à l'inférence elle ne voit QUE la
rétine — l'oracle est débranché (honnêteté CLAUDE.md §2).

ARCHITECTURE = scoreur par-rayon partagé + attention (soft-argmax) sur les rayons.
  - Un MLP plat (rétine→xy) OVERFITTE : il doit ré-apprendre la trigo bearing×depth sans biais → mémorise.
  - Ici la GÉOMÉTRIE est câblée (chaque rayon k connaît son bearing) et on n'APPREND que « quel rayon est
    la ressource » (scoreur sur [depth,R,G,B,sinθ,cosθ]) → couleur→sens reste apprise, mais le décodage
    position est exact → généralise comme le décodeur analytique (~0.16 m) au lieu de mémoriser.
  - Multi-ressources : un canal de score par ressource (food=rouge, eau=bleu) — chacun apprend sa couleur.

Sortie (par ressource) : pos NORMALISÉE (÷ max_range) → locate() remet en mètres via pos_scale.
Convention (x_right, z_fwd) IDENTIQUE à food_xz_from_radar → branchement direct dans le planner.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

RETINA_DIM = 144  # 36 rayons × 4 (depth, R, G, B)
MAX_RANGE = 10.0  # = Perception.MAX_RANGE (normalisation depth)


class RetinaPerceptionHead(nn.Module):
    def __init__(self, retina_dim: int = RETINA_DIM, hidden: int = 64, n_resources: int = 1,
                 max_range: float = MAX_RANGE):
        super().__init__()
        self.retina_dim = retina_dim
        self.n_rays = retina_dim // 4
        self.n_resources = n_resources  # 1 = bouffe ; 2 = bouffe + eau
        self.max_range = max_range
        # bearing connu de chaque rayon (k → k·2π/N), rayon 0 = forward (cf perception.gd::retina)
        ang = torch.tensor([2.0 * math.pi * k / self.n_rays for k in range(self.n_rays)])
        self.register_buffer("ray_sin", torch.sin(ang))  # [n_rays]
        self.register_buffer("ray_cos", torch.cos(ang))
        # scoreur PARTAGÉ par rayon : [depth, R, G, B, sinθ, cosθ] → un score par ressource
        self.scorer = nn.Sequential(
            nn.Linear(6, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_resources),
        )
        # offset depth→distance (le rayon touche la SURFACE de la sphère collision, ~0.35 m = 0.035 normalisé)
        self.depth_offset = nn.Parameter(torch.tensor(0.035))
        self.register_buffer("pos_scale", torch.tensor(max_range))  # locate() : normalisé → mètres

    def forward(self, retina: torch.Tensor) -> dict[str, torch.Tensor]:
        lead = retina.shape[:-1]
        r = retina.view(*lead, self.n_rays, 4)
        depth = r[..., 0]                                   # [..., n_rays]
        sin = self.ray_sin.expand_as(depth)
        cos = self.ray_cos.expand_as(depth)
        feat = torch.stack([r[..., 0], r[..., 1], r[..., 2], r[..., 3], sin, cos], dim=-1)  # [...,n_rays,6]
        scores = self.scorer(feat)                          # [..., n_rays, n_res]
        miss = (depth >= 0.999).unsqueeze(-1)               # rayon sans hit → ignoré
        scores = scores.masked_fill(miss, -1e9)
        attn = torch.softmax(scores, dim=-2)                # soft-argmax sur les rayons, [...,n_rays,n_res]
        # position candidate (NORMALISÉE) de chaque rayon : D̂ = depth + offset ; (x,z) = D̂·(sinθ, cosθ)
        d_norm = depth + self.depth_offset                  # [..., n_rays]
        cand = torch.stack([d_norm * sin, d_norm * cos], dim=-1)  # [..., n_rays, 2]
        pos = torch.einsum("...kr,...kd->...rd", attn, cand)      # [..., n_res, 2] (normalisé)
        conf_logit = scores.max(dim=-2).values              # [..., n_res] : -1e9 si tout miss
        # scores [...,n_rays,n_res] exposés comme LOGITS d'attention (supervision "pointe la + proche").
        return {"pos": pos, "conf_logit": conf_logit, "conf": torch.sigmoid(conf_logit), "scores": scores}

    @torch.no_grad()
    def locate(self, retina: torch.Tensor, conf_thresh: float = 0.5) -> list[tuple[float, float] | None]:
        """Inférence façon oracle : renvoie [(x_right, z_fwd) | None] par ressource (None si conf basse)."""
        if retina.dim() == 1:
            retina = retina.unsqueeze(0)
        out = self.forward(retina)
        pos = out["pos"][0] * self.pos_scale   # [n_res, 2] en mètres
        conf = out["conf"][0]
        res: list[tuple[float, float] | None] = []
        for i in range(self.n_resources):
            res.append((float(pos[i, 0]), float(pos[i, 1])) if float(conf[i]) >= conf_thresh else None)
        return res
