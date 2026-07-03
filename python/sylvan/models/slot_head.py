"""Slot de perception OBJECT-CENTRIC, AUTO-SUPERVISÉ (chantier WM pur, 2026-06-23).

Remplace `retina_head` (qui était supervisé sur des LABELS-ORACLE de position) par un slot APPRIS SANS label :
l'encodeur extrait, par ATTENTION GÉOMÉTRIQUE (soft-argmax sur les rayons d'angle CONNU → coordonnée par
construction), la position ego de l'objet ; il est entraîné UNIQUEMENT par consistance de transport sous
l'ego-motion (équivariance) + VICReg (cf train_slot_head.py). Pré-check (diag_fpure1c) : bearing MAE 3.8° /
position 0.17 m = ÉGAL/MEILLEUR que retina_head supervisé → pureté JEPA sans régression.

Interface = drop-in de RetinaPerceptionHead : `.locate(retina_tensor)` → [[x_right, z_fwd]] (frame agent), même
convention que food_rel0 / food_xz_from_radar. n_resources=1 (un slot par TYPE ; multi-type = plus tard).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .perception_head import RETINA_DIM

NRAY = RETINA_DIM // 4        # 36 rayons × [depth, R, G, B]
RANGE = 10.0                  # portée raycast (depth normalisé → mètres)
DEPTH_OFFSET = 0.35           # rayon de la sphère de collision (depth=surface) → distance ≈ depth*RANGE + OFFSET


class SelfSupervisedSlotHead(nn.Module):
    """retina(144) → position ego de l'objet le plus saillant, via soft-argmax géométrique sur les rayons.

    La position vit dans le repère agent PAR CONSTRUCTION (les angles de rayon θ_k sont connus), donc l'attention
    n'a qu'à SÉLECTIONNER le bon rayon — ce qui émerge de la seule auto-supervision (transport-consistance)."""

    def __init__(self, n_resources: int = 1) -> None:
        super().__init__()
        self.n_resources = n_resources
        # un scoreur d'attention par ressource (type) ; n_resources=1 pour l'instant (food).
        self.score = nn.ModuleList(
            nn.Sequential(nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 1))
            for _ in range(n_resources)
        )
        th = torch.tensor([k * 2.0 * math.pi / NRAY for k in range(NRAY)])
        self.register_buffer("sin", torch.sin(th))
        self.register_buffer("cos", torch.cos(th))
        # REQUÊTES-COULEUR par slot (chantier multi-ressource 2026-07-04, design cible de la recette
        # ajout-pulsion : « tête de lecture paramétrée par la requête-couleur » — même statut de pureté
        # que les tokens color-gatés de Mode-1 : une requête sur SON capteur, pas un oracle ; ressource
        # nouvelle = requête nouvelle, zéro retrain des autres slots). K=1 → None = chemin historique
        # BYTE-IDENTIQUE (saillance color-agnostique du slot promu). L'émergence pure sans requête
        # (compétition+répulsion seules) a été tentée et a dégénéré (slot mort) — négatif informatif.
        if n_resources > 1:
            q = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]][:n_resources])
            self.register_buffer("color_queries", q / q.norm(dim=-1, keepdim=True))
        else:
            self.color_queries = None

    def _attend(self, retina: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared helper: returns (dist, sal, a_list) without allocating positions.

        dist  [..., NRAY]          — ray distance in metres
        sal   [..., NRAY]          — saliency mask (coloured object, un-normalised)
        a_list list of [..., NRAY] — normalised learned attention per resource
        """
        r = retina.reshape(*retina.shape[:-1], NRAY, 4)
        depth, R, G, B = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
        dist = depth * RANGE + DEPTH_OFFSET                          # [..., NRAY]
        sat = torch.stack([R, G, B], -1).amax(-1) - torch.stack([R, G, B], -1).amin(-1)
        sal = sat.clamp(min=0.0) * torch.sigmoid(40.0 * (0.95 - depth))
        scores = [self.score[k](r).squeeze(-1) for k in range(self.n_resources)]
        a_list = [torch.softmax(s, dim=-1) for s in scores]
        if self.color_queries is not None:
            # SAILLANCE REQUÊTÉE-COULEUR : chaque slot ne « voit » que les rayons dont la teinte matche
            # sa requête (affinité cosinus seuillée). Ancre chaque slot sur SON type d'objet → pas de
            # slot mort ni de liage ambigu. K=1 → None → saillance agnostique historique, byte-identique.
            rgb = r[..., 1:4]
            rgbn = rgb / (rgb.norm(dim=-1, keepdim=True) + 1e-6)     # [..., NRAY, 3]
            aff = torch.einsum("...nc,kc->...kn", rgbn, self.color_queries)  # [..., K, NRAY]
            aff = (aff - 0.55).clamp(min=0.0)
            a_list = [a_list[k] * aff[..., k, :] for k in range(self.n_resources)]
        return dist, sal, a_list

    def positions(self, retina: torch.Tensor) -> torch.Tensor:
        """retina [..., 144] → [..., n_resources, 2] (x_right, z_fwd) en mètres.

        L'attention apprise est GATÉE par une SAILLANCE perceptuelle (le rayon a touché un objet COLORÉ ≠ vide) :
        saliency = saturation_couleur × hit. C'est la perception (pas un label de position, pas un oracle ;
        color-AGNOSTIQUE → général, §3) ; elle BRISE la sous-détermination de la transport-consistance (qui sinon
        verrouille aussi bien sur une direction VIDE opposée → 127° MAE, non robuste). La position fine ÉMERGE
        toujours de l'attention apprise + la consistance ; la saillance ne fait qu'ancrer 'sur un objet'."""
        dist, sal, a_list = self._attend(retina)
        outs = []
        for k in range(self.n_resources):
            w = a_list[k] * sal
            w = w / (w.sum(-1, keepdim=True) + 1e-6)
            px = (w * dist * self.sin).sum(-1); pz = (w * dist * self.cos).sum(-1)
            outs.append(torch.stack([px, pz], dim=-1))
        return torch.stack(outs, dim=-2)                             # [..., n_resources, 2]

    def positions_and_salience(self, retina: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """retina [..., 144] → (positions [..., n_resources, 2], salience [..., n_resources]).

        salience = un-normalised gated attention mass (a * sal).sum(-1) per resource.
        salience → 0 means no coloured object was hit (occluded / out of range).
        Output byte-identical to positions() for the positions tensor.
        """
        dist, sal, a_list = self._attend(retina)
        pos_outs, sal_outs = [], []
        for k in range(self.n_resources):
            aw = a_list[k] * sal                                     # [..., NRAY] un-normalised gated mass
            sal_outs.append(aw.sum(-1))                              # [...] scalar saliency per resource
            w = aw / (aw.sum(-1, keepdim=True) + 1e-6)
            px = (w * dist * self.sin).sum(-1); pz = (w * dist * self.cos).sum(-1)
            pos_outs.append(torch.stack([px, pz], dim=-1))
        return torch.stack(pos_outs, dim=-2), torch.stack(sal_outs, dim=-1)  # ([..., n_res, 2], [..., n_res])

    @torch.no_grad()
    def color_masses(self, retina: torch.Tensor) -> torch.Tensor:
        """[..., n_resources, 2] = masse d'attention gatée sur les rayons ROUGES vs BLEUS par slot.
        Sert à l'ASSIGNATION label-free slot→ressource (rouge=bouffe, bleu=eau) après entraînement."""
        r = retina.reshape(*retina.shape[:-1], NRAY, 4)
        red = (r[..., 1] > r[..., 3]).float()                        # R > B par rayon
        dist, sal, a_list = self._attend(retina)
        out = []
        for k in range(self.n_resources):
            aw = a_list[k] * sal
            out.append(torch.stack([(aw * red).sum(-1), (aw * (1.0 - red)).sum(-1)], dim=-1))
        return torch.stack(out, dim=-2)

    @torch.no_grad()
    def locate(self, retina: torch.Tensor) -> list[list[float]]:
        """Drop-in de RetinaPerceptionHead.locate : [[x,z], ...] (une entrée par ressource)."""
        pos = self.positions(retina.reshape(-1)[:RETINA_DIM])
        return [[float(pos[k, 0]), float(pos[k, 1])] for k in range(self.n_resources)]


def load_slot_head(path: str) -> SelfSupervisedSlotHead:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    head = SelfSupervisedSlotHead(n_resources=int(ck.get("n_resources", 1)))
    head.load_state_dict(ck["state_dict"])
    head.eval()
    return head
