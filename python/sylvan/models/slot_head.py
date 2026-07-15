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
            # requêtes PURE-CANAL, 1 par ressource : rouge=bouffe, bleu=eau, VERT=danger (2026-07-15).
            # Vert = seul canal libre → cosinus < 0.55 (seuil) avec rouge ET bleu → zéro fuite croisée
            # (le violet fuyait dans les deux). Byte-identique pour n_resources ≤ 2 (le slice garde red,blue).
            q = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]][:n_resources])
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
            # SOFTMAX MASQUÉ (K>1) : le gating saillance×affinité×proximité entre DANS le softmax
            # comme log-prior. Leçon (slot-eau effondré, 2026-07-04) : gater APRÈS le softmax crée une
            # région morte — le scoreur peut fuir les rayons de sa couleur (masse gatée → 0, position →
            # origine à coût quasi nul, gradient évanoui, irrécupérable). En log-prior, la masse somme
            # à 1 sur les rayons de MA couleur PAR CONSTRUCTION ; le scoreur ne peut que redistribuer.
            # Proximité incluse (sémantique planner : « le plus proche de cette couleur »).
            prox = ((1.0 - depth).clamp(min=0.0)) ** 2
            # READOUT GÉOMÉTRIQUE PUR (K>1, décision 2026-07-04 après 7 itérations) : le scoreur APPRIS
            # est retiré des logits — chaque variante apprise trouvait un optimum pathologique (collapse
            # origine 1.8 m, centroïde 64-67°, distorsions 15-19°) alors que le prior géométrique seul
            # = argmin-souple « le plus proche de ma couleur » ≈ plancher capteur (0.46-0.68 m). Même
            # leçon que slot_calib : c'est une GÉOMÉTRIE, pas une quantité à fitter. Prior de distance
            # −2/m = départage nearest-vs-centroïde (Δ2 m → e⁴≈55×). Zéro paramètre, zéro entraînement.
            # MASQUE COULEUR DUR (fix 2026-07-06) : les rayons de MAUVAISE couleur (aff==0) sont
            # EXCLUS du softmax (logit −inf), pas juste log(1e-8)=−18. Bug mesuré : sans ça, un rayon
            # BLEU (eau) PROCHE battait un rayon ROUGE (bouffe) LOIN via le prior −4/m·dist → le
            # slot-bouffe lisait la position de l'EAU (1.1 m au lieu de 6.0 m) → planner orbite un
            # fantôme en monde épars (bouffe-loin+eau-proche). Dense (bouffe proche) non affecté.
            # Toggle SYLVAN_SLOT_HARD_MASK (défaut 1=fix) : 0 reproduit l'ancien masque MOU pour l'A/B
            # de non-régression dense (le fix change aussi le dense multi-objet).
            import os as _os
            _hard = _os.environ.get("SYLVAN_SLOT_HARD_MASK", "1") != "0"
            NEG = -1e9
            a_list = []
            for k in range(self.n_resources):
                logit = torch.log(sal * aff[..., k, :] * prox + 1e-8) - 4.0 * dist
                if _hard:
                    logit = torch.where(aff[..., k, :] > 0.0, logit, torch.full_like(logit, NEG))
                a_list.append(torch.softmax(logit, dim=-1))
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
            if self.color_queries is not None:
                w = a_list[k]                      # softmax masqué : déjà une distribution propre
            else:
                w = a_list[k] * sal
                w = w / (w.sum(-1, keepdim=True) + 1e-6)
            px = (w * dist * self.sin).sum(-1); pz = (w * dist * self.cos).sum(-1)
            if self.color_queries is not None:
                # DÉCOUPLAGE direction/distance (K>1 ; diagnostic 2026-07-04 : bearing 1.5° parfait
                # mais distance ÉCRASÉE 1.07 vs 2.64 m — les fuites d'attention vers d'autres items
                # de la même couleur à d'autres azimuts s'ANNULENT vectoriellement → la norme fond).
                # Direction = soft-argmax vectoriel (robuste) ; distance = moyenne SCALAIRE pondérée
                # (pas d'annulation). Un seul item visible → strictement identique à l'ancien calcul.
                vec_norm = (px ** 2 + pz ** 2 + 1e-4).sqrt()  # eps DANS le sqrt (grad de sqrt(0) = inf)
                d_scalar = (w * dist).sum(-1)
                px = px / vec_norm * d_scalar
                pz = pz / vec_norm * d_scalar
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

    def visibility(self, retina: torch.Tensor) -> torch.Tensor:
        """[..., n_resources] ∈ [0,1] : max sur les rayons de saillance×affinité-couleur — « un rayon
        de MA couleur a-t-il touché quelque chose ? ». Quasi-binaire, robuste à l'échelle des masses
        (un seuil sur la masse d'attention brute dépend de aff×prox → fragile)."""
        r = retina.reshape(*retina.shape[:-1], NRAY, 4)
        depth = r[..., 0]
        rgb = r[..., 1:4]
        sat = rgb.amax(-1) - rgb.amin(-1)
        sal = sat.clamp(min=0.0) * torch.sigmoid(40.0 * (0.95 - depth))
        if self.color_queries is None:
            return sal.amax(-1, keepdim=True).expand(*sal.shape[:-1], self.n_resources)
        rgbn = rgb / (rgb.norm(dim=-1, keepdim=True) + 1e-6)
        aff = (torch.einsum("...nc,kc->...kn", rgbn, self.color_queries) - 0.55).clamp(min=0.0)
        return (aff * sal.unsqueeze(-2)).amax(-1)

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
