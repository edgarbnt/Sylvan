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
import os

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
        # Score APPRIS par ressource, sur la rétine BRUTE [B, 36, 4]. Entraîné par CONSISTANCE DE
        # TRANSPORT (zéro label). ⚠️ En multi-ressource il est INTÉGRALEMENT ÉCRASÉ par le readout
        # géométrique ci-dessous (audit A2, 2026-07-24) : 2498 paramètres calculés puis jetés.
        self.score = nn.ModuleList(
            nn.Sequential(nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(),
                          nn.Linear(32, 1))
            for _ in range(n_resources)
        )
        # ANGLES DES RAYONS — géométrie pure, doit refléter EXACTEMENT perception.gd. Avec un vrai
        # cône (SYLVAN_RETINA_FOV_DEG), les 36 rayons sont redistribués sur le champ ; le décodage
        # de position reste correct par construction puisqu'il n'utilise que ces angles CONNUS.
        # ⚠️ Ces buffers sont PERSISTANTS (présents dans le state_dict) : charger un checkpoint
        # restaure les angles 360°. serve_planner_command les recalcule APRÈS chargement.
        _fov = math.radians(float(os.environ.get("SYLVAN_RETINA_FOV_DEG", "360")))
        th = torch.tensor([(k if k <= NRAY // 2 else k - NRAY) * _fov / NRAY for k in range(NRAY)])
        self.register_buffer("sin", torch.sin(th))
        self.register_buffer("cos", torch.cos(th))
        # ⚰️ NÉGATIF BANKÉ (2026-07-30, code retiré le 2026-08-02) : un `affinity_net` (MLP 4→32→1
        # par ressource) devait remplacer les requêtes-couleur codées-main en classant chaque rayon.
        # Mesuré : 2,09 m contre 2,18 m au cosinus — RIEN. Cause structurelle : 39,9 % des rayons
        # d'arbres partagent le volume (depth,R,G,B) des rayons de nourriture, donc aucune fonction
        # du rayon SEUL ne les sépare. Même verdict pour la variante sur tokens encodeur (2,10 m) et
        # pour le score-token entraîné par transport (1,50 m). Détail :
        # `docs/diag_perception_consequence_2026-07-30.md`.
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
            # MARGE PAR-REQUÊTE (chantier P6-reopen, docs/design_perception_types.md — fix du Mur B) :
            # le seuil GLOBAL 0.55 faisait deux boulots (décrire l'affinité ET séparer les types) —
            # avec des requêtes APPRISES (couleurs vraies : cos(bleu-vrai, vert-vrai)=0.61 > 0.55),
            # un seuil global fuit structurellement. La marge devient PAR TYPE, MESURÉE de l'écart
            # réel entre groupes d'apparence (build_typed_slots). persistent=False : absent du
            # state_dict → tous les checkpoints existants chargent inchangés ; défaut 0.55 partout
            # = BIT-IDENTIQUE au seuil historique ; le WM typé la porte via meta["query_thr"].
            self.register_buffer("query_thr", torch.full((n_resources,), 0.55), persistent=False)
        else:
            self.color_queries = None
            self.query_thr = None
        # SAILLANCE DE PULSION APPRISE (chantier « perception pure de la faim », 2026-08-02).
        # Remplace la RÈGLE-COULEUR codée-main par un s(rgb) appris de la CONSÉQUENCE VÉCUE
        # (wm.ate) — la dernière clé-apparence structurelle du projet. La GÉOMÉTRIE du
        # soft-argmax ne change PAS d'une ligne : seule la SÉLECTION change de source.
        # Opt-in `SYLVAN_SLOT_DRIVE_SALIENCY="<idx>:<ckpt>[,<idx>:<ckpt>...]"`, défaut None =
        # chemin cosinus BYTE-IDENTIQUE. Les ressources sans tête gardent leur requête.
        self.drive_saliency: dict[int, object] = {}
        self._load_drive_saliency(os.environ.get("SYLVAN_SLOT_DRIVE_SALIENCY", ""))

    def _load_drive_saliency(self, spec: str) -> None:
        if not spec:
            return
        from .drive_saliency import load_drive_saliency  # import paresseux
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            idx_s, _, path = part.partition(":")
            k = int(idx_s)
            if not 0 <= k < self.n_resources:
                raise ValueError(f"slot {k} hors bornes (n_resources={self.n_resources})")
            head, ck = load_drive_saliency(path)
            # ⚠️ VOLONTAIREMENT hors de l'arbre de modules (dict simple, pas add_module) :
            # ces têtes ne doivent PAS entrer dans le state_dict, sinon tous les checkpoints
            # existants échoueraient au chargement strict. Elles sont gelées et en eval.
            head.requires_grad_(False)
            self.drive_saliency[k] = head
            print(f"[slot] SAILLANCE APPRISE sur le slot {k} ({ck.get('drive', '?')}) "
                  f"— la règle-couleur est court-circuitée · ρ̂={ck.get('rho_hat', float('nan')):.2f} m")

    def _affinity(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[..., NRAY, 3] -> (score [..., K, NRAY], seuil [K]).

        UNE seule source d'appartenance rayon→ressource, quelle que soit son origine : le
        cosinus codé-main, ou la saillance APPRISE là où une tête est branchée. Tout ce qui
        suit dans `_attend` est identique — c'est le point du chantier.
        """
        rgbn = rgb / (rgb.norm(dim=-1, keepdim=True) + 1e-6)
        score = torch.einsum("...nc,kc->...kn", rgbn, self.color_queries)
        thr = self.query_thr.clone()
        if self.drive_saliency:
            from .drive_saliency import SAL_THR
            score = score.clone()
            for k, head in self.drive_saliency.items():
                score[..., k, :] = head.s(rgb)
                thr[k] = SAL_THR
        return score, thr

    def _attend(self, retina: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Returns (dist, sal, a_list) — softmax attention weights per resource.

        dist    [..., NRAY]          — ray distance in metres
        sal     [..., NRAY]          — saliency mask (coloured object, un-normalised)
        a_list  list of [..., NRAY]  — normalised attention per resource
        """
        r = retina.reshape(*retina.shape[:-1], NRAY, 4)
        depth, R, G, B = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
        dist = depth * RANGE + DEPTH_OFFSET                          # [..., NRAY]
        # SAILLANCE SATURATION (K=1, byte-identique). Ancre l'attention sur un objet COLORÉ.
        sat = torch.stack([R, G, B], -1).amax(-1) - torch.stack([R, G, B], -1).amin(-1)
        sal = sat.clamp(min=0.0) * torch.sigmoid(40.0 * (0.95 - depth))
        scores = [self.score[k](r).squeeze(-1) for k in range(self.n_resources)]
        a_list = [torch.softmax(s, dim=-1) for s in scores]
        # REQUÊTES-COULEUR (K>1) : le readout GÉOMÉTRIQUE écrase intégralement le score appris
        # ci-dessus. C'est l'anomalie A2 de l'audit du 2026-07-24, assumée et documentée.
        if self.color_queries is not None:
            rgb = r[..., 1:4]
            cos, thr = self._affinity(rgb)
            aff = (cos - thr.unsqueeze(-1)).clamp(min=0.0)
            sal_cos = cos.amax(dim=-2).clamp(min=0.0)
            prox = ((1.0 - depth).clamp(min=0.0)) ** 2
            _hard = os.environ.get("SYLVAN_SLOT_HARD_MASK", "1") != "0"
            NEG = -1e9
            a_list = []
            for k in range(self.n_resources):
                logit = torch.log(sal_cos * aff[..., k, :] * prox + 1e-8) - 4.0 * dist
                if _hard:
                    logit = torch.where(aff[..., k, :] > 0.0, logit,
                                        torch.full_like(logit, NEG))
                a_list.append(torch.softmax(logit, dim=-1))
            sal = sal_cos
        return dist, sal, a_list

    def positions(self, retina: torch.Tensor) -> torch.Tensor:
        """retina [..., 144] → [..., n_resources, 2] (x_right, z_fwd) en mètres."""
        dist, sal, a_list = self._attend(retina)
        # attention masquée par requête-couleur : déjà une distribution propre, et le découplage
        # direction/distance corrige les fuites cross-azimut.
        gated = self.color_queries is not None
        outs = []
        for k in range(self.n_resources):
            if gated:
                w = a_list[k]                      # softmax déjà propre (masque couleur)
            else:
                w = a_list[k] * sal
                w = w / (w.sum(-1, keepdim=True) + 1e-6)
            px = (w * dist * self.sin).sum(-1); pz = (w * dist * self.cos).sum(-1)
            if gated:
                # DÉCOUPLAGE direction/distance (K>1 ; diagnostic 2026-07-04 : bearing 1.5° parfait
                # mais distance ÉCRASÉE 1.07 vs 2.64 m — les fuites d'attention vers d'autres items
                # de la même couleur à d'autres azimuts s'ANNULENT vectoriellement → la norme fond).
                # Direction = soft-argmax vectoriel (robuste) ; distance = moyenne SCALAIRE pondérée
                # (pas d'annulation). Un seul item visible → strictement identique à l'ancien calcul.
                vec_norm = (px ** 2 + pz ** 2 + 1e-4).sqrt()
                d_scalar = (w * dist).sum(-1)
                px = px / vec_norm * d_scalar
                pz = pz / vec_norm * d_scalar
            outs.append(torch.stack([px, pz], dim=-1))
        return torch.stack(outs, dim=-2)                             # [..., n_resources, 2]

    def positions_and_salience(self, retina: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor]:
        """retina [..., 144] → (positions [..., n_resources, 2], salience [..., n_resources]).

        salience = un-normalised gated attention mass (a * sal).sum(-1) per resource.
        salience → 0 means no coloured object was hit (occluded / out of range)."""
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
        # MÊME source d'affinité que `_attend` : sans ce miroir, la gate de visibilité servie
        # et le slot divergeraient dès qu'une tête apprise est branchée.
        score, thr = self._affinity(rgb)
        aff = (score - thr.unsqueeze(-1)).clamp(min=0.0)
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
