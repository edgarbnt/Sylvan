"""Corpus + cible + token du CRITIQUE-REPAS (monde conséquent `bosquets_v3_perish`, 2026-07-24).

UN SEUL ENDROIT pour trois conventions qui, dupliquées, ont déjà coûté cher au projet :
  1. les FRONTIÈRES d'épisode d'un log BC,
  2. la CIBLE (résidu-de-repas),
  3. le TOKEN drive-symétrique donné au critique.
Le diagnostic, l'entraîneur et le juge importent d'ici — ils ne peuvent donc plus diverger.

CIBLE = « repas dans les K ticks à venir » (le coût intrinsèque FUTUR de LeCun), et NON le temps de
survie (écrêté, quasi constant dans ce monde → mort comme signal) ni le niveau de faim (qui recopie
la faim courante).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

MEAL_JUMP = 5.0        # un repas = l'énergie remonte de plus de 5 points en un tick
TELEPORT_M = 0.5       # reset d'épisode = saut du torse > 0,5 m en (x,z) ; vitesse réelle 0,011 m/tick
RETINA_DIM = 144


def load_bc_corpus(path: Path | str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Log BC (une ligne par tick) → (obs [N,277], énergie [N], commandes [N,2], bornes d'épisodes).

    Les commandes RÉELLEMENT exécutées sont rendues parce que le critique-latent doit être entraîné
    sur des latents RÊVÉS sous ces commandes — sinon on entraîne sur une distribution (teacher-forced)
    et on déploie sur une autre (rêve open-loop), ce qui est le train ≠ déploiement classique.

    ⚠️ `torso0` = (x, z, YAW). Le 3ᵉ canal est un ANGLE : l'inclure dans une norme fait passer chaque
    enroulement à ±π (saut de 2π ≈ 6,28) pour un téléport — bug mesuré le 2026-07-24, il fabriquait
    94 fausses frontières sur 20 épisodes et corrompait labels ET split. On n'utilise que (x, z).
    """
    path = Path(path)
    proprio, retina, energy, torso, cmd = [], [], [], [], []
    with open(path / "ep_0000.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            proprio.append(r["obs"]["proprio"])
            retina.append(r["wm"]["retina0"])
            energy.append(r["obs"]["energy"])
            torso.append(r["wm"]["torso0"])
            cmd.append((r["wm"].get("cmd") or [0.0, 0.0])[:2])
    e = torch.tensor(energy, dtype=torch.float32)
    obs = torch.cat([
        torch.tensor(proprio, dtype=torch.float32),
        torch.tensor(retina, dtype=torch.float32),
        (e / 100.0).unsqueeze(1),
    ], dim=1)
    t = torch.tensor(torso, dtype=torch.float32)
    step = (t[1:, :2] - t[:-1, :2]).norm(dim=1)
    bounds = [0] + [int(i) + 1 for i in torch.nonzero(step > TELEPORT_M).flatten()] + [len(e)]
    return obs, e, torch.tensor(cmd, dtype=torch.float32), bounds


def meal_flags(e: torch.Tensor, bounds: list[int]) -> torch.Tensor:
    """1 au tick où l'énergie remonte (repas). Les frontières d'épisode sont EXCLUES : un reset
    remet l'énergie à 100 et serait sinon compté comme un repas (erreur commise le 2026-07-23)."""
    ate = torch.zeros_like(e)
    starts = set(bounds)
    for i in range(1, len(e)):
        if i not in starts and e[i] - e[i - 1] > MEAL_JUMP:
            ate[i] = 1.0
    return ate


def residual_label(ate: torch.Tensor, bounds: list[int], k: int) -> torch.Tensor:
    """y=1 s'il y a un repas dans les k ticks À VENIR, borné à l'épisode (pas de fuite inter-vies)."""
    y = torch.zeros_like(ate)
    for a, b in zip(bounds[:-1], bounds[1:]):
        c = torch.cat([torch.zeros(1), ate[a:b].cumsum(0)])
        for i in range(b - a):
            j = min(i + 1 + k, b - a)
            y[a + i] = 1.0 if c[j] - c[i + 1] > 0 else 0.0
    return y


def token(level: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
    """Token drive-symétrique [niveau, dist/10, |sin(bearing)|, cos(bearing), connu] — MÊME contrat
    que le critique de survie déployé (`scripts/train_survival_critic.token`).

    SYMÉTRIE MIROIR IMPOSÉE PAR CONSTRUCTION : |sin| et non sin. La VALEUR d'un état ne peut pas
    dépendre du côté où est la ressource — seule l'ACTION le peut. Mesuré sur le critique précédent :
    avec sin signé, écart miroir jusqu'à 0,13 (plus que l'effet de 3 m de distance), ce qui créait un
    optimum de valeur HORS-AXE (~30°) → le planner gardait la cible de biais et ORBITAIT. Une
    symétrie connue s'IMPOSE, elle ne se fitte pas.

    slot [..., 2] = (x_droite, z_avant) ; level [...] = énergie/100. `connu` vaut 1 ici (le slot est
    toujours produit) ; le canal existe pour le contrat multi-pulsion (une pulsion sans perception).
    """
    d = slot.norm(dim=-1)
    return torch.stack([
        level,
        d.clamp(max=10.0) / 10.0,
        slot[..., 0].abs() / (d + 1e-6),
        slot[..., 1] / (d + 1e-6),
        torch.ones_like(d),
    ], dim=-1)


def episode_split(bounds: list[int], n_total: int, frac_train: float = 0.7) -> tuple[torch.Tensor, torch.Tensor]:
    """Masques train/held-out PAR ÉPISODE. Un split par tick fuirait massivement : deux ticks
    voisins sont quasi identiques et la cible varie lentement."""
    n_ep = len(bounds) - 1
    n_tr = max(1, int(round(frac_train * n_ep)))
    tr = torch.zeros(n_total, dtype=torch.bool)
    for a, b in zip(bounds[:n_tr], bounds[1:n_tr + 1]):
        tr[a:b] = True
    return tr, ~tr


def auc(score: torch.Tensor, y: torch.Tensor) -> float:
    """AUC de Mann-Whitney (rangs), robuste aux ex-aequo."""
    pos, neg = int(y.sum()), int((1 - y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = score.argsort()
    ranks = torch.empty_like(score)
    ranks[order] = torch.arange(1, len(score) + 1, dtype=score.dtype)
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
