"""GATE GRATUIT DÉCISIF — la cible du critique dépend-elle de plus que la GÉOMÉTRIE ?

POURQUOI (principe de travail n°1 : diagnostiquer gratuitement avant tout entraînement).
Le critique appris est censé REMPLACER le coût analytique, dont le terme porteur est `-min_dist`
= la DISTANCE DU SLOT (plus le cap). Donc si la cible « résidu-de-repas » est déjà prédictible
par la géométrie que le coût analytique connaît DÉJÀ (distance + bearing), alors un critique
appris ne fera que ré-apprendre `-min_dist` : zéro gain, et on aura payé un entraînement + un
A/B pour rien. C'est très exactement le piège des 3 échecs historiques (on jugeait un R² poolé
sans jamais demander « ce signal est-il AILLEURS que dans la distance ? »).

Ce diagnostic ne s'entraîne PAS sur le monde : il lit un corpus BC déjà collecté, recalcule le
slot avec l'encodeur du WM VIVANT (train = déploiement) et compare deux prédicteurs de la MÊME
cible, en held-out PAR ÉPISODE (sans fuite) :

    GEO  = [dist_slot, cos(bearing), |sin(bearing)|]      <- ce que le coût analytique sait déjà
    FULL = GEO + [énergie]                                 <- + l'état interne (la pulsion)

CRITÈRES PRÉ-ENREGISTRÉS (avant de lancer) :
  * SUCCÈS  : AUC(FULL) − AUC(GEO) >= +0.02  -> il existe du signal APPRENABLE au-delà de la
              géométrie -> entraîner la tête-valeur est justifié.
  * KILL    : AUC(FULL) − AUC(GEO) <= +0.005 -> la cible est de la géométrie déguisée ; un
              critique ne battrait pas `-min_dist` -> NE PAS entraîner, chercher ailleurs.
  * entre les deux : marginal, à trancher avec la taille d'effet.
On rapporte aussi AUC(dist seule) comme repère, et la part d'états où la bouffe est hors-vue.

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_critic_beyond_geometry.py \
      [--corpus data/replay_buffer/critic_bosq_a] [--k 200]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from sylvan.critic_corpus import load_bc_corpus, meal_flags, residual_label, auc, episode_split
from sylvan.models.command_wm import CommandWorldModel

MEAL_JUMP = 5.0        # un repas = l'énergie remonte de plus de 5 points en un tick
TELEPORT_M = 0.5       # un reset d'episode = le torse saute de plus de 0,5 m en xz (vitesse reelle 0,011 m/tick -> aucun faux positif ; 19 frontieres detectees a 0,3 comme a 2,0 = separation nette)


def load_corpus(path: Path):
    """Lit le log BC (une ligne par tick) -> obs [N,277], énergie [N], bornes d'épisodes."""
    proprio, retina, energy, torso = [], [], [], []
    with open(path / "ep_0000.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            proprio.append(r["obs"]["proprio"])
            retina.append(r["wm"]["retina0"])
            energy.append(r["obs"]["energy"])
            torso.append(r["wm"]["torso0"])
    e = torch.tensor(energy, dtype=torch.float32)
    obs = torch.cat([
        torch.tensor(proprio, dtype=torch.float32),
        torch.tensor(retina, dtype=torch.float32),
        (e / 100.0).unsqueeze(1),
    ], dim=1)
    t = torch.tensor(torso, dtype=torch.float32)
    # Frontière d'épisode = TÉLÉPORT du torse (robuste), et non un saut d'énergie : un repas qui
    # plafonne à 100 est indiscernable d'un reset si on ne regarde que l'énergie.
    # ⚠️ torso0 = (x, z, YAW) : le 3e canal est un ANGLE. L'inclure dans la norme faisait passer
    # chaque enroulement de yaw (saut de 2π ≈ 6,28) pour un téléport -> 94 fausses frontières, donc
    # des labels et un split corrompus. On ne mesure que la POSITION (x, z).
    step = (t[1:, :2] - t[:-1, :2]).norm(dim=1)
    bounds = [0] + [int(i) + 1 for i in torch.nonzero(step > TELEPORT_M).flatten()] + [len(e)]
    return obs, e, bounds


def meal_flags(e: torch.Tensor, bounds: list[int]) -> torch.Tensor:
    """1 au tick où l'énergie remonte (repas). Les frontières d'épisode sont exclues."""
    ate = torch.zeros_like(e)
    starts = set(bounds)
    for i in range(1, len(e)):
        if i not in starts and e[i] - e[i - 1] > MEAL_JUMP:
            ate[i] = 1.0
    return ate


def residual_label(ate: torch.Tensor, bounds: list[int], k: int) -> torch.Tensor:
    """Cible LeCun : y=1 s'il y a un repas dans les k ticks À VENIR, borné à l'épisode."""
    y = torch.zeros_like(ate)
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = ate[a:b]
        c = torch.cat([torch.zeros(1), seg.cumsum(0)])          # c[i] = repas avant i
        for i in range(b - a):
            j = min(i + 1 + k, b - a)
            y[a + i] = 1.0 if c[j] - c[i + 1] > 0 else 0.0
    return y


def auc(score: torch.Tensor, y: torch.Tensor) -> float:
    """AUC de Mann-Whitney (rangs), robuste aux ex-aequo."""
    pos, neg = int(y.sum()), int((1 - y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = score.argsort()
    ranks = torch.empty_like(score)
    ranks[order] = torch.arange(1, len(score) + 1, dtype=score.dtype)
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def fit_logreg(x: torch.Tensor, y: torch.Tensor, epochs: int = 400) -> torch.nn.Module:
    """Régression logistique (features standardisées en amont). Déterministe."""
    torch.manual_seed(0)
    model = torch.nn.Linear(x.shape[1], 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(x).squeeze(1), y)
        loss.backward()
        opt.step()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/replay_buffer/critic_bosq_a")
    ap.add_argument("--wm", default="data/checkpoints/wm_objcentric_kin/wm_best.pt")
    ap.add_argument("--k", type=int, default=200)
    args = ap.parse_args()

    torch.manual_seed(0)
    obs, energy, _cmds, bounds = load_bc_corpus(Path(args.corpus))
    n_ep = len(bounds) - 1
    print(f"corpus {args.corpus} : {len(energy)} ticks, {n_ep} épisodes détectés (téléport torse)")

    pl = torch.load(args.wm, map_location="cpu", weights_only=False)
    meta = pl["meta"]
    wm = CommandWorldModel(
        obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
        predictor_arch=meta.get("predictor_arch", "shallow"),
        with_slot=True, slot_resources=meta.get("slot_resources", 1),
    )
    wm.load_state_dict(pl["model"])
    wm.eval()

    with torch.no_grad():                                    # slot = MÊME encodeur qu'en déploiement
        slot = torch.cat([wm.encode_slot(obs[i:i + 4096]) for i in range(0, len(obs), 4096)])
    dist = slot.norm(dim=1)
    bearing = torch.atan2(slot[:, 0], slot[:, 1])            # (x_right, z_fwd)
    geo = torch.stack([dist, torch.cos(bearing), torch.sin(bearing).abs()], dim=1)
    full = torch.cat([geo, (energy / 100.0).unsqueeze(1)], dim=1)

    ate = meal_flags(energy, bounds)
    y = residual_label(ate, bounds, args.k)
    print(f"repas détectés : {int(ate.sum())} | cible 'repas dans {args.k} ticks' : "
          f"{100 * float(y.mean()):.1f} % d'états positifs")
    print(f"slot : distance médiane {float(dist.median()):.2f} m | "
          f"états à plus de 10 m (hors portée rétine) : {100 * float((dist > 10).float().mean()):.1f} %")

    # Split HELD-OUT PAR ÉPISODE (pas par tick) : deux ticks voisins sont quasi identiques, un split
    # aléatoire fuirait massivement et gonflerait les deux AUC.
    n_tr = max(1, int(round(0.7 * n_ep)))
    tr = torch.zeros(len(y), dtype=torch.bool)
    for a, b in zip(bounds[:n_tr], bounds[1:n_tr + 1]):
        tr[a:b] = True
    te = ~tr
    print(f"split : {n_tr}/{n_ep} épisodes en train ({int(tr.sum())} ticks), "
          f"{int(te.sum())} ticks en held-out")

    results = {}
    for name, feat in (("dist seule", geo[:, :1]), ("GEO", geo), ("FULL (GEO+énergie)", full)):
        mu, sd = feat[tr].mean(0), feat[tr].std(0).clamp_min(1e-6)
        xs = (feat - mu) / sd
        model = fit_logreg(xs[tr], y[tr])
        with torch.no_grad():
            s = model(xs[te]).squeeze(1)
        results[name] = auc(s, y[te])
        print(f"  AUC held-out [{name:20s}] = {results[name]:.3f}")

    delta = results["FULL (GEO+énergie)"] - results["GEO"]
    print(f"\n  DELTA (FULL - GEO) = {delta:+.3f}")
    if delta >= 0.02:
        print("  -> SUCCÈS : signal APPRENABLE au-delà de la géométrie -> entraîner la tête-valeur")
    elif delta <= 0.005:
        print("  -> KILL : la cible est de la géométrie déguisée, un critique ne battrait pas -min_dist")
    else:
        print("  -> MARGINAL : effet trop petit pour justifier seul l'entraînement")


if __name__ == "__main__":
    main()
