"""A1 — LE LATENT PORTE-T-IL L'OBJET ? (sonde gratuite, juge de la réparation du substrat)

POURQUOI. L'audit de conformité JEPA (docs/audit_conformite_jepa.md) a identifié A1 comme le maillon
dont tout le reste dépend : si le latent ne porte pas la position de la ressource, aucun module
appris posé dessus ne peut lire la scène — ce qui explique d'un coup les 4 échecs du critique. Le
critère JEPA est qu'une représentation soit MAXIMALEMENT INFORMATIVE tout en restant prédictible ;
l'effondrement, c'est perdre l'information discriminante.

CE QUE CETTE SONDE TRANCHE. La première mesure (2026-07-24) utilisait une régression LINÉAIRE et
donnait R² ≈ 0. Mais un latent peut porter l'information de façon NON LINÉAIRE : conclure « l'info
est absente » depuis une sonde linéaire serait une faute de mesure. On compare donc :

  * sonde LINÉAIRE (ridge)  — l'information est-elle lisible SIMPLEMENT ?
  * sonde MLP               — l'information est-elle PRÉSENTE, même encodée non linéairement ?

Lecture :
  - MLP haut  + linéaire bas  -> l'info EST là, mal exposée. Pas besoin de ré-entraîner le WM :
                                 c'est la TÊTE qui doit être plus expressive.
  - MLP bas   + linéaire bas  -> l'info est ABSENTE du latent. Réparation du substrat justifiée.

Toujours held-out PAR ÉPISODE, et on rapporte une BASELINE (prédire la moyenne) pour que R² ait un
sens. Le WM est GELÉ : on ne mesure que ce qu'il contient déjà.

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_latent_carries_object.py \
      [--corpus DIR ...] [--depth 80]
"""
from __future__ import annotations

import argparse

import torch
from torch import nn

from sylvan.critic_corpus import load_bc_corpora
from sylvan.models.command_wm import CommandWorldModel


def r2(pred: torch.Tensor, truth: torch.Tensor) -> float:
    ss_res = ((pred - truth) ** 2).sum()
    ss_tot = ((truth - truth.mean(0)) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def probe_linear(xtr, ytr, xte) -> torch.Tensor:
    a = torch.cat([xtr, torch.ones(len(xtr), 1)], 1)
    w = torch.linalg.lstsq(a, ytr).solution
    return torch.cat([xte, torch.ones(len(xte), 1)], 1) @ w


def probe_mlp(xtr, ytr, xte, steps: int = 3000, hidden: int = 256) -> torch.Tensor:
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(xtr.shape[1], hidden), nn.SiLU(),
                        nn.Linear(hidden, hidden), nn.SiLU(),
                        nn.Linear(hidden, ytr.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(steps):
        idx = torch.randperm(len(xtr))[:1024]
        opt.zero_grad()
        loss = ((net(xtr[idx]) - ytr[idx]) ** 2).mean()
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        return net(xte)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+",
                    default=["data/replay_buffer/critic_bosq_ripe11",
                             "data/replay_buffer/critic_bosq_ripe12"])
    ap.add_argument("--wm", default="data/checkpoints/wm_objcentric_kin/wm_best.pt")
    ap.add_argument("--depth", type=int, default=80)   # profondeur du rêve où l'on sonde
    ap.add_argument("--stride", type=int, default=24)
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(1)

    obs, energy, cmds, bounds = load_bc_corpora(args.corpus)
    pl = torch.load(args.wm, map_location="cpu", weights_only=False)
    meta = pl["meta"]
    wm = CommandWorldModel(
        obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
        predictor_arch=meta.get("predictor_arch", "shallow"),
        with_slot=True, slot_resources=meta.get("slot_resources", 1),
    )
    wm.load_state_dict(pl["model"])
    wm.eval()                                          # WM GELÉ : on mesure ce qu'il contient déjà

    H = args.depth
    starts, is_tr = [], []
    n_ep = len(bounds) - 1
    split = bounds[max(1, int(round(0.7 * n_ep)))]
    for a, b in zip(bounds[:-1], bounds[1:]):
        for t in range(a, b - H - 1, args.stride):
            starts.append(t)
            is_tr.append(t < split)
    st = torch.tensor(starts)

    lat_l, slot_l = [], []
    with torch.no_grad():
        for i in range(0, len(st), 256):
            idx = st[i:i + 256]
            o = wm.rollout_open_loop(obs[idx], torch.stack([cmds[j:j + H] for j in idx]))
            lat_l.append(o["predicted_latents"][:, -1])
            slot_l.append(o["slot"][:, -1])
    lat, slot = torch.cat(lat_l), torch.cat(slot_l)
    tr = torch.tensor(is_tr)

    mu, sd = lat[tr].mean(0), lat[tr].std(0).clamp_min(1e-6)
    x = (lat - mu) / sd
    xtr, ytr, xte, yte = x[tr], slot[tr], x[~tr], slot[~tr]
    print(f"corpus {len(lat)} latents rêvés (profondeur {H}) | train {int(tr.sum())} / "
          f"held-out {int((~tr).sum())} | WM = {args.wm}")

    d_true = yte.norm(dim=-1)
    base = yte.mean(0).expand_as(yte)                  # prédire la moyenne = R² 0 par définition
    print(f"  BASELINE (moyenne)          R² slot {r2(base, yte):+.3f}")

    for name, pred in (("LINÉAIRE (ridge)", probe_linear(xtr, ytr, xte)),
                       ("MLP (2×256)", probe_mlp(xtr, ytr, xte))):
        d_pred = pred.norm(dim=-1)
        r2d = 1 - ((d_pred - d_true) ** 2).sum() / ((d_true - d_true.mean()) ** 2).sum()
        err = (pred - yte).norm(dim=-1)
        print(f"  {name:26s} R² slot {r2(pred, yte):+.3f} | R² distance {float(r2d):+.3f} | "
              f"erreur médiane {float(err.median()):.2f} m")

    print("\n  LECTURE : MLP nettement > linéaire -> l'info EST dans le latent, mal exposée "
          "(fix = tête plus expressive, PAS de ré-entraînement).")
    print("            MLP aussi bas que linéaire -> l'info est ABSENTE -> réparer le substrat.")


if __name__ == "__main__":
    main()
