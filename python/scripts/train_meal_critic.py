"""Entraîne le CRITIQUE-REPAS — la tête de valeur du monde conséquent `bosquets_v3_perish`.

RÔLE (LeCun) : c'est le *trainable critic* TC, qui prédit le COÛT INTRINSÈQUE FUTUR depuis un état,
au-delà de la portée du rollout du WM (0,8 m). Il est destiné à remplacer le terme porteur du coût
analytique (`-min_dist`) dans le classement des candidats du planner.

CIBLE = « repas dans K ticks » (cf `sylvan.critic_corpus`), et non la survie (saturée ici donc
morte) ni le niveau de faim (qui recopie la faim courante).

ENTRÉE = le token drive-symétrique [niveau, dist/10, |sin|, cos, connu] construit sur le SLOT lu par
l'encodeur du WM VIVANT — donc la même perception qu'en déploiement (train = déploiement).

⚠️ CE QUE CE SCRIPT NE PROUVE PAS. L'AUC held-out qu'il affiche est une mesure **POOLÉE** : elle
compare des états ENTRE EUX. Or le planner classe 117 candidats DANS LE MÊME ÉTAT — seule la
variance INTRA-état décide, et juger au poolé est très exactement le bug de mesure derrière les 3
échecs historiques du critique. L'AUC ici ne sert qu'à vérifier que l'entraînement a convergé.
LE JUGE est ailleurs : `diagnostics/diag_critic_intra_state.py` (classement des 21 candidats à un
fork conséquent, confronté aux issues RÉELLEMENT mesurées), puis un A/B pleine-politique.

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python -m scripts.train_meal_critic \
      [--corpus data/replay_buffer/critic_bosq_a] [--k 200] [--out data/checkpoints/meal_critic]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from sylvan.critic_corpus import (auc, episode_split, load_bc_corpus, meal_flags,
                                  residual_label, token)
from sylvan.models.command_wm import CommandWorldModel

TOK_DIM = 5


class MealCritic(nn.Module):
    """V(token) ∈ [0,1] = probabilité qu'un repas survienne dans les K ticks à venir.

    mu/sd en buffers → le checkpoint se recharge seul, sans refournir les stats du corpus (même
    convention que ValueHead, pour que le déploiement ne dépende pas d'un fichier annexe).
    """

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TOK_DIM, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("mu", torch.zeros(TOK_DIM))
        self.register_buffer("sd", torch.ones(TOK_DIM))

    def logit(self, tok: torch.Tensor) -> torch.Tensor:
        return self.net((tok - self.mu) / self.sd).squeeze(-1)

    def value(self, tok: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logit(tok))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/replay_buffer/critic_bosq_a")
    ap.add_argument("--wm", default="data/checkpoints/wm_objcentric_kin/wm_best.pt")
    ap.add_argument("--out", default="data/checkpoints/meal_critic")
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)                      # A/B non reproductible = A/B inutile
    torch.set_num_threads(1)

    obs, energy, bounds = load_bc_corpus(args.corpus)
    n_ep = len(bounds) - 1
    print(f"corpus {args.corpus} : {len(energy)} ticks, {n_ep} épisodes")

    pl = torch.load(args.wm, map_location="cpu", weights_only=False)
    meta = pl["meta"]
    wm = CommandWorldModel(
        obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
        predictor_arch=meta.get("predictor_arch", "shallow"),
        with_slot=True, slot_resources=meta.get("slot_resources", 1),
    )
    wm.load_state_dict(pl["model"])
    wm.eval()                                          # WM GELÉ : on n'entraîne que la tête
    with torch.no_grad():
        slot = torch.cat([wm.encode_slot(obs[i:i + 4096]) for i in range(0, len(obs), 4096)])

    x = token(energy / 100.0, slot)                    # [N, 5]
    ate = meal_flags(energy, bounds)
    y = residual_label(ate, bounds, args.k)
    tr, te = episode_split(bounds, len(y))
    print(f"repas {int(ate.sum())} | cible 'repas<{args.k} ticks' {100 * float(y.mean()):.1f} % positifs "
          f"| train {int(tr.sum())} ticks / held-out {int(te.sum())} ticks")
    if float(y[te].sum()) == 0:
        raise SystemExit("held-out sans aucun positif -> AUC indéfinie, corpus trop petit")

    model = MealCritic()
    with torch.no_grad():                              # STATS SUR LE TRAIN SEULEMENT (pas de fuite)
        model.mu.copy_(x[tr].mean(0))
        model.sd.copy_(x[tr].std(0).clamp_min(1e-6))

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    best_auc, best_state = -1.0, None
    for ep in range(args.epochs):
        model.train()
        opt.zero_grad()
        loss = lossf(model.logit(x[tr]), y[tr])
        loss.backward()
        opt.step()
        if (ep + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                a = auc(model.logit(x[te]), y[te])
            if a > best_auc:
                best_auc = a
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  ep {ep + 1:4d} | loss {float(loss):.4f} | AUC held-out {a:.3f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": best_state,
        "meta": {
            "k": args.k, "tok_dim": TOK_DIM, "corpus": str(args.corpus), "wm": str(args.wm),
            "token": "[niveau, dist/10, |sin|, cos, connu] — symétrie miroir par construction",
            "target": f"repas dans {args.k} ticks (résidu de repas, LeCun TC)",
            "auc_pooled_heldout": best_auc,
            "warning": "AUC POOLÉE, ne juge PAS le critique — juge = intra-état (21 candidats) + A/B",
        },
    }, out / "critic_best.pt")
    print(f"\n  -> {out / 'critic_best.pt'} | meilleure AUC POOLÉE held-out {best_auc:.3f}")
    print("  RAPPEL : cette AUC ne juge PAS le critique (question poolée). "
          "Juge = diag_critic_intra_state.py puis A/B pleine-politique.")


if __name__ == "__main__":
    main()
