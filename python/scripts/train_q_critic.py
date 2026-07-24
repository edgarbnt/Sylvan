"""CRITIQUE Q(s, a) — état FIABLE du WM + ce que le candidat obtient géométriquement.

POURQUOI CETTE FORME (mesurée, pas devinée). Le diagnostic A1 corrigé
(`docs/audit_conformite_jepa.md`) montre que :
  * le latent porte la PRÉSENCE et l'APPARENCE de l'objet à la profondeur 0 (R² 0,59 pour
    « bouffe en vue », 0,57 pour les rayons bouffe — donc potentiellement l'indice de maturité),
  * mais PAS ses coordonnées précises (R² 0,05),
  * et cette information SE DÉGRADE le long du rêve : 0,556 → 0,160 entre les profondeurs 0 et 79.

Les 4 critiques précédents lisaient le latent TERMINAL — le point le PLUS dégradé. D'où :

    Q(s, a)  =  V( latent à t=0 , slot TRANSPORTÉ à l'horizon )
                 └── état fiable ──┘   └── ce que le candidat OBTIENT ──┘

Le latent à t=0 dit CE QUE JE VOIS (scène, apparence) ; le slot transporté dit OÙ JE SERAI par
rapport à la ressource. Les deux sont l'ÉTAT DU WORLD-MODEL (donc architecturalement pur : on ne lit
NI la rétine brute, ce qui donnerait au coût sa propre perception, NI le latent dégradé). Le latent
apporte exactement ce que `-min_dist` ne peut pas voir.

CIBLE = le VRAI retour actualisé, calculé EXACTEMENT. Les épisodes sont complets, donc pas besoin de
bootstrap TD (et donc pas d'erreur de propagation, qui était le défaut mesuré de la version
précédente : V à 3 % de son ancre). Ce n'est PAS la cible « repas dans K ticks » à fenêtre fixe qui
avait échoué : ici l'horizon est la fin de l'épisode, borné seulement par γ.

TRAIN = DÉPLOIEMENT : le slot d'horizon est le slot TRANSPORTÉ par le rêve sous les commandes
réellement exécutées — la même quantité qu'au déploiement, pas le slot réel observé.

WM GELÉ (§3).

⚠️ Rien ici ne juge le critique. Juge = A/B PLEINE-POLITIQUE ; l'intra-état ne peut que DISQUALIFIER.

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python -m scripts.train_q_critic [--corpus DIR ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from sylvan.critic_corpus import load_bc_corpora, meal_flags, token
from sylvan.models.command_wm import CommandWorldModel


class QCritic(nn.Module):
    """Q(latent_t0, token_slot_horizon) = repas futurs actualisés attendus."""

    def __init__(self, latent_dim: int, tok_dim: int = 5, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + tok_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("mu", torch.zeros(latent_dim + tok_dim))
        self.register_buffer("sd", torch.ones(latent_dim + tok_dim))

    def forward(self, latent0: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        x = torch.cat([latent0, tok], dim=-1)
        return self.net((x - self.mu) / self.sd).squeeze(-1)


def true_return(ate: torch.Tensor, bounds: list[int], gamma: float) -> torch.Tensor:
    """Retour actualisé EXACT, borné à l'épisode (récurrence arrière). Pas d'approximation."""
    g = torch.zeros_like(ate)
    for a, b in zip(bounds[:-1], bounds[1:]):
        acc = 0.0
        for t in range(b - 1, a - 1, -1):
            acc = float(ate[t]) + gamma * acc
            g[t] = acc
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+",
                    default=["data/replay_buffer/critic_bosq_ripe11",
                             "data/replay_buffer/critic_bosq_ripe12",
                             "data/replay_buffer/critic_bosq_ripe13"])
    ap.add_argument("--wm", default="data/checkpoints/wm_objcentric_kin/wm_best.pt")
    ap.add_argument("--out", default="data/checkpoints/q_critic")
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--horizon", type=int, default=80)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    obs, energy, cmds, bounds = load_bc_corpora(args.corpus)
    ate = meal_flags(energy, bounds)
    G = true_return(ate, bounds, args.gamma)
    n_ep = len(bounds) - 1
    print(f"corpus : {len(energy)} ticks, {n_ep} vies, {int(ate.sum())} repas | "
          f"vrai retour moy {float(G.mean()):.3f} (médiane {float(G.median()):.3f})")

    pl = torch.load(args.wm, map_location="cpu", weights_only=False)
    meta = pl["meta"]
    wm = CommandWorldModel(
        obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
        predictor_arch=meta.get("predictor_arch", "shallow"),
        with_slot=True, slot_resources=meta.get("slot_resources", 1),
    )
    wm.load_state_dict(pl["model"])
    wm.eval()
    latent_dim = meta.get("latent_dim", 128)

    H = args.horizon
    split = bounds[max(1, int(round(0.7 * n_ep)))]
    starts, is_tr = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        for t in range(a, b - H - 1, args.stride):
            starts.append(t)
            is_tr.append(t < split)
    st = torch.tensor(starts)

    lat0_l, tok_l = [], []
    with torch.no_grad():
        for i in range(0, len(st), 256):
            idx = st[i:i + 256]
            o = wm.rollout_open_loop(obs[idx], torch.stack([cmds[j:j + H] for j in idx]))
            lat0_l.append(o["predicted_latents"][:, 0])          # état FIABLE (profondeur 0)
            tok_l.append(token(energy[idx] / 100.0, o["slot"][:, -1]))   # slot TRANSPORTÉ, comme au déploiement
    lat0, tok = torch.cat(lat0_l), torch.cat(tok_l)
    y = G[st + H - 1]                                            # vrai retour à l'horizon
    tr = torch.tensor(is_tr)
    te = ~tr
    print(f"  {len(lat0)} échantillons (train {int(tr.sum())} / held-out {int(te.sum())}) | "
          f"cible moy {float(y.mean()):.3f}")

    model = QCritic(latent_dim)
    with torch.no_grad():
        x = torch.cat([lat0, tok], dim=-1)
        model.mu.copy_(x[tr].mean(0))
        model.sd.copy_(x[tr].std(0).clamp_min(1e-6))

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best, best_state = -9e9, None
    for step in range(args.steps):
        model.train()
        idx = torch.randperm(int(tr.sum()))[:1024]
        opt.zero_grad()
        loss = ((model(lat0[tr][idx], tok[tr][idx]) - y[tr][idx]) ** 2).mean()
        loss.backward()
        opt.step()
        if (step + 1) % max(1, args.steps // 12) == 0:
            model.eval()
            with torch.no_grad():
                p = model(lat0[te], tok[te])
            yt = y[te]
            r2 = float(1 - ((p - yt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())
            pc = float(((p - p.mean()) / p.std() * (yt - yt.mean()) / yt.std()).mean())
            if r2 > best:
                best, best_state = r2, {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  pas {step + 1:5d} | loss {float(loss):.4f} | R² held-out {r2:+.3f} | corr {pc:+.3f}")

    # CONTRÔLE D'ABLATION : le latent apporte-t-il quelque chose au-delà de la géométrie ?
    # (si le token seul fait aussi bien, on a re-fabriqué -min_dist et il faut le DIRE)
    abl = QCritic(0, tok_dim=5)
    with torch.no_grad():
        abl.mu.copy_(tok[tr].mean(0))
        abl.sd.copy_(tok[tr].std(0).clamp_min(1e-6))
    opt2 = torch.optim.Adam(abl.parameters(), lr=1e-3)
    empty = torch.zeros(len(tok), 0)
    for _ in range(args.steps):
        idx = torch.randperm(int(tr.sum()))[:1024]
        opt2.zero_grad()
        ((abl(empty[tr][idx], tok[tr][idx]) - y[tr][idx]) ** 2).mean().backward()
        opt2.step()
    abl.eval()
    with torch.no_grad():
        pa = abl(empty[te], tok[te])
    yt = y[te]
    r2a = float(1 - ((pa - yt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())
    print(f"\n  ABLATION token SEUL (géométrie) : R² {r2a:+.3f}   vs   avec latent : R² {best:+.3f}")
    print(f"  -> apport du latent : {best - r2a:+.3f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "latent_dim": latent_dim, "hidden": 256,
                "meta": {"gamma": args.gamma, "horizon": H, "r2_heldout": best,
                         "r2_token_only": r2a, "corpus": [str(c) for c in args.corpus],
                         "form": "Q(latent_t0, token(slot transporté à H)) -> vrai retour actualisé",
                         "warning": "juge = A/B PLEINE-POLITIQUE"}},
               out / "critic_best.pt")
    print(f"  -> {out / 'critic_best.pt'}")


if __name__ == "__main__":
    main()
