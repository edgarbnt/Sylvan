"""CRITIQUE TD — valeur TERMINALE apprise par bootstrapping (forme TD-MPC).

POURQUOI CETTE RÉÉCRITURE (2026-07-24, après recherche). Les deux critiques précédents ont été
refusés par le juge intra-état, et la littérature explique pourquoi ils ne POUVAIENT pas marcher.

TD-MPC (Hansen et al., ICML 2022) résout exactement notre problème — un rollout court dans un WM
appris, qui doit quand même décider en fonction d'un avenir lointain. Son objectif est

    max_a  Σ_{i=0}^{H-1} γ^i R(z_{t+i}, a_{t+i})  +  γ^H · Q(z_{t+H}, a_{t+H})
                                                     ^^^^^^^^^^^^^^^^^^^^^^^
                                              la valeur TERMINALE, qui fait entrer
                                              l'information de long horizon dans un
                                              plan de COURT horizon.

Deux erreurs de nos versions précédentes, corrigées ici :

1. AGRÉGAT `mean` → TERMINAL. On moyennait V sur toute la trajectoire rêvée, ce qui traite V comme
   une récompense par pas. La forme correcte met V *au bout* du rêve : elle résume tout ce qui vient
   APRÈS l'horizon, au lieu de re-noter ce qui est dedans.

2. CIBLE MONTE-CARLO À FENÊTRE FIXE → TD BOOTSTRAPPÉE. On apprenait « repas dans K=200 ticks » :
   une fenêtre finie, donc structurellement incapable de voir au-delà de K. Le bootstrapping TD
   `V(z_t) ← r_t + γ·V(z_{t+1})` (réseau-cible retardé) propage la valeur de proche en proche depuis
   arbitrairement loin — c'est CE mécanisme qui donne la vision longue avec un rêve court.

Et surtout : NE PAS allonger le rollout. Mesuré en boucle fermée (horizon 300 → survie au plancher,
2 repas contre 11) et prédit par la littérature : sur un long rollout l'erreur du modèle se compose
et le planner optimise une fantaisie.

ALIGNEMENT TD PROPRE. On rêve sous les commandes RÉELLEMENT exécutées : le rêve suit donc la
trajectoire vécue, et la récompense réelle du tick t+d s'aligne exactement sur la profondeur d du
rêve. On peut donc bootstrapper LE LONG DU RÊVE, c'est-à-dire sur la distribution de déploiement.

WM GELÉ (§3). Récompense = 1 au tick d'un repas, 0 sinon → V(z) = nombre de repas futurs actualisé.

⚠️ Aucun chiffre d'ici ne juge le critique. LE juge est
`diagnostics/diag_critic_intra_state.py --critic-type td`.

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python -m scripts.train_td_critic \
      [--corpus DIR ...] [--gamma 0.999] [--horizon 80]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from sylvan.critic_corpus import load_bc_corpora, meal_flags
from sylvan.models.command_wm import CommandWorldModel


class TDValueHead(nn.Module):
    """V(latent) ∈ ℝ = repas futurs ACTUALISÉS depuis cet état.

    Sortie linéaire (pas de sigmoïde) : avec γ proche de 1 la valeur n'est pas une probabilité mais
    une somme actualisée, bornée par 1/(1−γ). mu/sd en buffers → checkpoint autonome.
    """

    def __init__(self, latent_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("mu", torch.zeros(latent_dim))
        self.register_buffer("sd", torch.ones(latent_dim))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net((latent - self.mu) / self.sd).squeeze(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", default=["data/replay_buffer/critic_bosq_a"])
    ap.add_argument("--wm", default="data/checkpoints/wm_objcentric_kin/wm_best.pt")
    ap.add_argument("--out", default="data/checkpoints/td_critic")
    ap.add_argument("--gamma", type=float, default=0.999)   # horizon effectif ~1/(1-γ) = 1000 ticks
    ap.add_argument("--horizon", type=int, default=80)      # MÊME rêve court qu'en déploiement
    ap.add_argument("--start-stride", type=int, default=8)
    ap.add_argument("--steps", type=int, default=8000)   # PAS de gradient (etait 400 = 2 ordres trop peu)
    ap.add_argument("--tau", type=float, default=0.01)      # EMA du réseau-cible (retard)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    obs, energy, cmds, bounds = load_bc_corpora(args.corpus)
    ate = meal_flags(energy, bounds)
    n_ep = len(bounds) - 1
    print(f"corpus : {len(energy)} ticks, {n_ep} épisodes, {int(ate.sum())} repas "
          f"| γ={args.gamma} (horizon effectif ≈ {1 / (1 - args.gamma):.0f} ticks)")

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

    n_tr_ep = max(1, int(round(0.7 * n_ep)))
    split_tick = bounds[n_tr_ep]
    starts, is_tr = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        for t in range(a, b - args.horizon - 1, args.start_stride):
            starts.append(t)
            is_tr.append(t < split_tick)
    starts_t = torch.tensor(starts)

    # Rêve sous les commandes VÉCUES -> la profondeur d du rêve correspond au tick t+d réel, donc la
    # récompense réelle s'aligne sur le latent rêvé. C'est ce qui rend le bootstrap TD légitime SUR
    # LA DISTRIBUTION DE DÉPLOIEMENT (et non sur des latents teacher-forced).
    lat_l, rew_l = [], []
    with torch.no_grad():
        for i in range(0, len(starts), 256):
            idx = starts_t[i:i + 256]
            seq = torch.stack([cmds[j:j + args.horizon] for j in idx])
            lat_l.append(wm.rollout_open_loop(obs[idx], seq)["predicted_latents"])
            rew_l.append(torch.stack([ate[j:j + args.horizon] for j in idx]))
    lat = torch.cat(lat_l)                                   # [N, H, L] latents rêvés
    rew = torch.cat(rew_l)                                   # [N, H]    repas réels alignés
    tr = torch.tensor(is_tr)
    print(f"  {len(lat)} rêves de {args.horizon} pas (train {int(tr.sum())} / held-out {int((~tr).sum())})"
          f" | repas dans les fenêtres : {int(rew.sum())}")

    model = TDValueHead(latent_dim)
    with torch.no_grad():                                    # stats sur le TRAIN seulement
        flat = lat[tr].reshape(-1, latent_dim)
        model.mu.copy_(flat.mean(0))
        model.sd.copy_(flat.std(0).clamp_min(1e-6))
    target = TDValueHead(latent_dim)                          # réseau-cible RETARDÉ (stabilité TD)
    target.load_state_dict(model.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ltr, rtr = lat[tr], rew[tr]
    lte, rte = lat[~tr], rew[~tr]

    # ANCRE ANALYTIQUE : en régime stationnaire, E[V] = (taux de repas)/(1−γ). Elle permet de MESURER
    # si la valeur a fini de se propager, au lieu de le supposer. Défaut mesuré le 2026-07-24 :
    # V=0,015 pour une ancre de 0,472, soit 31× trop petit -> valeur PLATE -> argmax = bruit ->
    # l'agent errait et mourait de faim (A/B closed-loop : 0 repas).
    anchor = float(ate.sum()) / len(energy) / (1.0 - args.gamma)

    def n_step_targets(z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Retours n-PAS sur toute la fenêtre du rêve, par récurrence arrière :
              G_{H-1} = V_cible(z_{H-1})            (bootstrap TERMINAL)
              G_d     = r_d + γ · G_{d+1}
        En TD 1-pas la valeur ne remonte QUE d'un tick par mise à jour — avec γ=0.999 il faut ~1000
        remontées, d'où la sous-propagation mesurée. Ici toute la fenêtre de H pas remonte en UNE
        mise à jour. Cible détachée + réseau-cible retardé (sinon la valeur poursuit sa propre sortie).
        """
        with torch.no_grad():
            g = target(z[:, -1])
            outs = []
            for d in range(z.shape[1] - 2, -1, -1):
                g = r[:, d] + args.gamma * g
                outs.append(g)
            return torch.stack(outs[::-1], dim=1)

    def td_loss(z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return ((model(z[:, :-1]) - n_step_targets(z, r)) ** 2).mean()

    print(f"  ancre E[V] = taux/(1-γ) = {anchor:.3f} | retours {args.horizon}-pas | "
          f"{args.steps} pas de gradient")
    for step in range(args.steps):
        model.train()
        idx = torch.randperm(len(ltr))[:256]
        opt.zero_grad()
        loss = td_loss(ltr[idx], rtr[idx])
        loss.backward()
        opt.step()
        with torch.no_grad():                                # EMA : cible = copie lente de V
            for p, pt in zip(model.parameters(), target.parameters()):
                pt.mul_(1 - args.tau).add_(args.tau * p)
        if (step + 1) % max(1, args.steps // 10) == 0:
            model.eval()
            with torch.no_grad():
                vte = model(lte)
            m = float(vte.mean())
            print(f"  pas {step + 1:6d} | loss {float(loss):.5f} | V held-out moy {m:.3f} "
                  f"écart-type {float(vte.std()):.3f} | {m / anchor * 100:5.1f} % de l'ancre")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "latent_dim": latent_dim, "hidden": 256,
        "meta": {
            "gamma": args.gamma, "horizon": args.horizon, "corpus": [str(c) for c in args.corpus],
            "wm": str(args.wm),
            "form": "TD-MPC : valeur TERMINALE, entraînée par bootstrapping (réseau-cible EMA)",
            "reward": "1 au tick d'un repas -> V = repas futurs actualisés",
            "usage": "score du candidat = γ^H · V(latent rêvé FINAL), jamais une moyenne",
            "warning": "rien ici ne juge le critique — juge = A/B PLEINE-POLITIQUE ; l'intra-etat ne peut que DISQUALIFIER",
            "anchor": anchor,
        },
    }, out / "critic_best.pt")
    print(f"\n  -> {out / 'critic_best.pt'}")
    print("  RAPPEL : juger avec diag_critic_intra_state.py --critic-type td (intra-état).")


if __name__ == "__main__":
    main()
