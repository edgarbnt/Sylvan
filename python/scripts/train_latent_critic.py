"""CRITIQUE-LATENT — la seule forme qui puisse battre `-min_dist`.

POURQUOI CETTE FORME. Le critique-token (`train_meal_critic.py`) a été REFUSÉ par le juge
intra-état : corrélation de rang +0,93 avec `-min_dist`, donc il avait simplement ré-appris la
géométrie. C'était prévisible — son entrée EST la géométrie (distance, cap, énergie). Sur la
géométrie seule il n'y a prouvablement rien à gagner, puisque c'est exactement ce que le coût
analytique calcule déjà.

Le LATENT du WM, lui, porte la SCÈNE : les autres bosquets, les buissons-marqueurs, ce qui est
derrière, l'occlusion — tout ce que le token à 5 dims jette. C'est donc la seule entrée qui
contienne une information que `-min_dist` n'a pas.

TRAIN = DÉPLOIEMENT (leçon dure du projet). Le planner applique V sur les latents RÊVÉS en
open-loop à des profondeurs 0..H. On entraîne donc sur exactement ça : latents rêvés sous les
commandes RÉELLEMENT exécutées, à profondeurs variées — et non sur des latents teacher-forced,
qui sont une autre distribution.

WM GELÉ : on n'entraîne que la tête (le substrat ne se refond pas pour une pulsion, §3).

⚠️ L'AUC affichée est POOLÉE et ne juge RIEN (cf train_meal_critic). LE juge est
`diagnostics/diag_critic_intra_state.py --critic-type latent`.

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python -m scripts.train_latent_critic \
      [--corpus data/replay_buffer/critic_bosq_a] [--k 200] [--horizon 80]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from sylvan.critic_corpus import auc, load_bc_corpus, meal_flags, residual_label
from sylvan.models.command_wm import CommandWorldModel
from sylvan.models.value_head import ValueHead


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/replay_buffer/critic_bosq_a")
    ap.add_argument("--wm", default="data/checkpoints/wm_objcentric_kin/wm_best.pt")
    ap.add_argument("--out", default="data/checkpoints/latent_critic")
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=80)
    ap.add_argument("--start-stride", type=int, default=8)    # 1 départ sur N (volume)
    ap.add_argument("--depth-stride", type=int, default=4)    # 1 profondeur sur N (couvre 0..H)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    obs, energy, cmds, bounds = load_bc_corpus(args.corpus)
    ate = meal_flags(energy, bounds)
    y_all = residual_label(ate, bounds, args.k)
    n_ep = len(bounds) - 1
    print(f"corpus {args.corpus} : {len(energy)} ticks, {n_ep} épisodes, {int(ate.sum())} repas")

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

    # Départs valides : le rêve de H pas doit rester DANS l'épisode (sinon on rêverait à travers
    # un reset, et le label viendrait d'une autre vie).
    n_tr_ep = max(1, int(round(0.7 * n_ep)))
    split_tick = bounds[n_tr_ep]
    starts, is_train = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        for t in range(a, b - args.horizon - 1, args.start_stride):
            starts.append(t)
            is_train.append(t < split_tick)
    starts_t = torch.tensor(starts)
    print(f"  départs : {len(starts)} (train {sum(is_train)} / held-out {len(starts) - sum(is_train)})")

    depths = list(range(0, args.horizon, args.depth_stride))
    lat_l, lab_l, tr_l = [], [], []
    with torch.no_grad():
        for i in range(0, len(starts), 256):
            idx = starts_t[i:i + 256]
            o0 = obs[idx]                                              # [B, obs_dim]
            seq = torch.stack([cmds[j:j + args.horizon] for j in idx])  # [B, H, 2] commandes VÉCUES
            lat = wm.rollout_open_loop(o0, seq)["predicted_latents"]    # [B, H, latent_dim]
            for d in depths:
                lat_l.append(lat[:, d])
                lab_l.append(y_all[idx + d])
                tr_l.append(torch.tensor([is_train[k] for k in range(i, min(i + 256, len(starts)))]))
    x = torch.cat(lat_l)
    y = torch.cat(lab_l)
    tr = torch.cat(tr_l)
    te = ~tr
    print(f"  latents rêvés : {len(x)} ({len(depths)} profondeurs) | "
          f"cible positive {100 * float(y.mean()):.1f} % | train {int(tr.sum())} / held-out {int(te.sum())}")
    if float(y[te].sum()) == 0:
        raise SystemExit("held-out sans positif -> AUC indéfinie")

    model = ValueHead(latent_dim)
    with torch.no_grad():                                   # stats sur le TRAIN seulement
        model.mu.copy_(x[tr].mean(0))
        model.sd.copy_(x[tr].std(0).clamp_min(1e-6))

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    best_auc, best_state = -1.0, None
    xt, yt = x[tr], y[tr]
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xt))[:8192]               # mini-batch (volume)
        opt.zero_grad()
        loss = lossf(model.logit(xt[perm]), yt[perm])
        loss.backward()
        opt.step()
        if (ep + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                a = auc(model.logit(x[te]), y[te])
            if a > best_auc:
                best_auc, best_state = a, {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  ep {ep + 1:4d} | loss {float(loss):.4f} | AUC POOLÉE held-out {a:.3f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": best_state, "latent_dim": latent_dim, "hidden": 256,
        "meta": {
            "k": args.k, "horizon": args.horizon, "corpus": str(args.corpus), "wm": str(args.wm),
            "input": "latent RÊVÉ open-loop (train = déploiement)",
            "target": f"repas dans {args.k} ticks",
            "auc_pooled_heldout": best_auc,
            "warning": "AUC POOLÉE, ne juge PAS — juge = diag_critic_intra_state.py --critic-type latent",
        },
    }, out / "critic_best.pt")
    print(f"\n  -> {out / 'critic_best.pt'} | AUC POOLÉE held-out {best_auc:.3f} (ne juge RIEN)")


if __name__ == "__main__":
    main()
