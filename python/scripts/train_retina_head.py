"""Étage 1 RÉTINE — entraîne la tête de perception apprise (🅐) : rétine BRUTE → position ressource.

Label = vraie position du simulateur (food_rel0/water_rel0), PAS l'oracle radar. À l'éval, la tête ne voit
QUE la rétine (oracle débranché) → on mesure la MAE de position en mètres = le GATE falsifiable (< 0.5 m)
avant de payer le retrain WM (étage 2). cf docs/scope_retina.md §5.

Usage:
  PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.train_retina_head \
    --runs data/replay_buffer/retina_head_a data/replay_buffer/retina_head_b \
    --out data/checkpoints/retina_head --epochs 40
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import torch
import torch.nn.functional as F

from sylvan.models.perception_head import RetinaPerceptionHead, RETINA_DIM

POS_SCALE = 10.0  # cibles normalisées par la portée (≈ MAX_RANGE) → optimisation Adam plus stable


def _visible_color(ret: list[float], color: str) -> float:
    """La ressource de cette COULEUR est-elle VISIBLE (un rayon touche un hit de la bonne dominante) ?
    Sémantique honnête : la tête reporte ce qu'elle VOIT, pas ce qui existe dans le monde (une cible entre
    2 rayons / occluse n'est pas perçue) → supervise la position seulement quand visible (pas de cible
    contradictoire). NB : la couleur sert ici à SÉLECTIONNER les échantillons (masque/label), pas d'entrée
    au modèle — la tête doit toujours APPRENDRE couleur→sens depuis la rétine brute."""
    for k in range(0, len(ret), 4):
        d, rr, gg, bb = ret[k], ret[k + 1], ret[k + 2], ret[k + 3]
        if d < 0.999:
            if color == "red" and rr > gg and rr > bb and rr > 0.3:
                return 1.0
            if color == "blue" and bb > rr and bb > gg and bb > 0.3:
                return 1.0
    return 0.0


def _nearest_ray(ret: list[float], color: str) -> int:
    """Index du rayon de la bonne COULEUR avec la profondeur MINIMALE (= cible visible la plus proche),
    -1 si aucun. Cible de la supervision d'attention : enseigne « pointe la ressource la plus proche »
    (même politique que l'oracle food_xz_from_radar). Sélection d'échantillon, pas une entrée du modèle."""
    best_k, best_d = -1, 1e9
    for k in range(0, len(ret), 4):
        ri = k // 4
        d, rr, gg, bb = ret[k], ret[k + 1], ret[k + 2], ret[k + 3]
        if d >= 0.999:
            continue
        is_c = (color == "red" and rr > gg and rr > bb and rr > 0.3) or \
               (color == "blue" and bb > rr and bb > gg and bb > 0.3)
        if is_c and d < best_d:
            best_d, best_k = d, ri
    return best_k


def _load(runs: list[str], n_resources: int):
    """Renvoie X[retina], Ypos[n_res,2], Yvis[n_res], et l'index d'épisode (pour split propre).
    Yvis = visibilité dans la rétine (cible présence + masque de la perte position)."""
    X, Ypos, Ypres, Yray, ep_id = [], [], [], [], []
    files = []
    for d in runs:
        files += sorted(glob.glob(os.path.join(d, "**", "episode_*.jsonl"), recursive=True))
        files += sorted(glob.glob(os.path.join(d, "episode_*.jsonl")))
    files = sorted(set(files))
    keys = ["food_rel0"] + (["water_rel0"] if n_resources > 1 else [])
    for ei, f in enumerate(files):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                wm = r.get("wm", {})
                ret = wm.get("retina0")
                if not ret or len(ret) != RETINA_DIM:
                    continue
                pos_row, pres_row, ray_row, ok = [], [], [], True
                for ki, k in enumerate(keys):
                    rel = wm.get(k) or []
                    if len(rel) != 3:
                        ok = False
                        break
                    color = "red" if ki == 0 else "blue"
                    pos_row.append([rel[0], rel[1]])
                    # présence = VISIBILITÉ dans la rétine de la couleur de cette ressource (food=rouge,
                    # water=bleu), PAS l'existence-monde. Masque la perte position aux échantillons vus.
                    pres_row.append(_visible_color(ret, color))
                    ray_row.append(_nearest_ray(ret, color))  # cible attention (-1 si non visible)
                if not ok:
                    continue
                X.append(ret)
                Ypos.append(pos_row)
                Ypres.append(pres_row)
                Yray.append(ray_row)
                ep_id.append(ei)
    if not X:
        raise SystemExit("Aucun enregistrement avec retina0 + labels trouvé dans %s" % runs)
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Ypos, dtype=torch.float32),
        torch.tensor(Ypres, dtype=torch.float32),
        torch.tensor(Yray, dtype=torch.long),
        torch.tensor(ep_id, dtype=torch.long),
        len(files),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="data/checkpoints/retina_head")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--n-resources", type=int, default=1)  # 1=bouffe, 2=bouffe+eau
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    args = ap.parse_args()

    torch.manual_seed(0)
    X, Ypos, Ypres, Yray, ep, n_files = _load(args.runs, args.n_resources)
    n_ep = int(ep.max().item()) + 1
    n_hold = max(1, int(round(n_ep * args.holdout_frac)))
    hold_eps = set(range(n_ep - n_hold, n_ep))  # derniers épisodes = held-out
    is_hold = torch.tensor([int(e.item()) in hold_eps for e in ep])
    tr, te = ~is_hold, is_hold
    print(f"[retina-head] {len(X)} records | {n_files} files | {n_ep} eps "
          f"| train {int(tr.sum())} / held-out {int(te.sum())} (eps {sorted(hold_eps)})")

    model = RetinaPerceptionHead(n_resources=args.n_resources)
    model.pos_scale = torch.tensor(POS_SCALE)  # locate() remettra en mètres
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    Xtr, Ptr, Ctr, Rtr = X[tr], Ypos[tr], Ypres[tr], Yray[tr]
    n = Xtr.shape[0]

    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        model.train()
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            out = model(Xtr[idx])
            pres = Ctr[idx]  # [b, n_res] = visibilité (masque + cible présence)
            pos_err = (out["pos"] - Ptr[idx] / POS_SCALE) ** 2  # [b, n_res, 2], espace normalisé
            pos_loss = (pos_err.sum(-1) * pres).sum() / (pres.sum().clamp(min=1.0))
            conf_loss = F.binary_cross_entropy_with_logits(out["conf_logit"], pres)
            # ATTENTION : pousse le scoreur à mettre EN TÊTE le rayon de la ressource la + proche
            # (sinon le soft-argmax moyenne entre plusieurs cibles → ~1-2 m). CE sur les samples visibles.
            scores = out["scores"]  # [b, n_rays, n_res]
            ray_tgt = Rtr[idx]      # [b, n_res], -1 si non visible
            attn_loss = scores.new_zeros(())
            for res in range(args.n_resources):
                m_ok = ray_tgt[:, res] >= 0
                if m_ok.any():
                    attn_loss = attn_loss + F.cross_entropy(scores[m_ok, :, res], ray_tgt[m_ok, res])
            loss = pos_loss + conf_loss + attn_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            mae = _eval_mae(model, X[te], Ypos[te], Ypres[te])
            print(f"  epoch {epoch:3d} | train_loss {tot / n:.4f} | held-out MAE_pos {mae:.3f} m")

    mae = _eval_mae(model, X[te], Ypos[te], Ypres[te])
    pres_acc = _eval_pres(model, X[te], Ypres[te])
    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, "head_best.pt")
    torch.save({"state_dict": model.state_dict(),
                "n_resources": args.n_resources,
                "retina_dim": RETINA_DIM,
                "heldout_mae_m": mae,
                "heldout_pres_acc": pres_acc}, ckpt)
    print(f"[retina-head] DONE | held-out MAE_pos = {mae:.3f} m | present_acc = {pres_acc:.3f}")
    gate = 0.5
    print(f"[retina-head] GATE (<{gate} m) : {'PASS ✅' if mae < gate else 'FAIL ❌'}  -> {ckpt}")


@torch.no_grad()
def _eval_mae(model, X, Ypos, Ypres) -> float:
    if len(X) == 0:
        return float("nan")
    model.eval()
    out = model(X)
    # MAE euclidienne en mètres sur les ressources VISIBLES uniquement (pos remise en mètres)
    d = torch.sqrt(((out["pos"] * POS_SCALE - Ypos) ** 2).sum(-1) + 1e-9)  # [b, n_res]
    mask = Ypres > 0.5
    if mask.sum() == 0:
        return float("nan")
    return float((d[mask]).mean())


@torch.no_grad()
def _eval_pres(model, X, Ypres) -> float:
    if len(X) == 0:
        return float("nan")
    model.eval()
    pred = (model(X)["conf"] > 0.5).float()
    return float((pred == (Ypres > 0.5).float()).float().mean())


if __name__ == "__main__":
    main()
