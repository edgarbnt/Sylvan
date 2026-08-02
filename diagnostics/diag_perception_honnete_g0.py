"""G0 GRATUIT — un signal du CAPTEUR sait-il dire « je n'ai rien sous les yeux » ?

Gate pré-inscrit dans `docs/design_perception_honnete.md`. Zéro Godot, zéro entraînement lourd.

LA QUESTION. L'entité invente une position 61 % du temps et ne le sait pas : sa jauge de visibilité
servie classe correctement 56 % des cas, soit à peine mieux que pile ou face. Existe-t-il, DANS LA
RÉTINE, un signal qui le dise ?

⚠️ POURQUOI PAS UNE CONFIANCE TIRÉE DU MODÈLE (réfuté AVANT de coder, deux raisons indépendantes) :
  1. négatif déjà banké du projet — la consistance de transport seule VERROUILLE SUR LES TRONCS
     (résidu = prey_speed x gap ; un arbre immobile est plus consistant qu'une proie qui fuit) ;
  2. « Why Model Uncertainty Fails as a Risk Signal » (arXiv 2607.16591) — pénaliser l'incertitude
     du MODÈLE augmente les collisions (26 % → 34 %) car elle est anticorrélée au danger réel
     (r < 0,15) : elle pousse vers le PRÉVISIBLE, pas vers le SÛR.
  ⇒ les candidats testés ici sont TOUS calculés depuis la rétine seule (world feedback).

BARRES PRÉ-ENREGISTRÉES (écrites avant de regarder les données) :
  G0-honnête  meilleur signal >= 80 % de bon classement — à 67 % (déjà mesuré en juillet) elle se
              tromperait une fois sur trois et abandonnerait une vraie proie aussi souvent, ce qui
              coûte plus que ça ne rapporte sur un budget de ~350 pas de vie.
  G0-contrôle la jauge SERVIE reste vers 56 % (sinon il n'y a rien à améliorer).
  🛑 STOP     meilleur signal < 70 % ⇒ le capteur ne porte pas l'information, ne pas payer la suite.

⚠️ CORRECTION POUR COMPARAISONS MULTIPLES OBLIGATOIRE : on teste ~6 signaux et on garde le meilleur.
   Le max sous l'hypothèse nulle est estimé PAR PERMUTATION — c'est la faute exacte commise le
   2026-08-02 sur le G0 critique-de-rang (seuil posé sur un max non corrigé).

⚠️ `food_rel0` est un ORACLE D'ÉVAL : il sert à fabriquer la vérité « un rayon touche-t-il vraiment
   la proie ? », jamais à alimenter un signal candidat.

Usage :
    PYTHONPATH=python SYLVAN_RETINA_FOV_DEG=120 env_pytorch_3.12/bin/python \
        diagnostics/diag_perception_honnete_g0.py
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import sys

import torch

sys.path.insert(0, "diagnostics")
from diag_drive_corpus import true_food_rays  # géométrie VALIDÉE, ne pas la recoder
from sylvan.models.command_wm import CommandWorldModel

BAR_OK = 0.80
STOP_OK = 0.70


def auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """Taux de bon classement (Mann-Whitney)."""
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    allv = torch.cat([pos, neg])
    rank = allv.argsort().argsort().float() + 1.0
    n1, n2 = float(pos.numel()), float(neg.numel())
    return float((rank[: pos.numel()].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=[f"data/replay_buffer/sp2_ref{s}" for s in (1, 2, 3)])
    ap.add_argument("--wm", default="data/checkpoints/wm_foret_v2_slot/wm_best.pt")
    ap.add_argument("--question", choices=("plus-proche", "une-proie"), default="plus-proche")
    args = ap.parse_args()

    payload = torch.load(args.wm, map_location="cpu", weights_only=False)
    wm = CommandWorldModel.from_checkpoint(payload)
    wm.eval()
    wm.requires_grad_(False)
    se = wm.slot_encoder

    X, F = [], []
    for run in args.runs:
        for f in sorted(glob.glob(f"{run}/*.jsonl")):
            for line in open(f):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                w = r.get("wm")
                if not w:
                    continue
                ret = w.get("retina0")
                if not ret or len(ret) != 144:
                    continue
                X.append(ret)
                F.append(w.get("food_rel0") or [0.0, 0.0, 0.0])
    X = torch.tensor(X)
    F = torch.tensor(F).view(-1, 3)
    print(f"{len(X)} ticks\n")

    # ── DEUX FORMULATIONS DE LA VÉRITÉ, jugées côte à côte ────────────────────────────────────
    # (1) « LA PLUS PROCHE » : un rayon touche-t-il la proie que `food_rel0` désigne ?
    # (2) « UNE PROIE »      : la position LUE par le slot est-elle réellement occupée ?
    #     Reformulation demandée le 2026-08-02 : `food_rel0` ne désigne que la proie la plus
    #     proche, donc la question (1) compte comme « invention » des cas où l'entité voit
    #     correctement une AUTRE proie. La question (2) n'est PAS tautologique : le slot rend un
    #     BARYCENTRE, qui peut tomber ENTRE deux proies, à un endroit où il n'y a rien.
    #     Vérité (2) : la position du slot coïncide-t-elle (< 0,6 m) avec l'extrémité d'un rayon
    #     touchant DE COULEUR DE PROIE (palette servie du monde — oracle d'ÉVAL) ?
    lab, vis_cone = true_food_rays(X.view(-1, 36, 4), F)
    touches = lab.any(dim=-1)
    keep = vis_cone            # on ne juge que là où la cible est dans le champ
    y = touches[keep]
    if args.question == "une-proie":
        from diag_drive_corpus import FOOD_HUES, RETINA_RANGE_M, DEPTH_OFFSET, ray_angles
        pal = torch.tensor(FOOD_HUES)
        pal = pal / pal.norm(dim=-1, keepdim=True)
        rr = X.view(-1, 36, 4)
        dd, cc = rr[..., 0], rr[..., 1:]
        ccn = cc / (cc.norm(dim=-1, keepdim=True) + 1e-6)
        is_prey = ((ccn @ pal.T).amax(-1) > 0.98) & (dd < 0.999)
        th = ray_angles()
        dm = dd * RETINA_RANGE_M + DEPTH_OFFSET
        px, pz = dm * torch.sin(th), dm * torch.cos(th)
        with torch.no_grad():
            slot = wm.slot_encoder.positions(X)[:, 0, :]
        d2 = torch.sqrt((px - slot[:, 0:1]) ** 2 + (pz - slot[:, 1:2]) ** 2)
        d2 = torch.where(is_prey, d2, torch.full_like(d2, 1e9))
        y = (d2.amin(-1) < 0.6)[keep]
        print("  QUESTION : « la position lue est-elle occupée par UNE proie ? »")
    else:
        print("  QUESTION : « un rayon touche-t-il LA proie la plus proche ? »")
    print(f"  jugés : {int(keep.sum())} ticks (cible dans le cône)")
    print(f"  dont un rayon touche VRAIMENT : {100 * float(y.float().mean()):.1f} %\n")
    if int(y.sum()) < 100 or int((~y).sum()) < 100:
        print("❌ classes trop déséquilibrées pour juger")
        return

    # --- SIGNAUX CANDIDATS, tous calculés depuis la RÉTINE seule -------------------------------
    with torch.no_grad():
        dist, sal, a_list = se._attend(X)
        served = se.visibility(X)[:, 0]
    att = a_list[0]                                  # attention softmax sur les 36 rayons
    r = X.view(-1, 36, 4)
    depth, rgb = r[..., 0], r[..., 1:]
    rgbn = rgb / (rgb.norm(dim=-1, keepdim=True) + 1e-6)
    q = payload["model"]["slot_encoder.color_queries"][0]
    thr = float(payload["meta"]["query_thr"][0])
    flagged = ((rgbn @ q) > thr) & (depth < 0.999)
    top2 = att.topk(2, dim=-1).values

    cand = {
        "nb de rayons retenus": flagged.float().sum(-1),
        "masse de saillance": (sal * flagged.float()).sum(-1),
        "piqué de l'attention (max)": att.amax(-1),
        "-entropie de l'attention": (att * (att + 1e-9).log()).sum(-1),
        "écart 1er-2e pic": top2[:, 0] - top2[:, 1],
        "proximité du plus proche retenu": (1.0 - torch.where(flagged, depth,
                                                              torch.ones_like(depth)).amin(-1)),
    }

    print("TAUX DE BON CLASSEMENT (50 % = pile ou face, barre 80 %) :")
    scores = {}
    for name, v in cand.items():
        a = auc(v[keep][y], v[keep][~y])
        scores[name] = v[keep]
        flag = "✅" if a >= BAR_OK else ("~" if a >= STOP_OK else "  ")
        print(f"  {flag} {name:34s} {100 * a:5.1f} %")
    a_served = auc(served[keep][y], served[keep][~y])
    print(f"\n  [contrôle] jauge SERVIE            {100 * a_served:5.1f} %   (attendu ~56 %)")

    best_name = max(cand, key=lambda k: auc(cand[k][keep][y], cand[k][keep][~y]))
    best = auc(cand[best_name][keep][y], cand[best_name][keep][~y])

    # --- CORRECTION POUR COMPARAISONS MULTIPLES (permutation) ----------------------------------
    random.seed(0)
    yv = y.clone()
    n_perm, ge = 2000, 0
    null = []
    for _ in range(n_perm):
        idx = torch.randperm(len(yv))
        ys = yv[idx]
        m = max(auc(v[ys], v[~ys]) for v in scores.values())
        null.append(m)
        if m >= best:
            ge += 1
    null.sort()
    print(f"\n  meilleur signal : « {best_name} » à {100 * best:.1f} %")
    print(f"  max attendu PAR HASARD sur {len(cand)} signaux : médiane "
          f"{100 * null[n_perm // 2]:.1f} %  ·  95e centile {100 * null[int(0.95 * n_perm)]:.1f} %")
    print(f"  p (permutation, corrigé) = {ge / n_perm:.3f}")

    print("\n" + "=" * 74)
    if best >= BAR_OK and ge / n_perm < 0.05:
        print(f"✅ G0-honnête PASSÉ — « {best_name} » classe correctement {100 * best:.0f} % des cas.")
        print("   Le capteur PORTE l'information. G1 (gate d'USAGE) est licencié.")
    elif best < STOP_OK:
        print(f"🛑 STOP pré-enregistré — meilleur signal {100 * best:.0f} %, sous la barre de 70 %.")
        print("   Le capteur ne porte pas l'information : quand aucun rayon ne touche la cible,")
        print("   il n'y a rien à extraire. Ne pas payer G1.")
    else:
        print(f"~ ZONE GRISE — {100 * best:.0f} %, entre 70 % et la barre de 80 %.")
        print("   Utilisable en principe, mais elle se tromperait encore trop souvent pour")
        print("   qu'un changement de comportement soit rentable. Décision owner.")


if __name__ == "__main__":
    main()
