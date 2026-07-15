"""SONDE GRATUITE — un slot requête-couleur localise-t-il le DANGER sur le WM GELÉ, sans retrain ?

CONTEXTE (docs/etat_critique.md). Le danger est perceptible (rétine capte le VERT). Le slot lit la
RÉTINE BRUTE, pas le latent (command_wm.py:130) → sa localisation est une attention géométrique par
COULEUR, sans paramètre appris (le scoreur est retiré des logits, slot_head.py:85). DONC un 3ᵉ slot
requête-VERT devrait localiser le danger SANS ré-entraîner le WM — SI le vert ne se confond pas avec
le rouge(bouffe)/bleu(eau). Les requêtes sont pure-canal, seuil cosinus 0.55.

CE QUE LA SONDE VÉRIFIE (aucun entraînement, aucun checkpoint, déterministe CPU) :
  A. SÉPARATION COULEUR (matrice de confusion) : chaque rayon qui a touché un objet coloré est étiqueté
     par sa VRAIE couleur rendue (rouge/bleu/vert = vérité-terrain, le retina_color est lu brut non
     éclairé), puis on regarde quelles requêtes il déclenche (cos>0.55). PROPRE = bloc-diagonal :
     rouge→rouge seul, bleu→bleu seul, vert→vert seul. Toute case hors-diagonale = FUITE (corruption).
  B. NON-CORRUPTION : positions bouffe/eau IDENTIQUES avec un slot-head 2-ressources vs 3-ressources
     (ajouter la requête verte ne doit PAS bouger les slots existants — requêtes indépendantes).
  C. LOCALISATION : le slot vert a-t-il une SAILLANCE haute quand le danger est en vue (position finie,
     distincte de bouffe/eau) ? La justesse géométrique est garantie par construction (argmin-souple
     sur les rayons verts = bord du danger) — on vérifie surtout qu'il s'ALLUME.

CRITÈRE (écrit AVANT) : SUCCÈS si (A) 0 fuite croisée ET (B) bouffe/eau bit-identiques ET (C) saillance
verte haute quand le danger est vu → le slot-danger marche SUR LE WM GELÉ, ZÉRO RETRAIN (leçon slot-1).
ÉCHEC si fuite → mauvaise couleur (revoir) ; si (C) faible → le danger n'est pas assez vu (revoir taille).

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_hazard_slot.py --selfcheck
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_hazard_slot.py --glob 'data/replay_buffer/hazslot'
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
from pathlib import Path

import torch

from sylvan.models.slot_head import NRAY, SelfSupervisedSlotHead

REF = {"rouge(bouffe)": (0.9, 0.3, 0.2), "bleu(eau)": (0.2, 0.5, 0.95), "vert(danger)": (0.1, 0.9, 0.15)}
QUERIES = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])  # rouge, bleu, vert (pur canal)
QNAMES = ["rouge", "bleu", "vert"]
THRESH = 0.55


def classify(rgb: tuple[float, float, float]) -> str | None:
    """Vérité-terrain : la couleur RENDUE (retina_color lu brut) → quel objet. None si rayon vide."""
    if max(rgb) - min(rgb) < 0.15:            # non saturé = pas d'objet coloré (miss / sol)
        return None
    best, bd = None, 1e9
    for name, ref in REF.items():
        d = sum((rgb[i] - ref[i]) ** 2 for i in range(3))
        if d < bd:
            bd, best = d, name
    return best


def fired_queries(rgb: tuple[float, float, float]) -> list[str]:
    v = torch.tensor(rgb)
    vn = v / (v.norm() + 1e-6)
    cos = QUERIES @ vn
    return [QNAMES[k] for k in range(3) if float(cos[k]) > THRESH]


def selfcheck() -> None:
    # rouge pur ne déclenche QUE la requête rouge ; idem bleu, vert
    assert fired_queries((0.9, 0.3, 0.2)) == ["rouge"], fired_queries((0.9, 0.3, 0.2))
    assert fired_queries((0.2, 0.5, 0.95)) == ["bleu"]
    assert fired_queries((0.1, 0.9, 0.15)) == ["vert"]
    # le VIOLET (l'ancien mauvais choix) fuit dans rouge ET bleu — c'est ce que la sonde a évité
    assert set(fired_queries((0.6, 0.12, 0.85))) == {"rouge", "bleu"}, fired_queries((0.6, 0.12, 0.85))
    assert classify((0.1, 0.9, 0.15)) == "vert(danger)" and classify((0.5, 0.5, 0.5)) is None
    print("[selfcheck] OK : requêtes pure-canal séparent rouge/bleu/vert ; le violet fuit (rouge+bleu)")


def load_retina(dirs: list[str], n: int = 4000) -> torch.Tensor:
    rows = []
    for d in dirs:
        f = Path(d) / "ep_0000.jsonl"
        if not f.exists():
            continue
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            ret = (r.get("wm") or {}).get("retina0")
            if ret and len(ret) == NRAY * 4:
                rows.append(ret)
            if len(rows) >= n:
                break
    return torch.tensor(rows, dtype=torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/replay_buffer/hazslot")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return

    ret = load_retina(sorted(globmod.glob(args.glob)))
    if ret.numel() == 0:
        print(f"AUCUNE rétine dans {args.glob}")
        return
    F = ret.shape[0]
    r = ret.reshape(F, NRAY, 4)

    # ── A. MATRICE DE CONFUSION couleur → requêtes déclenchées ──────────────────────────────────
    classes = list(REF) + ["(aucune requête)"]
    conf = {c: {q: 0 for q in QNAMES + ["aucune"]} for c in REF}
    seen = {c: 0 for c in REF}
    for f in range(F):
        for k in range(NRAY):
            rgb = (float(r[f, k, 1]), float(r[f, k, 2]), float(r[f, k, 3]))
            cls = classify(rgb)
            if cls is None:
                continue
            seen[cls] += 1
            fired = fired_queries(rgb)
            if not fired:
                conf[cls]["aucune"] += 1
            for q in fired:
                conf[cls][q] += 1

    print(f"\n=== A. SÉPARATION COULEUR — {F} frames, {NRAY} rayons ===")
    print(f"{'vrai objet':16}{'→rouge':>8}{'→bleu':>8}{'→vert':>8}{'→aucune':>9}   (rayons vus)")
    leaks = 0
    for c in REF:
        row = conf[c]
        print(f"{c:16}{row['rouge']:>8}{row['bleu']:>8}{row['vert']:>8}{row['aucune']:>9}   {seen[c]}")
        # fuite = déclenche une requête qui n'est PAS la sienne
        own = {"rouge(bouffe)": "rouge", "bleu(eau)": "bleu", "vert(danger)": "vert"}[c]
        leaks += sum(row[q] for q in QNAMES if q != own)
    print(f"→ fuites croisées (hors-diagonale) : {leaks}  {'✅ AUCUNE' if leaks == 0 else '❌ CORRUPTION'}")

    # ── B. NON-CORRUPTION bouffe/eau (2-res vs 3-res) ───────────────────────────────────────────
    torch.manual_seed(0)
    with torch.no_grad():
        h2 = SelfSupervisedSlotHead(n_resources=2).eval()
        h3 = SelfSupervisedSlotHead(n_resources=3).eval()
        p2, s2 = h2.positions_and_salience(ret)          # [F,2,2],[F,2]
        p3, s3 = h3.positions_and_salience(ret)          # [F,3,2],[F,3]
    dfw = float((p2 - p3[:, :2, :]).abs().max())
    print(f"\n=== B. NON-CORRUPTION bouffe/eau ===")
    print(f"écart max position bouffe+eau (slot-head 2-res vs 3-res) : {dfw:.2e} m  "
          f"{'✅ intactes' if dfw < 1e-4 else '❌ modifiées'}")

    # ── C. LOCALISATION du slot vert (danger) ───────────────────────────────────────────────────
    sal_g = s3[:, 2]                                     # saillance du slot vert
    visible = sal_g > 1e-3
    print(f"\n=== C. LOCALISATION du slot DANGER (vert) ===")
    print(f"danger vu (saillance>0) : {int(visible.sum())}/{F} frames ({100*float(visible.float().mean()):.0f}%)")
    if int(visible.sum()) > 0:
        pv = p3[visible, 2, :]
        dist = pv.norm(dim=-1)
        print(f"quand vu : distance médiane {float(dist.median()):.2f} m, "
              f"saillance médiane {float(sal_g[visible].median()):.3f}")
        # distinct de bouffe/eau ? (les 3 slots ne doivent pas pointer le même endroit)
        d_gf = float((p3[visible, 2, :] - p3[visible, 0, :]).norm(dim=-1).median())
        print(f"distance médiane slot-danger ↔ slot-bouffe : {d_gf:.2f} m (distinct si > ~0.3)")

    print("\n--- VERDICT (critère écrit AVANT) ---")
    ok = leaks == 0 and dfw < 1e-4 and int(visible.sum()) > 0.4 * F
    if ok:
        print("  ✅ SLOT-DANGER OK SUR LE WM GELÉ, ZÉRO RETRAIN : 0 fuite, bouffe/eau intactes, danger")
        print("     bien localisé quand vu. La leçon slot-1 tient : le slot était déjà là. Le gros retrain")
        print("     WM n'est PAS nécessaire pour PERCEVOIR le danger → brancher le 3ᵉ slot dans le planner.")
    else:
        print(f"  ⚠️ à revoir : fuites={leaks}, écart bouffe/eau={dfw:.1e}, vu={int(visible.sum())}/{F}")


if __name__ == "__main__":
    main()
