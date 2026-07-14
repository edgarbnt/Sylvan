"""SONDE GRATUITE — l'exploration produit-elle des VIES VARIEES ? (2026-07-15)

Le critique-correction (residu) a echoue son gate parce que le corpus = 57 vies sous UNE politique
deterministe : l'entite revit la meme vie, rien a apprendre. On veut de la DIVERSITE. Cette sonde
compare deux corpus (ex. exploration OFF vs persistante) et dit s'ils different VRAIMENT.

TROIS mesures par vie (reconstruites depuis les replans loggues : energie/soif + food/water ego + wm.cmd) :
  1. SURVIE       : nb de replans avant la mort (segments coupes aux respawns). Sa DISPERSION entre vies
                    dit si les vies ont des issues variees.
  2. COMMANDES    : ecart-type realise de omega. L'exploration doit l'AUGMENTER (sinon elle est lavee).
  3. ERRANCE      : (chemin parcouru vers la ressource) / (rapprochement net). 1.0 = ligne droite ;
                    >1 = detours. ⚠️ POLLUEE (constate 2026-07-15) : un refill (+0.4 < seuil de coupe 0.5)
                    ne coupe PAS la vie -> la ressource respawn DANS le segment -> la distance saute ->
                    errance artificiellement ~7-10. NE PAS s'y fier tel quel ; le signal PROPRE est la
                    DISPERSION DE SURVIE (si l'explo diversifie, des vies plus variees -> dispersion UP).

VERDICT (revise 2026-07-15) : la DISPERSION DE SURVIE est le juge (diversite des ISSUES). Si l'explo la
FAIT MONTER -> vies variees. Si elle la fait BAISSER -> l'explo COMPRIME (rend tout le monde un peu pire,
sans structure) = negatif. omega_std est trompeur (distrib bimodale ±0.6/0 -> smearing baisse le std).

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_life_diversity.py --selfcheck
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_life_diversity.py \
      --a data/replay_buffer/expl_off --b data/replay_buffer/expl_on
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path


def _segments(path: Path) -> list[list[dict]]:
    """Un fichier de replans → liste de vies (coupees aux respawns : les 2 drives remontent d'un coup)."""
    rows = []
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        o, p, w = r.get("obs") or {}, r.get("plan") or {}, r.get("wm") or {}
        if "energy" not in o:
            continue
        rows.append({"e": float(o["energy"]) / 100.0, "t": float(o["thirst"]) / 100.0,
                     "food": p.get("food"), "water": p.get("water"), "cmd": w.get("cmd")})
    segs, cur = [], []
    for row in rows:
        if cur and (row["e"] - cur[-1]["e"] > 0.5 or row["t"] - cur[-1]["t"] > 0.5):
            segs.append(cur)
            cur = []
        cur.append(row)
    if cur:
        segs.append(cur)
    return [s for s in segs if len(s) >= 8]


def _wander(seg: list[dict]) -> float | None:
    """Errance vers la ressource URGENTE : distance ego parcourue / rapprochement net. ~1 = droit."""
    urgent = "food" if seg[0]["e"] <= seg[0]["t"] else "water"
    ds = [math.hypot(*r[urgent]) for r in seg if r.get(urgent)]
    if len(ds) < 4:
        return None
    net = ds[0] - min(ds)                         # rapprochement net atteint
    path = sum(abs(ds[i] - ds[i - 1]) for i in range(1, len(ds)))   # variation totale de distance
    return path / net if net > 0.3 else None      # net trop faible → jamais approche, non mesurable


def summarize(glob_dir: str) -> dict:
    files = sorted(Path().glob(glob_dir + "*/ep_0000.jsonl")) or sorted(Path().glob(glob_dir + "/ep_0000.jsonl"))
    lives, omegas, wanders = [], [], []
    for f in files:
        for seg in _segments(f):
            lives.append(len(seg))
            omegas += [r["cmd"][1] for r in seg if r.get("cmd") and len(r["cmd"]) == 2]
            w = _wander(seg)
            if w is not None:
                wanders.append(w)
    return {"n_lives": len(lives),
            "surv_med": st.median(lives) if lives else 0,
            "surv_std": st.pstdev(lives) if len(lives) > 1 else 0.0,
            "omega_std": st.pstdev(omegas) if len(omegas) > 1 else 0.0,
            "wander_med": st.median(wanders) if wanders else float("nan"),
            "n_wander": len(wanders)}


def selfcheck() -> None:
    straight = [{"e": 0.5, "t": 0.9, "food": [0.0, 3.0 - 0.1 * i], "water": None, "cmd": [0.6, 0.0]}
                for i in range(20)]
    assert abs(_wander(straight) - 1.0) < 1e-6, "approche droite → errance 1.0"
    # detour : s'eloigne puis revient → errance > 1
    detour = ([{"e": 0.5, "t": 0.9, "food": [0.0, 3.0 + 0.1 * i], "water": None, "cmd": [0.6, 0.3]}
               for i in range(10)]
              + [{"e": 0.5, "t": 0.9, "food": [0.0, 4.0 - 0.2 * i], "water": None, "cmd": [0.6, -0.3]}
                 for i in range(18)])
    assert _wander(detour) > 1.5, "detour → errance nettement > 1"
    print("[selfcheck] OK : errance = 1.0 en ligne droite, >1.5 avec detour")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="data/replay_buffer/expl_off", help="corpus reference (ex. exploration OFF)")
    ap.add_argument("--b", default="data/replay_buffer/expl_on", help="corpus teste (ex. exploration ON)")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    A, B = summarize(args.a), summarize(args.b)
    print(f"\n{'':16}{'A (' + Path(args.a).name + ')':>22}{'B (' + Path(args.b).name + ')':>22}")
    print("-" * 60)
    for key, lab, fmt in [("n_lives", "vies", "d"), ("surv_med", "survie med (replans)", ".0f"),
                          ("surv_std", "survie dispersion", ".0f"), ("omega_std", "omega std realise", ".3f"),
                          ("wander_med", "errance med (1=droit)", ".2f"), ("n_wander", "vies mesurables", "d")]:
        print(f"{lab:16}{format(A[key], fmt):>22}{format(B[key], fmt):>22}")
    print("-" * 60)
    print("\n--- VERDICT (juge = DISPERSION DE SURVIE ; errance polluee, omega_std trompeur) ---")
    spread_ratio = B["surv_std"] / A["surv_std"] if A["surv_std"] > 1e-9 else float("nan")
    if spread_ratio > 1.15:
        print(f"  ✅ B DIVERSIFIE LES ISSUES : dispersion de survie ×{spread_ratio:.2f} (>1.15).")
        print("     → vies plus variees. Lancer la collecte complete pour le re-gate residu.")
    elif spread_ratio < 0.87:
        print(f"  ❌ B COMPRIME : dispersion de survie ×{spread_ratio:.2f} (<0.87). L'explo rend tout le")
        print("     monde un peu PIRE sans structure (bruit, pas diversite). Mauvais levier : le monde")
        print("     est simple, la politique quasi-optimale → perturber ne peut que degrader.")
    else:
        print(f"  ⚠️ B ≈ A : dispersion de survie ×{spread_ratio:.2f} (plate). Pas d'effet net.")


if __name__ == "__main__":
    main()
