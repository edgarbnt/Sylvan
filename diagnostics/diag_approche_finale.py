"""Pourquoi l'entité meurt de faim à un mètre de sa nourriture — l'approche TERMINALE.

Né d'une observation de l'owner (2026-08-02) devant `scripts/voir_faim_ab.sh` : « elle change
énormément d'avis, elle reste dans un rayon assez petit, elle fait plein d'aller-retours, elle
ne va pas chercher la bouffe ». Ce diagnostic mesure ce qui se passe VRAIMENT, et le verdict
n'est aucune des trois causes qu'on soupçonnait.

CE QUE ÇA MESURE, dans l'ordre où ça a fait tomber les hypothèses :
  1. La nourriture est-elle loin ?      → NON : à moins de 10 m 100 % du temps, à moins de 3 m
                                           45 % du temps, vue dans 50 % des ticks à 2,93 m.
                                           « elle ne va pas la chercher » est donc FAUX : elle
                                           est déjà à côté.
  2. À quelle distance cale-t-elle ?    → à 1,00 / 1,00 / 1,02 m (q10/q25/q50), pour un rayon de
                                           bouche de 1,00 m. Un empilement sur une valeur EXACTE
                                           n'est pas du bruit de visée.
  3. Est-elle bloquée physiquement ?    → NON. Le rendement du corps au bord (0,100 m par unité
                                           demandée) est MEILLEUR qu'en approche libre (0,078).
                                           Ce qui tombe, c'est la vitesse DEMANDÉE : 0,600 → 0,250.
                                           C'est une DÉCISION : le planner se croit arrivé.

LA CAUSE, géométrique et vérifiable dans le code du monde :
  · les baies forment une COURONNE de rayon 0,95 m (`food_manager.gd:_patch_radius`) autour d'un
    buisson-marqueur de rayon 0,55 m ;
  · ce que le planner sait viser est le BARYCENTRE de cette couronne (le slot est un soft-argmax
    sur les rayons, `slot_head.py`), donc le centre du buisson ;
  · depuis ce centre, CHAQUE baie est à 0,95 m, et la bouche fait 1,00 m
    ⇒ la marge de positionnement tolérée est de **5 cm** ;
  · l'erreur de visée mesurée du slot est de ~23° de gisement, soit ~40 cm à 1 m.
  ⇒ 40 cm d'erreur pour 5 cm de marge : l'échec est STRUCTUREL, pas comportemental. Mesuré :
    49 % des passages à moins de 3 m ne donnent aucun repas.

⚠️ CE QU'IL NE FAUT PAS EN CONCLURE (§2 du CLAUDE.md) : « il suffit d'agrandir eat_radius ».
   Élargir la bouche pour que ça passe est la fausse solution type — ça masque l'imprécision au
   lieu de la corriger. Le vrai défaut est que DEUX contraintes du monde sont incompatibles :
   les baies doivent tomber hors du buisson-marqueur (donc couronne ≥ ~0,6 m) ET la couronne doit
   rester bien à l'intérieur de la bouche (donc ≤ ~0,5 m pour une marge honnête). Avec un buisson
   de 0,55 m, aucune valeur ne satisfait les deux. Le levier propre est le buisson, pas la bouche.

Usage :
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_approche_finale.py \
        --runs data/replay_buffer/replan10
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st

EAT_RADIUS = 1.0  # food_manager.gd:eat_radius
RETINA_RANGE_M = 10.0
FOV_DEG = 120.0
NEAR_M = 3.0  # seuil d'une « tentative » d'approche


def load(runs: list[str]) -> list[list[tuple]]:
    """-> une liste par épisode de (dist_bouffe, vx_demandé, torse, a_mangé, vue)."""
    eps = []
    for run in runs:
        for f in sorted(glob.glob(f"{run}/*.jsonl")):
            rows = []
            for line in open(f):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                wm, ob = r.get("wm"), r.get("obs")
                if not wm:
                    continue
                fr = wm.get("food_rel0") or [0.0, 0.0, 0.0]
                d = math.hypot(fr[0], fr[1])
                bear = abs(math.degrees(math.atan2(fr[0], fr[1])))
                seen = fr[2] > 0.5 and bear <= FOV_DEG / 2 and d <= RETINA_RANGE_M
                torso = wm.get("torso0") or (ob or {}).get("torso")
                cmd = wm.get("cmd") or [0.0, 0.0]
                rows.append((d, cmd[0], torso, float(wm.get("ate", 0.0)) > 0.5, seen))
            if rows:
                eps.append(rows)
    return eps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["data/replay_buffer/replan10"])
    args = ap.parse_args()
    eps = load(args.runs)
    allr = [r for e in eps for r in e]
    if not allr:
        print("❌ aucun tick avec bloc `wm` — relancer la collecte avec SYLVAN_WM_COLLECT=1")
        return
    print(f"{len(eps)} vies · {len(allr)} ticks\n")

    print("1. LA NOURRITURE EST-ELLE LOIN ?")
    d = [r[0] for r in allr]
    print(f"   distance médiane à la baie la plus proche : {st.median(d):.2f} m")
    for lim in (3.0, 10.0):
        print(f"   à moins de {lim:4.1f} m : {100 * sum(1 for x in d if x <= lim) / len(d):.0f} % du temps")
    print(f"   vue (cône + portée) : {100 * sum(1 for r in allr if r[4]) / len(allr):.0f} % des ticks")

    print("\n2. OÙ CALE-T-ELLE ?")
    mins, ate_n, miss_n = [], 0, 0
    for rows in eps:
        i = 0
        while i < len(rows):
            if rows[i][0] < NEAR_M:
                j, mn, ate = i, 9e9, False
                while j < len(rows) and rows[j][0] < NEAR_M:
                    mn = min(mn, rows[j][0])
                    ate = ate or rows[j][3]
                    j += 1
                mins.append(mn)
                ate_n, miss_n = (ate_n + 1, miss_n) if ate else (ate_n, miss_n + 1)
                i = j
            else:
                i += 1
    if mins:
        q = st.quantiles(mins, n=100)
        print(f"   {len(mins)} approches sous {NEAR_M} m · repas {ate_n} · RATÉES {miss_n} "
              f"({100 * miss_n / len(mins):.0f} %)")
        print(f"   distance MINIMALE atteinte : q10={q[9]:.2f}  q25={q[24]:.2f}  q50={q[49]:.2f} m"
              f"   (bouche = {EAT_RADIUS} m)")
        print(f"   approches closant SOUS la bouche : "
              f"{100 * sum(1 for m in mins if m < EAT_RADIUS) / len(mins):.0f} %")

    print("\n3. BLOQUÉE, OU C'EST UN CHOIX ?")
    def band(lo: float, hi: float) -> tuple[float, float, int]:
        c, mv = [], []
        for rows in eps:
            for i in range(len(rows) - 1):
                if lo <= rows[i][0] <= hi and rows[i][2] and rows[i + 1][2]:
                    t0, t1 = rows[i][2], rows[i + 1][2]
                    c.append(abs(rows[i][1]))
                    mv.append(math.hypot(t1[0] - t0[0], t1[1] - t0[1]))
        return (st.median(c) if c else 0.0), (st.median(mv) if mv else 0.0), len(c)

    cs, ms, ns = band(0.9, 1.3)
    cf, mf, nf = band(2.0, 4.0)
    print(f"   au bord (0,9-1,3 m) : demandé {cs:.3f} → réel {ms:.4f} m/pas  (n={ns})")
    print(f"   en approche (2-4 m) : demandé {cf:.3f} → réel {mf:.4f} m/pas  (n={nf})")
    if cs > 0 and cf > 0:
        rs, rf = ms / cs, mf / cf
        print(f"   rendement du corps : bord {rs:.4f} · libre {rf:.4f}")
        if rs >= rf * 0.9:
            print("   ⇒ le corps OBÉIT au bord : elle n'est PAS bloquée. C'est la vitesse")
            print("     DEMANDÉE qui s'effondre — le planner se croit ARRIVÉ.")
        else:
            print("   ⇒ le corps n'obtient plus rien au bord : elle POUSSE contre un obstacle.")

    print(f"\nMARGE GÉOMÉTRIQUE DU MONDE : couronne 0,95 m, bouche {EAT_RADIUS} m → "
          f"{100 * (EAT_RADIUS - 0.95):.0f} cm de tolérance,")
    print("   pour une erreur de visée mesurée d'environ 40 cm à 1 m (23° de gisement).")


if __name__ == "__main__":
    main()
