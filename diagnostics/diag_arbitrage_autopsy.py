"""AUTOPSIE des morts étiquetées « arbitrage » — GRATUIT (0 run, 0 train), owner-demandée 2026-07-21.

POURQUOI : l'étiquette `morts-par-arbitrage` de diag_arbitrage_g0 dit littéralement « au dernier
replan où la ressource manquante était NOMINALEMENT atteignable, l'entité visait autre chose ».
Ça CONFOND au moins trois choses très différentes :
  (a) un VRAI RATÉ            — elle était proche/urgente, l'autre jauge tranquille, et on est parti ailleurs ;
  (b) un DILEMME inévitable   — l'autre jauge était AUSSI critique : servir l'une tuait l'autre ;
  (c) de la PORTÉE déguisée   — « atteignable » vient d'une formule ligne-droite optimiste ; à la
                                distance réelle, l'entité n'atteint quasi jamais (courbe mesurée).
Tant qu'on ne sépare pas ces trois cas, entraîner une tête pour « corriger l'arbitrage » revient à
soigner une maladie non diagnostiquée (PRINCIPE N°1).

CE QU'ON RECONSTRUIT au dernier replan utile (tout est DÉJÀ loggé : sf/sw, cible, positions ego des
deux ressources, niveaux des deux jauges) :
  - distances aux deux ressources et niveaux des deux jauges ;
  - les DEUX scores du planner (sf/sw) et l'écart RELATIF → le planner hésitait-il, ou était-il
    CONFIANT et faux ? (serré ⇒ une valeur apprise peut faire pencher ; large ⇒ c'est son MODÈLE
    qui est faux, et l'apprentissage n'est pas forcément la bonne réponse) ;
  - la manquante était-elle plus PROCHE que celle choisie ?

Lancement (racine) :
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_arbitrage_autopsy.py [--selfcheck]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import sys

sys.path.insert(0, "diagnostics")
from diag_arbitrage_g0 import (CONSUME_JUMP, DEATH_THR, FRESH_H, NEEDED_RES, SPEED_M_PER_TICK,
                               START_DRIVE, START_TOL)

PRACTICAL_REACH_M = 5.0     # au-delà, la courbe MESURÉE donne ~0-15 % d'atteinte => pas vraiment atteignable
DILEMMA_LEVEL = 30.0        # l'AUTRE jauge sous ce niveau = elle aussi critique => vrai dilemme
TIGHT_GAP = 0.05            # écart RELATIF |sf-sw|/max < 5 % => le planner était quasi indifférent
CLASSES = ["portée déguisée", "dilemme (2 jauges critiques)", "choix SERRÉ", "vrai RATÉ"]

RUNS = ["data/replay_buffer/arbgrad_graded_s1", "data/replay_buffer/arbgrad_graded_s2",
        "data/replay_buffer/arbgrad_sym_s1", "data/replay_buffer/arbgrad_sym_s2"]


def _open(p: str):
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p)


def load_lives(run: str) -> list[dict]:
    """Vies avec, par replan MULTI : distances aux 2 ressources, sf/sw, cible retenue."""
    lives: list[dict] = []
    cur: dict | None = None
    prev_e = prev_t = None
    prev_h = 100.0
    for ep in sorted(glob.glob(os.path.join(run, "ep_*.jsonl*"))):
        with _open(ep) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                o = rec["obs"]
                e, t, h = float(o["energy"]), float(o["thirst"]), float(o.get("health", 100.0))
                at_start = (abs(e - START_DRIVE) < START_TOL and abs(t - START_DRIVE) < START_TOL
                            and h >= FRESH_H)
                jumped = prev_e is not None and (abs(e - prev_e) > 1.0 or abs(t - prev_t) > 1.0
                                                 or h - prev_h > 1.0)
                if cur is None or (at_start and jumped):
                    if cur is not None:
                        lives.append(cur)
                    cur = {"drives": [], "plans": []}
                cur["drives"].append((e, t, h))
                p = rec.get("plan")
                if p is not None and "sf" in p and p.get("first") in ("food", "water"):
                    entry = {"i": len(cur["drives"]) - 1, "first": p["first"],
                             "sf": float(p["sf"]), "sw": float(p["sw"])}
                    for res in ("food", "water"):
                        pos = p.get(res)
                        if pos is not None:
                            entry[res] = math.hypot(float(pos[0]), float(pos[1]))
                    cur["plans"].append(entry)
                prev_e, prev_t, prev_h = e, t, h
    if cur is not None and cur["drives"]:
        lives.append(cur)
    return lives


def autopsy(life: dict, drains: tuple[float, float]) -> dict | None:
    """None si la vie n'est pas une mort-par-arbitrage ; sinon le dossier du dernier replan utile."""
    e, t, h = life["drives"][-1]
    if h <= DEATH_THR or (e > DEATH_THR and t > DEATH_THR):
        return None                                   # mort danger / vie tronquée
    dying = "energy" if e <= t else "thirst"
    needed = NEEDED_RES[dying]
    other = "water" if needed == "food" else "food"
    di = 0 if dying == "energy" else 1
    # dernier replan où la manquante était VUE et NOMINALEMENT atteignable (formule ligne droite)
    last = None
    for p in life["plans"]:
        if needed not in p:
            continue
        reserve = life["drives"][p["i"]][di]
        if p[needed] <= reserve * (SPEED_M_PER_TICK / drains[di]):
            last = p
    if last is None or last["first"] == needed:
        return None                                   # jamais atteignable, ou elle la visait => pas "arbitrage"
    lv = life["drives"][last["i"]]
    d_need = last[needed]
    d_other = last.get(other)
    lvl_other = lv[1 - di]
    gap = abs(last["sf"] - last["sw"]) / max(abs(last["sf"]), abs(last["sw"]), 1e-9)
    if d_need > PRACTICAL_REACH_M:
        k = "portée déguisée"
    elif lvl_other < DILEMMA_LEVEL:
        k = "dilemme (2 jauges critiques)"
    elif gap < TIGHT_GAP:
        k = "choix SERRÉ"
    else:
        k = "vrai RATÉ"
    return {"class": k, "d_need": d_need, "d_other": d_other, "lvl_need": lv[di],
            "lvl_other": lvl_other, "gap": gap,
            "closer": (d_other is not None and d_need < d_other)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=RUNS)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    runs = [r for r in args.runs if glob.glob(os.path.join(r, "ep_*.jsonl*"))]
    if not runs:
        raise SystemExit("aucun corpus trouvé")
    if args.selfcheck:
        lv = load_lives(runs[0])
        assert lv, "aucune vie"
        assert any(p.get("sf") is not None for l in lv for p in l["plans"]), "sf/sw absents du corpus"
        print(f"[selfcheck] OK — {len(lv)} vies, sf/sw présents")
        return

    for label, sel, drains in (("GRADED (0.05/0.035)", "graded", (0.05, 0.035)),
                               ("SYM (0.05/0.05)", "sym", (0.05, 0.05))):
        cases = []
        n_lives = 0
        for run in [r for r in runs if sel in r]:
            lives = load_lives(run)
            n_lives += len(lives)
            for lf in lives:
                a = autopsy(lf, drains)
                if a:
                    cases.append(a)
        if not cases:
            continue
        print(f"\n=== {label} — {n_lives} vies, {len(cases)} morts étiquetées « arbitrage » ===")
        for k in CLASSES:
            sub = [c for c in cases if c["class"] == k]
            if not sub:
                print(f"  {k:>28s} :   0")
                continue
            med_dn = sorted(c["d_need"] for c in sub)[len(sub) // 2]
            med_gap = sorted(c["gap"] for c in sub)[len(sub) // 2]
            closer = sum(1 for c in sub if c["closer"])
            print(f"  {k:>28s} : {len(sub):3d} ({100*len(sub)/len(cases):3.0f}%) | "
                  f"dist manquante méd {med_dn:.1f} m | écart score méd {100*med_gap:5.1f}% | "
                  f"manquante plus PROCHE {closer}/{len(sub)}")
        gaps = sorted(c["gap"] for c in cases)
        q = lambda p: gaps[int(p * (len(gaps) - 1))]
        print(f"  écart de score |sf-sw|/max — q25/méd/q75 : {100*q(.25):.1f}% / {100*q(.5):.1f}% / {100*q(.75):.1f}%")

    print("\nLecture : 'portée déguisée' + 'dilemme' = morts NON imputables à un mauvais arbitrage.")
    print("'choix SERRÉ' = le planner hésitait -> une valeur apprise pourrait faire pencher (G2 justifié).")
    print("'vrai RATÉ' avec écart LARGE = le planner était CONFIANT et faux -> son MODÈLE est en cause,")
    print("               l'apprentissage n'est pas forcément la bonne réponse (corriger le coût peut suffire).")


if __name__ == "__main__":
    main()
