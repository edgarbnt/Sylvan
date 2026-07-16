"""G0 du chantier P5 — perception par la CONSÉQUENCE (docs/design_purete_hjepa.md §P5).

Diag GRATUIT (0 train) : le corpus vécu porte-t-il de quoi apprendre « dangereux = ce qui a
précédé mes dégâts » sur la rétine BRUTE (36 rayons × (d,r,g,b)) ? Mesure les 4 gates G0
pré-enregistrés AVANT tout train :
  G0.1  ≥150 onsets-dégâts (1er tick de morsure après ≥20 ticks sains) avec rétine au tick ;
  G0.2  visibilité : ≥90 % des ticks-dégâts ont ≥1 rayon vert-règle touchant (sinon label=bruit) ;
  G0.3  contraste : ≥500 ticks proche-sans-dégât (rayon touchant <2 m, zéro dégât ±20 ticks)
        dont ≥100 avec rayon rouge/bleu proche (le confond « bouffe au cœur » est testable) ;
  G0.4  diversité : onsets répartis sur ≥2 des 3 secteurs angulaires (avant/flanc/arrière).
+ mesures ρ̂ candidates : distribution des distances min au point vert aux onsets (méd/q90/q95)
  — la portée-morsure VÉCUE que g(d) devra retrouver (gate G-ρ de la phase A).

⚠️ La règle-verte (green_points) sert ici d'ORACLE D'ÉVALUATION SEULEMENT (monde-jouet : le vert
EST la vérité rendue, cf diag_hazard_slot) — jamais de label d'entraînement.

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_saliency_corpus.py [--selfcheck]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

from scripts.train_sprint_critic import DEATH_RUNS
from scripts.train_waypoint_pain import _open_text
from sylvan.control.waypoint_layer import N_RAY, RETINA_RANGE_M, green_points

DMG_DROP = 0.3        # chute de santé/tick qui signe une morsure (mesuré : −0.5/pas en zone)
LIFE_JUMP = 45.0      # remontée d'un signal vital qui signe un respawn. ⚠️ > 40 = restore d'un
                      # repas/boisson (mesuré Phase 0 sprint) — sinon chaque repas fabriquerait une
                      # fausse frontière (et de faux onsets pendant les repas engouffrés). Mort-danger
                      # : santé <15→100 = +85 ; mort faim/soif : drive ~0→100 = +100 → captés.
CLEAN_TICKS = 20      # ticks sains requis avant une morsure pour compter un ONSET
NEAR_D = 0.2          # « proche » = d normalisé < 0.2 (2 m)
SAT = 0.15            # même seuil de saturation que la règle mur-vert


def _dominant(r: float, g: float, b: float) -> str | None:
    """Canal dominant saturé (même structure que la règle verte) — lentille d'ÉVAL seulement."""
    if max(r, g, b) - min(r, g, b) <= SAT:
        return None
    if g > r and g > b:
        return "green"
    if r > g and r > b:
        return "red"
    if b > r and b > g:
        return "blue"
    return None


def scan_run(run: Path) -> list[dict]:
    """1 entrée par tick : dégât ?, visibilité verte, distances/bearing du vert le plus proche,
    présence de rayons proches par couleur. Frontières de vie par remontée santé/énergie."""
    ticks: list[dict] = []
    prev_h = prev_e = prev_t = None
    life = 0
    for line in _open_text(run / "ep_0000.jsonl"):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        h, e = float(rec["obs"]["health"]), float(rec["obs"]["energy"])
        t = float(rec["obs"]["thirst"])
        ret = rec["wm"]["retina0"]
        if prev_h is not None and (h - prev_h > LIFE_JUMP or e - prev_e > LIFE_JUMP
                                   or t - prev_t > LIFE_JUMP):
            life += 1
            prev_h = None                        # pas de dégât inter-vies
        dmg = prev_h is not None and (prev_h - h) > DMG_DROP
        greens = green_points(ret)
        gd = [math.hypot(x, z) for x, z in greens]
        i_min = min(range(len(gd)), key=lambda i: gd[i]) if gd else None
        near_col = {"red": False, "blue": False, "green": False}
        near_any = False
        for k in range(N_RAY):
            d, r, g, b = ret[4 * k:4 * k + 4]
            if d >= 0.999 or d >= NEAR_D:
                continue
            near_any = True
            c = _dominant(r, g, b)
            if c:
                near_col[c] = True
        ticks.append({
            "life": life, "dmg": dmg,
            "green_vis": bool(greens),
            "gmin": gd[i_min] if i_min is not None else float("nan"),
            "gbear": math.degrees(math.atan2(greens[i_min][0], greens[i_min][1]))
                     if i_min is not None else float("nan"),
            "near_any": near_any, "near_red": near_col["red"], "near_blue": near_col["blue"],
        })
        prev_h, prev_e, prev_t = h, e, t
    return ticks


def sector(bearing_deg: float) -> str:
    a = abs(bearing_deg)
    return "avant" if a < 60.0 else ("flanc" if a < 120.0 else "arrière")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=DEATH_RUNS)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return

    n_dmg = n_dmg_green = n_onset = 0
    onset_dist: list[float] = []
    onset_sect: dict[str, int] = {"avant": 0, "flanc": 0, "arrière": 0}
    n_neg = n_neg_rb = 0
    dmg_dist_all: list[float] = []
    for run in args.runs:
        rows = scan_run(Path(run))
        # index des ticks-dégâts par vie (fenêtres ±CLEAN_TICKS du contraste)
        dmg_t = [i for i, r in enumerate(rows) if r["dmg"]]
        dmg_set = set(dmg_t)
        last_dmg: dict[int, int] = {}
        n_run_onsets = 0
        for i in dmg_t:
            r = rows[i]
            n_dmg += 1
            if r["green_vis"]:
                n_dmg_green += 1
                dmg_dist_all.append(r["gmin"])
            prev = last_dmg.get(r["life"])
            if prev is None or i - prev >= CLEAN_TICKS:
                n_onset += 1
                n_run_onsets += 1
                if r["green_vis"]:
                    onset_dist.append(r["gmin"])
                    onset_sect[sector(r["gbear"])] += 1
            last_dmg[r["life"]] = i
        for i, r in enumerate(rows):
            if not r["near_any"] or r["dmg"]:
                continue
            if any(j in dmg_set for j in range(i - CLEAN_TICKS, i + CLEAN_TICKS + 1)):
                continue
            n_neg += 1
            if r["near_red"] or r["near_blue"]:
                n_neg_rb += 1
        print(f"[g0] {Path(run).name}: {len(rows)} ticks, {len(dmg_t)} ticks-dégâts, "
              f"{n_run_onsets} onsets")

    vis = n_dmg_green / max(n_dmg, 1)
    print("\n[g0] === GATES G0 (pré-enregistrés, §P5) — BUT, pas proxy ===")
    g1 = n_onset >= 150
    g2 = vis >= 0.90
    g3 = n_neg >= 500 and n_neg_rb >= 100
    populated = [s for s, n in onset_sect.items() if n >= 10]
    g4 = len(populated) >= 2
    print(f"[g0] G0.1 onsets-dégâts (rétine au tick)  : {n_onset} (gate ≥150) → {'✅' if g1 else '❌'}")
    print(f"[g0] G0.2 visibilité verte aux dégâts     : {100 * vis:.1f}% de {n_dmg} ticks "
          f"(gate ≥90%) → {'✅' if g2 else '❌'}")
    print(f"[g0] G0.3 contraste proche-sans-dégât     : {n_neg} (gate ≥500), "
          f"dont rouge/bleu proche {n_neg_rb} (gate ≥100) → {'✅' if g3 else '❌'}")
    print(f"[g0] G0.4 diversité angulaire des onsets  : {onset_sect} "
          f"(secteurs ≥10 : {populated}) → {'✅' if g4 else '❌'}")
    if len(onset_dist) >= 20:
        qs = st.quantiles(onset_dist, n=20)
        print(f"\n[g0] ρ̂ candidates (dist min au point vert, ONSETS, n={len(onset_dist)}) : "
              f"méd={st.median(onset_dist):.2f} q90={qs[17]:.2f} q95={qs[18]:.2f} m")
    if len(dmg_dist_all) >= 20:
        print(f"[g0] dist min au vert sur TOUS ticks-dégâts : méd={st.median(dmg_dist_all):.2f} "
              f"q95={st.quantiles(dmg_dist_all, n=20)[18]:.2f} m")
    verdict = g1 and g2 and g3 and g4
    print(f"\n[g0] {'✅ G0 PASSÉ → phase A (train saillance) licenciée' if verdict else '❌ G0 ÉCHOUÉ → collecte ε seeds 3+4 AVANT tout train'}")


def selfcheck() -> None:
    assert N_RAY * 4 == 144 and RETINA_RANGE_M == 10.0
    # rayon k=9 (90° à droite), d=0.2, vert → point ≈ (2, 0)
    ret = [1.0, 0.0, 0.0, 0.0] * N_RAY
    ret[4 * 9:4 * 9 + 4] = [0.2, 0.0, 1.0, 0.0]
    pts = green_points(ret)
    assert len(pts) == 1 and abs(pts[0][0] - 2.0) < 1e-6 and abs(pts[0][1]) < 1e-6, pts
    assert _dominant(1.0, 0.0, 0.0) == "red" and _dominant(0.0, 0.0, 1.0) == "blue"
    assert _dominant(0.5, 0.5, 0.5) is None            # gris insaturé
    assert sector(30.0) == "avant" and sector(-100.0) == "flanc" and sector(170.0) == "arrière"
    print("[selfcheck] OK — conventions rétine/règle-verte/secteurs")


if __name__ == "__main__":
    main()
