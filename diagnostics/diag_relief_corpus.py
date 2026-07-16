"""G0 du volet P6 — requêtes de slot apprises du SOULAGEMENT vécu (design_purete_hjepa.md §P6).

Diag GRATUIT (0 train) : le corpus porte-t-il de quoi apprendre « nourrissant = ce qui a soulagé
MON drive » sur la rétine brute ? Les 3 gates G0 pré-enregistrés :
  G0.1  ≥100 soulagements ÉNERGIE et ≥100 SOIF avec rétine à t−1 et ≥1 rayon touchant <1.5 m ;
  G0.2  confond miroir : ≥30 repas ENGOUFFRÉS (rayon vert-règle proche pendant le soulagement
        énergie) — la requête-faim ne doit pas absorber le vert (symétrique du G-loc P5) ;
  G0.3  contraste : ≥500 ticks avec rayon coloré proche SANS soulagement ±10 ticks du drive.
+ mesure : couleur dominante des rayons proches à t−1 par type d'événement (l'oracle d'éval
  couleur-rendue, licite monde-jouet — jamais un label d'entraînement).

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_relief_corpus.py [--selfcheck]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.train_danger_saliency import LIFE_JUMP
from scripts.train_sprint_critic import DEATH_RUNS
from scripts.train_waypoint_pain import _open_text
from sylvan.control.waypoint_layer import N_RAY

RELIEF = 5.0        # remontée de drive/tick qui signe une consommation (convention pursuit_end)
NEAR_D = 0.15       # « proche » = d normalisé < 0.15 (1.5 m — l'objet consommé est au contact)
SAT = 0.15
CONTRAST_W = 10     # fenêtre ±ticks « sans soulagement » du contraste


def _dominant(r: float, g: float, b: float) -> str | None:
    if max(r, g, b) - min(r, g, b) <= SAT:
        return None
    if g > r and g > b:
        return "vert"
    if r > g and r > b:
        return "rouge"
    if b > r and b > g:
        return "bleu"
    return None


def near_colors(retina: list[float], near_d: float = NEAR_D) -> set[str]:
    out: set[str] = set()
    for k in range(N_RAY):
        d, r, g, b = retina[4 * k:4 * k + 4]
        if d >= 0.999 or d >= near_d:
            continue
        c = _dominant(r, g, b)
        if c:
            out.add(c)
    return out


def scan_run(run: Path) -> tuple[list[dict], list[dict]]:
    """→ (événements-soulagement [{drive, tick, near_colors}], ticks [{e_up, t_up, near}])."""
    events: list[dict] = []
    ticks: list[dict] = []
    prev = None
    prev_ret = None
    for i, line in enumerate(_open_text(run / "ep_0000.jsonl")):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        e, t = float(rec["obs"]["energy"]), float(rec["obs"]["thirst"])
        ret = rec["wm"]["retina0"]
        if prev is not None:
            de, dt = e - prev[0], t - prev[1]
            boundary = de > LIFE_JUMP or dt > LIFE_JUMP or \
                float(rec["obs"]["health"]) - prev[2] > LIFE_JUMP
            e_up = not boundary and RELIEF < de
            t_up = not boundary and RELIEF < dt
            if e_up or t_up:
                events.append({"drive": "energy" if e_up else "thirst", "tick": i,
                               "near": near_colors(prev_ret)})
            ticks.append({"e_up": e_up, "t_up": t_up, "near": near_colors(ret)})
        else:
            ticks.append({"e_up": False, "t_up": False, "near": set()})
        prev = (e, t, float(rec["obs"]["health"]))
        prev_ret = ret
    return events, ticks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=DEATH_RUNS)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return

    n_e = n_t = n_engulf = 0
    color_census: dict[tuple[str, str], int] = {}
    n_contrast = 0
    for run in args.runs:
        events, ticks = scan_run(Path(run))
        for ev in events:
            if not ev["near"]:
                continue
            if ev["drive"] == "energy":
                n_e += 1
                if "vert" in ev["near"]:
                    n_engulf += 1
            else:
                n_t += 1
            for c in ev["near"]:
                color_census[(ev["drive"], c)] = color_census.get((ev["drive"], c), 0) + 1
        # contraste : rayon coloré proche, AUCUN soulagement (é ou soif) à ±CONTRAST_W
        up_e = [i for i, tk in enumerate(ticks) if tk["e_up"]]
        up_t = [i for i, tk in enumerate(ticks) if tk["t_up"]]
        up_all = set(up_e) | set(up_t)
        for i, tk in enumerate(ticks):
            if not tk["near"]:
                continue
            if any(j in up_all for j in range(i - CONTRAST_W, i + CONTRAST_W + 1)):
                continue
            n_contrast += 1
        print(f"[g0-p6] {Path(run).name}: {len(events)} soulagements "
              f"({sum(1 for v in events if v['drive'] == 'energy')} énergie)")

    print("\n[g0-p6] === GATES G0 (pré-enregistrés §P6) ===")
    g1 = n_e >= 100 and n_t >= 100
    g2 = n_engulf >= 30
    g3 = n_contrast >= 500
    print(f"[g0-p6] G0.1 soulagements avec percept proche : énergie={n_e} soif={n_t} "
          f"(gates ≥100/≥100) → {'✅' if g1 else '❌'}")
    print(f"[g0-p6] G0.2 repas ENGOUFFRÉS (vert proche)   : {n_engulf} (gate ≥30) → {'✅' if g2 else '❌'}")
    print(f"[g0-p6] G0.3 contraste proche-sans-soulagement : {n_contrast} (gate ≥500) → {'✅' if g3 else '❌'}")
    print(f"[g0-p6] census couleurs proches par événement  : "
          f"{ {f'{d}/{c}': n for (d, c), n in sorted(color_census.items())} }")
    verdict = g1 and g2 and g3
    print(f"\n[g0-p6] {'✅ G0 PASSÉ → train des requêtes licencié' if verdict else '❌ G0 ÉCHOUÉ → collecte ε seeds 3+4 AVANT tout train'}")


def selfcheck() -> None:
    assert N_RAY * 4 == 144
    ret = [1.0, 0.0, 0.0, 0.0] * N_RAY
    ret[0:4] = [0.05, 1.0, 0.0, 0.0]                 # rouge à 0.5 m
    ret[4:8] = [0.10, 0.0, 1.0, 0.0]                 # vert à 1.0 m
    ret[8:12] = [0.30, 0.0, 0.0, 1.0]                # bleu à 3 m (PAS proche)
    assert near_colors(ret) == {"rouge", "vert"}, near_colors(ret)
    assert _dominant(0.5, 0.5, 0.5) is None
    print("[selfcheck] OK — near_colors/seuils")


if __name__ == "__main__":
    main()
