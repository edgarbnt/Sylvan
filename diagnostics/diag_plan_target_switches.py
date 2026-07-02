"""diag_plan_target_switches — l'hésitation est-elle RÉELLE au niveau du PLANNER, ou un artefact
du métrique H0 (inférence rétine, confondue par le plus-proche qui change d'identité à 5+5 items) ?

CONTEXTE (2026-07-03). Re-A/B = parité, mais hésitation « 85-87% de poursuites avortées » sur les
DEUX coûts et les DEUX couleurs → suspect = le métrique lui-même. Fix méthodologique : le serveur
logge désormais LA CIBLE DU PLANNER à chaque replan (clé additive `plan.target` dans le log BC,
`serve_planner_command._plan_target_record` : la ressource que le meilleur plan imaginé rapproche
le plus). Ce diag mesure l'hésitation sur cette VÉRITÉ-TERRAIN et la compare à l'inférence rétine
H0 sur les MÊMES fichiers.

MÉTRIQUES (mêmes définitions que H0, mais sur la cible du planner) :
- excess-switches par intervalle entre 2 consommations (1 switch = nécessaire) ;
- poursuites AVORTÉES = run de cible abandonné sans consommation de cette ressource.

CRITÈRES PRÉ-ENREGISTRÉS :
- ARTEFACT MAJEUR : avortées-vraies < 40% ET excess-vrai médian <= 1 (alors que l'inférence H0 sur
  les mêmes fichiers dit ~85%/2+) → l'« hésitation » était surtout un artefact de mesure ; le vrai
  problème est ailleurs (ex. conversion poursuite→consommation).
- RÉELLE : avortées-vraies >= 60% OU excess-vrai médian >= 2 → flip-flop décisionnel confirmé au
  niveau du planner → creuser replan/perception (EMA, slot) comme source du flip.
- Entre les deux : PARTIEL.

Lancer :  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_plan_target_switches.py \
              --files data/replay_buffer/hesit_probe_55/ep_0000.jsonl [--selfcheck]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_forage_hesitation import (  # noqa: E402  (mêmes définitions consommation/frontière)
    JUMP_MAX, JUMP_MIN, RESPAWN_JUMP,
)

AFTER_WIN = 3      # une conso dans les N replans suivant la fin d'un run compte pour ce run


def load_plans(path: str) -> list[dict]:
    """Lignes AVEC `plan` (une par replan) : {target, energy, thirst}."""
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            p = r.get("plan")
            if p is None:
                continue
            rows.append({"target": p.get("target", "none"),
                         "e": float(r["obs"]["energy"]), "t": float(r["obs"]["thirst"])})
    return rows


def split_eps(rows: list[dict]) -> list[list[dict]]:
    eps, cur = [], []
    for i, r in enumerate(rows):
        if cur and (r["e"] - cur[-1]["e"] > RESPAWN_JUMP or r["t"] - cur[-1]["t"] > RESPAWN_JUMP):
            eps.append(cur)
            cur = []
        cur.append(r)
    if cur:
        eps.append(cur)
    return [e for e in eps if len(e) > 10]


def analyze(ep: list[dict]) -> dict:
    events = []                                   # (replan_idx, 'food'|'water')
    for i in range(1, len(ep)):
        if JUMP_MIN < ep[i]["e"] - ep[i - 1]["e"] < JUMP_MAX:
            events.append((i, "food"))
        if JUMP_MIN < ep[i]["t"] - ep[i - 1]["t"] < JUMP_MAX:
            events.append((i, "water"))
    targets = [r["target"] for r in ep]
    # runs de cible (les 'none' prolongent le run courant — pas une décision de switch)
    runs: list[tuple[int, int, str]] = []
    for i, tg in enumerate(targets):
        if tg == "none":
            continue
        if runs and runs[-1][2] == tg:
            runs[-1] = (runs[-1][0], i, tg)
        else:
            runs.append((i, i, tg))
    aborted = sum(1 for a, b, tg in runs[:-1]
                  if not any(a <= idx <= b + AFTER_WIN and col == tg for idx, col in events))
    bounds = [0] + [i for i, _ in sorted(events)] + [len(targets)]
    excess = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 3:
            continue
        seg = [t for t in targets[a:b] if t != "none"]
        sw = sum(1 for i in range(1, len(seg)) if seg[i] != seg[i - 1])
        excess.append(max(0, sw - 1))
    frac_none = float(np.mean([t == "none" for t in targets]))
    return {"n_runs": len(runs), "aborted": aborted, "excess": excess,
            "meals": sum(1 for _, c in events if c == "food"),
            "drinks": sum(1 for _, c in events if c == "water"), "frac_none": frac_none}


def selfcheck() -> None:
    mk = lambda tg, e, t: {"target": tg, "e": e, "t": t}
    # jongleur propre : food (conso) puis water (conso) → 1 switch, 0 avorté, excess 0
    ep = [mk("food", 60, 60)] * 20 + [mk("water", 100, 60)] * 20 + [mk("water", 100, 100)]
    m = analyze(ep)
    assert m["meals"] == 1 and m["drinks"] == 1 and m["aborted"] == 0, m
    assert not m["excess"] or max(m["excess"]) == 0, m
    # zigzagueur : 5 alternances sans conso → runs avortés + excess élevé
    ep2 = sum([[mk("food", 50, 50)] * 5 + [mk("water", 50, 50)] * 5 for _ in range(3)], [])
    m2 = analyze(ep2)
    assert m2["n_runs"] == 6 and m2["aborted"] == 5 and m2["excess"][0] >= 4, m2
    # split par respawn (vrai respawn : drive quasi-mort → 100, saut ~+70 > seuil 50 strict)
    assert len(split_eps([mk("food", 30, 50)] * 12 + [mk("food", 100, 100)] * 12)) == 2
    print("[selfcheck] OK — jongleur propre, zigzagueur, split respawn")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=False,
                    default=["data/replay_buffer/hesit_probe_55/ep_0000.jsonl"])
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return

    eps = []
    for f in args.files:
        eps.extend(split_eps(load_plans(f)))
    if not eps:
        print("aucune ligne `plan` — le serveur doit tourner avec l'instrumentation cible (post 2026-07-03).")
        return
    res = [analyze(e) for e in eps]
    n_runs = sum(m["n_runs"] for m in res)
    denom = sum(max(0, m["n_runs"] - 1) for m in res)
    aborted = sum(m["aborted"] for m in res)
    all_excess = [x for m in res for x in m["excess"]]
    med_ex = st.median(all_excess) if all_excess else float("nan")
    frac_ab = aborted / max(1, denom)
    print(f"épisodes={len(res)} | repas={sum(m['meals'] for m in res)} "
          f"boissons={sum(m['drinks'] for m in res)} | runs de cible={n_runs} "
          f"| replans sans cible={100*float(np.mean([m['frac_none'] for m in res])):.0f}%")
    print("\n=== HÉSITATION VRAIE (cible du planner) vs inférence H0 ===")
    print(f"BUT   excess-switches/intervalle : médiane {med_ex:.1f} "
          f"moyenne {st.mean(all_excess) if all_excess else float('nan'):.2f} (n={len(all_excess)})")
    print(f"BUT   poursuites AVORTÉES        : {aborted}/{denom} = {100*frac_ab:.0f}%")
    print("(comparer : inférence rétine H0 sur les mêmes fichiers → diag_forage_hesitation --files ...)")

    print("\n--- VERDICT (critères pré-enregistrés) ---")
    if frac_ab < 0.40 and (not all_excess or med_ex <= 1.0):
        print("ARTEFACT MAJEUR : le planner ne flip-flop PAS (ou peu) — l'« hésitation » H0 était surtout")
        print("le plus-proche-rayon qui change d'identité. Le vrai problème = conversion poursuite→conso.")
    elif frac_ab >= 0.60 or med_ex >= 2.0:
        print("HÉSITATION RÉELLE au niveau du planner → creuser la source du flip (replan, EMA eau, slot).")
    else:
        print("PARTIEL → une part artefact, une part réelle ; lire les sous-scores.")


if __name__ == "__main__":
    main()
