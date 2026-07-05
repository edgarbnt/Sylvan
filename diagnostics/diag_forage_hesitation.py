"""diag_forage_hesitation — GATE H0 : l'« hésitation » du forager multi-drive est-elle RÉELLE
et coûteuse, ou un artefact visuel ?

CONTEXTE (2026-07-03). Owner : « il hésite énormément entre eau et bouffe, en plus de ne pas
toujours faire le bon choix ». Avant de payer le fix (planner rollout-de-survie), on OBJECTIVE
l'impression (principe de travail n°1) sur les buffers BC déjà sur disque — collectés avec LE PLANNER
(coût designed + survival_weight) aux commandes dans le monde multi-drive
(`data/replay_buffer/mode1_bc_{a,b}/ep_0000.jsonl` : obs{proprio,energy,thirst} + wm{retina0,cmd}).
Ces chiffres = BASELINE d'hésitation contre laquelle juger le futur planner refill-aware.

MÉTHODE (gratuit, offline) :
- Consommations = sauts de drive (+5..+45) : Δenergy→repas (ROUGE), Δthirst→boisson (BLEU).
  Frontière d'épisode = saut > 50 (respawn).
- Poursuite inférée depuis la rétine (color-gating exact `obs._color_gated_depths`) : la ressource
  dont la profondeur lissée DÉCROÎT nettement sur une fenêtre = celle qu'il poursuit (sticky sinon).
- Par intervalle entre 2 consommations : 1 switch de poursuite est NÉCESSAIRE (après avoir mangé,
  se tourner vers l'autre) ; le reste = « excess switches » = hésitation.
- Poursuite AVORTÉE = segment de poursuite abandonné sans consommation de cette ressource.

CRITÈRES PRÉ-ENREGISTRÉS (avant le run) :
- HÉSITATION RÉELLE  : médiane excess-switches/intervalle >= 2  OU  fraction avortées >= 30%.
- ARTEFACT VISUEL    : médiane excess <= 1  ET  avortées < 10%.
- Entre les deux     : PARTIEL (lire les sous-scores).
- Secondaire (coût)  : les intervalles avec hésitation finissent-ils avec un drive-min plus bas ?
- Rapporté sans seuil: taux d'inversions de signe de ω commandé (proxy zigzag visuel).

Lancer :  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_forage_hesitation.py \
              [--files data/replay_buffer/mode1_bc_a/ep_0000.jsonl ...] [--selfcheck]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_mode1_death_cause import CLOSE_DEPTH, VISIBLE  # noqa: E402
from sylvan.control.mode1.obs import BLUE, RED, _color_gated_depths  # noqa: E402

JUMP_MIN, JUMP_MAX = 5.0, 45.0     # drive jump = consumption (refill +40, capped)
RESPAWN_JUMP = 50.0                # bigger jump = respawn (episode boundary)
SMOOTH_W = 15                      # depth smoothing window (log lines)
TREND_W = 25                       # pursuit trend window
TREND_EPS = 0.01                   # smoothed-depth decrease over TREND_W to count as pursuing
OMEGA_MIN = 0.1                    # |omega| floor for sign-flip counting

CRIT = {"excess_real": 2.0, "aborted_real": 0.30, "excess_artefact": 1.0, "aborted_artefact": 0.10}


# ---- loading -----------------------------------------------------------------------------------
def load_runs(files: list[str]) -> list[dict]:
    """Each file -> {'e','t','dred','dblue','omega'} arrays (planner-at-the-wheel BC logs)."""
    runs = []
    for path in files:
        e, t, dr, db, om = [], [], [], [], []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                e.append(float(r["obs"]["energy"]))
                t.append(float(r["obs"]["thirst"]))
                ret = r["wm"]["retina0"]
                dr.append(min(_color_gated_depths(ret, RED)))
                db.append(min(_color_gated_depths(ret, BLUE)))
                om.append(float(r["wm"]["cmd"][1]))
        runs.append({"e": np.asarray(e), "t": np.asarray(t), "dred": np.asarray(dr),
                     "dblue": np.asarray(db), "omega": np.asarray(om), "src": path})
    return runs


def split_episodes_by_respawn(run: dict) -> list[tuple[int, int]]:
    de = np.diff(run["e"]); dt = np.diff(run["t"])
    cuts = [0] + [i + 1 for i in range(len(de)) if de[i] > RESPAWN_JUMP or dt[i] > RESPAWN_JUMP] \
        + [len(run["e"])]
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b - a > 2 * TREND_W]


# ---- pursuit inference -------------------------------------------------------------------------
def smooth(x: np.ndarray, w: int) -> np.ndarray:
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def pursuit_labels(dred: np.ndarray, dblue: np.ndarray) -> list[str | None]:
    """Per line: 'red'/'blue' if that resource's smoothed depth clearly decreases over TREND_W
    (the more decreasing one wins); otherwise carry the previous label (sticky)."""
    sr, sb = smooth(dred, SMOOTH_W), smooth(dblue, SMOOTH_W)
    labels: list[str | None] = []
    cur: str | None = None
    for i in range(len(sr)):
        j = max(0, i - TREND_W)
        tr, tb = sr[i] - sr[j], sb[i] - sb[j]      # negative = approaching
        cand = None
        if tr < -TREND_EPS or tb < -TREND_EPS:
            cand = "red" if tr <= tb else "blue"
        if cand is not None:
            cur = cand
        labels.append(cur)
    return labels


def consumption_events(e: np.ndarray, t: np.ndarray) -> list[tuple[int, str]]:
    ev = []
    for i in range(len(e) - 1):
        if JUMP_MIN < e[i + 1] - e[i] < JUMP_MAX:
            ev.append((i + 1, "red"))
        if JUMP_MIN < t[i + 1] - t[i] < JUMP_MAX:
            ev.append((i + 1, "blue"))
    return ev


# ---- per-episode metrics -----------------------------------------------------------------------
def analyze_episode(e, t, dred, dblue, omega) -> dict:
    labels = pursuit_labels(dred, dblue)
    events = consumption_events(e, t)

    # pursuit segments (maximal runs of one label)
    segs: list[tuple[int, int, str]] = []
    for i, lab in enumerate(labels):
        if lab is None:
            continue
        if segs and segs[-1][2] == lab and segs[-1][1] == i - 1:
            segs[-1] = (segs[-1][0], i, lab)
        elif segs and segs[-1][2] == lab:
            segs[-1] = (segs[-1][0], i, lab)   # bridge over None gaps (sticky)
        else:
            segs.append((i, i, lab))
    # aborted = segment whose resource was NOT consumed during it (or within TREND_W after)
    aborted = 0
    for a, b, lab in segs[:-1]:                # last segment = episode end, not an abandonment
        if not any(a <= idx <= b + TREND_W and col == lab for idx, col in events):
            aborted += 1
    # excess switches per inter-consumption interval (1 switch is the necessary re-target)
    bounds = [0] + [idx for idx, _ in sorted(events)] + [len(labels)]
    excess_l = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < TREND_W:
            continue
        sw = sum(1 for i in range(a + 1, b)
                 if labels[i] is not None and labels[i - 1] is not None and labels[i] != labels[i - 1])
        excess_l.append(max(0, sw - 1))
    # omega sign flips at pursuit cadence
    om = omega[np.abs(omega) >= OMEGA_MIN]
    flips = float((np.sign(om[1:]) != np.sign(om[:-1])).mean()) if len(om) > 1 else 0.0
    # cost probe: min drive reached in hesitant vs clean intervals
    min_drive = float(min(e.min(), t.min()))
    return {"n_meals": sum(1 for _, c in events if c == "red"),
            "n_drinks": sum(1 for _, c in events if c == "blue"),
            "n_segments": len(segs), "aborted": aborted,
            "excess": excess_l, "flip_rate": flips, "min_drive": min_drive,
            "len": len(labels)}


# ---- selfcheck ---------------------------------------------------------------------------------
def _mk(dr: float, db: float) -> tuple[float, float]:
    return dr, db


def selfcheck() -> None:
    n = 400
    # clean juggler: approach red 0..150 (eat at 150), then blue (drink at 300)
    dred = np.concatenate([np.linspace(0.8, 0.1, 150), np.full(250, 0.8)])
    dblue = np.concatenate([np.full(150, 0.8), np.linspace(0.8, 0.1, 150), np.full(100, 0.8)])
    e = np.full(n, 80.0); t = np.full(n, 80.0)
    e[151:] += 20.0                             # meal jump at 150->151
    t[301:] += 15.0                             # drink jump at 300->301
    m = analyze_episode(e, t, dred, dblue, np.zeros(n))
    assert m["n_meals"] == 1 and m["n_drinks"] == 1, m
    assert m["aborted"] == 0, m
    assert st.median(m["excess"]) <= 1, m
    # ditherer: zigzag red/blue every 60 lines, never consumes
    seg = []
    for k in range(6):
        app = np.linspace(0.8, 0.4, 60)
        seg.append(app if k % 2 == 0 else np.full(60, 0.8))
    dred2 = np.concatenate(seg)
    dblue2 = np.concatenate([np.full(60, 0.8) if k % 2 == 0 else np.linspace(0.8, 0.4, 60)
                             for k in range(6)])
    e2 = np.full(360, 60.0); t2 = np.full(360, 60.0)
    m2 = analyze_episode(e2, t2, dred2, dblue2, np.zeros(360))
    assert m2["n_segments"] >= 4 and m2["aborted"] >= 3, m2
    assert m2["excess"] and m2["excess"][0] >= 3, m2
    print("[selfcheck] OK — jongleur propre (0 avorté, excess<=1) vs zigzagueur (avortés, excess>=3)")


# ---- main --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+",
                    default=["data/replay_buffer/mode1_bc_a/ep_0000.jsonl",
                             "data/replay_buffer/mode1_bc_b/ep_0000.jsonl"])
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return

    runs = load_runs(args.files)
    eps = []
    for run in runs:
        for a, b in split_episodes_by_respawn(run):
            eps.append(analyze_episode(run["e"][a:b], run["t"][a:b],
                                       run["dred"][a:b], run["dblue"][a:b], run["omega"][a:b]))
    if not eps:
        print("aucun épisode exploitable")
        return

    all_excess = [x for m in eps for x in m["excess"]]
    n_seg = sum(m["n_segments"] for m in eps)
    n_ab = sum(m["aborted"] for m in eps)
    denom = sum(max(0, m["n_segments"] - 1) for m in eps)  # last segment per episode excluded
    frac_ab = n_ab / max(1, denom)
    med_excess = st.median(all_excess) if all_excess else float("nan")
    mean_excess = st.mean(all_excess) if all_excess else float("nan")
    flip = st.median([m["flip_rate"] for m in eps])
    meals = sum(m["n_meals"] for m in eps); drinks = sum(m["n_drinks"] for m in eps)

    # cost probe: hesitant intervals (excess>=2) vs clean — min drive of the episode they sit in
    hes_eps = [m for m in eps if m["excess"] and max(m["excess"]) >= 2]
    clean_eps = [m for m in eps if not m["excess"] or max(m["excess"]) < 2]
    md_h = st.median([m["min_drive"] for m in hes_eps]) if hes_eps else float("nan")
    md_c = st.median([m["min_drive"] for m in clean_eps]) if clean_eps else float("nan")

    print(f"épisodes={len(eps)} | repas={meals} boissons={drinks} | segments poursuite={n_seg}")
    print("\n=== TABLE BUT (hésitation) vs proxy (il mange quand même) ===")
    print(f"BUT   excess-switches/intervalle : médiane {med_excess:.1f}  moyenne {mean_excess:.2f} "
          f"(n intervalles={len(all_excess)})")
    print(f"BUT   poursuites AVORTÉES        : {n_ab}/{denom} = {100*frac_ab:.0f}%")
    print(f"info  taux d'inversion signe ω   : {100*flip:.0f}% (proxy zigzag visuel, pas de seuil)")
    print(f"info  coût : min-drive épisodes hésitants {md_h:.1f} vs propres {md_c:.1f} "
          f"(n={len(hes_eps)}/{len(clean_eps)})")

    print("\n--- VERDICT (critères pré-enregistrés) ---")
    if med_excess >= CRIT["excess_real"] or frac_ab >= CRIT["aborted_real"]:
        print(f"HÉSITATION RÉELLE (médiane excess {med_excess:.1f} >= {CRIT['excess_real']} "
              f"ou avortées {100*frac_ab:.0f}% >= {100*CRIT['aborted_real']:.0f}%) "
              "→ baseline chiffrée posée ; le planner refill-aware devra la faire baisser.")
    elif med_excess <= CRIT["excess_artefact"] and frac_ab < CRIT["aborted_artefact"]:
        print("ARTEFACT VISUEL (peu de switches inutiles, peu d'avortées) → ne rien dépenser sur "
              "l'« hésitation » ; le vrai sujet reste le CHOIX (myopie), déjà traité par le rollout.")
    else:
        print("PARTIEL → hésitation présente mais modérée ; à re-mesurer en A/B avec le nouveau planner.")


if __name__ == "__main__":
    main()
