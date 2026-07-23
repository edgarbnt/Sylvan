"""GATE DU CRITIQUE, sur le VÉCU — avec un contrôle de cohérence BLOQUANT. FREE (0 run/train).

CE QUI A TUÉ LA VERSION PRÉCÉDENTE (2026-07-23, rétracté le jour même). Le gate d'avant jugeait une
tête sur un SIMULATEUR maison, avec pour cible `faim(τ+δ)`. Deux défauts :
  1. la cible encodait le TIMING : à repas ÉGAL, manger tard score plus haut que manger tôt (1,00
     contre 0,90) parce qu'il reste moins de temps pour re-drainer. La tête a appris cet artefact ;
  2. le simulateur classait à l'INVERSE du coût analytique — un coût dont on sait qu'il fait vivre
     l'entité (1,40 repas, 50 % de survie en vies). Rien ne l'avait vérifié.

D'OÙ LA STRUCTURE DE CELUI-CI. Un CONTRÔLE DE COHÉRENCE ouvre le gate et peut le fermer seul :
    « le coût qui fait VIVRE l'entité est-il positivement corrélé à ma cible ? »
S'il ne l'est pas, la cible ne mesure pas la qualité d'une décision et RIEN d'autre n'est
interprétable. C'est exactement le test qui manquait, et il est bloquant.

TROIS CHANGEMENTS SUR LA CIBLE :
  * comptée en ÉVÉNEMENTS (repas observés), jamais dérivée d'un état final — pas de timing encodé ;
  * mesurée sur le VÉCU réel (corpus), pas sur un simulateur dont la fidélité est inconnue ;
  * fenêtre TRONQUÉE À LA VIE : un tick à moins de δ de la mort est écarté, sinon la cible mélange
    « n'a pas mangé » et « n'a pas eu le temps ».

CE QUE CE GATE NE FAIT PAS. Il ne juge PAS un classement de candidats : le vécu n'observe qu'UN
candidat par état, et fabriquer les autres nous a déjà coûté un faux positif. Il répond à la
question en amont : **la cible est-elle prédictible du tout, et mieux que ce que l'inné sait déjà ?**
Si non, aucun classement ne vaudra la peine d'être tenté.

Usage:
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_critic_gate_vecu.py \
      --runs data/replay_buffer/<tag1> data/replay_buffer/<tag2> [--selfcheck]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st

import numpy as np

DRAIN = 0.05
RESTORE = 40.0
MEAL_JUMP = 5.0          # guards.CONSUME_JUMP : un repas est un saut de jauge > 5
TELEPORT_M = 1.0         # frontière de vie : un respawn déplace de plus d'un mètre

# --------------------------------------------------------------- CRITÈRES PRÉ-INSCRITS
BAR_COHERENCE = 0.0      # BLOQUANT : corr(score du plan exécuté, cible) doit être > 0
BAR_R2 = 0.10            # la tête doit expliquer au moins ça, sur des VIES jamais vues
BAR_VS_INNE = 0.05       # et dépasser l'inné d'au moins ça


def load(runs: list[str]) -> list[dict]:
    """Charge les ticks, découpés en VIES (frontière = téléport de respawn)."""
    lives: list[dict] = []
    for run in runs:
        for f in sorted(glob.glob(f"{run}/*.jsonl")):
            E, POS, PLAN, RET = [], [], [], []
            for line in open(f):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                E.append(r["obs"]["energy"])
                t = r["wm"]["torso0"]
                POS.append((t[0], t[1]))
                # En mono-pulsion la branche plan_wm_slot ne loggue QUE target+reason (verifie) :
                # les coordonnees ne sont PAS dans le corpus. On lit donc la geometrie a la SOURCE,
                # la retine, qui est loggee a chaque tick. `plan` ne sert plus qu a reperer les
                # ticks de REPLAN (ceux ou une decision est reellement prise).
                PLAN.append(r.get("plan"))
                RET.append(r["wm"].get("retina0"))
            cuts = [0] + [i + 1 for i in range(len(POS) - 1)
                          if math.hypot(POS[i + 1][0] - POS[i][0],
                                        POS[i + 1][1] - POS[i][1]) > TELEPORT_M] + [len(E)]
            for a, b in zip(cuts, cuts[1:]):
                if b - a > 300:
                    lives.append(dict(E=E[a:b], POS=POS[a:b], PLAN=PLAN[a:b], RET=RET[a:b]))
    return lives


def meals_ahead(E: list[float], i: int, delta: int) -> int | None:
    """Repas COMPTÉS dans [i, i+delta). None si la vie s'arrête avant la fin de la fenêtre —
    sinon la cible confond « n'a pas mangé » et « n'a pas eu le temps »."""
    if i + delta >= len(E):
        return None
    return sum(1 for k in range(i + 1, i + delta) if E[k] - E[k - 1] > MEAL_JUMP)


NRAY, RANGE_M, DEPTH_OFF, RED_THR = 36, 10.0, 0.35, 0.55
FOV_DEG = 120.0          # preset bosquets_v2 : cone reel, rayons REDISTRIBUES


def nearest_food(retina: list[float]) -> tuple[float, float] | None:
    """(distance, bearing) de la baie rouge la plus proche, lue dans la RETINE elle-meme.
    Angles conformes au FOV SERVI (perception.gd) : rayon 0 devant, index croissant a droite."""
    if not retina:
        return None
    best = None
    for k in range(NRAY):
        depth, r, g, b = retina[k * 4: k * 4 + 4]
        if depth >= 0.999:
            continue
        n = math.sqrt(r * r + g * g + b * b)
        if n < 1e-6 or r / n < RED_THR:
            continue
        d = depth * RANGE_M + DEPTH_OFF
        if best is None or d < best[0]:
            kk = k if k <= NRAY // 2 else k - NRAY
            best = (d, math.radians(kk * FOV_DEG / NRAY))
    return best


def features(life: dict, i: int) -> list[float] | None:
    """Ce que la tete voit au moment de decider : faim + geometrie percue de la ressource."""
    f = nearest_food(life["RET"][i])
    if f is None:
        return [life["E"][i] / 100.0, 1.2, 0.0, 1.0, 0.0]   # rien en vue : etat DECLARE, pas ecarte
    d, brg = f
    return [life["E"][i] / 100.0, d / 10.0, math.cos(brg), abs(math.sin(brg)), 1.0]


def innate(life: dict, i: int) -> float | None:
    """L INNE = -distance percue a la ressource.

    Le score mono-pulsion est `-min_dist + heading_weight*mean_align` (command_planner.py:578).
    `-min_dist` en est le terme DOMINANT, et la distance PERCUE en est la valeur au premier ordre.
    ⚠️ C est une APPROXIMATION declaree : le corpus ne loggue pas le score reel en mono-pulsion.
    Mais c est l approximation que le planner MINIMISE vraiment — contrairement au proxy de ce matin,
    qui OMETTAIT ce terme et classait sous le hasard.
    Rien en vue -> tres loin (12 m), ce qui est bien ce que le planner "voit".
    """
    f = nearest_food(life["RET"][i])
    return -(f[0] if f else 12.0)


def build(lives: list[dict], delta: int):
    X, Y, I, L = [], [], [], []
    for li, life in enumerate(lives):
        for i in range(len(life["E"])):
            if life["PLAN"][i] is None:
                continue                       # uniquement les ticks de REPLAN
            y = meals_ahead(life["E"], i, delta)
            if y is None:
                continue
            x = features(life, i)
            inn = innate(life, i)
            if x is None or inn is None:
                continue
            X.append(x); Y.append(float(y)); I.append(inn); L.append(li)
    return np.array(X), np.array(Y), np.array(I), np.array(L)


def r2(pred, y):
    ss = float(((y - pred) ** 2).sum())
    tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss / tot if tot > 1e-12 else 0.0


def fit(Xtr, Ytr, Xte, lam=1e-2):
    A = np.c_[Xtr, np.ones(len(Xtr))]
    w = np.linalg.lstsq(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Ytr, rcond=None)[0]
    return np.c_[Xte, np.ones(len(Xte))] @ w


def selfcheck() -> int:
    E = [100.0] * 10 + [50.0] + [95.0] + [90.0] * 10      # un saut de +45 au tick 11
    assert meals_ahead(E, 0, 15) == 1, meals_ahead(E, 0, 15)
    assert meals_ahead(E, 0, 100) is None, "une fenêtre qui dépasse la vie doit être écartée"
    print("  [ok] repas comptés en ÉVÉNEMENTS, fenêtre tronquée à la vie")

    # la cible ne doit PAS dépendre du moment du repas — c'est l'artefact rétracté ce matin.
    # Deux vies identiques à un détail près : l'une mange tôt (tick 20), l'autre tard (tick 150).
    def life_eating_at(t_eat: int, n: int = 200) -> list[float]:
        e, out = 50.0, []
        for k in range(n):
            e -= 0.05
            if k == t_eat:
                e = min(100.0, e + RESTORE)     # LE repas : un vrai saut vers le haut
            out.append(e)
        return out
    early, late = life_eating_at(20), life_eating_at(150)
    assert meals_ahead(early, 0, 180) == meals_ahead(late, 0, 180) == 1, \
        (meals_ahead(early, 0, 180), meals_ahead(late, 0, 180))
    print("  [ok] manger tôt et manger tard comptent PAREIL (l'artefact de fenêtre est éliminé)")

    y = np.array([0.0, 1.0, 2.0, 1.0]); assert abs(r2(y, y) - 1.0) < 1e-9
    assert r2(np.full(4, y.mean()), y) < 1e-9
    print("  [ok] R² : parfait sur l'oracle, nul sur la moyenne")
    print("SELFCHECK PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[])
    ap.add_argument("--delta", type=int, default=600)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()

    lives = load(a.runs)
    X, Y, I, L = build(lives, a.delta)
    print("=" * 78)
    print("GATE DU CRITIQUE — sur le VÉCU, contrôle de cohérence BLOQUANT")
    print("=" * 78)
    print(f"  {len(lives)} vies, {len(Y)} instants de replan exploitables, delta={a.delta}")
    if len(Y) < 200 or len(set(L)) < 4:
        print("  [NUL] pas assez de données."); return 0
    print(f"  cible : repas COMPTÉS dans la fenêtre — "
          f"moyenne {Y.mean():.2f}, min {Y.min():.0f}, max {Y.max():.0f}\n")

    # ── CONTRÔLE DE COHÉRENCE, bloquant ────────────────────────────────────────────────
    coh = float(np.corrcoef(I, Y)[0, 1])
    print(f"  CONTRÔLE DE COHÉRENCE : corr(inné exécuté, cible) = {coh:+.3f}")
    if not (coh > BAR_COHERENCE):
        print(f"    [NUL] ≤ {BAR_COHERENCE}. Le coût qui fait VIVRE l'entité n'est pas corrélé à la")
        print( "    cible : c'est la CIBLE qui est fausse, pas le coût. Rien d'autre n'est lisible.")
        print( "    (C'est exactement ce qui a fait rétracter le gate précédent.)")
        return 0
    print("    [ok] le gate est ouvert : la cible mesure quelque chose que l'inné capture déjà.\n")

    # ── prédictibilité, split PAR VIE (jamais par tick : fuite) ─────────────────────────
    folds, sc_l, sc_i = 4, [], []
    for f in range(folds):
        te = np.isin(L, [li for li in np.unique(L) if li % folds == f])
        tr = ~te
        if te.sum() < 50 or tr.sum() < 50:
            continue
        sc_l.append(r2(fit(X[tr], Y[tr], X[te]), Y[te]))
        # l'inné recalibré (une droite) : on juge son INFORMATION, pas son échelle
        sc_i.append(r2(fit(I[tr].reshape(-1, 1), Y[tr], I[te].reshape(-1, 1)), Y[te]))
    rl, ri = st.mean(sc_l), st.mean(sc_i)
    print(f"  {'prédicteur':<28s} {'R² (vies jamais vues)':>24s}")
    print(f"  {'inné (recalibré)':<28s} {ri:>+24.3f}")
    print(f"  {'tête apprise (ridge)':<28s} {rl:>+24.3f}")
    print(f"  {'gain':<28s} {rl - ri:>+24.3f}")

    print("\n  VERDICT :")
    if rl > BAR_R2 and (rl - ri) > BAR_VS_INNE:
        print(f"    [PASS] la cible est prédictible ({rl:+.3f}) et la tête dépasse l'inné "
              f"({rl - ri:+.3f}).")
    elif rl <= BAR_R2:
        print(f"    [ÉCHEC] cible peu prédictible ({rl:+.3f} ≤ {BAR_R2}) — rien à apprendre ici.")
    else:
        print(f"    [ÉCHEC] la tête ne dépasse pas l'inné ({rl - ri:+.3f} ≤ {BAR_VS_INNE}).")
    print("\n  ⚠️ Ce gate ne juge PAS un classement de candidats (le vécu n'en observe qu'un).")
    print("     Il dit si la cible vaut la peine, pas si une tête saura ranger 117 options.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
