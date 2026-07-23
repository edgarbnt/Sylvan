"""GATE HORS-LIGNE du critique appris — une tête peut-elle CLASSER les candidats ? FREE.

POURQUOI CE GATE EXISTE. Trois critiques appris ont échoué, et la recherche du 2026-07-23 a montré
que les trois échecs sont UN SEUL bug de mesure : on jugeait au **R² poolé**, dominé par la variance
INTER-état (où se trouve l'agent), alors que le planner compare 117 candidats DANS LE MÊME ÉTAT.
Un modèle qui prédit parfaitement la moyenne par état et classe au hasard obtient déjà un bon R²
poolé — c'est littéralement le « inné +0,437 » qu'on avait pris pour une référence solide.
Ce gate ne mesure donc QUE le rang intra-état.

LE PIÈGE QU'IL FALLAIT RÉSOUDRE. Vérifier un classement exige la vraie cible pour PLUSIEURS
candidats du même instant. Un corpus n'en observe qu'UN (celui qui a été exécuté). La sortie n'est
ni le rêve du WM (biaisé, circulaire) ni une re-collecte (chère) : le monde est GELÉ et le corps
CINÉMATIQUE, donc la vraie cible des 117 candidats se CALCULE — c'est le simulateur déjà validé par
diag_consequence_g0 (selfcheck : variance intra exactement nulle à horizon trivial).

CE QUE MESURE CE GATE — le PLAFOND. Les cibles sont exactes, sans bruit d'observation. Si une tête
ne sait pas classer AVEC des cibles parfaites, elle ne le saura jamais avec des cibles vécues.
Un échec ici est donc décisif ; un succès est nécessaire, pas suffisant.

CIBLE (LeCun, `‖IC(s_{τ+δ}) − TC(s_τ)‖²` — mais en RÉSIDU) :
    y = (faim(τ+δ) − faim(τ) + drain·δ) / restore  ≈ nombre de repas dans la fenêtre
Régresser le NIVEAU absolu recopierait la faim courante (déterministe et déjà connue) : c'est le
mécanisme exact de l'échec n°2. Mesuré bimodal (0 repas 46 % / 1 repas 47 % / 2 repas 6 %).

Usage:
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_critic_rank_gate.py [--selfcheck]
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import random
import statistics as st
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "_cons", str(Path(__file__).with_name("diag_consequence_g0.py")))
_CONS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CONS)

DRAIN, RESTORE = _CONS.DRAIN, _CONS.RESTORE
CANDS = [(vx, om) for vx in _CONS.VX_GRID for om in _CONS.OM_GRID]

# --------------------------------------------------------------- CRITÈRES PRÉ-INSCRITS
#     écrits AVANT de lancer — la barre ne bouge pas après avoir vu les chiffres
HEADING_WEIGHT = 2.0    # command_planner.py:62, defaut SERVI par le harnais
HEADING_FAR_GATE = 2.0  # command_planner.py:69

BAR_PAIRWISE = 0.65     # accuracy pairwise intra-état, sur les paires DÉPARTAGEABLES
BAR_TAU = 0.30          # Kendall tau intra-état moyen
BAR_VS_INNE = 0.05      # il faut battre le coût analytique d'au moins ça en pairwise
PERM_LO, PERM_HI = 0.45, 0.55   # contrôle : cibles permutées -> le hasard, sinon fuite


def sample_states(n: int, seed: int) -> list[tuple]:
    """États de départ tirés dans le monde GELÉ. Chaque état porte son propre monde (bosquets
    partiellement épuisés) : c'est ce qui fait varier la cible d'un état à l'autre."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        patches = _CONS.make_world(rng)
        r = _CONS.PATCH_SPACING * 0.7 * math.sqrt(rng.random())
        a = rng.random() * 2 * math.pi
        x, z = r * math.cos(a), r * math.sin(a)
        yaw = rng.random() * 2 * math.pi
        hunger = rng.uniform(30.0, 95.0)
        for q in patches:
            if rng.random() < 0.4:
                q[2] = max(0.0, q[2] - 1)
        out.append((x, z, yaw, hunger, patches))
    return out


def token(state: tuple, cand: tuple) -> list[float]:
    """Ce que la tête VOIT : l'état perçu + le candidat. Volontairement pauvre et interprétable —
    on mesure si l'information SUFFIT, pas si un gros réseau la trouve.
    Contient la faim (échec n°3 : un critique aveugle à la cause dominante ne peut rien prédire)."""
    x, z, yaw, hunger, patches = state
    vx, om = cand
    best = None
    for q in patches:
        if q[2] <= 0:
            continue
        d = math.hypot(q[0] - x, q[1] - z)
        if best is None or d < best[0]:
            brg = math.atan2(q[0] - x, q[1] - z) - yaw
            best = (d, (brg + math.pi) % (2 * math.pi) - math.pi)
    d, brg = best if best else (15.0, 0.0)
    # bearing APRÈS le virage commandé : c'est là que le candidat agit
    brg_after = brg - om / 0.6 * _CONS.TURN * 60
    return [hunger / 100.0, d / 10.0, math.cos(brg), abs(math.sin(brg)),
            math.cos(brg_after), abs(math.sin(brg_after)), vx, abs(om),
            1.0 if best else 0.0]


def innate_score(state: tuple, cand: tuple, replan: int, horizon: int = 80) -> float:
    """LE VRAI coût analytique du planner en mono-pulsion, reproduit à l'identique.

    Source : command_planner.py:578-583, branche `plan_wm_slot` — celle qu'émettent réellement les
    corpus mono-drive (`reason: "plan_wm_slot"`, vérifié). Formule :
        score = −min_dist + heading_weight·mean_align + energy_weight·E_fin − done_penalty·P(chute)
    avec `mean_align = mean(cos(bearing) · min(dist/heading_far_gate, 1))`.

    Ce qui est reproduit FIDÈLEMENT : `−min_dist` (distance MINIMALE atteinte sur toute la
    trajectoire — le terme DOMINANT, et celui que mon premier proxy avait omis, d'où un classement
    sous le hasard) et le terme de cap avec sa porte de distance. Coefficients = les défauts servis
    (heading_weight 2.0, heading_far_gate 2.0).

    Ce qui est OMIS, et déclaré : `energy_weight·E_fin` et `done_penalty·P(chute)` viennent de têtes
    du WM. En corps cinématique `has_fallen` est câblé à False (sylvan_agent.gd:756) donc la
    pénalité de chute est nulle par construction ; et E_fin ne dépend quasiment pas du candidat sur
    un horizon de 80 pas (drain constant). Leur omission ne peut donc pas inverser un classement.
    """
    x, z, yaw, _hunger, patches = state
    vx, om = cand
    tgt = None
    for q in patches:
        if q[2] <= 0:
            continue
        d = math.hypot(q[0] - x, q[1] - z)
        if tgt is None or d < tgt[0]:
            tgt = (d, q)
    if tgt is None:
        return 0.0
    _, q = tgt

    # rollout du candidat sur l'horizon du planner (analytique : le corps obéit exactement)
    min_dist, aligns = float("inf"), []
    cx, cz, cyaw = x, z, yaw
    for _ in range(horizon):
        cyaw += om / 0.6 * _CONS.TURN
        step = vx / 0.75 * _CONS.SPEED
        cx += math.sin(cyaw) * step
        cz += math.cos(cyaw) * step
        d = math.hypot(q[0] - cx, q[1] - cz)
        min_dist = min(min_dist, d)
        brg = math.atan2(q[0] - cx, q[1] - cz) - cyaw
        brg = (brg + math.pi) % (2 * math.pi) - math.pi
        aligns.append(math.cos(brg) * min(d / HEADING_FAR_GATE, 1.0))
    return -min_dist + HEADING_WEIGHT * (sum(aligns) / len(aligns))


def build(states: list[tuple], delta: int, replan: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pour chaque état, la VRAIE cible des 117 candidats, calculée analytiquement."""
    X, Y, G = [], [], []
    for gi, sdata in enumerate(states):
        x, z, yaw, hunger, patches = sdata
        for cand in CANDS:
            h_end = _CONS.simulate(x, z, yaw, hunger, patches, cand[0], cand[1],
                                   replan, delta, 0.0)
            Y.append((h_end - hunger + DRAIN * delta) / RESTORE)   # ~ repas dans la fenêtre
            X.append(token(sdata, cand))
            G.append(gi)
    return np.array(X), np.array(Y), np.array(G)


def rank_metrics(pred: np.ndarray, y: np.ndarray, g: np.ndarray) -> dict:
    """Uniquement du RANG, uniquement INTRA-état. Les paires ex-æquo en cible sont ÉCARTÉES :
    les compter comme des succès gonflerait le score sans qu'aucune décision soit prise."""
    accs, taus, regrets = [], [], []
    for gi in np.unique(g):
        m = g == gi
        yy, pp = y[m], pred[m]
        ok = tot = conc = disc = 0
        for i in range(len(yy)):
            for j in range(i + 1, len(yy)):
                if abs(yy[i] - yy[j]) < 1e-9:
                    continue                      # non départageable
                tot += 1
                same = (pp[i] > pp[j]) == (yy[i] > yy[j])
                ok += same
                conc += same
                disc += not same
        if tot == 0:
            continue
        accs.append(ok / tot)
        taus.append((conc - disc) / tot)
        regrets.append(float(yy.max() - yy[int(np.argmax(pp))]) * RESTORE)  # en pts de jauge
    return dict(pairwise=st.mean(accs) if accs else 0.5,
                tau=st.mean(taus) if taus else 0.0,
                regret=st.mean(regrets) if regrets else 0.0,
                n_states=len(accs))


def fit_ridge(Xtr, Ytr, Xte, lam=1e-2):
    A = np.c_[Xtr, np.ones(len(Xtr))]
    w = np.linalg.lstsq(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Ytr, rcond=None)[0]
    return np.c_[Xte, np.ones(len(Xte))] @ w


def run(n_states: int, delta: int, replan: int, seed: int, folds: int = 4) -> dict:
    states = sample_states(n_states, seed)
    X, Y, G = build(states, delta, replan)
    inn = np.array([innate_score(states[G[i]], CANDS[i % len(CANDS)], replan) for i in range(len(X))])

    # SPLIT PAR ÉTAT (l'analogue du split par vie) : jamais deux candidats du même état
    # de part et d'autre de la frontière, sinon la tête a déjà vu la réponse.
    res = {k: [] for k in ("learned", "innate", "perm")}
    rng = np.random.default_rng(seed)
    for f in range(folds):
        te_states = [gi for gi in np.unique(G) if gi % folds == f]
        te = np.isin(G, te_states)
        tr = ~te
        if te.sum() < len(CANDS) or tr.sum() < len(CANDS):
            continue
        res["learned"].append(rank_metrics(fit_ridge(X[tr], Y[tr], X[te]), Y[te], G[te]))
        res["innate"].append(rank_metrics(inn[te], Y[te], G[te]))
        # CONTRÔLE : cibles permutées DANS chaque état -> toute structure est détruite
        Yp = Y.copy()
        for gi in te_states:
            m = G == gi
            v = Yp[m]
            rng.shuffle(v)
            Yp[m] = v
        res["perm"].append(rank_metrics(fit_ridge(X[tr], Y[tr], X[te]), Yp[te], G[te]))
    agg = {k: {m: st.mean([r[m] for r in v]) for m in ("pairwise", "tau", "regret")}
           for k, v in res.items() if v}
    return agg


def selfcheck() -> int:
    states = sample_states(6, seed=0)
    X, Y, G = build(states, delta=600, replan=60)
    assert len(X) == 6 * len(CANDS), (len(X), len(CANDS))
    print(f"  [ok] {len(CANDS)} candidats x 6 etats = {len(X)} exemples, cibles calculees")

    # un ORACLE (la cible elle-meme) doit obtenir un rang parfait
    o = rank_metrics(Y, Y, G)
    assert o["pairwise"] > 0.999 and o["regret"] < 1e-6, o
    print(f"  [ok] oracle : pairwise {o['pairwise']:.3f}, regret {o['regret']:.3f} — la metrique est saine")

    # un predicteur ALEATOIRE doit tomber au hasard
    rng = np.random.default_rng(0)
    r = rank_metrics(rng.normal(size=len(Y)), Y, G)
    assert 0.35 < r["pairwise"] < 0.65, r
    print(f"  [ok] hasard : pairwise {r['pairwise']:.3f} ~ 0.5")

    # inverser la cible doit inverser le rang
    inv = rank_metrics(-Y, Y, G)
    assert inv["pairwise"] < 0.001, inv
    print(f"  [ok] predicteur inverse : pairwise {inv['pairwise']:.3f} ~ 0")
    print("SELFCHECK PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--states", type=int, default=120)
    ap.add_argument("--delta", type=int, default=600)
    ap.add_argument("--replan", type=int, default=60, help="60 = valeur retenue le 2026-07-23")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()

    print("=" * 78)
    print("GATE HORS-LIGNE DU CRITIQUE — rang INTRA-ÉTAT, jamais le R² poolé")
    print("=" * 78)
    print(f"  monde gelé bosquets_v2, replan={a.replan}, delta={a.delta}, {a.states} états, "
          f"{len(CANDS)} candidats/état")
    print(f"  CRITÈRES PRÉ-INSCRITS : pairwise > {BAR_PAIRWISE} ET tau > {BAR_TAU} "
          f"ET appris − inné > {BAR_VS_INNE}")
    print(f"  CONTRÔLE : cibles permutées doivent retomber dans [{PERM_LO}, {PERM_HI}]\n")

    agg = run(a.states, a.delta, a.replan, a.seed)
    print(f"  {'prédicteur':<26s} {'pairwise':>10s} {'Kendall tau':>13s} {'regret@1 (pts)':>16s}")
    for k, lbl in (("innate", "VRAI coût analytique"), ("learned", "tête apprise (ridge)"),
                   ("perm", "contrôle permuté")):
        if k in agg:
            m = agg[k]
            print(f"  {lbl:<26s} {m['pairwise']:>10.3f} {m['tau']:>13.3f} {m['regret']:>16.2f}")

    print("\n  VERDICT :")
    if "learned" not in agg:
        print("    [NUL] pas assez de données"); return 0
    L, I, P = agg["learned"], agg.get("innate", {}), agg.get("perm", {})
    gain = L["pairwise"] - I.get("pairwise", 0.5)
    if P and not (PERM_LO <= P["pairwise"] <= PERM_HI):
        print(f"    [NUL] contrôle permuté à {P['pairwise']:.3f}, hors [{PERM_LO}, {PERM_HI}] — fuite.")
    elif L["pairwise"] > BAR_PAIRWISE and L["tau"] > BAR_TAU and gain > BAR_VS_INNE:
        print(f"    [PASS] la tête classe ({L['pairwise']:.3f}) et bat l'inné de {gain:+.3f}.")
    elif gain <= BAR_VS_INNE:
        print(f"    [ÉCHEC] la tête ne bat pas l'inné ({gain:+.3f} ≤ {BAR_VS_INNE}). "
              "L'inné reste le meilleur classeur — ne pas promouvoir.")
    else:
        print(f"    [ÉCHEC] pairwise {L['pairwise']:.3f} ou tau {L['tau']:.3f} sous la barre.")
    print("\n  ⚠️ Ce gate mesure un PLAFOND : cibles exactes, aucun bruit d'observation.")
    print("     Un échec ici est décisif ; un succès est nécessaire, pas suffisant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
