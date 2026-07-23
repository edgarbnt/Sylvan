"""G0 CONSÉQUENCE — que faudrait-il pour qu'une COMMANDE engage l'avenir ? FREE (0 run/Godot/train).

LA QUESTION, posée par le KILL du 2026-07-23. Le critique appris est mort hors-ligne sur un chiffre :
`Var_intra-état / Var_totale = 8,2 %` (barre 10 %). Traduit : dans un état donné, la faim future ne
dépend presque pas de la commande choisie maintenant. Le planner classe 117 candidats qui mènent
tous au même endroit. Aucune tête de valeur ne peut classer ce qui ne diffère pas.

Deux causes possibles, que le chiffre d'hier ne SÉPARE pas :
  (A) LE CORPS est trop récupérable — il obéit exactement à (vx, ω) et pivote sur place, donc un
      mauvais virage se défait au tick suivant ;
  (B) LE PLANNER replanifie trop souvent — même avec un corps engageant, re-décider tous les
      10 ticks efface le choix précédent.

Elles n'ont pas le même prix. (A) = changer le corps, donc probablement recollecter le WM. (B) =
un paramètre du planner. Avant de payer (A), il faut savoir si (B) suffit.

CE QUE FAIT LA SONDE. On simule le monde GELÉ (bosquets_v1) hors moteur, et pour chaque état
échantillonné on déroule les 117 candidats du planner : chacun est TENU pendant `replan` ticks,
puis une politique gloutonne prend le relais jusqu'à τ+δ. On mesure alors la dispersion de la faim
finale ENTRE CANDIDATS partant du MÊME état — c'est exactement la variance intra-état, mais cette
fois avec de VRAIS contrefactuels, ce que le corpus ne pouvait pas fournir (il n'observe qu'un seul
candidat exécuté par tick).

MODÈLE DE CORPS PARAMÉTRÉ. `tau` = constante de temps d'inertie, en ticks. tau=0 reproduit le corps
actuel (obéissance exacte, mesurée 0,011 m/tick et 0,015 rad/tick). tau>0 filtre la commande :
la vitesse et la rotation mettent ~tau ticks à suivre l'ordre, donc un virage engage réellement.

CE QUE CETTE SONDE NE DIT PAS. Elle borne ce que la GÉOMÉTRIE autorise, pas ce que l'entité fera.
Un ratio élevé ne promet pas qu'un critique apprendra — il dit seulement qu'il aurait quelque chose
à apprendre. Un ratio bas, lui, est décisif : il n'y a rien à classer, et c'est une propriété du
substrat, pas de la tête.

Usage:
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_consequence_g0.py [--selfcheck]
"""

from __future__ import annotations

import argparse
import math
import random
import statistics as st

# --------------------------------------------------------------- constantes MESURÉES (preset gelé)
SPEED = 0.011          # m/tick à vx=0.75 (mesuré, téléports filtrés)
TURN = 0.015           # rad/tick à |ω|=0.6 (mesuré sur le torse)
DRAIN = 0.05           # points de jauge / tick
RESTORE = 40.0
GAUGE_MAX = 100.0
EAT_R = 1.0
PATCH_SPACING = 9.0
N_PATCH = 4
BERRIES = 2

# grille de candidats du planner (command_planner.py:52-53)
VX_GRID = (0.55, 0.65, 0.75)
OM_GRID = (-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6)

# barre pré-inscrite, reprise du KILL d'hier
BAR_INTRA = 10.0       # % — en dessous, il n'y a rien à classer


def make_world(rng: random.Random) -> list[list[float]]:
    """4 bosquets sur un carré de côté PATCH_SPACING, chacun avec ses baies. Géométrie du preset."""
    h = PATCH_SPACING / 2.0
    return [[x, z, float(BERRIES)] for x, z in ((-h, -h), (h, h), (-h, h), (h, -h))]


def simulate(x: float, z: float, yaw: float, hunger: float, patches: list[list[float]],
             vx: float, om: float, replan: int, delta: int, tau: float) -> float:
    """Tenir (vx, om) pendant `replan` ticks, puis glouton jusqu'à delta. Rend la faim finale.

    `tau` = inertie en ticks. À tau=0 la commande est suivie exactement (corps actuel). À tau>0,
    la vitesse et la rotation relaxent vers la consigne avec un facteur exp(-1/tau) : un virage
    engage le corps au-delà du tick où il est décidé.
    """
    p = [list(q) for q in patches]
    v_cur = om_cur = 0.0
    alpha = 1.0 if tau <= 0 else 1.0 - math.exp(-1.0 / tau)

    for t in range(delta):
        if t < replan:
            v_t, om_t = vx, om                      # le candidat, TENU
        else:
            # relais glouton : viser le bosquet non vide le plus proche
            best = None
            for q in p:
                if q[2] <= 0:
                    continue
                d = math.hypot(q[0] - x, q[1] - z)
                if best is None or d < best[0]:
                    best = (d, q)
            if best is None:
                v_t, om_t = 0.75, 0.0
            else:
                _, q = best
                brg = math.atan2(q[0] - x, q[1] - z) - yaw
                brg = (brg + math.pi) % (2 * math.pi) - math.pi
                om_t = max(-0.6, min(0.6, brg / (TURN * 10)))
                v_t = 0.75 if abs(brg) < 0.6 else 0.55

        v_cur += alpha * (v_t - v_cur)
        om_cur += alpha * (om_t - om_cur)
        yaw += om_cur / 0.6 * TURN
        step = v_cur / 0.75 * SPEED
        x += math.sin(yaw) * step
        z += math.cos(yaw) * step

        hunger -= DRAIN
        for q in p:
            if q[2] > 0 and math.hypot(q[0] - x, q[1] - z) <= EAT_R:
                q[2] -= 1
                hunger = min(GAUGE_MAX, hunger + RESTORE)
                break
        if hunger <= 0:
            return 0.0
    return hunger


def assess(tau: float, replan: int, delta: int, n_states: int, seed: int) -> dict:
    """Dispersion de la faim finale ENTRE candidats, à état de départ égal."""
    rng = random.Random(seed)
    cands = [(vx, om) for vx in VX_GRID for om in OM_GRID]
    per_state_var, all_vals = [], []
    for _ in range(n_states):
        patches = make_world(rng)
        r = PATCH_SPACING * 0.7 * math.sqrt(rng.random())
        a = rng.random() * 2 * math.pi
        x, z = r * math.cos(a), r * math.sin(a)
        yaw = rng.random() * 2 * math.pi
        hunger = rng.uniform(30.0, 95.0)
        for q in patches:                            # épuisement partiel réaliste
            if rng.random() < 0.4:
                q[2] = max(0.0, q[2] - 1)
        vals = [simulate(x, z, yaw, hunger, patches, vx, om, replan, delta, tau)
                for vx, om in cands]
        per_state_var.append(st.pvariance(vals))
        all_vals += vals
    v_intra = st.mean(per_state_var)
    v_tot = st.pvariance(all_vals)
    return dict(intra=v_intra, tot=v_tot,
                ratio=100.0 * v_intra / v_tot if v_tot > 1e-9 else 0.0)


def selfcheck() -> int:
    # tau=0 doit reproduire l'obéissance exacte : un pas droit avance de SPEED
    h = simulate(0.0, 0.0, 0.0, 100.0, [[99.0, 99.0, 0.0]], 0.75, 0.0, 1, 1, 0.0)
    assert abs((100.0 - h) - DRAIN) < 1e-9, h
    print(f"  [ok] tau=0 : un tick coûte exactement {DRAIN} de jauge (obéissance exacte)")

    # avec inertie, le corps met du temps à atteindre la consigne -> il avance MOINS vite au début
    def dist(tau):
        x = z = yaw = 0.0
        v = 0.0
        alpha = 1.0 if tau <= 0 else 1.0 - math.exp(-1.0 / tau)
        for _ in range(10):
            v += alpha * (0.75 - v)
            z += v / 0.75 * SPEED
        return z
    d0, d5 = dist(0.0), dist(5.0)
    assert d0 > d5 > 0, (d0, d5)
    print(f"  [ok] inertie : 10 ticks parcourent {d0:.4f} m à tau=0 contre {d5:.4f} m à tau=5")

    # manger doit remonter la jauge
    h2 = simulate(0.0, 0.0, 0.0, 50.0, [[0.0, 0.5, 1.0]], 0.75, 0.0, 1, 1, 0.0)
    assert h2 > 50.0, h2
    print(f"  [ok] manger remonte la jauge : 50 -> {h2:.1f}")

    # un état où AUCUN candidat ne change rien doit donner une variance intra nulle
    r = assess(tau=0.0, replan=1, delta=2, n_states=3, seed=0)
    assert r["intra"] < 1e-6, r
    print(f"  [ok] horizon trivial (delta=2) : variance intra {r['intra']:.2e} ~ 0")
    print("SELFCHECK PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--delta", type=int, default=600)
    ap.add_argument("--states", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()

    print("=" * 78)
    print("G0 CONSÉQUENCE — une commande engage-t-elle l'avenir ?")
    print("=" * 78)
    print(f"  monde gelé bosquets_v1, delta={a.delta} ticks, {a.states} états, "
          f"{len(VX_GRID) * len(OM_GRID)} candidats par état")
    print(f"  barre pré-inscrite (KILL du 2026-07-23) : Var_intra / Var_totale > {BAR_INTRA} %\n")

    print("  AXE A — INERTIE DU CORPS (tau en ticks ; 0 = corps actuel), replan=10 (actuel)")
    print(f"  {'tau':>6s} {'Var intra':>11s} {'Var totale':>11s} {'ratio':>8s}   verdict")
    for tau in (0.0, 5.0, 15.0, 30.0, 60.0):
        r = assess(tau, replan=10, delta=a.delta, n_states=a.states, seed=a.seed)
        v = "au-dessus de la barre" if r["ratio"] > BAR_INTRA else "rien à classer"
        print(f"  {tau:>6.0f} {r['intra']:>11.2f} {r['tot']:>11.2f} {r['ratio']:>7.1f}%   {v}")

    print("\n  AXE B — ENGAGEMENT DU PLANNER (replan en ticks ; 10 = actuel), corps INCHANGÉ (tau=0)")
    print(f"  {'replan':>6s} {'Var intra':>11s} {'Var totale':>11s} {'ratio':>8s}   verdict")
    for rp in (10, 30, 60, 120, 300):
        r = assess(0.0, replan=rp, delta=a.delta, n_states=a.states, seed=a.seed)
        v = "au-dessus de la barre" if r["ratio"] > BAR_INTRA else "rien à classer"
        print(f"  {rp:>6d} {r['intra']:>11.2f} {r['tot']:>11.2f} {r['ratio']:>7.1f}%   {v}")

    print("\n  LECTURE : si l'axe B suffit, le levier est un PARAMÈTRE du planner (gratuit).")
    print("            Si seul l'axe A marche, il faut changer le CORPS (et recollecter le WM).")
    print("            Si aucun ne marche, la conséquence ne vient pas de là — chercher ailleurs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
