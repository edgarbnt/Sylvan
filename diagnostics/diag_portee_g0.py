"""G0 GRATUIT — QU'EST-CE QUI l'empêche d'attraper une proie qui est déjà à portée ?

Aucun Godot, aucun WM, aucun entraînement : simulation analytique avec les constantes MESURÉES
du corps servi, en ablation FACTORIELLE. On ne cherche pas à confirmer une intuition, on cherche
lequel des facteurs candidats fait s'effondrer la capture.

POINT DE DÉPART (mesuré le 2026-08-02 sur corpus réel) :
  · la proie est à moins de 3 m 45 % du temps, vue 50 % des ticks à 2,93 m
  · MAIS seulement 7 % des approches closent sous 1 m, et 49 % des approches sous 3 m ne
    donnent AUCUN repas
  · dans le dernier mètre, le rapprochement est de ZÉRO exact alors qu'elle bouge vite
  · elle se déplace en permanence à ~50° de la direction de sa cible

CE QUE `diag_prey_interception.py` A DÉJÀ TRANCHÉ (négatif banké, ne pas y revenir) : viser où la
proie SERA au lieu de où elle EST n'apporte RIEN (+0,0 % de capture à tous les ratios). La piste
« interception/prédiction » est morte. ⚠️ Mais ce diagnostic tournait avec `SPEED=0.011` — le corps
d'AVANT. Le corps servi fait **0,0469 m/tick** (mesuré), donc le ratio proie/agent réel est 0,49,
un régime où ce même simulateur prédit ~100 % de capture. **Il y a donc un facteur qui manque à ce
modèle**, et c'est lui qu'on cherche ici.

LES QUATRE CANDIDATS, tous MESURÉS aujourd'hui, testés en ablation :
  1. VIRAGE       — rayon de braquage 2,00 m en croisière pour une bouche de 1,00 m : elle ne peut
                    pas virer À L'INTÉRIEUR de sa propre allonge.
  2. RALENTI      — dans le dernier mètre la vitesse DEMANDÉE tombe de 0,600 à 0,250 (soit
                    0,025 m/tick réels), c'est-à-dire la vitesse de la proie (0,023) : le
                    rapprochement net devient ~0.
  3. VISÉE        — 23° d'erreur de gisement médiane sur la position lue.
  4. INTERMITTENCE— la cible n'est vue que 50 % des ticks (hors-cône, occlusion).

Usage :
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_portee_g0.py
"""

from __future__ import annotations

import argparse
import itertools
import math

import torch

# --- constantes MESURÉES du corps servi (2026-08-02) ------------------------------------------
SPEED = 0.0469  # m/tick — croisière mesurée (diag_viabilite_monde)
PREY = 0.023  # m/tick — SYLVAN_FOOD_PREY_SPEED du preset foret_v1
CAPTURE = 1.0  # m — eat_radius
# ⚠️ BUDGET RÉALISTE, corrigé le 2026-08-02. Une 1ʳᵉ version laissait 2000 pas (= 94 m de trajet) :
# les 16 lignes de l'ablation rendaient 100 % et le test ne mesurait RIEN. Avec autant de temps,
# même un mauvais poursuivant finit par attraper une proie à 3-11 m. La contrainte réelle est
# MÉTABOLIQUE : 60 de jauge / 0,154 par pas = 389 pas d'autonomie, et il faut aussi BOIRE dans ce
# budget. On juge donc « attrape-t-elle DANS SON BUDGET », la seule question qui décide de sa vie.
MAX_TICKS = 390

TURN_FREE = 0.075  # rad/tick — la valeur OPTIMISTE de diag_prey_interception (rayon 0,63 m)
TURN_REAL = SPEED / 2.00  # rad/tick — rayon de braquage MESURÉ 2,00 m en croisière
SLOW_SPEED = 0.025  # m/tick — vitesse réelle mesurée dans le dernier mètre
SLOW_RANGE = 1.5  # m — distance sous laquelle elle ralentit
BEARING_ERR = math.radians(23.1)  # erreur de gisement médiane mesurée
SEEN_P = 0.50  # part des ticks où la cible est réellement vue


def run(n: int, seed: int, turn: float, slow: bool, bearing: bool, blind: bool) -> float:
    """n poursuites indépendantes → taux de capture. Poursuite pure (l'interception est réfutée)."""
    g = torch.Generator().manual_seed(seed)
    d0 = 3.0 + 8.0 * torch.rand(n, generator=g)
    a0 = 2 * math.pi * torch.rand(n, generator=g)
    prey = torch.stack([d0 * torch.cos(a0), d0 * torch.sin(a0)], 1)
    pos = torch.zeros(n, 2)
    head = 2 * math.pi * torch.rand(n, generator=g)
    pdir = 2 * math.pi * torch.rand(n, generator=g)
    pdir = torch.stack([torch.cos(pdir), torch.sin(pdir)], 1)
    caught = torch.zeros(n, dtype=torch.bool)
    last = prey.clone()  # dernière position CONNUE (sert quand la cible n'est pas vue)

    for _ in range(MAX_TICKS):
        rel = prey - pos
        dist = rel.norm(dim=1)
        caught |= (~caught) & (dist <= CAPTURE)
        if bool(caught.all()):
            break

        # INTERMITTENCE : si la cible n'est pas vue ce tick, on vise sa dernière position connue.
        if blind:
            seen = torch.rand(n, generator=g) < SEEN_P
            last = torch.where(seen.unsqueeze(1), prey, last)
            target = last
        else:
            target = prey

        want = torch.atan2(target[:, 1] - pos[:, 1], target[:, 0] - pos[:, 0])
        if bearing:  # VISÉE bruitée — l'erreur de gisement de la perception servie
            want = want + torch.randn(n, generator=g) * BEARING_ERR
        err = torch.remainder(want - head + math.pi, 2 * math.pi) - math.pi
        head = head + err.clamp(-turn, turn)

        v = torch.full((n,), SPEED)
        if slow:  # RALENTI terminal — ce que le planner fait réellement près du but
            v = torch.where(dist < SLOW_RANGE, torch.full_like(v, SLOW_SPEED), v)
        step = torch.stack([torch.cos(head), torch.sin(head)], 1) * v.unsqueeze(1)
        pos = torch.where(caught.unsqueeze(1), pos, pos + step)

        away = (prey - pos) / (prey - pos).norm(dim=1, keepdim=True).clamp_min(1e-6)
        near = (dist < 3.0).float().unsqueeze(1)
        nd = pdir * 0.97 + away * 0.03 + near * away * 0.15
        pdir = nd / nd.norm(dim=1, keepdim=True).clamp_min(1e-6)
        prey = torch.where(caught.unsqueeze(1), prey, prey + pdir * PREY)

    return float(caught.float().mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"corps MESURÉ : {SPEED:.4f} m/tick · proie {PREY:.3f} (ratio {PREY / SPEED:.2f}) · "
          f"bouche {CAPTURE} m · {args.n} poursuites/ligne")
    print(f"virage : libre {TURN_FREE:.4f} rad/tick (rayon {SPEED / TURN_FREE:.2f} m) "
          f"| RÉEL {TURN_REAL:.4f} (rayon 2,00 m)\n")

    # CONTRÔLE : à proie immobile et sans handicap, la capture doit être totale. Sinon le
    # simulateur est biaisé et rien de ce qui suit ne vaut.
    ctrl = run(args.n, args.seed, TURN_FREE, False, False, False)
    print(f"CONTRÔLE (aucun handicap) : capture {100 * ctrl:.1f} %  "
          f"{'✅' if ctrl > 0.95 else '❌ simulateur biaisé, résultats non exploitables'}\n")

    print("ABLATION FACTORIELLE — capture % :")
    print(f"  {'virage':>8} {'ralenti':>8} {'visée':>7} {'intermit.':>10} | {'capture':>8}")
    rows = []
    for tr, sl, be, bl in itertools.product((False, True), repeat=4):
        rate = run(args.n, args.seed, TURN_REAL if tr else TURN_FREE, sl, be, bl)
        rows.append(((tr, sl, be, bl), rate))
        print(f"  {'RÉEL' if tr else 'libre':>8} {'oui' if sl else 'non':>8} "
              f"{'23°' if be else '0°':>7} {'50%' if bl else '100%':>10} | {100 * rate:7.1f} %")

    base = dict(rows)[(False, False, False, False)]
    print("\nCOÛT DE CHAQUE FACTEUR SEUL (par rapport au contrôle) :")
    for i, name in enumerate(("virage réel (rayon 2 m)", "ralenti terminal",
                              "visée 23°", "intermittence 50 %")):
        key = tuple(j == i for j in range(4))
        r = dict(rows)[key]
        print(f"  {name:26s} : {100 * base:5.1f} % → {100 * r:5.1f} %   "
              f"(perte {100 * (base - r):5.1f} pts)")

    worst = min(rows, key=lambda kv: kv[1])
    print(f"\n  pire combinaison : {100 * worst[1]:.1f} % de capture")
    print("\n  RÉALITÉ MESURÉE, pour comparaison : 49-51 % des approches sous 3 m donnent un")
    print("  repas, et 7 % seulement closent sous 1 m. La ligne du tableau qui s'en approche")
    print("  DÉSIGNE le facteur limitant — c'est lui qu'il faut traiter, et lui seul.")


if __name__ == "__main__":
    main()
