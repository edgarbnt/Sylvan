"""SONDE GRATUITE de la QUEUE ANALYTIQUE du coût survie (`_survival_extension`).

POURQUOI. L'A/B `nominal_speed` 0.02 -> 0.010 a rendu un KILL (portée [6,8) m : 47,2 -> 32,5 %) :
mettre la VRAIE vitesse du corps DÉGRADE l'entité. La règle pré-inscrite exige de NOMMER ce que la
valeur fausse compensait — mais on ne peut le nommer qu'en regardant le bon code. ⚠️ Ma première
attribution était FAUSSE : j'avais accusé `deficit = relu(d/spd x drain - niveau)`
(command_planner.py:1084), qui est dans la branche `else` de `surv_mode` donc INERTE en config
vivante (`surv_mode` retourne à L1063). Le chemin réellement exécuté est `_survival_extension`.

CE QUE FAIT LA SONDE. Elle appelle `_survival_extension` DIRECTEMENT sur des états synthétiques
(aucun WM, aucun Godot, aucun run) et décompose son score. Le score vaut `time + margin_w x margin` ;
en appelant deux fois — une fois avec `margin_w=0` — on isole les deux termes sans toucher au code :
    time   = score(margin_w=0)
    margin = (score(margin_w=W) - time) / W
On peut alors dire LEQUEL des deux termes retourne la préférence quand la vitesse est corrigée.

CLI :
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_survival_tail.py [--selfcheck]
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from sylvan.control.planning.command_planner import _survival_extension  # noqa: E402

# Valeurs SERVIES par les harnais vivants (collect_arb_graded.sh), pas les défauts du code.
DRAIN, DRAIN_T, RESTORE = 0.0005, 0.00035, 0.4
CAP, MARGIN_W = 3000.0, 200.0
TURN_RATE = 0.015           # MESURÉ correct en Phase 1
SPD_DECLARED, SPD_MEASURED = 0.02, 0.010


def decompose(df: float, dw: float, e: float, t: float, *, spd: float,
              dist_fw: float = 4.0, bearing_f: float = 0.0,
              bearing_w: float = 0.0) -> dict:
    """Score de la queue pour (bouffe d'abord / eau d'abord), décomposé en `time` et `margin`."""
    def call(margin_w: float) -> tuple[torch.Tensor, torch.Tensor]:
        return _survival_extension(
            torch.tensor([df]), torch.tensor([dw]),
            torch.tensor([e]), torch.tensor([t]),
            torch.ones(1), torch.zeros(1),
            dist_fw, DRAIN, RESTORE, spd, CAP, margin_w,
            turn_f=torch.tensor([abs(bearing_f) / TURN_RATE]),
            turn_w=torch.tensor([abs(bearing_w) / TURN_RATE]),
            gamma=0.0, drain_t=DRAIN_T,
        )
    tf0, tw0 = call(0.0)
    tfW, twW = call(MARGIN_W)
    time_f, time_w = float(tf0), float(tw0)
    return {
        "score_food": float(tfW), "score_water": float(twW),
        "time_food": time_f, "time_water": time_w,
        "margin_food": (float(tfW) - time_f) / MARGIN_W,
        "margin_water": (float(twW) - time_w) / MARGIN_W,
    }


def _table() -> None:
    """La question DÉCISIVE : à partir de quelle distance la queue abandonne-t-elle la bouffe ?

    On fixe l'eau à 4 m et on éloigne la bouffe. `sign(score_food - score_water)` dit quel ordre la
    queue préfère. Si la vraie vitesse fait basculer plus TÔT vers l'eau, on tient le mécanisme.
    """
    print("Queue analytique du coût survie — l'eau est fixée à 4 m, la bouffe s'éloigne.")
    print("État : énergie 0.50, soif 0.50 (les deux à mi-jauge), cibles DEVANT (virage nul).\n")
    print(f"{'d_bouffe':>9} | {'DÉCLARÉ spd=0.020':^34} | {'MESURÉ spd=0.010':^34}")
    print(f"{'':>9} | {'Δscore':>9} {'Δtime':>9} {'Δmargin':>9} {'pref':>4} "
          f"| {'Δscore':>9} {'Δtime':>9} {'Δmargin':>9} {'pref':>4}")
    print("-" * 92)
    for df in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0):
        cells = []
        for spd in (SPD_DECLARED, SPD_MEASURED):
            r = decompose(df, 4.0, 0.50, 0.50, spd=spd)
            ds = r["score_food"] - r["score_water"]
            dt = r["time_food"] - r["time_water"]
            dm = r["margin_food"] - r["margin_water"]
            cells.append(f"{ds:9.1f} {dt:9.1f} {dm:9.3f} {'FOOD' if ds > 0 else 'EAU':>4}")
        print(f"{df:9.1f} | {cells[0]} | {cells[1]}")

    print("\nLecture : `pref` = l'ordre que la queue préfère. Un basculement vers EAU plus TÔT sous la")
    print("vitesse MESURÉE = la queue abandonne la bouffe lointaine plus tôt — le mécanisme du KILL.")
    print("`Δtime` vs `Δmargin` dit LEQUEL des deux termes porte le basculement (margin est pondéré")
    print(f"par margin_w={MARGIN_W:.0f}, donc un Δmargin de 0.05 pèse {0.05 * MARGIN_W:.0f} points de score).")


def _mortality() -> None:
    """Le premier leg est-il jugé MORTEL ? (le vrai « abandon » : le candidat meurt en route)"""
    print("\n\nMORTALITÉ IMAGINÉE DU PREMIER LEG (bouffe d'abord, eau à 4 m).")
    print("La queue déclare le candidat mort si le trajet dépasse le temps-avant-mort.")
    print(f"{'énergie':>8} | {'d_max atteignable DÉCLARÉ 0.020':>32} | {'MESURÉ 0.010':>14}")
    print("-" * 62)
    for e in (0.2, 0.3, 0.4, 0.5, 0.7, 0.9):
        row = []
        for spd in (SPD_DECLARED, SPD_MEASURED):
            dmax = 0.0
            for df in [x / 10 for x in range(5, 161)]:
                r = decompose(df, 4.0, e, 0.9, spd=spd)
                if r["margin_food"] > 0.0:          # margin > 0 <=> arrivée VIVANTE au 1er leg
                    dmax = df
                else:
                    break
            row.append(dmax)
        print(f"{e:8.2f} | {row[0]:32.1f} | {row[1]:14.1f}")
    print("\nLecture : distance max à laquelle la queue croit ARRIVER VIVANTE. Sous la vraie vitesse")
    print("elle est mécaniquement divisée par 2 — tout ce qui est au-delà est score comme fatal.")


def _selfcheck() -> None:
    # non-régression : drain_t=None doit être bit-identique à drain_t=drain (corps symétrique)
    a = _survival_extension(torch.tensor([3.0]), torch.tensor([4.0]), torch.tensor([0.5]),
                            torch.tensor([0.5]), torch.ones(1), torch.zeros(1),
                            4.0, DRAIN, RESTORE, 0.01, CAP, MARGIN_W, gamma=0.0, drain_t=None)
    b = _survival_extension(torch.tensor([3.0]), torch.tensor([4.0]), torch.tensor([0.5]),
                            torch.tensor([0.5]), torch.ones(1), torch.zeros(1),
                            4.0, DRAIN, RESTORE, 0.01, CAP, MARGIN_W, gamma=0.0, drain_t=DRAIN)
    assert float(a[0]) == float(b[0]), "drain_t=None doit être identique à drain_t=drain"
    # la décomposition doit reconstituer le score
    r = decompose(3.0, 4.0, 0.5, 0.5, spd=0.01)
    assert abs((r["time_food"] + MARGIN_W * r["margin_food"]) - r["score_food"]) < 1e-3
    # doubler la vitesse doit réduire le trajet donc AUGMENTER la marge d'arrivée
    slow = decompose(6.0, 4.0, 0.5, 0.5, spd=0.010)["margin_food"]
    fast = decompose(6.0, 4.0, 0.5, 0.5, spd=0.020)["margin_food"]
    assert fast > slow, f"marge attendue plus grande a vitesse imaginee plus haute : {fast} vs {slow}"
    print("selfcheck OK — non-régression drain_t, décomposition exacte, monotonie de la marge")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    if ap.parse_args().selfcheck:
        _selfcheck()
        return
    _table()
    _mortality()


if __name__ == "__main__":
    main()
