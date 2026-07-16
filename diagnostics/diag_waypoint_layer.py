"""SMOKE SYNTHÉTIQUE GRATUIT — l'étage waypoint décide-t-il ce qu'il doit, hors ligne ?

Scénarios joués sans Godot ni WM (géométrie pure, déterministe) :
  A. Monde plat (zéro vert) → DIRECT, aucun commit (G0 structurel : l'étage est transparent).
  B. Mur vert ENTRE l'entité et la cible → le direct est pénalisé, un waypoint LATÉRAL commit
     (l'échappée est un changement de mode — TangentBug).
  C. Machine à états : wp devant + commandes d'avance → événement « reached » avant timeout.
  D. Commandes nulles → événement « timeout » à timeout_steps.
  E. Hystérésis : du vert HORS de la ligne ne fait pas quitter le direct.
  F. maybe_decide : première décision, re-check périodique, re-décision sur cible téléportée.

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_waypoint_layer.py
"""

from __future__ import annotations

import math

from sylvan.control.waypoint_layer import (N_RAY, RETINA_RANGE_M, WaypointConfig, WaypointLayer,
                                           green_points)


def retina_with_greens(points: list[tuple[float, float]]) -> list[float]:
    """Rétine 144 vide + un rayon VERT par point ego (x_right, z_fwd) demandé."""
    ret = [1.0, 0.0, 0.0, 0.0] * N_RAY
    for x, z in points:
        k = round(math.atan2(x, z) / (2.0 * math.pi / N_RAY)) % N_RAY
        d = math.hypot(x, z) / RETINA_RANGE_M
        ret[4 * k:4 * k + 4] = [min(d, 0.99), 0.1, 0.9, 0.15]
    return ret


def main() -> None:
    cfg = WaypointConfig()

    # A. monde plat → direct, pas de commit
    lay = WaypointLayer(cfg)
    rec = lay.maybe_decide("food", (0.0, 4.0), retina_with_greens([]))
    assert rec is not None and rec["choice"] == "direct" and not lay.active(), rec
    print(f"A. plat → direct sans commit ✓  {rec}")

    # B. mur vert entre entité et cible → waypoint latéral commité
    lay = WaypointLayer(cfg)
    greens = [(-0.4, 2.0), (0.0, 2.0), (0.4, 2.0)]
    assert len(green_points(retina_with_greens(greens))) == 3
    rec = lay.decide("food", (0.0, 4.0), retina_with_greens(greens))
    assert rec["choice"] == "waypoint" and lay.active(), rec
    assert abs(lay.wp[0]) > 1.0, f"waypoint pas latéral : {lay.wp}"
    assert rec["cost_direct"] > rec["cost_best_wp"], rec
    print(f"B. mur vert → wp latéral commité ✓  {rec}")

    # C. atteinte : wp devant, avance → 'reached' avant timeout
    lay = WaypointLayer(cfg)
    lay.wp, lay.target_id, lay.leg_steps = (0.0, 2.5), "food", 0
    steps = 0
    while lay.active() and steps < cfg.timeout_steps:
        lay.tick((0.75, 0.0))
        steps += 1
    assert lay.consume_event() == "reached" and lay.n_reached == 1, (steps, lay.n_timeouts)
    print(f"C. atteinte en {steps} pas (< timeout {cfg.timeout_steps}) ✓")

    # D. timeout : commandes nulles
    lay = WaypointLayer(cfg)
    lay.wp, lay.target_id, lay.leg_steps = (0.0, 2.5), "food", 0
    for _ in range(cfg.timeout_steps + 1):
        lay.tick((0.0, 0.0))
    assert lay.consume_event() == "timeout" and lay.n_timeouts == 1
    print(f"D. timeout à {cfg.timeout_steps} pas ✓")

    # E. hystérésis : vert hors de la ligne → direct conservé
    lay = WaypointLayer(cfg)
    rec = lay.decide("food", (0.0, 4.0), retina_with_greens([(3.0, 2.0)]))
    assert rec["choice"] == "direct" and not lay.active(), rec
    print(f"E. vert hors-ligne → direct conservé ✓  {rec}")

    # F. cadence de maybe_decide : 1re décision, silence, re-check périodique, re-décision sur saut
    lay = WaypointLayer(cfg)
    ret = retina_with_greens([])
    assert lay.maybe_decide("food", (0.0, 4.0), ret) is not None          # 1re fois
    quiet = [lay.maybe_decide("food", (0.0, 3.9), ret) for _ in range(cfg.recheck_every - 1)]
    assert all(r is None for r in quiet), quiet                            # silence entre re-checks
    assert lay.maybe_decide("food", (0.0, 3.8), ret) is not None          # re-check périodique
    assert lay.maybe_decide("food", (2.5, -3.0), ret) is not None         # cible téléportée (>1.5 m)
    assert lay.maybe_decide("water", (1.0, 1.0), ret) is not None         # changement de cible
    print("F. cadence maybe_decide (1re / silence / périodique / saut / cible) ✓")

    print("\nSMOKE OK — l'étage waypoint est géométriquement et machinalement sain (offline).")


if __name__ == "__main__":
    main()
