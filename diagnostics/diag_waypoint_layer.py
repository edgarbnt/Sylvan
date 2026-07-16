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

    # G. LA GÉOMÉTRIE DE L'ÉCHEC G1 v0 : cible LOIN (7 m) derrière un NUAGE vert à mi-chemin →
    #    l'anneau seul donnait best_wp≈direct (le 2ᵉ segment reconverge) ; un candidat TANGENT
    #    (posé à côté de l'OBSTACLE) doit maintenant gagner avec un 2ᵉ segment réellement dégagé.
    lay = WaypointLayer(cfg)
    cloud = [(x, 3.85 + dz) for x in (-1.0, -0.5, 0.0, 0.5, 1.0) for dz in (-0.4, 0.4)]
    rec = lay.decide("food", (0.0, 7.0), retina_with_greens(cloud))
    assert rec["choice"] == "waypoint" and lay.active(), rec
    assert rec["cost_best_wp"] < rec["cost_direct"] * (1 - cfg.hysteresis), rec
    from sylvan.control.waypoint_layer import route_cost
    _, intr_committed = route_cost(lay.wp, (0.0, 7.0), green_points(retina_with_greens(cloud)), cfg)
    assert intr_committed < 0.2, f"le wp commité ne dégage pas la route : intr={intr_committed:.2f} wp={lay.wp}"
    print(f"G. cible 7 m derrière nuage vert → wp tangent commité, route dégagée ✓  wp={lay.wp} {rec}")

    # H. patience de bascule : 1 replan de flip = bruit (leg conservé) ; 2 consécutifs = abort
    lay = WaypointLayer(cfg)
    lay.wp, lay.target_id, lay.leg_steps = (2.0, 2.0), "food", 0
    lay.note_first_target("water")
    assert lay.active(), "abort au 1er flip (bruit) : patience non respectée"
    lay.note_first_target("food")          # retour → streak reset
    lay.note_first_target("water")
    assert lay.active(), "abort après flip isolé post-reset"
    lay.note_first_target("water")         # 2e consécutif → vraie bascule
    assert not lay.active() and lay.n_aborts == 1
    print("H. patience de bascule (1 flip=bruit conservé, 2 consécutifs=abort) ✓")

    # I. featurizer critique-waypoint : INVARIANCE MIROIR par construction (canonicalisation)
    from sylvan.control.waypoint_layer import WP_FEAT_DIM, candidate_features
    wp_c, tg_c, gr_c = (-1.8, 1.2), (0.5, 4.0), [(-0.7, 2.1), (0.4, 2.6)]
    f1 = candidate_features(wp_c, tg_c, gr_c)
    f2 = candidate_features((-wp_c[0], wp_c[1]), (-tg_c[0], tg_c[1]), [(-x, z) for x, z in gr_c])
    assert len(f1) == WP_FEAT_DIM and all(abs(a - b) < 1e-9 for a, b in zip(f1, f2)), (f1, f2)
    fd = candidate_features(tg_c, tg_c, gr_c)
    assert fd[-1] == 1.0 and f1[-1] == 0.0, (fd, f1)          # flag is_direct
    print("I. featurizer : miroir-invariant + flag direct ✓")

    # J. exploration + log : ε=1 → tout choix est exploré, uniforme, loggé, reproductible par seed
    import json as _json
    import os as _os
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        _os.environ["SYLVAN_WP_EXPLORE_EPS"] = "1.0"
        _os.environ["SYLVAN_WP_EXPLORE_SEED"] = "7"
        _os.environ["SYLVAN_WP_LOG"] = td
        try:
            lay = WaypointLayer(cfg)
            chosen = []
            for _ in range(30):
                lay.decide("food", (0.0, 5.0), retina_with_greens([(0.0, 2.5)]))
                chosen.append(lay.wp)
            rows = [_json.loads(l) for l in open(f"{td}/decisions.jsonl")]
        finally:
            for k in ("SYLVAN_WP_EXPLORE_EPS", "SYLVAN_WP_EXPLORE_SEED", "SYLVAN_WP_LOG"):
                _os.environ.pop(k, None)
    assert len(rows) == 30 and all(r["explore"] for r in rows)
    assert len({r["chosen"] for r in rows}) >= 5, "exploration pas assez uniforme"
    assert all(len(r["feats"]) == len(r["costs"]) and len(r["feats"][0]) == WP_FEAT_DIM for r in rows)
    assert any(r["chosen"] == 0 for r in rows), "le DIRECT doit aussi être explorable"
    print(f"J. exploration ε=1 : 30 décisions loggées, {len({r['chosen'] for r in rows})} candidats "
          f"distincts choisis, feats {WP_FEAT_DIM}-d ✓")

    # K. mode DOULEUR APPRISE (si le checkpoint des gates v2 existe) : mêmes comportements
    #    qualitatifs que l'analytique, SANS marge verte codée-main dans le scoring.
    import os as _os2
    ck = "data/checkpoints/waypoint_pain/pain_best.pt"
    if _os2.path.exists(ck):
        _os2.environ["SYLVAN_WP_PAIN_CRITIC"] = ck
        try:
            lay = WaypointLayer(cfg)
            assert lay.pain_critic is not None
            r_flat = lay.decide("food", (0.0, 4.0), retina_with_greens([]))
            assert r_flat["choice"] == "direct", r_flat            # plat → direct (douleur ~0 partout)
            lay2 = WaypointLayer(cfg)
            r_blk = lay2.decide("food", (0.0, 7.0), retina_with_greens(cloud))
            assert r_blk["choice"] == "waypoint" and lay2.active(), r_blk   # gardé → détour
        finally:
            _os2.environ.pop("SYLVAN_WP_PAIN_CRITIC", None)
        print(f"K. mode douleur : plat→direct, cible gardée→wp commité (zéro marge main) ✓  wp={lay2.wp}")
    else:
        print("K. (sauté : pas de checkpoint waypoint_pain)")

    print("\nSMOKE OK — l'étage waypoint est géométriquement et machinalement sain (offline).")


if __name__ == "__main__":
    main()
