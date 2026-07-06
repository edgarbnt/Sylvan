"""diag_orbit_scoring, GRATUIT (offline, WM promu, aucun run Godot).

QUESTION (2026-07-06). Mesure : le planner (cout survie) n'orbite les cibles LOINTAINES parce que
son score est PLAT et l'omega choisi est DECORRELE de la direction bouffe (corr -0.09). On veut
savoir POURQUOI, pour cibler le fix (pur > echafaudage flagge) :
  (a) AUCUN candidat n'APPROCHE la bouffe lointaine dans le reve (spread min_df ~0) -> l'horizon 80
      / le jeu de candidats ne peut pas produire l'approche -> fix = terme d'attraction/heading a
      distance (echafaudage flagge) ou horizon effectif plus long ;
  (b) des candidats APPROCHENT (min_df varie) mais le SCORE ne les classe pas premiers
      (corr(score, -min_df) faible) -> l'AGREGATION du score est cassee a distance -> fix = scoring.

On balaie la distance de la bouffe (proche 3 m ou l'agent close, loin 6-8 m ou il orbite) et on lit,
via plan(debug_scores=True), par candidat : (omega du 1er pas, min_df, df_end, score).

CRITERES PRE-ENREGISTRES :
- si a distance LOIN : spread(min_df) < 0.5 m -> cause (a) (aucun candidat n'approche) ;
- si spread(min_df) >= 1 m mais corr(score, -min_df) < 0.3 -> cause (b) (agregation cassee) ;
- si corr(omega_1er, vers-bouffe) monte avec la distance proche->loin -> confirme le decrochage.

Lancer : PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_orbit_scoring.py [--selfcheck]
"""
from __future__ import annotations

import argparse
import math

import torch

WM_CKPT = "data/checkpoints/wm_objcentric_s2/wm_best.pt"


def _corr(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / ((vx * vy) ** 0.5 + 1e-9)


def selfcheck() -> None:
    assert abs(_corr([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-6
    assert abs(_corr([1, 2, 3], [3, 2, 1]) + 1.0) < 1e-6
    print("[selfcheck] OK (corr)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return

    import os
    os.environ["SYLVAN_PLANNER_COST"] = "survival"
    os.environ["SYLVAN_PLANNER_DRAIN"] = "0.0005"
    os.environ["SYLVAN_PLANNER_RESTORE"] = "0.4"
    os.environ["SYLVAN_MULTI_SLOT2"] = "0"   # bouffe via override (retine vide -> gate slot2 mettrait None)
    os.environ["SYLVAN_MULTI_FOOD_SLOT"] = "0"
    from sylvan.control.planning.command_planner import CommandPlanner, CommandPlanConfig
    from sylvan.models.command_wm import CommandWorldModel

    pl = torch.load(WM_CKPT, map_location="cpu", weights_only=False)
    m = pl["meta"]
    wm = CommandWorldModel(obs_dim=m["obs_dim"], proprio_dim=m["proprio_dim"],
                           predictor_arch=m.get("predictor_arch", "shallow"),
                           with_slot=True, slot_resources=2)
    wm.load_state_dict(pl["model"])
    wm.eval()
    wm.food_idx = 0
    wm.water_idx = 1
    _H = int(os.environ.get("SYLVAN_DIAG_HORIZON", "80"))
    planner = CommandPlanner(wm, CommandPlanConfig(horizon=_H, cost_mode="survival"))
    print(f"[diag] horizon={_H} far_align={os.environ.get('SYLVAN_PLANNER_FAR_ALIGN','0')} "
          f"gain={os.environ.get('SYLVAN_PLANNER_ALIGN_GAIN','?')} pivot={os.environ.get('SYLVAN_PLANNER_PIVOT','0')}")

    # VRAI proprio+retine (le reve du WM est irrealiste depuis une obs a zero) : on prend un etat
    # reel d'un buffer de foraging, on n'override QUE la position bouffe (le point qu'on teste).
    import glob as _g
    import json as _j
    obs = torch.zeros(m["obs_dim"])
    _seen = 0
    for f in sorted(_g.glob("data/replay_buffer/mode1_bc_a/ep_0000.jsonl")):
        for line in open(f):
            r = _j.loads(line)
            pr = r["obs"]["proprio"]; ret = r["wm"]["retina0"]
            if len(pr) == 132 and len(ret) == 144:
                _seen += 1
                if _seen >= 300:      # etat EN MOUVEMENT (pas le depart a l'arret)
                    obs = torch.tensor(pr + ret + [float(r["obs"]["energy"]) / 100.0], dtype=torch.float32)
                    break
        break
    assert obs.abs().sum() > 0, "pas de vrai proprio charge"
    water = [0.0] * 36
    water[9] = 0.7            # eau ~2 m a droite

    print(f"{'dist bouffe':>12}{'bearing':>9}{'vx choisi':>11}{'omega choisi':>14}{'spread min_df':>15}"
          f"{'mindf_best':>12}{'dfend_best':>12}{'straight20':>11}{'corr(om,f)':>12}")
    for dist in (3.0, 5.0, 7.0):
        for bearing_deg in (30.0, 90.0):      # bouffe a 30 deg (front-lat) et 90 deg (plein cote)
            b = math.radians(bearing_deg)
            food = (dist * math.sin(b), dist * math.cos(b))   # (x_right, z_fwd)
            out = planner.plan(obs, radar=[0.0] * 12, water_radar=water,
                               energy=0.30, thirst=0.90,
                               override_pos=True, food_override=food, water_override=(2.0, 0.0),
                               debug_scores=True)
            scores = out.get("scores")
            mindf = out.get("min_df")
            cmd0 = out.get("cand_cmd0")
            if not scores:
                print(f"{dist:>12.1f}{bearing_deg:>9.0f}  (pas de debug -> branche differente: {out.get('reason')})")
                continue
            oms = [c[1] for c in cmd0]
            # "vers-bouffe" : omega du signe qui reduit le bearing. food a droite (x>0) -> tourner a droite.
            toward = [om * (1.0 if food[0] > 0 else -1.0) for om in oms]
            spread = max(mindf) - min(mindf)
            c_score = _corr(scores, [-d for d in mindf])
            c_om = _corr(scores, toward)     # le score prefere-t-il tourner vers la bouffe ?
            vx_best, om_best = out["command"][0], out["command"][1]
            best_i = max(range(len(scores)), key=lambda i: scores[i])
            dfend = out.get("df_end")
            mindf_best = mindf[best_i]
            dfend_best = dfend[best_i] if dfend else float("nan")
            # COMMIT vs SPIRALE : fraction de pas DROITS (|om|<0.2) du candidat choisi, sur les 20 premiers
            # pas (ce qui sera EXÉCUTÉ avant le prochain replan) — commit droit = fraction haute, spirale = 0.
            seq = planner._cmd_seqs[best_i]
            straight20 = float((seq[:20, 1].abs() < 0.2).float().mean())
            print(f"{dist:>12.1f}{bearing_deg:>9.0f}{vx_best:>11.2f}{om_best:>14.2f}{spread:>15.2f}"
                  f"{mindf_best:>12.2f}{dfend_best:>12.2f}{straight20:>11.2f}{c_om:>12.2f}")

    print("\n(a) spread min_df ~0 a distance loin -> aucun candidat n'approche (horizon/candidats) ;")
    print("(b) spread grand mais corr(score,-min_df) faible -> agregation du score cassee a distance.")


if __name__ == "__main__":
    main()
