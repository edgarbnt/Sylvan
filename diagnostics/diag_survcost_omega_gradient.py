"""diag_survcost_omega_gradient — sonde GRATUITE post-KILL : le coût survie a-t-il un gradient
de VIRAGE (en ω), ou est-il ~plat (knife-edge) comme l'hypothèse n°1 le prédit ?

CONTEXTE (2026-07-03). A/B forage_ab_survcost = KILL (ON 2190 vs OFF 2600) avec MOINS
d'engagements (151 poursuites vs 262, morts de faim à 0 repas). Hypothèse falsifiable n°1
(précédent : knife-edge 2026-06-18) : la phase-2 de `_survival_extension` ignore l'ORIENTATION de
fin d'arc (chaque candidat est traité comme téléporté-cap-sur-cible) → sur un horizon de 80 pas les
positions de fin diffèrent peu → score ~plat en ω → le choix se fait au bruit, l'entité n'engage pas.

MÉTHODE (offline, vrai WM promu, vraies proprios des logs A/B ; `plan(debug_scores=True)`) :
géométries synthétiques contrôlées via override_pos (with_slot débranché LE TEMPS DE LA SONDE —
on sonde le COÛT, pas la perception) : ressource URGENTE placée front/side/rear × near/far,
l'autre ressource à l'opposé. Pour chaque (frame réelle, géométrie, mode designed|survival) :
  - Δ_toward = moyenne(score des candidats qui TOURNENT VERS la cible urgente, |ω|>=0.3)
             − moyenne(score des candidats qui tournent À L'OPPOSÉ), normalisée par le std des scores.
  - argmax-vers : le candidat choisi tourne-t-il vers la cible urgente (side/rear) ?

CRITÈRES PRÉ-ENREGISTRÉS :
  - HYPOTHÈSE CONFIRMÉE si, sur REAR : argmax-vers(survival) < 0.5 ET argmax-vers(designed) >= 0.7,
    OU Δ_toward(survival) < 0.3 × Δ_toward(designed).
  - INFIRMÉE si survival >= designed sur les deux mesures REAR → chercher ailleurs (marge, drain).

Lancer :  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_survcost_omega_gradient.py [--selfcheck]

RÉSULTATS :
- RUN 1 (2026-07-03, code du KILL) : verdict formel « mitigé » (ma métrique Δ_toward, basée sur l'ω
  du 1er pas, est polluée par les candidats 2-segments) MAIS découverte sans ambiguïté = **score
  survie PLAT (std=0.000) sur TOUTES les géométries far** : tout le monde survit au cap (égalité de
  temps) et la marge min-sur-toute-l'alternance converge vers le régime permanent (les différences
  initiales ±0.04 sont absorbées) → au-delà de ~2 m l'entité choisissait AU BRUIT. Cause racine plus
  profonde que l'orientation seule.
- FIX (même jour, `_survival_extension`) : (1) marge = PREMIÈRE arrivée seulement (le plan ne
  contrôle que son 1er leg) ; (2) 1er leg paie le temps de virage |bearing fin d'arc|/surv_turn_rate.
- RUN 2 (post-fix) : gradient restauré — std far 0.000→64-318 ; argmax-vers : side 0.83-0.92 et
  rear 0.67 vs designed 0.42-0.50 → le coût survie s'engage désormais MIEUX que le designed sur la
  sonde. (La ligne « INFIRMÉE » du run 2 compare aux critères écrits pour l'ANCIEN code — caduque.)
  Gate cheap du re-A/B : PASSÉ. Juge final = forage_ab_survcost.sh (mêmes critères pré-enregistrés).
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch

from sylvan.control.planning.command_planner import CommandPlanConfig, CommandPlanner
from sylvan.models.command_wm import CommandWorldModel

WM_CKPT = "data/checkpoints/wm_objcentric_s1/wm_best.pt"
FRAMES_SRC = "data/replay_buffer/ab_survcost_ON/ep_0000.jsonl"
N_FRAMES = 12
E_LOW, T_HIGH = 0.25, 0.85          # la FAIM est l'urgente (cas symétrique testé aussi)
GEOMS = {"front": 0.0, "side": math.pi / 2, "rear": math.pi}
DISTS = {"near": 2.0, "far": 5.0}
OMEGA_MIN_TURN = 0.3


def bearing_to_xz(bearing: float, dist: float) -> tuple[float, float]:
    return (dist * math.sin(bearing), dist * math.cos(bearing))  # (x_right, z_fwd)


def load_frames(path: str, n: int) -> list[torch.Tensor]:
    """Vraies obs WM-rétine (proprio132 ++ retina144 ++ énergie1) espacées dans le log A/B."""
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    step = max(1, len(rows) // n)
    frames = []
    for r in rows[:: step][:n]:
        obs = list(r["obs"]["proprio"]) + list(r["wm"]["retina0"]) + [E_LOW]
        frames.append(torch.tensor(obs, dtype=torch.float32))
    return frames


def probe(planner: CommandPlanner, obs: torch.Tensor, urgent_food: bool,
          bearing: float, dist: float) -> dict:
    urgent_xy = bearing_to_xz(bearing, dist)
    other_xy = bearing_to_xz(bearing + math.pi, 4.0)      # l'autre ressource à l'opposé, 4 m
    food_xy, water_xy = (urgent_xy, other_xy) if urgent_food else (other_xy, urgent_xy)
    e, t = (E_LOW, T_HIGH) if urgent_food else (T_HIGH, E_LOW)
    out = planner.plan(obs, radar=[0.0] * 12, override_pos=True,
                       food_override=food_xy, water_override=water_xy,
                       energy=e, thirst=t, debug_scores=True)
    scores = np.asarray(out["scores"], dtype=np.float64)
    om0 = np.asarray([c[1] for c in out["cand_cmd0"]], dtype=np.float64)
    toward_sign = 1.0 if math.sin(bearing) > 0 else -1.0  # bearing>0 = à droite = ω>0 (convention radar)
    toward = om0 * toward_sign >= OMEGA_MIN_TURN
    away = om0 * toward_sign <= -OMEGA_MIN_TURN
    std = float(scores.std()) + 1e-9
    d_toward = float((scores[toward].mean() - scores[away].mean()) / std) if toward.any() and away.any() else float("nan")
    chose_toward = float(out["command"][1]) * toward_sign > 0.0
    return {"d_toward": d_toward, "chose_toward": chose_toward, "std": std,
            "reason": out["reason"]}


def selfcheck() -> None:
    x, z = bearing_to_xz(0.0, 2.0)
    assert abs(x) < 1e-9 and abs(z - 2.0) < 1e-9          # front = droit devant (+z)
    x, z = bearing_to_xz(math.pi / 2, 2.0)
    assert abs(x - 2.0) < 1e-9 and abs(z) < 1e-9          # side = à droite (+x)
    x, z = bearing_to_xz(math.pi, 2.0)
    assert abs(x) < 1e-6 and abs(z + 2.0) < 1e-9          # rear = derrière (−z)
    print("[selfcheck] OK — géométrie bearing→(x_right,z_fwd)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return

    payload = torch.load(WM_CKPT, map_location="cpu", weights_only=False)
    meta = payload["meta"]
    wm = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
                           predictor_arch=meta.get("predictor_arch", "shallow"),
                           with_slot=meta.get("with_slot", False),
                           slot_resources=meta.get("slot_resources", 1))
    wm.load_state_dict(payload["model"])
    wm.eval()
    wm.with_slot = False                                   # sonde du COÛT : géométrie contrôlée par override

    # Mêmes réglages que l'A/B (horizon 80 ; bras ON : drain/restore calés éco de vie).
    cfg_designed = CommandPlanConfig(horizon=80)
    cfg_surv = CommandPlanConfig(horizon=80, cost_mode="survival",
                                 resource_drain=0.0005, resource_restore=0.4)
    planners = {"designed": CommandPlanner(wm, cfg_designed), "survival": CommandPlanner(wm, cfg_surv)}

    frames = load_frames(FRAMES_SRC, N_FRAMES)
    print(f"frames réelles={len(frames)} (source {FRAMES_SRC})")
    print(f"{'géométrie':<12}{'mode':<10}{'Δ_toward (norm)':>16}{'argmax-vers':>13}{'std score':>11}")
    summary: dict[tuple[str, str], dict] = {}
    for gname, bearing in GEOMS.items():
        for dname, dist in DISTS.items():
            for mode, planner in planners.items():
                res = [probe(planner, o, uf, bearing, dist)
                       for o in frames for uf in (True, False)]
                d = float(np.nanmean([r["d_toward"] for r in res]))
                frac = float(np.mean([r["chose_toward"] for r in res]))
                std = float(np.mean([r["std"] for r in res]))
                summary[(f"{gname}-{dname}", mode)] = {"d": d, "frac": frac}
                print(f"{gname}-{dname:<7}{mode:<10}{d:>16.2f}{frac:>13.2f}{std:>11.3f}")

    print("\n--- VERDICT (critères pré-enregistrés, géométries REAR) ---")
    rear = {(g, m): v for (g, m), v in summary.items() if g.startswith("rear")}
    confirmed = infirmed = 0
    for dname in DISTS:
        g = f"rear-{dname}"
        s, dgn = rear[(g, "survival")], rear[(g, "designed")]
        conf = (s["frac"] < 0.5 and dgn["frac"] >= 0.7) or (not math.isnan(s["d"]) and not math.isnan(dgn["d"])
                                                            and dgn["d"] > 0 and s["d"] < 0.3 * dgn["d"])
        inf = s["frac"] >= dgn["frac"] and (math.isnan(dgn["d"]) or s["d"] >= dgn["d"])
        print(f"{g}: survival frac={s['frac']:.2f} Δ={s['d']:.2f} | designed frac={dgn['frac']:.2f} "
              f"Δ={dgn['d']:.2f} → {'CONFIRME' if conf else ('INFIRME' if inf else 'mitigé')}")
        confirmed += conf
        infirmed += inf
    if confirmed:
        print("\nHYPOTHÈSE CONFIRMÉE (score survie ~plat/faux en ω sur cible derrière) → fix principiel :")
        print("temps de trajet phase-2 += |bearing fin d'arc|/ω_max (tourner coûte du temps → de la survie).")
    elif infirmed == len(DISTS):
        print("\nHYPOTHÈSE INFIRMÉE → le déficit d'engagement vient d'ailleurs (marge ? calibrage drain ?)"
              " — ne PAS appliquer le fix orientation, re-diagnostiquer.")
    else:
        print("\nMITIGÉ → lire les sous-scores par géométrie avant de trancher.")


if __name__ == "__main__":
    main()
