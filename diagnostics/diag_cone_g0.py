"""G0 of the cone-vision chantier (docs/design_vision_cone.md) — FREE (0 run/Godot/train).

Gates the expensive WM retrain: applies a frontal cone OFFLINE to the existing 360-degree corpora
and measures, per half-angle theta, whether a cone creates the structure we claim BEFORE paying
anything:
  (A) ACTIVE-PERCEPTION need: how often is the needed resource OUT of the cone (must turn to see)?
      In 360 this is 0 (everything visible). A substantial fraction => the cone forces orientation.
  (B) MEMORY place: seen-then-lost-and-feasible per 24 lives, cone-gated (in 360 it was ~0 = the
      memory G0 STOP). Reuses the dead-reckon feasibility of diag_memory_g0.

Everything is computed from the TRUE bearing of each sighting (|ray|*10 deg): "in cone at theta"
== |bearing| <= theta. First-order caveat (stated in the design doc): the corpora were lived under
360 vision, so this uses the orientations it ACTUALLY had; a real cone would change behaviour
(turn-to-scan). If even its 360-trajectory puts the resource out-of-cone often, a cone definitely
creates out-of-sight -> that is the gate.

Pre-registered verdict (design doc): pick the SMALLEST theta with (A) substantial AND (B) feasible
seen-then-lost > noise (5/24 lives) AND reachable. If no theta creates the place (or it is
infeasible) -> STOP, 360 stays, the retrain is not paid.

Run (repo root):
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_cone_g0.py [--selfcheck]
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

sys.path.insert(0, "diagnostics")
from diag_memory_g0 import (CAP_RADIUS, CRIT_DRIVE, DEATH_THR, EGO_CKPT, LOST_MIN_STEPS, MAX_AGE,
                            N_RAYS, REACH_PER_UNIT, _needy_sighting, load_lives)
from diag_arbitrage_g0 import NEEDED_RES

sys.path.insert(0, "python")
from sylvan.control.slot_memory import transport_geom
from sylvan.models.egomotion_head import load_egomotion_head
from sylvan.control.mode1.obs import RED, BLUE

THETAS = [90, 120, 150, 180]           # half-angles (180 = full 360, the baseline)
DEFAULT_RUNS = [
    "data/replay_buffer/critic_kin_g24as1", "data/replay_buffer/critic_kin_g24as2",
    "data/replay_buffer/critic_kin_g24bs1", "data/replay_buffer/critic_kin_g24bs2",
    "data/replay_buffer/critic_kin_spx3", "data/replay_buffer/critic_kin_spx4",
    "data/replay_buffer/critic_kin_judge1", "data/replay_buffer/critic_kin_judge2",
    "data/replay_buffer/critic_kin_pure1", "data/replay_buffer/critic_kin_pure2",
    "data/replay_buffer/critic_kin_arb3", "data/replay_buffer/critic_kin_arb4",
]


def sighting(retina: list[float], color: str) -> tuple[float, float] | None:
    """(distance_m, |bearing|_deg) of the closest ray of this colour, or None.

    _needy_sighting returns (x_right, z_fwd, signed_ray, depth_norm); the ray is spaced 10 deg
    (ray 0 = forward), so |bearing| = |signed_ray| * 10.
    """
    s = _needy_sighting(retina, color)
    if s is None:
        return None
    x, z, ray, _ = s
    return math.hypot(x, z), abs(ray) * 10.0


def measure_active(lives: list[dict], theta: float) -> dict:
    """Over under-pressure ticks: needed resource in-cone / out-cone-in-range / out-of-range."""
    c = {"in_cone": 0, "out_cone_in_range": 0, "out_of_range": 0}
    for lf in lives:
        for i in range(len(lf["drives"])):
            e, t, h = lf["drives"][i]
            if min(e, t) >= 50 or min(e, t) <= DEATH_THR:
                continue
            urg = "energy" if e <= t else "thirst"
            col = RED if NEEDED_RES[urg] == "food" else BLUE
            r = lf["retina"][i]
            if len(r) < 4 * N_RAYS:
                continue
            s = sighting(r, col)
            if s is None:
                c["out_of_range"] += 1
            elif s[1] <= theta:
                c["in_cone"] += 1
            else:
                c["out_cone_in_range"] += 1
    return c


def seen_then_lost_cone(life: dict, ego, theta: float, reach_factor: float = 0.5) -> str | None:
    """Cone-gated seen-then-lost class for a DRIVE-death life (None if not a drive death).

    'seen' == sighted AND in-cone (|bearing| <= theta). Feasibility = dead-reckon the last in-cone
    position to the critical onset and check it stays within the sustainable capture radius.
    """
    e, t, h = life["drives"][-1]
    if h <= DEATH_THR or (e > DEATH_THR and t > DEATH_THR):
        return None                              # danger death or truncated: not a drive death
    dying = "energy" if e <= t else "thirst"
    needed = NEEDED_RES[dying]
    di = 0 if dying == "energy" else 1
    col = RED if needed == "food" else BLUE
    n = len(life["drives"])
    crit_start = 0
    for i in range(n - 1, -1, -1):
        if life["drives"][i][di] >= CRIT_DRIVE:
            crit_start = i
            break
    last_before = None
    seen_in_crit = False
    seen_ever = False
    for i in range(n):
        r = life["retina"][i]
        if len(r) < 4 * N_RAYS:
            continue
        s = _needy_sighting(r, col)
        if s is None or abs(s[2]) * 10.0 > theta:    # not sighted, or out of cone
            continue
        seen_ever = True
        if i >= crit_start:
            seen_in_crit = True
        else:
            last_before = (s, i)
    if not seen_ever:
        return "jamais_vue_cone"
    if seen_in_crit or last_before is None or crit_start - last_before[1] < LOST_MIN_STEPS:
        return "seen_in_critical"
    (sx, sz, _, _), t0 = last_before
    belief = [sx, sz]
    age = 0
    for i in range(t0, crit_start):
        pr = life["proprio"][i]
        if len(pr) != 132:
            continue
        dyaw, dfwd, dlat = ego.predict(pr)
        belief = transport_geom(belief, dyaw, dfwd, dlat)
        age += 1
    dist = math.hypot(belief[0], belief[1])
    reserve = life["drives"][crit_start][di]
    reach = reserve * REACH_PER_UNIT * reach_factor
    if age <= MAX_AGE and age * 0.002 <= CAP_RADIUS and dist <= reach:
        return "seen_then_lost_feasible"
    return "seen_then_lost_infeasible"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    runs = [r for r in args.runs if glob.glob(os.path.join(r, "ep_*.jsonl*"))]
    if not runs:
        raise SystemExit("aucun run trouvé")
    ego = load_egomotion_head(EGO_CKPT)
    all_lives = [lf for run in runs for lf in load_lives(run)]
    n_lives = len(all_lives)

    if args.selfcheck:
        assert n_lives > 100, "trop peu de vies"
        a180 = measure_active(all_lives[:50], 180)
        assert a180["out_cone_in_range"] == 0, "à 360 rien ne doit être hors-cône"
        a90 = measure_active(all_lives[:50], 90)
        assert a90["out_cone_in_range"] > 0, "à ±90 une part doit être hors-cône"
        print(f"[selfcheck] {n_lives} vies ; 360 hors-cône=0 ✓ ; ±90 hors-cône>0 ✓")
        return

    print(f"{n_lives} vies, {len(runs)} runs. Cône appliqué OFFLINE (1er ordre : orientations vécues en 360).\n")
    print("(A) BESOIN DE PERCEPTION ACTIVE — ressource urgente HORS-CÔNE (doit tourner pour voir) :")
    print(f"{'demi-angle':>11s} | {'in-cône':>8s} | {'HORS-cône en portée':>20s} | {'hors-portée':>11s}")
    for th in THETAS:
        c = measure_active(all_lives, th)
        tot = sum(c.values())
        lbl = "360° (réf)" if th == 180 else f"±{th}°"
        print(f"{lbl:>11s} | {100*c['in_cone']/tot:>7.1f}% | {100*c['out_cone_in_range']/tot:>19.1f}% | {100*c['out_of_range']/tot:>10.1f}%")

    print("\n(B) PLACE MÉMOIRE — 'vu-puis-perdu FAISABLE' cône-gaté (dead-reckon, conservateur) :")
    print(f"{'demi-angle':>11s} | {'faisable/24 vies':>16s} | {'brut':>5s} | {'seen_in_crit':>12s} | {'jamais_vue':>10s}")
    for th in THETAS:
        cnt = {"seen_then_lost_feasible": 0, "seen_then_lost_infeasible": 0,
               "seen_in_critical": 0, "jamais_vue_cone": 0}
        for lf in all_lives:
            k = seen_then_lost_cone(lf, ego, th)
            if k is not None:
                cnt[k] += 1
        feas = cnt["seen_then_lost_feasible"]
        feas24 = feas * 24.0 / max(n_lives, 1)
        lbl = "360° (réf)" if th == 180 else f"±{th}°"
        print(f"{lbl:>11s} | {feas24:>15.1f} | {feas:>5d} | {cnt['seen_in_critical']:>12d} | {cnt['jamais_vue_cone']:>10d}")

    print("\nVerdict (docs/design_vision_cone.md §G0) : plus petit θ avec (A) hors-cône substantiel")
    print("ET (B) faisable > 5/24 vies -> LICENCIÉ (retrain gaté). Sinon -> STOP, 360° reste.")


if __name__ == "__main__":
    main()
