"""G0 of the spatial-memory chantier (docs/design_memoire_spatiale.md) — FREE (0 run/Godot/train).

Reopens a CLOSED negative (occlusion gate "no gain") on a NEW falsifiable hypothesis: that gate
tested the WRONG question (never-seen objects under an OOD mask on a 360 WM). Here we quantify
memory's actual SWEET SPOT on the already-paid corpora: among ARBITRATION deaths, how many had the
needy resource SEEN-THEN-LOST (perceived in the retina, then out of view when it became critical)
= convertible by memory, vs NEVER-SEEN (perception/exploration, memory powerless) vs
SEEN-IN-CRITICAL (still in view = a choice/range problem, already refuted).

For the seen-then-lost deaths, replay the dead-reckoned belief (transport_geom + EgomotionHead on
the logged proprio, exactly like MultiSlotMemory) from the last sighting to the critical moment and
check it would have stayed within the sustainable capture radius = the NET place. Edge-of-field
control (dette #2 of the map): was the last sighting at the retina edge (noisy seed → ghost risk)?

Pre-registered verdict (the design doc):
  convertible seen-then-lost > noise (±5 / 24 lives) AND feasibility OK -> chantier licensed (G1);
  <= noise, OR dominated by never-seen -> the occlusion negative STANDS, STOP (real lock is
  perception/exploration); feasibility KO (drift > radius) -> short memory insufficient, STOP.

Run (repo root):
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_memory_g0.py [--selfcheck]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import sys

sys.path.insert(0, "diagnostics")
from diag_arbitrage_g0 import (CONSUME_JUMP, DEATH_THR, FRESH_H, NEEDED_RES, START_DRIVE,
                               START_TOL, classify_life)

sys.path.insert(0, "python")
from sylvan.control.mode1.obs import BLUE, RED, _color_gated_depths
from sylvan.control.slot_memory import transport_geom
from sylvan.models.egomotion_head import load_egomotion_head

RANGE_M = 10.0                 # perception.gd MAX_RANGE
N_RAYS = 36
DRAIN = 0.05                   # measured drive drain / step
SPEED = 0.01                   # MEASURED 2026-07-21: per-tick displacement is exactly 0.0100 m
                               # (p50=p90=p99). The old 0.02 was 2x TOO HIGH and made metabolic
                               # reach 2x optimistic -> deaths wrongly labelled "arbitration".
REACH_PER_UNIT = SPEED / DRAIN  # metres of reach per unit of drive reserve
CRIT_DRIVE = 30.0              # "critical" = needy drive entered the danger zone
LOST_MIN_STEPS = 30            # out of view at least this long before critical = truly lost
MAX_AGE = 500                  # geometric belief lifetime (drift ~0.2 m/100 steps vs radius 1.0 m)
CAP_RADIUS = 1.0               # capture radius (sustainable reach target)
EDGE_RAY = 9                   # |ray| beyond this (~90 deg from forward) = retina edge
EGO_CKPT = "data/checkpoints/egomotion_head/best.pt"

DEFAULT_RUNS = [
    "data/replay_buffer/critic_kin_g24as1", "data/replay_buffer/critic_kin_g24as2",
    "data/replay_buffer/critic_kin_g24bs1", "data/replay_buffer/critic_kin_g24bs2",
    "data/replay_buffer/critic_kin_spx3", "data/replay_buffer/critic_kin_spx4",
    "data/replay_buffer/critic_kin_judge1", "data/replay_buffer/critic_kin_judge2",
    "data/replay_buffer/critic_kin_pure1", "data/replay_buffer/critic_kin_pure2",
    "data/replay_buffer/critic_kin_arb3", "data/replay_buffer/critic_kin_arb4",
]

CLASSES = ["seen_then_lost", "seen_in_critical", "never_seen", "no_retina"]


def _open(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def _needy_sighting(retina: list[float], color: str) -> tuple[float, float, int, float] | None:
    """Closest ego position (x_right, z_fwd), its ray index and normalized depth, or None."""
    depths = _color_gated_depths(retina, color)
    m = min(depths)
    if m >= 0.999:
        return None
    k = depths.index(m)
    b = 2.0 * math.pi * k / N_RAYS
    ray = k if k <= N_RAYS // 2 else k - N_RAYS       # signed: 0 = forward, +/- to the sides
    return (m * RANGE_M * math.sin(b), m * RANGE_M * math.cos(b), ray, m)


def load_lives(run: str) -> list[dict]:
    """Life dicts compatible with classify_life, plus per-tick retina + proprio."""
    lives: list[dict] = []
    cur: dict | None = None
    prev_e = prev_t = None
    prev_h = 100.0
    for ep in sorted(glob.glob(os.path.join(run, "ep_*.jsonl*"))):
        with _open(ep) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                o = rec["obs"]
                e, t, h = float(o["energy"]), float(o["thirst"]), float(o.get("health", 100.0))
                at_start = (abs(e - START_DRIVE) < START_TOL and abs(t - START_DRIVE) < START_TOL
                            and h >= FRESH_H)
                jumped = prev_e is not None and (abs(e - prev_e) > 1.0 or abs(t - prev_t) > 1.0
                                                 or h - prev_h > 1.0)
                if cur is None or (at_start and jumped):
                    if cur is not None:
                        lives.append(cur)
                    cur = {"drives": [], "plans": [], "meals": 0, "drinks": 0,
                           "consume_ticks": [], "retina": [], "proprio": []}
                else:
                    if e - prev_e > CONSUME_JUMP:
                        cur["meals"] += 1
                        cur["consume_ticks"].append((len(cur["drives"]), "food"))
                    if t - prev_t > CONSUME_JUMP:
                        cur["drinks"] += 1
                        cur["consume_ticks"].append((len(cur["drives"]), "water"))
                i = len(cur["drives"])
                cur["drives"].append((e, t, h))
                cur["retina"].append(rec["wm"].get("retina0") or [])
                cur["proprio"].append(o.get("proprio") or [])
                p = rec.get("plan")
                if p is not None:
                    entry = {"i": i, "target": p.get("target", "none")}
                    for res in ("food", "water"):
                        pos = p.get(res)
                        if pos is not None:
                            entry[res] = math.hypot(float(pos[0]), float(pos[1]))
                    cur["plans"].append(entry)
                prev_e, prev_t, prev_h = e, t, h
    if cur is not None and cur["drives"]:
        lives.append(cur)
    return lives


def analyze_death(life: dict, ego, reach_factor: float) -> dict:
    """For an arbitration death: memory-class + (if seen-then-lost) feasibility of the belief."""
    e, t, _ = life["drives"][-1]
    dying = "energy" if e <= t else "thirst"
    needed = NEEDED_RES[dying]
    di = 0 if dying == "energy" else 1
    color = RED if needed == "food" else BLUE
    n = len(life["drives"])
    # critical window = from the last time the needy drive was still >= CRIT_DRIVE to death
    crit_start = 0
    for i in range(n - 1, -1, -1):
        if life["drives"][i][di] >= CRIT_DRIVE:
            crit_start = i
            break
    # per-tick sightings of the needy resource
    last_before = None       # (pos, ray, tick) of the last sighting BEFORE the critical window
    seen_in_crit = False
    seen_ever = False
    has_retina = any(len(r) >= 4 * N_RAYS for r in life["retina"])
    for i in range(n):
        r = life["retina"][i]
        if len(r) < 4 * N_RAYS:
            continue
        s = _needy_sighting(r, color)
        if s is None:
            continue
        seen_ever = True
        if i >= crit_start:
            seen_in_crit = True
        else:
            last_before = (s, i)
    if not has_retina:
        return {"class": "no_retina"}
    if not seen_ever:
        return {"class": "never_seen"}
    if seen_in_crit or last_before is None:
        return {"class": "seen_in_critical"}
    if crit_start - last_before[1] < LOST_MIN_STEPS:
        return {"class": "seen_in_critical"}   # lost only just before critical = effectively in view
    # seen-then-lost: replay the dead-reckoned belief from last sighting to critical onset
    (sx, sz, ray, _), t0 = last_before
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
    drift = age * 0.002
    feasible = age <= MAX_AGE and drift <= CAP_RADIUS and dist <= reach
    return {"class": "seen_then_lost", "feasible": feasible, "age": age,
            "dist": dist, "reach": reach, "edge": abs(ray) > EDGE_RAY}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    runs = [r for r in args.runs if glob.glob(os.path.join(r, "ep_*.jsonl*"))]
    if not runs:
        raise SystemExit("aucun run trouvé")
    ego = load_egomotion_head(EGO_CKPT)

    if args.selfcheck:
        lives = load_lives(runs[0])
        assert lives, "aucune vie"
        assert any(len(r) == 4 * N_RAYS for lf in lives for r in lf["retina"]), "rétine absente"
        assert all(len(p) == 132 for lf in lives for p in lf["proprio"] if p), "proprio != 132"
        dy, df, dl = ego.predict(next(p for lf in lives for p in lf["proprio"] if len(p) == 132))
        assert all(math.isfinite(v) for v in (dy, df, dl)), "egomotion NaN"
        # a sighting must round-trip to a finite ego position
        for lf in lives:
            for r in lf["retina"]:
                if len(r) == 4 * N_RAYS and _needy_sighting(r, RED):
                    break
        print(f"[selfcheck] {len(lives)} vies, rétine+proprio OK, egomotion OK")
        return

    for factor in (1.0, 0.5):
        pooled = {c: 0 for c in CLASSES}
        feasible = edge = n_lives = n_arb = 0
        per_run = []
        for run in runs:
            lives = load_lives(run)
            n_lives += len(lives)
            cnt = {c: 0 for c in CLASSES}
            for lf in lives:
                cl = classify_life(lf, factor)["class"]
                if not cl.startswith("arbitrage"):
                    continue
                n_arb += 1
                d = analyze_death(lf, ego, factor)
                cnt[d["class"]] += 1
                pooled[d["class"]] += 1
                if d["class"] == "seen_then_lost":
                    feasible += int(d["feasible"])
                    edge += int(d["edge"])
            per_run.append((os.path.basename(run), cnt))
        stl = pooled["seen_then_lost"]
        stl24 = stl * 24.0 / max(n_lives, 1)
        feas24 = feasible * 24.0 / max(n_lives, 1)
        tag = "OPTIMISTE" if factor == 1.0 else "CONSERVATEUR"
        print(f"\n===== Atteignabilité {tag} (facteur {factor}) — {n_lives} vies, "
              f"{n_arb} morts-par-arbitrage =====")
        print(f"{'run':22s} " + " ".join(f"{c[:12]:>13s}" for c in CLASSES))
        for name, cnt in per_run:
            print(f"{name:22s} " + " ".join(f"{cnt[c]:13d}" for c in CLASSES))
        print(f"{'POOLÉ':22s} " + " ".join(f"{pooled[c]:13d}" for c in CLASSES))
        print("  --- BUT vs proxy ---")
        print(f"  BUT   : seen-then-lost FAISABLES /24 vies = {feas24:.1f}  (barre du bruit = 5.0)")
        print(f"          (seen-then-lost bruts {stl} → {stl24:.1f}/24 ; faisables {feasible}/{stl} ; "
              f"bord-de-champ {edge}/{stl})")
        print(f"  proxy : morts-par-arbitrage totales {n_arb} "
              f"(never-seen {pooled['never_seen']} = perception/exploration, PAS mémoire ; "
              f"seen-in-critical {pooled['seen_in_critical']} = choix, réfuté)")

    print("\nInterprétation (pré-enregistrée, docs/design_memoire_spatiale.md §G0) :")
    print("  seen-then-lost FAISABLES (conservateur) > 5/24 vies -> chantier LICENCIÉ (G1).")
    print("  <= 5/24, OU dominé par never-seen -> le négatif occlusion TIENT, STOP")
    print("  (le vrai verrou est perception/exploration, chantier distinct).")


if __name__ == "__main__":
    main()
