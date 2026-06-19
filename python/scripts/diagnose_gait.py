"""Objective gait diagnostic for J_walk (reads per-foot contact from the buffer).

Velocity (fwd_vel) tells us the body moves; it does NOT tell us HOW (drift vs dive
vs a real step cycle). This reads the reward-only per-foot contacts
(left_contact/right_contact, in obs.metrics) and quantifies the GAIT:

  - single_support_frac : fraction of steps with exactly one foot down (a foot in
    the air mid-step) — high in walking, ~0 in flat-footed standing/shuffling.
  - double_support_frac : both feet down (standing / shuffling).
  - flight_frac         : both feet up (hopping/jumping/falling) — should stay low.
  - alternations_per_100 : sole-contact foot switches per 100 steps — the walking
    RHYTHM. Hopping on one foot = ~0; a real left-right gait = many.
  - mean_fwd_vel, mean_len, fall_rate for context.

A real walking gait shows high single_support_frac AND high alternations_per_100
WITH forward velocity. Drift = low single_support; dive = forward but low/no
alternation; hop = single_support but low alternation.

Usage (from python/):
    python3 -m scripts.diagnose_gait --prefix ppo_walk4            # latest matching run
    python3 -m scripts.diagnose_gait --run wm_v1_clean
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sylvan.constants import REPLAY_BUFFER_DIR
from sylvan.buffer.reader import iter_episodes

# ⚠️ QUADRUPED (2026-06-08): this script's GAIT CONCEPTS (single-support %, L/R alternation) are
# BIPEDAL and don't directly apply to a 4-legged quad (whose gaits are trot/pace/gallop over 4 feet).
# It needs a quad-specific rewrite (e.g. diagonal-pair contact phase). Indices below are updated to
# the 94-d quad layout so the values read are at least correct, but interpret the bipedal metrics
# with care. QUAD proprio layout (sylvan_agent._rebuild_proprioception):
# [0]=height,[1-3]=lin vel,[4-6]=ang vel, [7-60]=9 bodies×6 axes, [61-64]=4 foot contacts
# (fl,fr,bl,br), [65-67]=com, [68-79]=12 dof angles, [80-91]=12 velocities, [92-93]=gait clock.
# Map the 2 "bipedal" contacts onto the two FRONT feet (fl=61, fr=62) as a rough proxy.
LEFT_CONTACT_IDX = 61   # front-left foot
RIGHT_CONTACT_IDX = 62  # front-right foot
# 12 dof angles at [68..79], order per leg = [hip_x, hip_z, knee_x]; front knees are dof 2 & 5.
LEFT_KNEE_IDX = 70   # fl_knee (68 + 2)
RIGHT_KNEE_IDX = 73  # fr_knee (68 + 5)
# proprio[7..12] = trunk basis (y axis, then -z axis); [11] = -trunk.basis.z.y. Unchanged (trunk first).
# forward_lean = max(0, -trunk_forward.y) with trunk_forward=+basis.z = max(0, proprio[11]).
TORSO_FWD_Y_IDX = 11


def _latest_run(prefix: str) -> Path | None:
    runs = sorted(REPLAY_BUFFER_DIR.glob(f"{prefix}*"), key=lambda p: p.name)
    return runs[-1] if runs else None


def _gait_of_episode(ep) -> dict:
    n = len(ep)
    single = double = flight = 0
    alternations = 0
    last_sole = -1
    for t in ep:
        lc = float(t.obs.proprio[LEFT_CONTACT_IDX]) > 0.5
        rc = float(t.obs.proprio[RIGHT_CONTACT_IDX]) > 0.5
        c = int(lc) + int(rc)
        if c == 1:
            single += 1
            sole = 0 if lc else 1
            if last_sole >= 0 and sole != last_sole:
                alternations += 1
            last_sole = sole
        elif c == 2:
            double += 1
        else:
            flight += 1
    fwd = [float(t.obs.metrics.get("forward_velocity", 0.0)) for t in ep]
    # Knee flexion (0=straight): mean bend, and mean alternation |L-R| (one bent, one straight).
    knee_bend = [0.5 * (float(t.obs.proprio[LEFT_KNEE_IDX]) + float(t.obs.proprio[RIGHT_KNEE_IDX])) for t in ep]
    knee_diff = [abs(float(t.obs.proprio[LEFT_KNEE_IDX]) - float(t.obs.proprio[RIGHT_KNEE_IDX])) for t in ep]
    fwd_lean = [max(0.0, float(t.obs.proprio[TORSO_FWD_Y_IDX])) for t in ep]
    return {
        "len": n,
        "single": single, "double": double, "flight": flight,
        "alternations": alternations,
        "fwd_sum": sum(fwd),
        "knee_bend_sum": sum(knee_bend),
        "knee_diff_sum": sum(knee_diff),
        "fwd_lean_sum": sum(fwd_lean),
        "fell": ep[-1].done,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Objective gait diagnostic (per-foot contact).")
    ap.add_argument("--prefix", default=None, help="Latest run matching this prefix.")
    ap.add_argument("--run", default=None, help="Exact run name or path.")
    args = ap.parse_args()

    if args.run:
        run_dir = Path(args.run) if Path(args.run).is_dir() else REPLAY_BUFFER_DIR / args.run
    elif args.prefix:
        run_dir = _latest_run(args.prefix)
    else:
        raise SystemExit("pass --prefix or --run")
    if not run_dir or not run_dir.is_dir():
        raise SystemExit(f"[gait] run not found: {run_dir}")

    episodes = [ep for ep in iter_episodes(run_dir) if ep]
    if not episodes:
        raise SystemExit(f"[gait] no episodes in {run_dir}")

    g = [_gait_of_episode(ep) for ep in episodes]
    total_steps = sum(x["len"] for x in g)
    n = len(g)
    single = sum(x["single"] for x in g)
    double = sum(x["double"] for x in g)
    flight = sum(x["flight"] for x in g)
    alt = sum(x["alternations"] for x in g)
    fwd = sum(x["fwd_sum"] for x in g)
    falls = sum(1 for x in g if x["fell"])

    print(f"[gait] {run_dir.name} | episodes={n} steps={total_steps}")
    print(f"[gait] mean_len={total_steps/n:.0f}  fall_rate={falls/n*100:.0f}%  mean_fwd_vel={fwd/total_steps:+.3f}")
    print(f"[gait] single_support={single/total_steps*100:5.1f}%  double={double/total_steps*100:5.1f}%  "
          f"flight={flight/total_steps*100:4.1f}%")
    knee_bend = sum(x["knee_bend_sum"] for x in g) / total_steps
    knee_diff = sum(x["knee_diff_sum"] for x in g) / total_steps
    print(f"[gait] alternations_per_100_steps={alt/total_steps*100:.2f}  (walking rhythm; hop/drift ~0)")
    fwd_lean = sum(x["fwd_lean_sum"] for x in g) / total_steps
    print(f"[gait] knee_bend(mean rad, 0=straight)={knee_bend:.3f}  knee_alternation|L-R|={knee_diff:.3f}  "
          f"(user: stiff straight legs = ~0; bent stepping = higher)")
    print(f"[gait] forward_lean(0=upright, higher=leaning fwd)={fwd_lean:.3f}  (owner: 'penché en avant' → topples fwd)")
    # crude interpretation
    ss = single / total_steps
    ar = alt / total_steps * 100
    if ss > 0.25 and ar > 3.0 and fwd / total_steps > 0.15:
        verdict = "WALKING-like (single-support + alternation + forward)"
    elif fwd / total_steps > 0.15 and ar < 1.0:
        verdict = "DIVE/DRIFT (forward but no step rhythm)"
    elif ss > 0.25 and ar < 1.0:
        verdict = "HOP (single-support but no alternation)"
    else:
        verdict = "FLAT-FOOTED (standing/shuffling)"
    print(f"[gait] -> {verdict}")


if __name__ == "__main__":
    main()
