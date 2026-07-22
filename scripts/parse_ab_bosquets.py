"""Pooled verdict for the memory A/B in the patch world.

Reads the Godot free-logs of both arms across seeds, pools the lives, and renders the verdict
against the criteria PRE-REGISTERED in docs/prereg_ab_memoire_bosquets.md — which are duplicated
as constants here so the script cannot silently drift from the pre-registration.

Primary metric is MEALS PER LIFE, not survival: survival saturates at the 3000-tick cap and is
dominated by variance (ETAT_DES_LIEUX §5). The starvation floor (100/0.05 = 2000 ticks) is
reported separately — a life sitting exactly there ate nothing, and an arm full of those means the
world broke, not that the arm lost.

Usage:
  PYTHONPATH=python env_pytorch_3.12/bin/python scripts/parse_ab_bosquets.py \
      --off /tmp/bosq_moff_..._s1_free.log /tmp/bosq_moff_..._s2_free.log \
      --on  /tmp/bosq_mon_..._s1_free.log  /tmp/bosq_mon_..._s2_free.log
"""

from __future__ import annotations

import argparse
import re
import statistics as st

# ---- criteria, copied from the pre-registration (docs/prereg_ab_memoire_bosquets.md)
GAIN_MEALS_MIN = 0.8      # PASS on the primary metric
GAIN_FULL_MIN = 10.0      # confirmation on the secondary (points)
FLOOR_TICKS = 2000.0      # 100 / 0.05 — died without ever eating
FLOOR_TOL = 60.0
NUL_FLOOR_FRAC = 0.60     # an arm this full of floor-deaths voids the verdict
EPISODE_CAP = 2999

_PAT = re.compile(r"Episode (\d+) \| Step (\d+) .* Energy: ([\d.]+) \| Thirst: ([\d.]+)")


def lives(path: str) -> list[tuple[int, int]]:
    """Returns [(ticks survived, meals)] for each episode in a Godot free-log."""
    eps: dict[int, list[tuple[int, float, float]]] = {}
    with open(path) as fh:
        for line in fh:
            m = _PAT.search(line)
            if m:
                eps.setdefault(int(m.group(1)), []).append(
                    (int(m.group(2)), float(m.group(3)), float(m.group(4))))
    out = []
    for ep in sorted(eps):
        rows = sorted(eps[ep])
        meals = sum(1 for i in range(1, len(rows)) if rows[i][1] - rows[i - 1][1] > 5)
        out.append((rows[-1][0], meals))
    return out


def describe(arm: list[tuple[int, int]]) -> dict:
    surv = [x[0] for x in arm]
    meals = [x[1] for x in arm]
    return {
        "n": len(arm),
        "meals_med": st.median(meals) if meals else 0.0,
        "meals_mean": st.mean(meals) if meals else 0.0,
        "surv_med": st.median(surv) if surv else 0.0,
        "full": 100.0 * sum(1 for s in surv if s >= EPISODE_CAP) / max(1, len(surv)),
        "floor": 100.0 * sum(1 for s in surv if abs(s - FLOOR_TICKS) < FLOOR_TOL) / max(1, len(surv)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", nargs="+", required=True, help="free-logs of the MEM=off arm, one per seed")
    ap.add_argument("--on", nargs="+", required=True, help="free-logs of the MEM=on arm, one per seed")
    a = ap.parse_args()

    per_seed = []
    print(f"  {'bras':>6s} {'graine':>7s} {'n':>4s} {'repas med':>10s} {'repas moy':>10s} "
          f"{'survie med':>11s} {'pleins':>8s} {'plancher':>9s}")
    arms: dict[str, list[tuple[int, int]]] = {"off": [], "on": []}
    for name, paths in (("off", a.off), ("on", a.on)):
        for i, p in enumerate(paths):
            lv = lives(p)
            arms[name] += lv
            d = describe(lv)
            per_seed.append((name, i + 1, d))
            print(f"  {name:>6s} {i+1:>7d} {d['n']:>4d} {d['meals_med']:>10.1f} {d['meals_mean']:>10.2f} "
                  f"{d['surv_med']:>11.0f} {d['full']:>7.0f}% {d['floor']:>8.0f}%")

    off, on = describe(arms["off"]), describe(arms["on"])
    print(f"\n  POOLÉ  off: n={off['n']}  repas med={off['meals_med']:.1f} moy={off['meals_mean']:.2f}"
          f"  pleins={off['full']:.0f}%  plancher={off['floor']:.0f}%")
    print(f"  POOLÉ  on : n={on['n']}  repas med={on['meals_med']:.1f} moy={on['meals_mean']:.2f}"
          f"  pleins={on['full']:.0f}%  plancher={on['floor']:.0f}%")

    gain_med = on["meals_med"] - off["meals_med"]
    gain_mean = on["meals_mean"] - off["meals_mean"]
    gain_full = on["full"] - off["full"]
    print(f"\n  ÉCART repas médians  : {gain_med:+.1f}   (barre pré-inscrite ≥ +{GAIN_MEALS_MIN})")
    print(f"  ÉCART repas moyens   : {gain_mean:+.2f}  (indicatif, non pré-inscrit)")
    print(f"  ÉCART épisodes pleins: {gain_full:+.0f} pts (confirmation ≥ +{GAIN_FULL_MIN})")

    # direction per seed — a pooled gain carried by a single seed is not a result
    dirs = []
    for s in (1, 2):
        o = next((d for n, i, d in per_seed if n == "off" and i == s), None)
        v = next((d for n, i, d in per_seed if n == "on" and i == s), None)
        if o and v:
            dirs.append(v["meals_mean"] - o["meals_mean"])
    if dirs:
        print(f"  direction par graine : " + "  ".join(f"graine{i+1} {d:+.2f}" for i, d in enumerate(dirs)))

    print("\n  VERDICT :")
    if off["floor"] >= 100 * NUL_FLOOR_FRAC or on["floor"] >= 100 * NUL_FLOOR_FRAC:
        print("    [NUL] un bras meurt majoritairement au plancher de famine — le monde est cassé, "
              "verdict void (ce n'est pas un négatif).")
    elif gain_med < 0 or gain_mean < 0:
        print("    [KILL] la mémoire NUIT sur la métrique primaire. Négatif à banker.")
    elif gain_med >= GAIN_MEALS_MIN and all(d > 0 for d in dirs):
        extra = "confirmé par les pleins" if gain_full >= GAIN_FULL_MIN else "sans confirmation sur les pleins"
        print(f"    [PASS] la mémoire paie dans ce monde ({extra}).")
    elif gain_med >= GAIN_MEALS_MIN:
        print("    [PARTIEL] gain poolé atteint mais la direction s'inverse sur une graine — "
              "non concluant, re-mesurer avant de conclure.")
    else:
        print(f"    [ÉCHEC] gain {gain_med:+.1f} sous la barre +{GAIN_MEALS_MIN}. "
              "Le monde n'exige pas (encore) la mémoire.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
