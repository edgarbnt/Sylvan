"""G0 of the obstacle+memory direction (docs/design_vision_cone.md redirect / design_memoire_spatiale.md
reopening (b)) — FREE (0 run/train), on EXISTING obstacle-world corpora.

The memory G0 STOP and the cone G0 split both hinged on the SPARSE 360 world having almost no
"reachable-but-invisible" moments. The obstacle world is different: a SOLID wall physically hides a
resource that is still REACHABLE (walk around). This measures that place directly.

"Blind-to-reachable-food" tick = food is NOT visible now, BUT was seen at close (<= SEEN_REACH_M)
within the last LOST_MIN_STEPS, with NO consumption since (so it was not eaten, and it is not out of
range — it is OCCLUDED). That is exactly the situation where remembering "food is behind the wall, go
around" would help. In the sparse 360 world this is ~0 (a 360 retina sees any in-range food); in the
wall world it should be substantial. Contrast is the verdict.

Run (repo root):
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_obstacle_memory_g0.py [--selfcheck]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import sys

sys.path.insert(0, "python")
from sylvan.control.mode1.obs import RED, _color_gated_depths
from sylvan.control.slot_memory import transport_geom
from sylvan.models.egomotion_head import load_egomotion_head

RANGE_M = 10.0
N_RAYS = 36
SEEN_REACH_M = 5.0          # "seen at reachable range" (metabolic reach ~ energy*0.4; 5m is safe)
LOST_MIN_STEPS = 30         # lost for at least this long = a real occlusion episode, not a blink
FORAGE_MAX_E = 60.0         # under foraging pressure (energy below this)
CONSUME_JUMP = 5.0
STILL_REACH_M = 3.5         # dead-reckoned belief still within this = OCCLUDED-reachable (memory job)
EGO_CKPT = "data/checkpoints/egomotion_head/best.pt"

OBSTACLE_RUNS = ["data/replay_buffer/obstacle_g1nav", "data/replay_buffer/obstacle_g2nav"]
SPARSE_RUNS = ["data/replay_buffer/critic_kin_pure1", "data/replay_buffer/critic_kin_pure2"]


def _open(p: str):
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p)


def food_pos(retina: list[float]) -> tuple[float, float] | None:
    """Ego (x_right, z_fwd) of the closest RED ray, or None if no food ray hits."""
    if len(retina) < 4 * N_RAYS:
        return None
    d = _color_gated_depths(retina, RED)
    m = min(d)
    if m >= 0.999:
        return None
    b = 2.0 * math.pi * d.index(m) / N_RAYS
    return m * RANGE_M * math.sin(b), m * RANGE_M * math.cos(b)


def measure(run: str, ego) -> dict:
    """Stream all ticks; count blind-to-reachable-food among foraging ticks + episode stats.

    A blind episode is refined into OCCLUDED-reachable (dead-reckoned belief of the last-seen food
    stays within STILL_REACH_M -> the food is still close, the agent just cannot see it = the memory
    job) vs WALKED-AWAY (belief drifts beyond reach -> the agent left it, memory would not help).
    """
    ticks = []
    for ep in sorted(glob.glob(os.path.join(run, "ep_*.jsonl*"))):
        with _open(ep) as f:
            for line in f:
                line = line.strip()
                if line:
                    ticks.append(json.loads(line))
    n = len(ticks)
    energy = [float(t["obs"]["energy"]) for t in ticks]
    retinas = [t["wm"].get("retina0") or [] for t in ticks]
    proprios = [t["obs"].get("proprio") or [] for t in ticks]
    fpos = [food_pos(r) for r in retinas]
    fdist = [math.hypot(*p) if p is not None else None for p in fpos]
    # consumption ticks (energy jump up)
    consume = [i for i in range(1, n) if energy[i] - energy[i - 1] > CONSUME_JUMP]
    forage = blind = occluded = 0
    last_seen_reach: int | None = None   # last tick food was seen at reachable range, AFTER any meal
    belief: list[float] | None = None    # dead-reckoned ego position of that last-seen food
    episodes = 0
    in_episode = False
    ep_lens: list[int] = []
    ep_start = 0
    consume_ptr = 0
    for i in range(n):
        # a consumption resets the belief (the food was eaten and respawns elsewhere) -> so
        # last_seen_reach always points to a sighting AFTER the last meal, making a "no meal since"
        # check unnecessary by construction.
        if consume_ptr < len(consume) and consume[consume_ptr] == i:
            consume_ptr += 1
            last_seen_reach, belief = None, None
            if in_episode:
                ep_lens.append(i - ep_start); in_episode = False
        elif belief is not None and len(proprios[i - 1]) == 132:
            dyaw, dfwd, dlat = ego.predict(proprios[i - 1])   # dead-reckon the belief one step
            belief = transport_geom(belief, dyaw, dfwd, dlat)
        if fdist[i] is not None:
            if fdist[i] <= SEEN_REACH_M:
                last_seen_reach, belief = i, list(fpos[i])
            if in_episode:
                ep_lens.append(i - ep_start); in_episode = False   # re-acquired (visible again)
        if energy[i] >= FORAGE_MAX_E:
            continue
        forage += 1
        # blind-to-reachable = not visible now, but seen reachable recently and not eaten since
        if fdist[i] is None and last_seen_reach is not None and i - last_seen_reach >= LOST_MIN_STEPS:
            blind += 1
            still_close = belief is not None and math.hypot(belief[0], belief[1]) <= STILL_REACH_M
            if still_close:
                occluded += 1                                  # occluded-reachable = the memory job
            if not in_episode:
                episodes += 1; in_episode = True; ep_start = i
        elif in_episode:
            ep_lens.append(i - ep_start); in_episode = False
    if in_episode:
        ep_lens.append(n - ep_start)
    med_len = sorted(ep_lens)[len(ep_lens) // 2] if ep_lens else 0
    food_vis = sum(1 for x in fdist if x is not None) / max(n, 1)
    return {"ticks": n, "forage": forage, "blind": blind, "occluded": occluded,
            "blind_frac": blind / max(forage, 1), "occ_frac": occluded / max(forage, 1),
            "episodes": episodes, "med_ep_len": med_len, "food_vis": food_vis, "meals": len(consume)}


def report(name: str, runs: list[str], ego) -> None:
    runs = [r for r in runs if glob.glob(os.path.join(r, "ep_*.jsonl*"))]
    agg = {"forage": 0, "blind": 0, "occluded": 0}
    vis = []
    for run in runs:
        m = measure(run, ego)
        for k in agg:
            agg[k] += m[k]
        vis.append(m["food_vis"])
        print(f"  {os.path.basename(run):22s} food-vu {100*m['food_vis']:4.0f}% | "
              f"OCCLUÉE-atteignable {100*m['occ_frac']:5.1f}% | (aveugle-tot {100*m['blind_frac']:4.1f}%) | "
              f"épisodes {m['episodes']:3d} | repas {m['meals']}")
    of = agg["occluded"] / max(agg["forage"], 1)
    bf = agg["blind"] / max(agg["forage"], 1)
    print(f"  {'POOLÉ ' + name:22s} food-vu {100*sum(vis)/len(vis):4.0f}% | "
          f"OCCLUÉE-atteignable {100*of:5.1f}% des ticks-forage | (aveugle-tot {100*bf:4.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    ego = load_egomotion_head(EGO_CKPT)
    if args.selfcheck:
        runs = [r for r in OBSTACLE_RUNS if glob.glob(os.path.join(r, "ep_*.jsonl*"))]
        assert runs, "aucun corpus obstacle trouvé"
        m = measure(runs[0], ego)
        assert m["ticks"] > 1000, "corpus trop petit"
        assert 0.0 <= m["occ_frac"] <= m["blind_frac"] <= 1.0, "occluded doit être <= blind"
        assert m["food_vis"] < 1.0, "food toujours visible = pas d'occlusion ?"
        print(f"[selfcheck] OK ({m['ticks']} ticks, food-vu {100*m['food_vis']:.0f}%)")
        return
    print("OCCLUÉE-atteignable = bouffe vue à portée puis INVISIBLE, belief dead-reckon TOUJOURS proche")
    print("(<3.5m) = elle est là, cachée, atteignable = LE travail de la mémoire.\n")
    print("=== MONDE OBSTACLE (mur solide occluant, food-only) ===")
    report("obstacle", OBSTACLE_RUNS, ego)
    print("\n=== MONDE ÉPARSE 360° (réf multi-drive — pas de mur) ===")
    report("sparse", SPARSE_RUNS, ego)
    print("\nVerdict : si l'obstacle montre une part SUBSTANTIELLE d'OCCLUÉE-atteignable (belief reste")
    print("proche) NETTEMENT > éparse -> la place mémoire 'atteignable-mais-invisible' EXISTE ->")
    print("chantier obstacle+mémoire LICENCIÉ (test-en-vies gaté). Sinon -> STOP.")


if __name__ == "__main__":
    main()
