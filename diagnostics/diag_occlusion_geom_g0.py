"""G0-geometry — can a world be OCCLUDING enough to demand memory and still be NAVIGABLE?

FREE: pure 2D geometry, 0 Godot, 0 run, 0 training.

WHY. diag_monde_bosquets_g0 showed the « bosquets » world demands COMMITMENT, not memory:
holding one point equals a full spatial map. The only cell of the sweep that separated them was
p_vis = 0.12 — a resource visible 12 % of the time when in range, against 0.58 measured in the
40-tree forest. And that density is unreachable: 45 trees is the navigable window, 54 trees
freezes the entity (immobile 85 %, 0 meals).

HYPOTHESIS UNDER TEST: occlusion per unit of NAVIGATIONAL obstruction. Many thin trunks cost a
lot of navigability and hide little; a few WIDE opaque masses should hide a lot while occupying
little floor. If true, there is a configuration with low p_vis at 45-tree navigability.

HOW p_vis IS COMPUTED — a faithful 2D replica of perception.gd, not a hand-wave:
  36 rays, 10 deg apart, 360 deg, range 10.0 m, cast from the torso (perception.gd:29-51).
  A patch is SEEN iff some ray reaches it before hitting an obstacle. This captures BOTH
  occlusion AND angular under-sampling: a 0.35 m collider at 8 m subtends 5 deg, i.e. half a
  ray step, so it is missed roughly half the time even with no trees in the way. That
  under-sampling is the already-measured flicker (3309 eclipses, median 5 ticks) and it is a
  real part of today's 0.58 — modelling it matters.

HOW NAVIGABILITY IS COMPUTED: configuration-space percolation. Obstacles are dilated by the
body radius, the free space is flood-filled from spawn, and we report the reachable fraction of
the arena. Rationale: _kin_collide (sylvan_agent.gd:684-709) is a SINGLE ray with NO sliding —
the body stops dead on contact — so the entity freezes when the free space pinches into
disconnected pockets, not when coverage is high.

WHAT THIS DIAG DOES NOT SAY. It judges geometry only. It cannot predict the entity's behaviour,
and the navigability proxy is validated against exactly two measured anchors.

Usage:
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_occlusion_geom_g0.py
  ... [--samples 400] [--selfcheck]
"""

from __future__ import annotations

import argparse
import math

import numpy as np

# ------------------------------------------------------------------ measured constants
RETINA = 10.0          # perception.gd:14 MAX_RANGE
NRAY = 36              # perception.gd:22
RAY_STEP = 2 * math.pi / NRAY
BODY_R = 0.35          # _KIN_SKIN, sylvan_agent.gd:680
TRUNK_R = 0.35         # forest_solid.gd _trunk_r
PATCH_R = 0.35         # food_manager.gd:205 perception collider
ARENA_R = 11.0         # forest_solid.gd _radius_max
RING_MIN = 2.5         # forest_solid.gd _radius_min
CLEAR_R = 2.0          # forest_solid.gd _clear_r (keep-out around spawn)

# The two MEASURED anchors that calibrate the navigability proxy.
ANCHOR_OK = 45         # navigable window: immobile 5.4 %, full speed
ANCHOR_FROZEN = 54     # immobile 85 %, 0 meals

# ------------------------------------------------------------------ PRE-REGISTERED criteria
#              written BEFORE running — never moved afterwards
PROXY_MIN_SEPARATION = 20.0   # the proxy must put >=20 points between the two anchors, else NUL
TARGET_PVIS = 0.20            # occlusion needed for memory to pay (sweep said 0.12; 0.20 generous)
NAV_TOLERANCE = 5.0           # a candidate may lose at most 5 points of reachability vs 45 trees


def sample_positions(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform IN AREA over the arena disc (r = R*sqrt(u)) — the law the Godot code does NOT
    use today (it samples r linearly, so density goes as 1/r; noted in the design doc)."""
    u = rng.random(n)
    ang = rng.random(n) * 2 * math.pi
    r = ARENA_R * np.sqrt(u)
    return np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)


def place_obstacles(count: int, radius: float, rng: np.random.Generator,
                    min_gap: float = 0.0) -> np.ndarray:
    """Rejection sampling, uniform in area, keeping clear of spawn. Mirrors forest_solid.gd
    except that sampling is uniform-in-area (the fix) and min_gap is honoured."""
    out: list[tuple[float, float]] = []
    for _ in range(count):
        for _try in range(60):
            u = rng.random()
            ang = rng.random() * 2 * math.pi
            r = math.sqrt(RING_MIN ** 2 + u * (ARENA_R ** 2 - RING_MIN ** 2))
            x, y = r * math.cos(ang), r * math.sin(ang)
            if math.hypot(x, y) < CLEAR_R + radius:
                continue
            if min_gap > 0 and any(math.hypot(x - ox, y - oy) < min_gap + 2 * radius
                                   for ox, oy in out):
                continue
            out.append((x, y))
            break
    return np.array(out) if out else np.zeros((0, 2))


def ray_hits_disc(ox: float, oy: float, dx: float, dy: float,
                  cx: float, cy: float, r: float) -> float:
    """Distance along the ray to the first intersection with a disc, or inf."""
    fx, fy = ox - cx, oy - cy
    b = fx * dx + fy * dy
    c = fx * fx + fy * fy - r * r
    disc = b * b - c
    if disc < 0:
        return math.inf
    s = math.sqrt(disc)
    for t in (-b - s, -b + s):
        if t > 1e-6:
            return t
    return math.inf


def p_visible(patches: np.ndarray, obstacles: np.ndarray, obs_r: float,
              positions: np.ndarray, yaws: np.ndarray) -> float:
    """Fraction of (position, patch) pairs IN RANGE where some retina ray reaches the patch."""
    in_range = 0
    seen = 0
    for (ax, ay), yaw in zip(positions, yaws):
        for px, py in patches:
            if math.hypot(px - ax, py - ay) > RETINA:
                continue
            in_range += 1
            hit = False
            for k in range(NRAY):
                ang = yaw + k * RAY_STEP
                dx, dy = math.cos(ang), math.sin(ang)
                d_patch = ray_hits_disc(ax, ay, dx, dy, px, py, PATCH_R)
                if d_patch > RETINA:
                    continue
                blocked = False
                for ox, oy in obstacles:
                    if ray_hits_disc(ax, ay, dx, dy, ox, oy, obs_r) < d_patch:
                        blocked = True
                        break
                if not blocked:
                    hit = True
                    break
            seen += hit
    return seen / in_range if in_range else 1.0


def navigability(obstacles: np.ndarray, obs_r: float, res: float = 0.12) -> float:
    """Reachable fraction of the arena in configuration space (obstacles dilated by BODY_R),
    flood-filled from spawn. Percentage points."""
    n = int(2 * ARENA_R / res) + 1
    xs = np.linspace(-ARENA_R, ARENA_R, n)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    inside = gx ** 2 + gy ** 2 <= ARENA_R ** 2
    free = inside.copy()
    grow = obs_r + BODY_R
    for ox, oy in obstacles:
        free &= (gx - ox) ** 2 + (gy - oy) ** 2 > grow ** 2

    # flood fill from the spawn cell
    start = (int(n // 2), int(n // 2))
    if not free[start]:
        return 0.0
    reach = np.zeros_like(free)
    stack = [start]
    reach[start] = True
    while stack:
        i, j = stack.pop()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < n and 0 <= b < n and free[a, b] and not reach[a, b]:
                reach[a, b] = True
                stack.append((a, b))
    return 100.0 * reach.sum() / inside.sum()


def immobile_fraction(obstacles: np.ndarray, obs_r: float, seed: int,
                      sliding: bool = False, ticks: int = 4000) -> float:
    """Fraction of ticks where the body barely moves, under the EXACT live collision rule.

    _kin_collide (sylvan_agent.gd:684-709): one ray along the motion, overshooting by _KIN_SKIN;
    on hit the body is placed at hit_dist - SKIN and STOPS — there is no tangential projection.
    `sliding=True` adds the projection that the live code does not have, to separate "the world
    is too dense" from "the body cannot get around a trunk".
    """
    rng = np.random.default_rng(seed)
    ax = ay = 0.0
    step = 0.011
    tx, ty = 0.0, 0.0
    immobile = 0
    stuck = 0
    REPLAN = 10          # the live planner replans every 10 ticks (serve_planner_command)
    for t in range(ticks):
        if math.hypot(tx - ax, ty - ay) < 0.5 or stuck >= REPLAN:
            u, ang = rng.random(), rng.random() * 2 * math.pi
            r = ARENA_R * math.sqrt(u)
            tx, ty = r * math.cos(ang), r * math.sin(ang)
        dx, dy = tx - ax, ty - ay
        d = math.hypot(dx, dy) or 1e-9
        dx, dy = dx / d, dy / d

        # EXACTLY the live rule: cast against the REAL obstacle, stop BODY_R short of it.
        allowed = step
        for ox, oy in obstacles:
            hit = ray_hits_disc(ax, ay, dx, dy, ox, oy, obs_r)
            allowed = min(allowed, max(0.0, hit - BODY_R))
        nx, ny = ax + dx * allowed, ay + dy * allowed

        if sliding and allowed < step * 0.5:
            # project the residual motion onto the tangent of the blocking obstacle
            best = None
            for ox, oy in obstacles:
                dd = math.hypot(ax - ox, ay - oy)
                if dd < obs_r + BODY_R + step * 3 and (best is None or dd < best[0]):
                    best = (dd, ox, oy)
            if best is not None:
                _, ox, oy = best
                nxr, nyr = (ax - ox), (ay - oy)
                nn = math.hypot(nxr, nyr) or 1e-9
                nxr, nyr = nxr / nn, nyr / nn
                tdx, tdy = -nyr, nxr
                if tdx * dx + tdy * dy < 0:
                    tdx, tdy = nyr, -nxr
                res = step - allowed
                cx, cy = nx + tdx * res, ny + tdy * res
                if all(math.hypot(cx - ox2, cy - oy2) > obs_r + BODY_R
                       for ox2, oy2 in obstacles):
                    nx, ny = cx, cy

        moved = math.hypot(nx - ax, ny - ay)
        if moved < step * 0.2:
            immobile += 1
            stuck += 1
        else:
            stuck = 0
        ax, ay = nx, ny
        if math.hypot(ax, ay) > ARENA_R:          # keep it inside the arena
            ax, ay = ax * 0.99, ay * 0.99
            tx, ty = 0.0, 0.0
    return 100.0 * immobile / ticks


def evaluate(count: int, radius: float, seed: int, samples: int,
             min_gap: float = 0.0) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    obs = place_obstacles(count, radius, rng, min_gap)
    h = 4.5
    patches = np.array([(-h, -h), (h, h), (-h, h), (h, -h)], dtype=float)
    pos = sample_positions(samples, rng)
    yaws = rng.random(samples) * 2 * math.pi
    return p_visible(patches, obs, radius, pos, yaws), navigability(obs, radius), len(obs)


def selfcheck() -> int:
    empty = np.zeros((0, 2))
    nav = navigability(empty, TRUNK_R)
    assert nav > 99.0, f"empty arena should be fully reachable, got {nav:.1f}"
    print(f"  [ok] empty arena reachable at {nav:.1f} %")

    pv, _, _ = evaluate(0, TRUNK_R, seed=0, samples=200)
    assert pv > 0.55, f"with no obstacle p_vis should be high, got {pv:.2f}"
    print(f"  [ok] no obstacle -> p_vis {pv:.2f} (residual loss = angular under-sampling only)")

    # a solid ring of large masses must strangle reachability
    ring = np.array([(6.0 * math.cos(a), 6.0 * math.sin(a))
                     for a in np.linspace(0, 2 * math.pi, 40, endpoint=False)])
    nav_ring = navigability(ring, 1.0)
    assert nav_ring < 40.0, f"a closed ring should trap the agent, got {nav_ring:.1f}"
    print(f"  [ok] closed ring traps the agent ({nav_ring:.1f} % reachable)")
    print("SELFCHECK PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()

    def avg(count: int, radius: float, min_gap: float = 0.0) -> tuple[float, float, float]:
        rs = [evaluate(count, radius, s, a.samples, min_gap) for s in range(a.seeds)]
        return (float(np.mean([r[0] for r in rs])),
                float(np.mean([r[1] for r in rs])),
                float(np.mean([r[2] for r in rs])))

    print("=" * 78)
    print("G0-GEOMETRY — occlusion vs navigability")
    print("=" * 78)
    print("  PRE-REGISTERED:")
    print(f"    proxy VALID only if navigability({ANCHOR_OK} trees) - navigability"
          f"({ANCHOR_FROZEN} trees) >= {PROXY_MIN_SEPARATION} points")
    print(f"    PASS  : a config reaches p_vis <= {TARGET_PVIS} with navigability within "
          f"{NAV_TOLERANCE} points of the {ANCHOR_OK}-tree anchor")
    print("    FAIL  : none does -> occlusion cannot demand memory in this body")
    print()

    print("  --- STEP 1: calibrate the proxy on the two measured anchors ---")
    pv_ok, nav_ok, n_ok = avg(ANCHOR_OK, TRUNK_R)
    pv_fr, nav_fr, n_fr = avg(ANCHOR_FROZEN, TRUNK_R)
    print(f"    {ANCHOR_OK} trees (MEASURED navigable, immobile 5.4 %) : "
          f"p_vis {pv_ok:.3f}  navigability {nav_ok:5.1f} %  ({n_ok:.0f} placed)")
    print(f"    {ANCHOR_FROZEN} trees (MEASURED frozen, immobile 85 %)  : "
          f"p_vis {pv_fr:.3f}  navigability {nav_fr:5.1f} %  ({n_fr:.0f} placed)")
    sep = nav_ok - nav_fr
    print(f"    separation = {sep:.1f} points (need >= {PROXY_MIN_SEPARATION})")
    if sep < PROXY_MIN_SEPARATION:
        print("\n    [NUL] The proxy does NOT separate the two measured anchors. It cannot be "
              "used to judge a new configuration. Verdict void, not negative.")
        print("    (Reported anyway below, as an exploration — but it grounds no decision.)")
    else:
        print("    [ok] proxy separates the anchors, it may be used")

    print("\n  --- STEP 2: sweep few-and-wide vs many-and-thin ---")
    print(f"  {'count':>6s} {'radius':>7s} {'placed':>7s} {'p_vis':>7s} {'navig':>7s}   verdict")
    best: tuple[float, float, float, int, float] | None = None
    for radius, counts in ((0.35, (45, 54)), (0.8, (8, 14, 20)),
                           (1.5, (4, 7, 10)), (2.5, (3, 5, 7))):
        for count in counts:
            pv, nav, placed = avg(count, radius, min_gap=1.3 if radius <= 0.5 else 0.0)
            ok = pv <= TARGET_PVIS and nav >= nav_ok - NAV_TOLERANCE
            tag = "<<< PASS" if ok else ("occlusion ok, trop fermé" if pv <= TARGET_PVIS
                                         else "pas assez occultant")
            print(f"  {count:6d} {radius:7.2f} {placed:7.0f} {pv:7.3f} {nav:7.1f}   {tag}")
            if ok and (best is None or pv < best[0]):
                best = (pv, nav, radius, count, placed)

    print("\n  VERDICT:")
    if sep < PROXY_MIN_SEPARATION:
        print("    [NUL] proxy invalid (see step 1) — no decision licensed.")
    elif best is None:
        print(f"    [X] FAIL — no configuration reaches p_vis <= {TARGET_PVIS} while staying "
              f"within {NAV_TOLERANCE} points of the {ANCHOR_OK}-tree navigability.")
        print("        Occlusion cannot make memory pay in this body. Say so; do not build.")
    else:
        pv, nav, radius, count, placed = best
        print(f"    [OK] PASS — {count} masses of radius {radius} m: p_vis {pv:.3f}, "
              f"navigability {nav:.1f} % (anchor {nav_ok:.1f} %).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
