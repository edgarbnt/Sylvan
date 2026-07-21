"""G0 of the « bosquets » world (docs/design_monde_bosquets.md) — FREE (0 run/Godot/train).

Question: does the PROPOSED world demand anything beyond myopia? We simulate it outside the
engine and run four policies through it, a ladder of increasing memory. The GAP between them
IS the measure of what the world requires. If a memoryless baseline already survives, no
learned stage will ever be needed, and the world must be re-dimensioned BEFORE a line of
GDScript is written. The baseline that matters is `sticky` (reactive + one remembered point):
beating a purely reactive agent proves little, beating `sticky` is the real bar.

Why simulating is legitimate: the body is kinematic (it obeys (vx, omega) exactly — verified
by diag_candidate_divergence), the metabolism is measured on corpora, the geometry is a design
choice. Nothing here needs the WM. This is an UPPER BOUND on what the world can demand.

WHAT THIS DIAG DOES NOT SAY. It judges the WORLD, not the entity. A PASS licenses building;
it does not predict that the real entity will exploit it. Occlusion is modelled as a spatially
coherent draw per (patch, 1.5 m cell), not as real tree geometry — assumed and declared.

Constants are MEASURED, not declared (ETAT_DES_LIEUX rule: no constant grounds a verdict
before being measured on the corpus):
  speed 0.011 m/tick ...... median per-tick displacement, teleports filtered, 5 forest corpora
  drain 0.05 / 0.035 ...... median negative per-tick delta, gauges on a 0-100 scale
  restore 39.95 ........... median positive jump (we PROPOSE 20 = half a pellet, see below)
  wander 1.08 ............. real path / straight line (ETAT_DES_LIEUX §1)
  p_visible 0.58 .......... 1 - 0.418, fraction of ticks food IS visible at 40 trees
                            (forest gate, docs/design_audit_peremption.md)
  retina 10.0 m ........... perception.gd:14 MAX_RANGE — NOT the 12 m the docs claim

Usage:
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_monde_bosquets_g0.py
  ... [--seeds 12] [--gap 9.0] [--regrow 600] [--portions 4] [--restore 20] [--selfcheck]
"""

from __future__ import annotations

import argparse
import math
import random
import statistics as st
from dataclasses import dataclass, field

# --------------------------------------------------------------- measured constants
SPEED = 0.011          # m/tick
DRAIN_E = 0.05         # gauge points per tick
DRAIN_T = 0.035
WANDER = 1.08          # real path / straight line
P_VIS = 0.58           # visible given in-range, in a 40-tree forest
RETINA = 10.0          # m
EPISODE = 3000         # ticks (SYLVAN_MAX_EPISODE_STEPS)
INIT = 70.0            # SYLVAN_INIT_ENERGY / _THIRST
GAUGE_MAX = 100.0
CAPTURE = 1.0          # eat_radius (food_manager.gd:24)

# --------------------------------------------------------------- PRE-REGISTERED criteria
#             written BEFORE running — never move the bar afterwards
KILL_GREEDY_MAX = 1800   # memoryless greedy must die below this
PASS_MEM_MIN = 2600      # perfect-memory greedy must exceed this
PASS_GAP_MIN = 800       # and the gap must be at least this
NUL_GAP_MAX = 300        # below this gap the world demands NOTHING -> re-dimension
RANDOM_MAX = 1000        # random must die: otherwise the world feeds by itself


@dataclass
class Patch:
    x: float
    y: float
    kind: str            # "food" | "water"
    portions: int
    cap: int
    last_regrow: int = 0

    def dist(self, ax: float, ay: float) -> float:
        return math.hypot(self.x - ax, self.y - ay)


@dataclass
class World:
    gap: float
    regrow: int
    cap: int
    restore: float
    rng: random.Random
    occ_seed: int = 0
    p_vis: float = P_VIS
    alias_r: float = 0.0    # 0 = stock readable at any range (today's world)
    fov_deg: float = 360.0  # 360 = omnidirectional retina (today's world)
    patches: list[Patch] = field(default_factory=list)

    def reset(self) -> None:
        h = self.gap / 2.0
        # Square of side `gap`: food on one diagonal, water on the other. No placement is a
        # function of the agent's position (CLAUDE.md: design the structure, never aim at it).
        coords = [(-h, -h, "food"), (h, h, "food"), (-h, h, "water"), (h, -h, "water")]
        self.patches = [Patch(x, y, k, self.cap, self.cap) for x, y, k in coords]

    def step_regrow(self, t: int) -> None:
        for p in self.patches:
            if p.portions < p.cap and t - p.last_regrow >= self.regrow:
                p.portions += 1
                p.last_regrow = t

    def apparent_portions(self, p: Patch, ax: float, ay: float) -> int:
        """What the RETINA can report about a patch's stock from where the agent stands.

        PERCEPTUAL ALIASING — the key idea. Beyond `alias_r` a depleted patch and a full one
        look IDENTICAL: you see the bush, not whether berries are left. This is not an
        occlusion trick and it does not shrink anything: a 1.5 m bush subtends 21 deg at 8 m
        and is reliably sampled, while a 0.35 m berry subtends 5 deg = half a ray step and is
        not. The aliasing is a property the sensor ALREADY has; the world just has to put the
        decision-relevant state at that scale.

        Consequence: seeing a patch tells you nothing about whether it is worth the crossing.
        Only memory of what you emptied, and when, does. That is the textbook condition for a
        memoryless policy to be strictly sub-optimal.
        """
        if self.alias_r <= 0.0 or p.dist(ax, ay) <= self.alias_r:
            return p.portions
        return p.cap                      # looks full from afar, whatever it holds

    def visible(self, i: int, p: Patch, ax: float, ay: float, face: float = 0.0) -> bool:
        """In range AND not occluded.

        Occlusion is SPATIALLY COHERENT, not redrawn each tick: a trunk hides a patch from a
        region of the arena, and stepping sideways is what reveals it. Drawing per-tick instead
        produces 5-tick flicker — the sampling artefact already measured on the real retina
        (3309 eclipses, median 5 ticks) — and it makes any target-following policy thrash.
        """
        if p.dist(ax, ay) > RETINA:
            return False
        if self.fov_deg < 360.0:
            # A cone makes LOOKING an action: what is behind simply is not sensed, and the only
            # way to sense it is to turn the body. `face` is the heading, i.e. the direction the
            # body last moved in.
            bearing = math.atan2(p.y - ay, p.x - ax) - face
            bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
            if abs(math.degrees(bearing)) > self.fov_deg / 2.0:
                return False
        cell = (i * 73856093) ^ (int(ax // 1.5) * 19349663) ^ (int(ay // 1.5) * 83492791)
        cell ^= self.occ_seed * 2654435761
        return ((cell * 2654435761) & 0xFFFF) / 65536.0 < self.p_vis


POLICIES = ("random", "greedy", "sticky", "memory")
# ~9 m of travel = one full crossing. "I hold ONE position and walk to it until I arrive."
# This is deliberately generous: it is what the live planner already does via the slot, so it
# is the honest bar. Anything less would be a straw baseline.
STICKY_TICKS = 900


def simulate(policy: str, world: World, seed: int) -> tuple[int, int]:
    """Run one life. Returns (ticks survived, consumptions).

    The four policies are a ladder of increasing memory:
      random  — no perception used at all (floor)
      greedy  — purely reactive: acts only on what is visible THIS tick
      sticky  — greedy + keeps walking to the LAST seen position for STICKY_TICKS. One
                remembered point, no map. This is the STRONG memoryless baseline: if the
                world still demands more than this, it demands genuine spatial memory.
      memory  — remembers every patch seen, with a regrowth belief per patch
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    rng = random.Random(seed)
    world.rng = rng
    world.occ_seed = seed
    world.reset()
    ax = ay = 0.0
    e, th = INIT, INIT
    # remembered belief: index -> (x, y, kind, portions when last seen, tick when last seen)
    known: dict[int, tuple[float, float, str, int, int]] = {}
    sticky: int | None = None
    consumed = 0
    heading = rng.uniform(0.0, 2 * math.pi)
    face = heading          # body heading; a cone can only sense around it

    for t in range(EPISODE):
        e -= DRAIN_E
        th -= DRAIN_T
        if e <= 0 or th <= 0:
            return t, consumed
        world.step_regrow(t)

        # which drive runs out first at the current rates
        need = "food" if (e / DRAIN_E) < (th / DRAIN_T) else "water"
        seen = [i for i, p in enumerate(world.patches) if world.visible(i, p, ax, ay, face)]

        if policy == "memory":
            for i in seen:
                p = world.patches[i]
                if world.alias_r <= 0.0 or p.dist(ax, ay) <= world.alias_r:
                    known[i] = (p.x, p.y, p.kind, p.portions, t)   # stock actually read
                elif i not in known:
                    known[i] = (p.x, p.y, p.kind, p.cap, t)        # position learned, stock guessed
                # else: keep the existing belief — a distant sighting carries no stock news

        target: int | None
        if policy == "random":
            target = None
        elif policy in ("greedy", "sticky"):
            # myopic: only what is visible RIGHT NOW and non-empty
            cand = [i for i in seen
                    if world.patches[i].kind == need
                    and world.apparent_portions(world.patches[i], ax, ay) > 0]
            cand = cand or [i for i in seen
                            if world.apparent_portions(world.patches[i], ax, ay) > 0]
            target = min(cand, key=lambda i: world.patches[i].dist(ax, ay)) if cand else None
            if policy == "sticky":
                # COMMITMENT: hold the current target until reached, seen empty, or stale.
                # Re-deciding every tick is what makes pure greedy oscillate between two
                # patches pulling opposite ways and stall between them — measured in the
                # real entity too (2022 target switches). Committing removes that failure,
                # so this is the strongest baseline that still carries no map.
                if sticky is not None:
                    if sticky in seen:
                        sp = world.patches[sticky]
                        known[sticky] = (sp.x, sp.y, sp.kind, sp.portions, t)
                        if world.apparent_portions(sp, ax, ay) <= 0:
                            sticky = None
                    elif t - known[sticky][4] > STICKY_TICKS:
                        sticky = None
                if sticky is None and target is not None:
                    p = world.patches[target]
                    known[target] = (p.x, p.y, p.kind, p.portions, t)
                    sticky = target
                target = sticky
        else:
            scored: list[tuple[float, int]] = []
            for i, (px, py, kind, seen_portions, t_seen) in known.items():
                # Belief only: what I last SAW there, plus what should have regrown since.
                # It never reads the patch's true state — that is the whole point of the test.
                est = min(world.cap, seen_portions + (t - t_seen) // world.regrow)
                if est <= 0:
                    continue
                travel = math.hypot(px - ax, py - ay) * WANDER / SPEED
                scored.append((travel + (0.0 if kind == need else 400.0), i))
            target = min(scored)[1] if scored else None

        if target is not None:
            p = world.patches[target]
            dx, dy = p.x - ax, p.y - ay
            d = math.hypot(dx, dy)
            if d <= CAPTURE:
                if p.portions > 0:
                    if p.kind == "food":
                        e = min(GAUGE_MAX, e + world.restore)
                    else:
                        th = min(GAUGE_MAX, th + world.restore)
                    p.portions -= 1
                    consumed += 1
                    if policy in ("memory", "sticky"):
                        known[target] = (p.x, p.y, p.kind, p.portions, t)
                    if policy == "sticky" and p.portions <= 0:
                        sticky = None          # exhausted: free to re-acquire
            else:
                step = SPEED / WANDER          # wandering eats into useful progress
                ax += dx / d * step
                ay += dy / d * step
        else:
            # CORRELATED random walk: hold the current heading, drift it slowly. A memoryless
            # forager keeps going rather than teleporting its intent to a far random waypoint —
            # picking a fresh distant waypoint every time sight is lost actively UNDOES the
            # approach and would fabricate the deficit this diag is meant to detect.
            heading += rng.gauss(0.0, 0.04)
            ax += math.cos(heading) * SPEED / WANDER
            ay += math.sin(heading) * SPEED / WANDER
            r = world.gap * 1.2                      # soft reflect at the arena edge
            if math.hypot(ax, ay) > r:
                heading = math.atan2(-ay, -ax) + rng.gauss(0.0, 0.3)

    return EPISODE, consumed


def run(a: argparse.Namespace, p_vis: float = P_VIS) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = {}
    for pol in POLICIES:
        w = World(a.gap, a.regrow, a.portions, a.restore, random.Random(0), p_vis=p_vis,
                  alias_r=getattr(a, 'alias', 0.0),
                  fov_deg=getattr(a, 'fov', 360.0))
        out[pol] = [simulate(pol, w, s) for s in range(a.seeds)]
    return out


def selfcheck() -> int:
    """Validate the simulator on worlds whose answer is known in advance."""
    ns = argparse.Namespace(seeds=6, gap=9.0, regrow=600, portions=4, restore=20.0)

    # 1. starvation floor: no portions at all -> everyone dies on the energy deadline
    ns.portions = 0
    r = run(ns)
    deadline = INIT / DRAIN_E
    for pol, v in r.items():
        surv = st.median([x[0] for x in v])
        assert surv <= deadline + 5, f"{pol} outlived an empty world: {surv} > {deadline}"
    print(f"  [ok] empty world kills every policy by tick {deadline:.0f}")

    # 2. generous world: patches on top of the agent, instant regrowth, NOTHING occluded.
    #    p_vis=1.0 matters — with occlusion on, a near-static agent stays inside one spatial
    #    cell and can be permanently blind there, which is a property of the occlusion model,
    #    not of the policy. This check isolates "is food reachable when it is visible".
    ns.portions, ns.gap, ns.regrow = 99, 1.0, 1
    r = run(ns, p_vis=1.0)
    g = st.median([x[0] for x in r["greedy"]])
    assert g >= EPISODE - 1, f"greedy starved in a trivially generous world: {g}"
    print("  [ok] generous unoccluded world lets the myopic greedy reach the episode cap")

    # 3. memory can never be worse than greedy when both see everything (gap >= 0)
    ns.portions, ns.gap, ns.regrow = 4, 9.0, 600
    r = run(ns)
    gm = st.median([x[0] for x in r["memory"]]) - st.median([x[0] for x in r["greedy"]])
    assert gm >= -200, f"memory did worse than greedy by {-gm:.0f} ticks — policy bug"
    print(f"  [ok] memory is not worse than greedy (gap {gm:+.0f})")

    # 4. determinism: same seeds -> same results
    assert run(ns)["greedy"] == r["greedy"], "simulation is not deterministic"
    print("  [ok] deterministic across runs")
    print("SELFCHECK PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--gap", type=float, default=9.0, help="patch square side, metres")
    ap.add_argument("--regrow", type=int, default=600, help="ticks per regrown portion")
    ap.add_argument("--portions", type=int, default=4, help="portions per patch when full")
    ap.add_argument("--restore", type=float, default=20.0, help="gauge points per portion")
    ap.add_argument("--alias", type=float, default=0.0,
                    help="range beyond which a patch's STOCK is unreadable (0 = off)")
    ap.add_argument("--fov", type=float, default=360.0,
                    help="frontal cone in degrees (360 = omnidirectional)")
    ap.add_argument("--pvis", type=float, default=P_VIS,
                    help="probability a patch is visible when in range (occlusion)")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()

    if a.selfcheck:
        return selfcheck()

    if a.restore <= 5.0:
        print(f"  !! restore={a.restore} <= guards.CONSUME_JUMP (5.0): real consumptions "
              "would be INVISIBLE to the instruments and every verdict would be void.")

    print("=" * 78)
    print("G0 — « BOSQUETS » WORLD: does it demand anything beyond myopia?")
    print("=" * 78)
    print(f"  4 patches (2 food / 2 water), square side {a.gap} m, {a.portions} portions, "
          f"restore {a.restore}, regrowth {a.regrow} ticks")
    print(f"  one crossing: {a.gap / SPEED:.0f} ticks = {a.gap / SPEED * DRAIN_E:.0f} energy "
          f"/ {a.gap / SPEED * DRAIN_T:.0f} thirst points")
    print(f"  need over a life: {EPISODE * (DRAIN_E + DRAIN_T) / a.restore:.1f} portions ; "
          f"world supplies {4 * (EPISODE / a.regrow):.0f}")
    print()
    print("  PRE-REGISTERED CRITERIA (written before launch):")
    print(f"    PASS         : greedy < {KILL_GREEDY_MAX} AND memory > {PASS_MEM_MIN} "
          f"AND gap >= {PASS_GAP_MIN}")
    print(f"    DEMANDS NOTHING : gap < {NUL_GAP_MAX} -> re-dimension BEFORE building")
    print(f"    TOO HARD     : memory < {KILL_GREEDY_MAX} -> world not viable")
    print(f"    SUSPECT      : random > {RANDOM_MAX} -> world feeds by itself")
    print()

    res = run(a, p_vis=a.pvis)
    print(f"  {'policy':10s} {'surv med':>9s} {'min':>6s} {'max':>6s} {'cons med':>9s} {'full':>8s}")
    med: dict[str, float] = {}
    for pol in POLICIES:
        surv = [r[0] for r in res[pol]]
        cons = [r[1] for r in res[pol]]
        med[pol] = st.median(surv)
        full = sum(1 for s in surv if s >= EPISODE)
        print(f"  {pol:10s} {med[pol]:9.0f} {min(surv):6d} {max(surv):6d} "
              f"{st.median(cons):9.1f} {full:5d}/{len(surv)}")

    gap = med["memory"] - med["greedy"]
    print(f"\n  GAP memory - greedy = {gap:+.0f} ticks")
    print("\n  VERDICT:")
    if med["random"] > RANDOM_MAX:
        print(f"    /!\\ SUSPECT — random survives {med['random']:.0f} > {RANDOM_MAX}.")
    if med["memory"] < KILL_GREEDY_MAX:
        print("    [X] TOO HARD — even perfect memory dies. World not viable.")
    elif gap < NUL_GAP_MAX:
        print(f"    [X] DEMANDS NOTHING — gap {gap:.0f} < {NUL_GAP_MAX}. Do not build: "
              "re-dimension (patches further apart, slower regrowth).")
    elif med["greedy"] < KILL_GREEDY_MAX and med["memory"] > PASS_MEM_MIN and gap >= PASS_GAP_MIN:
        print("    [OK] PASS — the world demands memory. Building licensed.")
    else:
        print(f"    [~] PARTIAL — greedy {med['greedy']:.0f} (< {KILL_GREEDY_MAX}?), "
              f"memory {med['memory']:.0f} (> {PASS_MEM_MIN}?), gap {gap:.0f}. "
              "Direction right, dimensioning to adjust.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
