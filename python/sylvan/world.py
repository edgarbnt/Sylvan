"""FROZEN WORLD PRESETS — one source of truth for what the world IS.

WHY THIS EXISTS. Until now the world lived in ~40 scattered `SYLVAN_*` environment variables per
harness, while Python carried a hand-maintained duplicate of the same numbers. That arrangement
produced, in a single afternoon:
  * `heading_weight=2.0` copied from a multi-drive harness (where it is INERT) into a single-drive
    one (where it is ACTIVE, command_planner.py:580) — a retired scaffolding silently switched back
    on for a whole A/B;
  * `SYLVAN_RETINA_FOV_DEG` needing to reach TWO separate processes (Godot casts the rays, the
    planner server decodes them) with nothing guaranteeing it reached both;
  * `surv_turn_rate` still modelling a body that had been made 4x faster an hour earlier.

"Freezing the world" is not a promise you keep by being careful; it is a mechanism. A preset is
declared once here, every consumer DERIVES from it, and a corpus carries the preset that produced
it so that a stale reading fails loudly instead of quietly.

WHAT A PRESET IS NOT. It does not describe the entity — no planner weights, no checkpoints, no
scaffolding flags. Those change per experiment; the world must not.

Usage:
  from sylvan.world import BOSQUETS_V1
  env = BOSQUETS_V1.to_env()            # what to hand Godot + the planner server
  PYTHONPATH=python env_pytorch_3.12/bin/python -m sylvan.world --selfcheck
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class WorldPreset:
    """An immutable description of a world. Frozen on purpose: a preset is a fact, not a knob."""

    name: str

    # --- perception (Godot casts the rays, Python decodes them; BOTH read these) -----------
    retina_rays: int = 36
    retina_range_m: float = 10.0        # perception.gd MAX_RANGE — NOT the 12 m the old docs claim
    retina_fov_deg: float = 360.0       # 360 = omnidirectional; a real cone REDISTRIBUTES the rays

    # --- body ------------------------------------------------------------------------------
    kin_speed: float = 0.8              # env unit; measured displacement 0.011 m/tick
    kin_turn: float = 1.5               # env unit; measured 0.015 rad/tick at 1.5

    # --- metabolism -------------------------------------------------------------------------
    energy_drain: float = 0.05          # gauge points per tick, gauge is 0-100
    thirst_drain: float | None = None   # None = single drive (thirst disabled)
    restore_per_item: float = 40.0
    init_energy: float = 100.0
    gauge_max: float = 100.0
    episode_steps: int = 3000

    # --- resources --------------------------------------------------------------------------
    patches_per_resource: int = 4
    items_total: int = 2                # berries spread over the patches
    patch_radius_m: float = 0.95        # OUTER radius of the berry RING (must be < eat_radius)
    patch_spacing_min_m: float = 9.0
    patch_spacing_max_m: float = 11.0
    regrow_ticks: int = 2500
    perish_ticks: int = 0        # 0 = OFF ; >0 = une baie non mangee perit apres ce delai
    spawn_annulus_m: tuple[float, float] = (3.0, 11.0)
    eat_radius_m: float = 1.0

    # ----------------------------------------------------------------------------------------
    @property
    def ray_step_deg(self) -> float:
        return self.retina_fov_deg / self.retina_rays

    @property
    def ticks_per_meal(self) -> float:
        """Ticks of drain covered by one full item, both drives summed."""
        drain = self.energy_drain + (self.thirst_drain or 0.0)
        return self.restore_per_item / drain

    @property
    def meals_needed(self) -> float:
        """Meals a life actually requires — total drain MINUS the starting reservoir.

        The reservoir is the part everyone forgets: over `episode_steps` the entity burns
        `drain * steps`, but it STARTS with `init_energy` already banked. Dividing total drain by
        restore (and getting 3.75 here) overstates the need by 3x; the real figure is 1.25.
        """
        drain = self.energy_drain + (self.thirst_drain or 0.0)
        deficit = drain * self.episode_steps - self.init_energy
        return max(0.0, deficit) / self.restore_per_item

    @property
    def starvation_floor_ticks(self) -> float:
        """How long a life lasts having eaten NOTHING — the floor any verdict must sit above."""
        return self.init_energy / self.energy_drain

    def to_env(self) -> dict[str, str]:
        """The environment the world needs. Handed to BOTH Godot and the planner server: the FOV
        is read by each of them independently, and a preset that reaches only one silently decodes
        the cone retina with the 360-degree angle table."""
        env = {
            "SYLVAN_RETINA_FOV_DEG": f"{self.retina_fov_deg}",
            "SYLVAN_KINEMATIC": "1",
            "SYLVAN_KIN_SPEED": f"{self.kin_speed}",
            "SYLVAN_KIN_TURN": f"{self.kin_turn}",
            "SYLVAN_ENERGY_DRAIN": f"{self.energy_drain}",
            "SYLVAN_INIT_ENERGY": f"{self.init_energy}",
            "SYLVAN_MAX_EPISODE_STEPS": f"{self.episode_steps}",
            "SYLVAN_EAT_RADIUS": f"{self.eat_radius_m}",
            "SYLVAN_FOOD_COUNT": f"{self.items_total}",
            "SYLVAN_FOOD_ENERGY_PER": f"{self.restore_per_item}",
            "SYLVAN_FOOD_PATCHES": f"{self.patches_per_resource}",
            "SYLVAN_FOOD_PATCH_RADIUS": f"{self.patch_radius_m}",
            "SYLVAN_FOOD_PATCH_SPACING": f"{self.patch_spacing_min_m}",
            "SYLVAN_FOOD_PATCH_SPACING_MAX": f"{self.patch_spacing_max_m}",
            "SYLVAN_FOOD_REGROW": f"{self.regrow_ticks}",
            "SYLVAN_FOOD_PERISH": f"{self.perish_ticks}",
            "SYLVAN_FOOD_MIN_RADIUS": f"{self.spawn_annulus_m[0]}",
            "SYLVAN_FOOD_SPAWN_RADIUS": f"{self.spawn_annulus_m[1]}",
        }
        if self.thirst_drain is not None:
            env["SYLVAN_THIRST_DRAIN"] = f"{self.thirst_drain}"
            env["SYLVAN_WATER_COUNT"] = f"{self.items_total}"
            env["SYLVAN_WATER_PATCHES"] = f"{self.patches_per_resource}"
            env["SYLVAN_WATER_REGROW"] = f"{self.regrow_ticks}"
            env["SYLVAN_DRINK_RADIUS"] = f"{self.eat_radius_m}"
        return env

    def as_dict(self) -> dict:
        """Serialised form, to be written into every corpus so the corpus is self-describing."""
        return dataclasses.asdict(self)


# ============================================================================================
# THE FROZEN PRESETS. Adding a preset is cheap; MUTATING one silently invalidates every number
# ever measured under it. Make a new one instead.
# ============================================================================================

#: The world as it was before 2026-07-22: open arena, food respawning 2-4.5 m FROM THE AGENT.
#: Kept only so old corpora remain interpretable — it demands nothing and should not be built on.
PERPETUAL_V0 = WorldPreset(
    name="perpetual_v0",
    retina_fov_deg=360.0, kin_turn=1.5,
    thirst_drain=0.05, patches_per_resource=0, items_total=5,
    spawn_annulus_m=(2.0, 8.0),
)

#: ADOPTED 2026-07-22. Fixed depleting patches + real cone + fast turn + memory.
#: ⚠️ Adopted on an owner JUDGEMENT after a FAILED calibration gate (dispersion criterion 354 < 400,
#: itself provably ill-posed on a bounded lifetime). Never re-read as "this passed its gate".
#: Measured: 55 % full episodes, 15 % at the starvation floor, 30 % intermediate lives, 1.40 meals.
BOSQUETS_V1 = WorldPreset(
    name="bosquets_v1",
    retina_fov_deg=120.0,        # real cone: 36 rays REDISTRIBUTED to 3.33 deg, not zeroed
    kin_turn=6.0,                # x4 — at 1.5 a full scan cost 89 % of the inter-meal budget
    thirst_drain=None,           # single drive: two segregated patch sets are arithmetically
                                 # unsurvivable (a drive switch costs 1.9x the inter-meal budget)
    patches_per_resource=4,
    items_total=2,
    regrow_ticks=2500,
)


#: ABLATION of BOSQUETS_V1: the cone WITHOUT the fast body. Isolates what the turn rate carries.
#: The +2.17 meals of 2026-07-22 was measured with FOV and kin_turn changed TOGETHER, so nothing in
#: that data separates them. The claim it rested on — "the 360 retina compensates a body too slow to
#: afford looking around" — comes from arithmetic (a full scan costs 89 % of the inter-meal budget at
#: 1.5) and from a SIMULATION (0/20 full lives), never from lives. This preset tests it.
BOSQUETS_V1_SLOWTURN = dataclasses.replace(BOSQUETS_V1, name="bosquets_v1_slowturn", kin_turn=1.5)

#: ⭐ RESULTAT 2026-07-22 : l ablation ci-dessus est INDISCERNABLE de BOSQUETS_V1. A monde egal
#: (B=2, 20 vies, graine 1) : 1,40 repas / 50 % pleins / 10 % plancher DES DEUX COTES, ecart +0,00,
#: IC95 [-0,40, +0,40]. Les runs sont bien distincts (8/20 vies de duree differente) et la rotation
#: a bien change (p99 mesure 0,0151 contre 0,0602 rad/tick) : ce sont les AGREGATS qui coincident.
#: ⇒ LA VITESSE DE ROTATION NE PORTE RIEN. Le cone seul produit l effet.
#: POURQUOI mon arithmetique ("un tour complet coute 89 % du budget") ne s appliquait pas : elle
#: chiffrait le prix d un BALAYAGE, or l entite ne balaie JAMAIS — il n existe aucun terme
#: d information dans le cout du planner (verifie : 0 occurrence). J ai calcule le prix d un
#: comportement que l agent n a pas.
#: ⇒ V2 = le monde adopte SANS le changement de corps. La dette des constantes calibrees sur
#: l ancien corps (surv_turn_rate=0.015 en tete) est ANNULEE : elles redeviennent valides.
BOSQUETS_V2 = dataclasses.replace(BOSQUETS_V1, name="bosquets_v2", kin_turn=1.5)

#: LEVIER CONSÉQUENCE (2026-07-23) : v2 + baies PÉRISSABLES. Une baie non mangée 300 ticks (~3,3 m
#: la baie que l'agent visait SAUTE sur un autre bosquet (elle ne disparaît PAS : ce monde n'a que
#: 2 baies, disparaître = famine, mesuré 0 repas). Densité constante -> survie préservée, mais un
#: choix trop lent perd son trajet. Attaque la RÉCUPÉRABILITÉ. Gate = scripts/cf_fork_distribution.sh
#: (taux de conséquence doit monter SANS effondrer la survie). GRATUIT côté WM (règle de monde).
BOSQUETS_V3_PERISH = dataclasses.replace(BOSQUETS_V2, name="bosquets_v3_perish", perish_ticks=800)


def selfcheck() -> int:
    """Check the presets against constants MEASURED on corpora, not against their declarations."""
    p = BOSQUETS_V1

    assert p.starvation_floor_ticks == 2000.0, p.starvation_floor_ticks
    print(f"  [ok] starvation floor {p.starvation_floor_ticks:.0f} ticks (matches the 2000 measured)")

    # the figure I got wrong all session: total drain / restore = 3.75, but the reservoir covers most
    naive = (p.energy_drain * p.episode_steps) / p.restore_per_item
    assert abs(naive - 3.75) < 1e-6 and abs(p.meals_needed - 1.25) < 1e-6, (naive, p.meals_needed)
    print(f"  [ok] meals needed {p.meals_needed:.2f}, not the naive {naive:.2f} "
          "(the reservoir covers the rest)")

    assert p.patch_radius_m < p.eat_radius_m, "berry ring must fit inside the mouth"
    print(f"  [ok] berry ring {p.patch_radius_m} m < eat radius {p.eat_radius_m} m")

    assert abs(p.ray_step_deg - 120.0 / 36) < 1e-9
    print(f"  [ok] cone gives {p.ray_step_deg:.2f} deg between rays "
          f"({360.0 / p.retina_rays / p.ray_step_deg:.1f}x finer than 360 deg)")

    env = p.to_env()
    assert "SYLVAN_THIRST_DRAIN" not in env and "SYLVAN_WATER_COUNT" not in env, \
        "single-drive preset must not emit any water variable"
    assert env["SYLVAN_RETINA_FOV_DEG"] == "120.0"
    print(f"  [ok] to_env() emits {len(env)} variables, no water in single drive")

    # a preset is a fact: mutating one must be impossible
    try:
        p.kin_turn = 99.0            # type: ignore[misc]
        raise AssertionError("preset is mutable — it must be frozen")
    except dataclasses.FrozenInstanceError:
        print("  [ok] presets are immutable")

    ab = BOSQUETS_V1_SLOWTURN
    assert ab.kin_turn == 1.5 and ab.retina_fov_deg == BOSQUETS_V1.retina_fov_deg, "ablation must vary ONLY the turn"
    diff = [f.name for f in dataclasses.fields(ab)
            if f.name != "name" and getattr(ab, f.name) != getattr(BOSQUETS_V1, f.name)]
    assert diff == ["kin_turn"], f"ablation differs on more than the turn rate: {diff}"
    print(f"  [ok] slowturn ablation differs from bosquets_v1 on exactly {diff}")

    assert BOSQUETS_V2.kin_turn == 1.5 and BOSQUETS_V2.retina_fov_deg == 120.0
    print("  [ok] bosquets_v2 = le cone SANS le changement de corps (ablation mesuree equivalente)")

    assert PERPETUAL_V0.retina_fov_deg == 360.0 and PERPETUAL_V0.patches_per_resource == 0
    print("  [ok] perpetual_v0 still describes the pre-patch world")
    print("SELFCHECK PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--preset", default="bosquets_v1")
    ap.add_argument("--env", action="store_true", help="print the preset as shell exports")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    p = {"bosquets_v1": BOSQUETS_V1, "bosquets_v1_slowturn": BOSQUETS_V1_SLOWTURN, "bosquets_v2": BOSQUETS_V2, "bosquets_v3_perish": BOSQUETS_V3_PERISH,
     "perpetual_v0": PERPETUAL_V0}[a.preset]
    if a.env:
        for k, v in p.to_env().items():
            print(f"export {k}={v}")
    else:
        print(f"{p.name}: floor={p.starvation_floor_ticks:.0f} ticks, "
              f"meals needed={p.meals_needed:.2f}, ray step={p.ray_step_deg:.2f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
