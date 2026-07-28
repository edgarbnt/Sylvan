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
    # SPEED FAN (§2.13). The body accepts any vx; the fan is the envelope its CONSUMERS may use —
    # the planner's grid and the collection's babbling range. It lives in the preset for exactly the
    # reason this file exists: it must reach TWO processes, and a fan that reaches only the babbler
    # gives a WM trained on speeds the planner never commands (or the reverse).
    vx_fan: tuple[float, ...] = ()      # () = leave every consumer on its own default
    speed_cost: float = 0.0             # k in "energy/tick = k * vx^2" (0 = OFF, locomotion is free)
    # TERRAIN FACTOR — the fraction of intended displacement a POLICY actually realises, averaged over
    # a life. 1.0 = open ground (the old, implicit, WRONG assumption for a forest). MEASURED under the
    # planner by G11 (not the arena mean, not the babbling median): it is the budget aggregate that
    # converts commanded speed into ground actually covered. A budget identity that ignores it
    # overstates the travel per life by 1/factor.
    terrain_factor: float = 1.0

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
    ripe_cue: bool = False       # True = la LUMINOSITE du buisson encode l'age de sa baie (invisible aux slots)
    ripe_decay: float = 0.0      # 0 = OFF ; 0.75 = une baie mure ne rend plus que 25 % de son energie
    prey_speed: float = 0.0      # m/tick ; la nourriture SE DEPLACE (0 = OFF). Agent = 0.011 m/tick mesure
    n_types: int = 0             # types de proies visibles (0 = OFF, max 4)
    type_values: tuple = ()      # multiplicateur de valeur nutritive PAR TYPE -- ARBITRAIRE
    spawn_annulus_m: tuple[float, float] = (3.0, 11.0)
    eat_radius_m: float = 1.0
    food_type_hues: tuple = ()   # teintes PAR TYPE "r,g,b;r,g,b;..." — () = les TYPE_COLORS du code

    # --- forêt (§2.2/§2.3/§2.8/§2.9) — tout à 0/False = monde d'avant, bit-identique -----------
    forest_count: int = 0               # arbres SOLIDES (occlusion + collision), 0 = aucune forêt
    forest_stands: int = 0              # peuplements Neyman-Scott (0 = semis uniforme)
    forest_clearings: int = 0           # clairières (disques d'exclusion)
    forest_appearance_var: float = 0.0  # jitter de teinte PAR ARBRE, hors des cônes ressource
    forest_radius_var: float = 0.0      # jitter de RAYON de tronc PAR ARBRE (géométrie : collision + rétine)
    forest_ring_m: tuple = ()           # anneau où les arbres se dispersent — () = défaut du code
    terrain_slow: float = 0.0           # pente du ralentissement par arbre proche (0 = OFF)
    terrain_radius_m: float = 2.5
    terrain_floor: float = 0.25
    distractor_count: int = 0           # animaux mobiles NON comestibles
    distractor_speed: float = 0.0       # m/tick — 0 = défaut du code (0,006, calibré sur l'ancien corps)
    gaze: bool = False                  # tête mobile — porte la proprioception de 132 à 133
    water_puddle_period: int = 0        # ticks du cycle de rétrécissement d'une flaque (0 = OFF)
    # DANGER — zones FIXES (§6quinquies B) : incluses comme STRUCTURE SPATIALE (elles contraignent
    # les trajets comme les arbres), PAS comme levier d'apprentissage — « évite la zone marquée » est
    # probablement dérivable, on n'en attend aucun gain côté critique. Le risque de BLESSURE à la
    # chasse reste DIFFÉRÉ (échec P2-bis mesuré). Une TROISIÈME classe de conséquence perceptible
    # (dégâts) est aussi ce qui rend les requêtes-couleur APPRENABLES par contingence — sans elle
    # l'encodeur reste aveugle au vert et le verrou A2 ne peut pas tomber.
    # ENCOMBREMENT DU CORPS (demi-longueur, demi-largeur) en mètres — () = sphère historique 0,35 m.
    # MESURÉ sur l'habillage servi : museau à 1,024 m du centre, demi-largeur 0,213 m. Sans lui, le
    # corps s'arrête 0,35 m avant un arbre alors que son maillage en projette 1,02 : on VOIT 0,674 m
    # de loup entrer dans le tronc. C'est une propriété du CORPS, donc servie même sans habillage.
    kin_body_extent: tuple = ()
    hazard_count: int = 0
    hazard_engulf_p: float = 0.0        # part des zones qui engloutissent (vs simple contact)
    health_regen: float = 0.0           # la santé redevient une économie cyclique, pas un budget

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
            # gauge_max existait dans ce fichier depuis le début et n'était SERVI nulle part : le
            # champ était muet, la jauge restait aux 100 codés dans homeostasis.gd. On le branche
            # (2026-07-28) parce que c'est lui qui décide si un repas est encaissable — voir le
            # commentaire de reset_state(). L'obs du WM lit energy/max_energy, donc elle reste
            # dans [0,1] quel que soit le plafond : agrandir la jauge ne déplace pas l'échelle
            # d'observation, il déplace seulement ce qu'un repas peut y écrire.
            "SYLVAN_MAX_ENERGY": f"{self.gauge_max}",
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
            "SYLVAN_FOOD_RIPE_CUE": "1" if self.ripe_cue else "0",
            "SYLVAN_FOOD_RIPE_DECAY": f"{self.ripe_decay}",
            "SYLVAN_FOOD_PREY_SPEED": f"{self.prey_speed}",
            "SYLVAN_FOOD_TYPES": f"{self.n_types}",
            "SYLVAN_FOOD_TYPE_VALUES": ",".join(str(v) for v in self.type_values),
            "SYLVAN_FOOD_MIN_RADIUS": f"{self.spawn_annulus_m[0]}",
            "SYLVAN_FOOD_SPAWN_RADIUS": f"{self.spawn_annulus_m[1]}",
        }
        if self.thirst_drain is not None:
            env["SYLVAN_THIRST_DRAIN"] = f"{self.thirst_drain}"
            env["SYLVAN_WATER_COUNT"] = f"{self.items_total}"
            env["SYLVAN_WATER_PATCHES"] = f"{self.patches_per_resource}"
            env["SYLVAN_WATER_REGROW"] = f"{self.regrow_ticks}"
            env["SYLVAN_DRINK_RADIUS"] = f"{self.eat_radius_m}"
            # PANNE SILENCIEUSE RÉPARÉE (2026-07-28) : la géométrie des bosquets n'était servie qu'à
            # la NOURRITURE. L'eau retombait donc sur les défauts du GDScript — écart mini 9,0 m et
            # PAS de borne haute — un intervalle que le placeur ne peut pas satisfaire pour 180
            # bosquets dans 3820 m². Mesuré dans le visuel : « WATER : 2/180 bosquets placés » contre
            # « FOOD : 176/180 », et les 3 vies observées sont mortes de SOIF, énergie quasi pleine,
            # sans une seule gorgée. Un corpus collecté ainsi n'aurait contenu AUCUN événement de
            # boisson : la seconde pulsion aurait été absente du monde qu'on prétend lui apprendre.
            env["SYLVAN_WATER_PATCH_RADIUS"] = f"{self.patch_radius_m}"
            env["SYLVAN_WATER_PATCH_SPACING"] = f"{self.patch_spacing_min_m}"
            env["SYLVAN_WATER_PATCH_SPACING_MAX"] = f"{self.patch_spacing_max_m}"
            # ...et l'ANNEAU, sans quoi le correctif ci-dessus ne sert à rien : le gestionnaire d'eau
            # retombait sur ses défauts d'export (1,5-7,0 m), un disque 26x plus petit que celui de la
            # nourriture (3,0-35,0 m). Toute l'eau du monde tenait autour du point de départ, et il n'y
            # rentre que ~15 bosquets à 3 m d'écart — d'où « 15/180 » une fois l'écart réparé.
            env["SYLVAN_WATER_MIN_RADIUS"] = f"{self.spawn_annulus_m[0]}"
            env["SYLVAN_WATER_SPAWN_RADIUS"] = f"{self.spawn_annulus_m[1]}"
            # La soif DOIT partir du même niveau que l'énergie. Sans ça les deux jauges ont des
            # planchers de famine différents (init/drain) alors que rien ne l'a décidé : une pulsion
            # tuerait systématiquement avant l'autre, et l'arbitrage mesuré serait un artefact.
            env["SYLVAN_INIT_THIRST"] = f"{self.init_energy}"
            env["SYLVAN_MAX_THIRST"] = f"{self.gauge_max}"   # même plafond, même raison que les drains
            if self.water_puddle_period > 0:
                env["SYLVAN_WATER_PUDDLE_PERIOD"] = f"{self.water_puddle_period}"
        if self.food_type_hues:
            env["SYLVAN_FOOD_TYPE_HUES"] = ";".join(",".join(f"{c}" for c in h)
                                                    for h in self.food_type_hues)
        if self.vx_fan:
            # Les DEUX consommateurs, ou aucun : le babillage de la collecte ET la grille du planner.
            # Un éventail qui n'atteint que l'un des deux donne un WM entraîné sur des vitesses que
            # le planner ne commande jamais (ou l'inverse) — la panne même que ce fichier existe pour
            # empêcher, déjà payée sur SYLVAN_RETINA_FOV_DEG.
            env["SYLVAN_WM_VX_MIN"] = f"{min(self.vx_fan)}"
            env["SYLVAN_WM_VX_MAX"] = f"{max(self.vx_fan)}"
            env["SYLVAN_PLANNER_VX_GRID"] = ",".join(f"{v}" for v in self.vx_fan)
        if self.speed_cost > 0.0:
            env["SYLVAN_SPEED_COST"] = f"{self.speed_cost}"
        if self.forest_count > 0:
            env["SYLVAN_FOREST_COUNT"] = f"{self.forest_count}"
            env["SYLVAN_FOREST_STANDS"] = f"{self.forest_stands}"
            env["SYLVAN_FOREST_CLEARINGS"] = f"{self.forest_clearings}"
            if self.forest_appearance_var > 0.0:
                env["SYLVAN_FOREST_APPEARANCE_VAR"] = f"{self.forest_appearance_var}"
            if self.forest_radius_var > 0.0:
                env["SYLVAN_FOREST_RADIUS_VAR"] = f"{self.forest_radius_var}"
            if self.forest_ring_m:
                env["SYLVAN_FOREST_RADIUS_MIN"] = f"{self.forest_ring_m[0]}"
                env["SYLVAN_FOREST_RADIUS_MAX"] = f"{self.forest_ring_m[1]}"
        if self.terrain_slow > 0.0:
            env["SYLVAN_TERRAIN_SLOW"] = f"{self.terrain_slow}"
            env["SYLVAN_TERRAIN_RADIUS"] = f"{self.terrain_radius_m}"
            env["SYLVAN_TERRAIN_FLOOR"] = f"{self.terrain_floor}"
        if self.distractor_count > 0:
            env["SYLVAN_DISTRACTOR_COUNT"] = f"{self.distractor_count}"
            if self.distractor_speed > 0.0:
                env["SYLVAN_DISTRACTOR_SPEED"] = f"{self.distractor_speed}"
        if self.gaze:
            env["SYLVAN_GAZE"] = "1"
        if self.kin_body_extent:
            env["SYLVAN_KIN_BODY_EXTENT"] = ",".join(f"{v}" for v in self.kin_body_extent)
        if self.hazard_count > 0:
            env["SYLVAN_HAZARD_COUNT"] = f"{self.hazard_count}"
            env["SYLVAN_HAZARD_ENGULF_P"] = f"{self.hazard_engulf_p}"
        if self.health_regen > 0.0:
            env["SYLVAN_HEALTH_REGEN"] = f"{self.health_regen}"
        return env

    # ── IDENTITÉS DE L'ÉVENTAIL DE VITESSE (§2.13) ────────────────────────────────────────────
    # Le coût de locomotion RE-COUPLE des grandeurs que G2 traitait comme indépendantes : aller vite
    # consomme, donc exige plus de repas, donc plus de trajet. Ces trois propriétés rendent le
    # couplage lisible au lieu de le laisser dans une tête.
    @property
    def cheapest_vx(self) -> float:
        """La vitesse la MOINS chère au mètre : vx* = sqrt(D_total / k).

        En dessous, le drain passif domine (on paie du temps) ; au-dessus, le coût de locomotion
        domine (on paie de la puissance). C'est ce point qui rend le sprint un pari : il n'existe
        que si k > 0, et il n'est un ARBITRAGE que s'il tombe À L'INTÉRIEUR de l'éventail.
        """
        if self.speed_cost <= 0.0:
            return float("inf")     # locomotion gratuite → la vitesse maximale domine toujours
        drain = self.energy_drain + (self.thirst_drain or 0.0)
        return math.sqrt(drain / self.speed_cost)

    def drain_at(self, vx: float) -> float:
        """Consommation TOTALE par tick à la vitesse vx (drain passif + locomotion)."""
        return self.energy_drain + (self.thirst_drain or 0.0) + self.speed_cost * vx * vx

    def events_at(self, vx: float) -> float:
        """Événements que la vie EXIGE si l'entité croise à vx — réservoirs des DEUX jauges déduits.

        Avec un coût de locomotion, « événements par vie » cesse d'être une constante du monde :
        c'est une conséquence de la politique de vitesse de l'entité. Le monde offre une bande.
        """
        reservoirs = self.init_energy * (2 if self.thirst_drain is not None else 1)
        return max(0.0, self.drain_at(vx) * self.episode_steps - reservoirs) / self.restore_per_item

    def metres_per_event_budget(self, vx: float, margin: float = 1.2) -> float:
        """Trajet MAXIMAL par événement que le budget tolère à la vitesse vx (identité (3) de G2).

        C'est la contrainte que la DENSITÉ du monde doit satisfaire — et le seul terme qui ne se
        dérive pas (il contient du comportement) : il se MESURE, jamais ne se postule.

        🚨 LE TRAJET EST PONDÉRÉ PAR terrain_factor. Sans lui on comptait la distance qu'un corps
        parcourrait sur sol dégagé (facteur implicite 1,0) ; G11 a mesuré 0,635 sous la politique.
        Ignorer le terrain surestimait le budget de 1/0,635 = 1,57x — c'est l'erreur que cette
        pondération corrige, et la raison de la re-calibration du 2026-07-25.
        """
        travel = (self.kin_speed * vx * self.terrain_factor / 60.0) * self.episode_steps
        ev = self.events_at(vx)
        return travel / (margin * ev) if ev > 0 else float("inf")

    def travel_budget(self, vx: float) -> float:
        """Sol RÉELLEMENT parcouru sur une vie à la vitesse vx (terrain inclus). Le numérateur de la
        joignabilité, rendu explicite parce que la re-calibration en dépend directement."""
        return (self.kin_speed * vx * self.terrain_factor / 60.0) * self.episode_steps

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

#: MATURITÉ VISIBLE (2026-07-24) : v3 + la LUMINOSITÉ du buisson-marqueur encode l'âge de sa baie
#: (vif = fraîche, sombre = imminente). Le buisson est à cos 0,40/0,45 des requêtes rouge/bleu, donc
#: SOUS le seuil 0,55 -> ses rayons sont exclus EN DUR des deux slots ; et l'affinité étant un
#: COSINUS, elle est invariante par changement d'échelle (mesuré : 0,402/0,453 de x1,0 à x0,2).
#: ⇒ l'indice est PROUVABLEMENT invisible aux slots (donc à `-min_dist`) et visible dans la rétine :
#: seul un critique qui lit la SCÈNE (le latent) peut s'en servir. C'est ce qui donne du travail au
#: critique sans câbler la réponse dans la perception (§2/§3).
BOSQUETS_V4_RIPE = dataclasses.replace(BOSQUETS_V3_PERISH, name="bosquets_v4_ripe", ripe_cue=True)

#: MATURITÉ QUI COMPTE (2026-07-24) : v4 + la maturité BAISSE la valeur nutritive (une baie sur le
#: point de se relocaliser ne rend plus que 25 % de son énergie). C'est le premier signal du monde
#: qui soit à la fois PERCEPTIBLE (luminosité du buisson, lisible dans le latent à R² 0,65),
#: NON-GÉOMÉTRIQUE (indépendant de la distance) et PRÉDICTIBLE (fonction déterministe de l'âge —
#: contrairement au saut de relocalisation, aléatoire ET invisible au rêve).
#: MESURE qui motive ce preset : sur v4, la géométrie SEULE prédit le retour mieux que
#: géométrie+latent (R² 0,179 vs 0,149) -> tout ce que le latent portait en plus était redondant ou
#: sans effet. Ici la maturité change l'ISSUE, donc elle devient apprenable par un critique.
BOSQUETS_V5_VALUE = dataclasses.replace(BOSQUETS_V4_RIPE, name="bosquets_v5_value", ripe_decay=0.75)

#: PROIE (2026-07-24) — la nourriture SE DÉPLACE au lieu d'attendre. Premier changement qui sort de
#: « décorer un point fixe » : viser où la proie EST devient DÉMONTRABLEMENT sous-optimal, donc le
#: prédicteur du WM devient enfin porteur (il faut simuler la trajectoire de l'autre).
#: SPÉC. issue du test GRATUIT `diagnostics/diag_prey_interception.py`, mesurée avant tout code :
#:   - la proie doit avoir du mouvement TRANSVERSAL et NE PAS FUIR (une fuite converge vers une
#:     trajectoire radiale, contre laquelle poursuite et interception coïncident -> gain NUL) ;
#:   - elle doit être RAPIDE : à 0,9x la vitesse de l'agent, interception 67,5 % vs poursuite 56,2 %
#:     de capture ; à 1,2x, 33,0 % vs 18,7 %. En dessous de 0,6x le gain est nul.
#: Base = bosquets_v2 (monde figé calibré), SANS périssable (le saut aléatoire confondrait la mesure).
#: Les buissons restent des repères FIXES pendant que la nourriture bouge.
BOSQUETS_V6_PREY = dataclasses.replace(BOSQUETS_V2, name="bosquets_v6_prey", prey_speed=0.0099)

#: TYPES ARBITRAIRES (2026-07-24) — v6 + 4 types de proies dont la valeur nutritive est ARBITRAIRE.
#: C'est la SEULE condition mesurée où un critique devient NÉCESSAIRE et pas seulement utile
#: (`diagnostics/diag_arbitrary_headroom.py`) : une formule ajustée, à qui on donne même le type comme
#: scalaire, plafonne à 49,5 % de la marge oracle, quand un modèle APPRIS atteint 69,7 %. Aucune
#: formule ne peut contenir une table de correspondance arbitraire — il faut avoir goûté.
#: Les 4 teintes ont été choisies PAR MESURE contre les requêtes réelles du WM : toutes dans le cône
#: bouffe (cos rouge 0,79-0,99 > seuil 0,55), hors du cône eau (cos bleu < 0,45), écart RGB mutuel
#: 0,187. Le type ne change donc QUE l'apparence, jamais la localisation par le slot.
#: Valeurs volontairement NON MONOTONES en l'indice de teinte : rien dans l'apparence ne les ordonne.
BOSQUETS_V7_TYPES = dataclasses.replace(BOSQUETS_V6_PREY, name="bosquets_v7_types",
                                        n_types=4, type_values=(1.0, 0.25, 1.5, 0.6))


#: ⭐ LA FORÊT (2026-07-25) — le monde complet de `docs/design_foret_complete.md`, celui que la
#: collecte et le retrain du WM vont servir. Il empile les 9 briques bâties et gatées (G1-G9) :
#: forêt structurée + terrain qui ralentit + regard + palette séparable + flaques + distracteurs +
#: apparence variable des troncs + éventail de vitesse facturé.
#:
#: LA CALIBRATION N'EST PAS COPIÉE DE G2 — ELLE EST RE-RÉSOLUE, et il faut dire pourquoi. G2 a
#: tranché le candidat D (vitesse x4,3 + drain 0,2333 + densité x3) sur un monde à UNE pulsion et
#: SANS coût de locomotion. Les deux décisions owner du 2026-07-24 (eau incluse, éventail facturé)
#: cassent cette arithmétique de deux façons mesurables :
#:   * DEUX jauges → le plancher de famine se lit par jauge (init/drain_jauge) : le tenir sous 25 %
#:     de la vie EXIGE 0,1333/jauge, soit 0,2666 de drain passif à lui seul — tout le budget de G2 ;
#:   * un COÛT de locomotion → à l'optimum économique le coût de locomotion ÉGALE le drain passif
#:     (propriété de sqrt(D/k)), donc la consommation à la croisière VAUT LE DOUBLE du drain passif.
#: Les deux ensemble mettaient les événements/vie à 35, hors de la bande [10, 30].
#: RÉSOLUTION V1 (2026-07-25 matin) : réservoir 75, repas 80 → 13,1 événements à la croisière, un
#: plancher 25,0 %, un optimum de vitesse au trot. MAIS elle supposait un budget de trajet de 84,9 m
#: (facteur terrain implicite 1,0) — un fantasme, réfuté le soir même.
#:
#: 🚨 RE-CALIBRATION V2 (2026-07-25 soir, sur MESURE). La sonde de portée G11 (planner-probe) a
#: mesuré, sous une VRAIE politique et non un babillage, les deux chiffres qui manquaient :
#:   * FACTEUR TERRAIN VÉCU = 0,635 (moyenne, l'agrégat de budget). Le budget de trajet réel n'est
#:     donc pas 84,9 m mais 2,83 x 0,6 x 0,635 / 60 x 3000 = 53,9 m/vie au trot. La V1 le surestimait
#:     de 1,57x — la calibration métabolique ignorait purement le terrain.
#:   * TRAJET PAR REPAS = 7,65 m sous le WM ACTUEL (borne HAUTE : il est OOD en forêt ; l'ancre
#:     sans-forêt à 13,38 m confirme que 7-13 m est sa compétence, pas une pénalité forêt).
#: CONSÉQUENCES, dérivées sans relâcher aucun critère (§2) :
#:   - drain 0,10 → 0,08/jauge : à 53,9 m de budget, 13 événements exigés étaient INJOIGNABLES (il
#:     aurait fallu 3,45 m/repas = ~48 sites, qui saturerait après retrain). 10 événements exigés est
#:     le bas HONNÊTE de la bande 10-30 — le budget réel ne porte pas le haut à 3000 ticks.
#:   - speed_cost 0,5556 → 0,444, lié au drain pour garder l'optimum au trot (cheapest_vx = 0,6).
#:   - init 75 → 60, restore 80 → 84 : plancher tenu à 25,0 %, 10 événements à la croisière.
#:   - densité 12 → 18 sites : vise ~4,5 m/repas pour un forageur COMPÉTENT (post-retrain), tolérable
#:     par le budget pour 10 événements. PAS tuné au 7,65 m OOD (qui sur-densifierait).
#: ⚠️ CE QUI RESTE IN VIVO (nuance 3 du pair) : le prochain pas est le RETRAIN, pas le critique ; le
#: WM a besoin de COUVERTURE + contacts, pas de la fréquence EXACTE des repas. Le calage fin du drain
#: (un seul chiffre) se fait après le retrain, sur la vitesse de croisière que la politique CHOISIRA
#: (inconnue avant). Cette calibration vise « l'entité mange raisonnablement », pas une perfection
#: pré-collecte — un objectif mouvant de toute façon. Sonde : diagnostics/diag_foret_g11_portee.py.
FORET_V1 = dataclasses.replace(
    BOSQUETS_V7_TYPES,
    name="foret_v1",
    # corps : l'éventail. vx=0.25 redonne 0.0118 m/tick = le corps d'AUJOURD'HUI (la marche reste le
    # régime déjà calibré) ; vx=1.0 donne 0.0472 m/tick = la cible du candidat D. L'éventail s'ouvre
    # donc VERS LE HAUT, sans invalider ce qui a été mesuré en bas.
    # kin_speed RELEVÉ 2,83 → 6,4 (2026-07-26, décision owner). Le budget de trajet est LINÉAIRE en
    # kin_speed, et c'est exactement le facteur qui manquait : le premier gate closed-loop a mesuré
    # 10,20 m de trajet par repas contre 4,49 m tolérés (dépassement 2,27x), donc un monde qui ne
    # supportait que ~4,4 événements par vie là où §2.13 en demande 10-30. x2,26 sur la vitesse porte
    # le budget de 53,9 m à ~122 m/vie au trot, soit 10 événements AU TRAJET MESURÉ — sans toucher à
    # la densité (qui dégraderait le slot, erreur 1,43 m à 60 % d'occupation) ni aux drains ni à la
    # cible. Le coût k·vx² est indépendant de kin_speed : les identités métaboliques restent vraies.
    # ⚠️ La marche ne reproduit plus le corps historique (0,0333 m/tick au lieu de 0,0118).
    # RELEVÉ UNE SECONDE FOIS 6,4 → 8,0 (2026-07-26) pour prendre de la MARGE : à 6,4 le budget
    # toléré (10,16 m/repas) égalait le trajet mesuré (10,20 m) — une marge de 1,00x, c'est-à-dire
    # un cycle de 85 min joué à pile ou face. À 8,0 la tolérance passe à ~12,7 m, soit 1,25x.
    kin_speed=8.0,
    # ROTATION RELEVÉE 1,5 → 6,0 (audit S1). Le rayon de braquage vaut kin_speed·vx/kin_turn : à 1,5
    # il faisait 3,20 m quand l'écart libre entre deux arbres voisins en fait 1,54 — le corps ne
    # pouvait PAS slalomer, seulement contourner un massif entier. À 6,0 il braque en 0,80 m.
    # Ce n'est pas un réglage de confort : une rotation calibrée pour un corps 10x plus lent devient
    # géométriquement incompatible avec la forêt dès qu'on accélère.
    kin_turn=6.0,
    vx_fan=(0.25, 0.60, 1.00),          # marcher / trotter / sprinter (§2.13)
    # speed_cost lié au drain par k = (D_énergie + D_soif) / 0.6² : garde l'optimum au mètre SUR le
    # trot (cheapest_vx = 0,6). Avec drain 0,08+0,08, k = 0,16/0,36 = 0,4444.
    speed_cost=0.4444,
    # TERRAIN mesuré sous la politique par G11 = 0,635 (pas la moyenne d'arène, pas la médiane du
    # babillage) : c'est lui qui ramène le budget de trajet de 84,9 m (fantasme) à 53,9 m (réel).
    terrain_factor=0.635,
    # métabolisme RE-CALIBRÉ sur le budget réel (2026-07-25). drain 0,08/jauge → 10 événements exigés
    # à la croisière (bas de la bande 10-30, honnête : 53,9 m ne portent pas le haut) ; init 60 tient
    # le plancher à 25,0 % (60/0,08 = 750 ticks) ; restore 84 = un repas remplit une petite réserve.
    energy_drain=0.08, thirst_drain=0.08, init_energy=60.0, restore_per_item=84.0,
    # ressources : densité RE-CALIBRÉE 12 → 18. Le budget réel (53,9 m) ne tolère que 4,5 m de trajet
    # par événement pour 10 repas ; 12 bosquets donnaient 7,65 m (WM OOD, G11). +50 % de densité vise
    # ~5 m pour un forageur COMPÉTENT (post-retrain), SANS carpetter (pas les ~48 sites qu'exigerait le
    # chiffre OOD, qui saturerait après retrain). Le chiffre exact reste affiné IN VIVO (nuance 3).
    # ARÈNE x10 EN AIRE (rayon 11 -> 35 m) : 70 m de diamètre = 31 longueurs de corps, contre 10
    # avant — une arène de 10 corps est une pièce, pas un monde. Le coût de calcul a été MESURÉ avant
    # de s'engager (§6quater D, gate jamais passé jusqu'ici) : +3,6 % seulement pour x10 d'aire et
    # x6 d'objets. La DENSITÉ de ressources est conservée, donc le trajet par repas (~10,2 m mesuré)
    # et toute la calibration métabolique restent valides.
    spawn_annulus_m=(3.0, 35.0),
    # DENSITÉ DIVISÉE PAR 7 (2026-07-28), sur mesure et non sur goût. La densité conservée ci-dessus
    # posait une ressource tous les 2 m ; le planner n'avait donc jamais à chasser, il se rechargeait
    # en passant. Sondé à masse de nourriture CONSTANTE (probe_foret_densite.sh, 6 vies/condition,
    # même graine, même WM) :
    #        bosquets   repas/vie   énergie AU repas   encaissé   survie méd   trajet/repas
    #            180        5,3            73             21          412         8,5 m
    #             60        3,5            51             32          632         9,0 m
    #             25        3,0            56             28          856        12,5 m
    # La survie DOUBLE en éclaircissant, et le trajet reste très sous le budget (15,2 m). C'est le
    # critère H1 pré-enregistré qui passe (-17 pts d'énergie-au-repas) ; H2 a échoué (1,36x contre
    # 1,50x exigé) — d'où la seconde correction, le plafond de jauge, SANS laquelle éclaircir ne
    # suffisait pas (856 ticks restent loin des 3000).
    # items_total INCHANGÉ à 180 : la masse de nourriture du monde ne bouge pas, seule sa
    # concentration change. 180 baies sur 25 bosquets = ~7 par bosquet, c'est-à-dire un vrai
    # bosquet — là où 180 bosquets d'UNE baie n'étaient pas des bosquets du tout.
    patches_per_resource=25, items_total=180,
    # L'ÉCART SUIT LA DENSITÉ, sinon « moins de bosquets » veut dire « tous tassés dans un coin » :
    # mesuré, à écart figé 3-6 m le placeur exige que chaque bosquet touche la chaîne existante et
    # 25 bosquets se serraient à 3,10 m, exactement comme 180. On prend l'écart typique d'un semis
    # uniforme, sqrt(aire/n) = 12,4 m, borné à [0,70x, 1,50x] — mesuré 8,68 m d'écart réel.
    patch_spacing_min_m=8.7, patch_spacing_max_m=18.5,
    # PLAFOND DE JAUGE 100 -> 200 (owner, 2026-07-28), le champ gauge_max enfin servi. Un repas rend
    # 84 points : sur une jauge de 100 il n'est encaissable en entier qu'à 16 d'énergie, un niveau
    # qu'un planner qui cherche à survivre ne s'autorise jamais — mesuré, elle mange entre 73 et 51
    # et n'encaisse que 25-38 % du repas. La valeur servie était donc indélivrable, et le calibrage
    # « 10 événements/vie » était bâti dessus. init reste à 60 : agrandir le PLAFOND rend le repas
    # encaissable, agrandir le DÉPART aurait doublé la réserve et adouci le monde (plancher 25 % ->
    # 50 %). Avec (200, 60) le plancher de famine reste 750 ticks = 25,0 % et le compte retombe
    # exactement sur la cible pré-enregistrée : 10,0 événements/vie.
    gauge_max=200.0,
    # HORLOGES RAMENÉES DANS LA VIE (audit S3) : une vie sans manger dure 375 ticks. Une repousse à
    # 2 500 ticks ne se déclenchait JAMAIS (le monde était un garde-manger qui se vide) et un cycle
    # de flaque à 300 tenait à peine un cycle. Une mécanique qui ne se déclenche pas n'existe pas.
    regrow_ticks=300, water_puddle_period=150,
    # perception : la palette séparable validée par G5 (les TYPE_COLORS par défaut sont des multiples
    # scalaires l'un de l'autre — cosinus mutuel ~1,0, donc illisibles par construction)
    food_type_hues=((0.9, 0.12, 0.1), (0.9, 0.55, 0.08), (0.85, 0.1, 0.45), (0.8, 0.42, 0.42)),
    # MOBILES REMIS À L'ÉCHELLE DU CORPS (audit S2). La proie était à 0,0099 m/tick — calibré quand
    # l'agent faisait 0,011. Face à une croisière de 0,0508 elle était à 0,19x, soit QUASI IMMOBILE :
    # `diag_prey_interception` a mesuré qu'en dessous de 0,6x le gain de l'interception sur la
    # poursuite est NUL, donc toute la brique « nourriture mobile » était décorative. Remise à 0,9x,
    # la valeur où l'interception paie (67,5 % de capture contre 56,2 %).
    prey_speed=0.046,
    distractor_speed=0.030,             # 0,6x la croisière : ils bougent VRAIMENT pour ce corps
    # forêt : occlusion + couvert + variation d'apparence, à la densité navigable mesurée (45 arbres ;
    # 54 → immobile 85 % du temps, plafond dur du §3)
    # FORÊT À L'ÉCHELLE D'UN LOUP (2026-07-28, audit diag_world_scale). 191 arbres sur 3 820 m² =
    # 20 m²/tige, soit la borne basse d'une forêt réelle (500-1000 tiges/ha) — au lieu du fourré de
    # 7,8 m²/tige où l'écart libre entre voisins (0,70 m) était plus petit que le rayon de braquage.
    forest_count=191, forest_stands=60, forest_clearings=30, forest_appearance_var=0.15,
    forest_radius_var=0.35,             # troncs de 0,23 à 0,47 m : la GÉOMÉTRIE varie, donc s'apprend
    forest_ring_m=(2.5, 35.0),
    terrain_slow=0.6, terrain_radius_m=2.5, terrain_floor=0.25,
    distractor_count=60,                # densité conservée (6 sur 380 m² -> 60 sur 3 820)
    gaze=True,                          # proprio 132 → 133 : incompatible avec les checkpoints actuels
    # DANGER : réglages du monde-danger déjà promu (2026-07-17). Il n'est PAS là pour ajouter une
    # difficulté — il est là parce que la spec l'inclut (§6quinquies B) ET parce qu'une TROISIÈME
    # conséquence perceptible est ce qui rend le lien apparence→conséquence apprenable : sans elle,
    # les requêtes-couleur restent codées-main (verrou A2) et le tronc-brun survit au retrain.
    hazard_count=10, hazard_engulf_p=0.5, health_regen=0.05,   # densité conservée sur l'arène x10
    # kin_body_extent VOLONTAIREMENT NON SERVI (owner, 2026-07-28). L'encombrement réel est mesuré
    # (museau 1,024 m, demi-largeur 0,213 m) et le mécanisme est committé — mais l'activer coince le
    # planner 43,9 % du temps et effondre le budget à 45 m/vie contre les 152 exigés. On garde donc
    # le corps tel qu'il est CONÇU depuis le pivot cinématique — une sphère de 0,35 m — et c'est
    # l'HABILLAGE qu'on recule pour qu'il cesse de traverser les troncs. Rien ne change côté
    # dynamique, donc rien à re-mesurer ; le vrai volume attend un corps qui en aura besoin.
)


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

    # ── LA FORÊT : chaque critère de calibration est VÉRIFIÉ, pas déclaré ────────────────────
    f = FORET_V1
    floor = f.init_energy / f.energy_drain
    assert abs(floor / f.episode_steps - 0.25) < 0.005, floor / f.episode_steps
    assert f.thirst_drain == f.energy_drain, "les deux jauges doivent avoir le MÊME plancher"
    print(f"  [ok] foret_v1 : plancher de famine {floor:.0f} ticks = "
          f"{floor / f.episode_steps * 100:.1f} % de la vie (<= 25 %, jauges symétriques)")

    vstar = f.cheapest_vx
    assert min(f.vx_fan) < vstar < max(f.vx_fan), (f.vx_fan, vstar)
    assert abs(vstar - 0.60) < 0.01, vstar

    def per_m(vx: float) -> float:
        return f.drain_at(vx) / (f.kin_speed * vx / 60.0)
    assert per_m(vstar) < per_m(min(f.vx_fan)) and per_m(vstar) < per_m(max(f.vx_fan))
    marge = per_m(max(f.vx_fan)) / per_m(vstar) - 1.0
    assert marge >= 0.02, marge
    print(f"  [ok] foret_v1 : optimum au mètre vx*={vstar:.2f} INTÉRIEUR à l'éventail "
          f"{f.vx_fan} | sprinter coûte {marge * 100:+.0f} % de plus au mètre que trotter "
          "→ la vitesse est un pari, pas un choix gratuit")

    ev = [f.events_at(vx) for vx in f.vx_fan]
    assert 9.9 <= ev[1] <= 30.0, ev               # RE-CALIBRÉ au bas de la bande (budget réel) : ≈10
    print(f"  [ok] foret_v1 : événements/vie {ev[0]:.1f} (marche) / {ev[1]:.1f} (trot) / "
          f"{ev[2]:.1f} (sprint) — bas de la bande [10,30] à la CROISIÈRE (le budget réel ne porte "
          "pas le haut) ; sprinter en permanence reste délibérément inabordable")

    # LA CORRECTION TERRAIN EST VIVE : le budget de trajet réel (terrain 0,635) est STRICTEMENT sous
    # le budget sol-dégagé (1,0) qu'on supposait à tort. Sans cette assertion, un retour silencieux
    # du facteur à 1,0 repasserait inaperçu — la panne même que ce fichier existe pour empêcher.
    real = f.travel_budget(vstar)
    naive = (f.kin_speed * vstar / 60.0) * f.episode_steps
    assert f.terrain_factor < 1.0 and abs(real - naive * f.terrain_factor) < 1e-6
    assert abs(real - 152.4) < 2.0, real   # 8.0 x 0.6 x 0.635 / 60 x 3000
    print(f"  [ok] foret_v1 : budget de trajet RÉEL {real:.1f} m/vie au trot (terrain "
          f"{f.terrain_factor}) vs {naive:.1f} m si sol dégagé — la V1 surestimait de "
          f"{naive / real:.2f}x")

    # SYMÉTRIE DES DEUX PULSIONS — la garde qui manquait le 2026-07-28. Ces clés décrivent la
    # GÉOMÉTRIE des bosquets ; servie à la nourriture seule, elle laisse l'eau retomber sur les
    # défauts du GDScript (écart mini 9,0 m, pas de borne haute) que le placeur ne peut pas
    # satisfaire → 2 bosquets d'eau sur 180, et une entité qui meurt de soif sans jamais boire.
    # Le bug était invisible : aucune erreur, un monde qui se charge, un corpus qui aurait
    # simplement été dépourvu de la seconde pulsion. On compare les VALEURS, pas la présence.
    fenv = f.to_env()
    for suffix in ("COUNT", "PATCHES", "PATCH_RADIUS", "PATCH_SPACING",
                   "PATCH_SPACING_MAX", "REGROW", "MIN_RADIUS", "SPAWN_RADIUS"):
        fk, wk = f"SYLVAN_FOOD_{suffix}", f"SYLVAN_WATER_{suffix}"
        assert wk in fenv, f"{wk} n'est pas servi — l'eau retombera sur le défaut du GDScript"
        assert fenv[fk] == fenv[wk], f"géométrie asymétrique : {fk}={fenv[fk]} mais {wk}={fenv[wk]}"
    # UN REPAS DOIT ÊTRE ENCAISSABLE. C'est l'incohérence trouvée le 2026-07-28 : restore 84 sur une
    # jauge de 100 n'est réalisable qu'à 16 d'énergie, et l'entité mange à 51-73 (MESURÉ sur trois
    # densités). Toute la calibration en événements/vie suppose que le repas est encaissé EN ENTIER ;
    # si le plafond ne le permet pas, cette calibration est fausse et rien ne le dit. On exige donc
    # que le repas tienne depuis le niveau où elle mange RÉELLEMENT, pas depuis un niveau théorique.
    EATS_AT = 56.0                       # médiane mesurée à 25 bosquets (probe_foret_densite.sh)
    assert f.gauge_max - EATS_AT >= f.restore_per_item, (
        f"repas indélivrable : {f.restore_per_item} pts servis mais seulement "
        f"{f.gauge_max - EATS_AT} encaissables depuis {EATS_AT} (plafond {f.gauge_max})")
    floor_ticks = f.init_energy / f.energy_drain
    assert abs(floor_ticks / f.episode_steps - 0.25) < 0.005, floor_ticks
    print(f"  [ok] foret_v1 : repas {f.restore_per_item:.0f} pts ENCAISSABLE depuis {EATS_AT:.0f} "
          f"(plafond {f.gauge_max:.0f}) et plancher de famine inchangé à "
          f"{floor_ticks / f.episode_steps * 100:.1f} % — agrandir la jauge n'adoucit rien")

    print(f"  [ok] foret_v1 : les 2 pulsions reçoivent la MÊME géométrie de bosquets "
          f"(écart {f.patch_spacing_min_m}-{f.patch_spacing_max_m} m, "
          f"{f.patches_per_resource} bosquets, rayon {f.patch_radius_m} m) — "
          "sans cette garde, l'eau était placée à 2/180")

    # JOIGNABILITÉ : le trajet toléré par événement, et l'honnêteté sur ce qui le rend atteignable.
    allowed = f.metres_per_event_budget(vstar)          # 53,9 / (1,2 x 10) = 4,49 m
    # Ce chiffre décrit la DENSITÉ SERVIE, il n'est donc pas transposable d'un monde à l'autre : le
    # 10,20 m d'avant avait été mesuré à 180 bosquets et aurait menti ici (2026-07-28). Re-mesuré à
    # la densité du preset par diag_foret_g11_portee.measure() sur le corpus dens_25 de la sonde.
    TPM_MEASURED = 12.50   # 25 bosquets, 6 vies, 18 repas — probe_foret_densite.sh
    assert TPM_MEASURED <= allowed, (
        f"trajet/repas mesuré {TPM_MEASURED} m > {allowed:.2f} m tolérés : à cette densité l'entité "
        "ne peut pas joindre ses ressources dans son budget, quel que soit son cerveau")
    print(f"  ⚠️  foret_v1 : budget tolère {allowed:.2f} m/repas pour {ev[1]:.0f} événements | MESURÉ "
          f"à {f.patches_per_resource} bosquets (6 vies, 18 repas) : {TPM_MEASURED} m — la marge "
          f"({allowed / TPM_MEASURED:.2f}x) s'ajoute au 1,2x déjà pris dans le budget, mais elle "
          "reste MINCE et le corpus qui la mesure est petit")

    # LES TROIS CONSÉQUENCES doivent exister dans le monde COLLECTÉ, sinon l'encodeur reste aveugle à
    # celle qui manque et le lien apparence→conséquence n'est pas apprenable pour elle (verrou A2 :
    # build_typed_slots exige K=3 groupes et la bijection food→énergie / eau→soif / vert→dégâts).
    assert f.hazard_count > 0 and f.thirst_drain is not None, \
        "les 3 conséquences (nourrir / abreuver / blesser) doivent être servies dans la MÊME collecte"
    print(f"  [ok] foret_v1 : 3 conséquences servies (nourriture, eau, danger x{f.hazard_count}) — "
          "condition pour que les requêtes-couleur soient APPRENABLES (verrou A2)")

    env = f.to_env()
    for k in ("SYLVAN_FOREST_COUNT", "SYLVAN_TERRAIN_SLOW", "SYLVAN_GAZE", "SYLVAN_SPEED_COST",
              "SYLVAN_DISTRACTOR_COUNT", "SYLVAN_FOOD_TYPE_HUES", "SYLVAN_WATER_PUDDLE_PERIOD",
              "SYLVAN_INIT_THIRST", "SYLVAN_WM_VX_MIN", "SYLVAN_PLANNER_VX_GRID",
              "SYLVAN_HAZARD_COUNT", "SYLVAN_HEALTH_REGEN"):
        assert k in env, f"foret_v1 n'émet pas {k} — un corpus ne se décrirait pas lui-même"
    assert env["SYLVAN_INIT_THIRST"] == env["SYLVAN_INIT_ENERGY"]
    print(f"  [ok] foret_v1 : to_env() émet {len(env)} variables, dont les 9 briques de la forêt "
          "(un preset muet rendrait le corpus indescriptible)")

    # les presets d'avant ne bougent PAS d'un iota : tout ce qui précède est opt-in
    base = BOSQUETS_V7_TYPES.to_env()
    nouveaux = [k for k in base if k in ("SYLVAN_FOREST_COUNT", "SYLVAN_SPEED_COST", "SYLVAN_GAZE")]
    assert not nouveaux, nouveaux
    assert "SYLVAN_INIT_THIRST" not in base, "bosquets_v7 est mono-pulsion : aucune variable eau"
    print("  [ok] les presets d'avant n'émettent aucune variable forêt (les chiffres historiques "
          "restent reproductibles)")
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
    p = {"bosquets_v1": BOSQUETS_V1, "bosquets_v1_slowturn": BOSQUETS_V1_SLOWTURN, "bosquets_v2": BOSQUETS_V2, "bosquets_v3_perish": BOSQUETS_V3_PERISH, "bosquets_v4_ripe": BOSQUETS_V4_RIPE, "bosquets_v5_value": BOSQUETS_V5_VALUE, "bosquets_v6_prey": BOSQUETS_V6_PREY, "bosquets_v7_types": BOSQUETS_V7_TYPES,
     "perpetual_v0": PERPETUAL_V0, "foret_v1": FORET_V1}[a.preset]
    if a.env:
        for k, v in p.to_env().items():
            # GUILLEMETS SEULEMENT SI LA VALEUR EN A BESOIN. La palette de teintes contient des
            # « ; », que le shell lit comme un séparateur de commandes : sans guillemets,
            # `eval "$(... --env)"` coupe la ligne et sert une palette TRONQUÉE.
            # Mais les citer TOUTES casse l'autre style de lecture, `sed 's/^export //' | tr` , que
            # SIX harnais utilisent — ils recevaient alors `"120.0"` guillemets compris, et le
            # serveur mourait sur un float illisible. Mesuré en lançant le viewer.
            # On cite donc au strict nécessaire : les valeurs simples restent lisibles par les deux
            # styles, et seules celles qui contiennent un caractère spécial exigent `eval`.
            print(f'export {k}="{v}"' if any(c in v for c in ' ;"\'$&|<>()') else f'export {k}={v}')
    else:
        print(f"{p.name}: floor={p.starvation_floor_ticks:.0f} ticks, "
              f"meals needed={p.meals_needed:.2f}, ray step={p.ray_step_deg:.2f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
