"""AUDIT D'ÉCHELLE DU MONDE — chaque constante confrontée au CORPS RÉELLEMENT SERVI.

POURQUOI CET OUTIL EXISTE. Le corps est passé de kin_speed 0,8 à 8,0 — un facteur DIX — en deux
décisions successives, chacune justifiée isolément. Mais presque toutes les constantes du monde ont
été calibrées AUTOUR de l'ancien corps : la vitesse des proies (« 0,9x l'agent », mesurée quand
l'agent faisait 0,011 m/tick), l'horloge de repousse, la portée de la rétine, la densité des arbres,
le rayon des zones de danger. Aucune ne lève d'erreur en devenant absurde : elle devient simplement
INERTE, et on continue de croire que la mécanique agit.

C'est le mode de panne le plus coûteux du projet — pas le crash, le réglage silencieusement mort.
Le contrat de monde (`diag_world_contract`) vérifie que le monde SERT ce qu'on demande ; celui-ci
vérifie que ce qu'on demande a encore un SENS pour le corps qui l'habite.

MÉTHODE. Zéro Godot, zéro corpus : pure arithmétique sur le preset gelé. Chaque test oppose une
constante du monde à une capacité MESURÉE du corps, avec un seuil pré-enregistré et sa raison. Un
échec ne dit pas « le monde est mauvais », il dit « cette mécanique ne peut plus produire son effet ».

CRITÈRES PRÉ-ENREGISTRÉS (chacun cite la source de son seuil) :
  S1 MANŒUVRABILITÉ . le rayon de braquage doit tenir dans l'espace libre entre deux arbres voisins,
                      sinon le corps ne peut PAS slalomer — il ne peut que s'arrêter ou contourner
                      le massif entier. Seuil : rayon de braquage <= écart libre moyen.
  S2 MOBILES ....... proies et distracteurs doivent aller à >= 0,6x la vitesse de croisière : en
                     dessous, `diag_prey_interception` a MESURÉ que le gain de l'interception sur la
                     poursuite est nul (§2.4). Sous ce seuil, « nourriture mobile » est décoratif.
  S3 HORLOGES ...... toute horloge du monde (repousse, cycle des flaques) doit se déclencher au moins
                     une fois dans une vie, sinon la mécanique n'existe pas du point de vue de
                     l'entité. Seuil : période <= durée de vie.
  S4 ÉTENDUE ....... l'arène doit faire >= 20 longueurs de corps, et l'entité ne doit pas pouvoir la
                     traverser plus de ~4 fois par vie : au-delà elle en fait le tour et l'espace
                     cesse d'être une ressource (ni mémoire, ni exploration, ni choix de site).
  S5 DENSITÉ ....... l'écart LIBRE entre deux arbres voisins doit dépasser la largeur du corps,
                     sinon le corps ne passe pas — et un modèle de collision par RAYON le laisse
                     traverser quand même, ce qui fabrique un mensonge visuel (mesuré : museau dans
                     le tronc) et une dynamique que le WM apprendra comme du bruit.
  S6 PERCEPTION .... la rétine doit donner >= l'horizon du planner en ticks d'avance, sinon l'entité
                     voit une ressource trop tard pour que son plan puisse l'atteindre.
  S7 BOUCHE ........ la bouche doit être traversée en >= 4 ticks, sinon on la franchit entre deux
                     pas de décision et manger devient une loterie (seuil hérité de G2).

CLI :
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_world_scale.py
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_world_scale.py --preset foret_v1
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_world_scale.py --selfcheck
"""

from __future__ import annotations

import argparse
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from sylvan.world import FORET_V1, WorldPreset  # noqa: E402

TICKS_PER_S = 60.0
BODY_LEN_M = 2.221          # longueur du maillage servi, MESURÉE au chargement ([wolf] encombrement)
BODY_HALF_W = 0.213         # demi-largeur MESURÉE
PREY_MIN_RATIO = 0.6        # §2.4, mesuré par diag_prey_interception : en dessous, gain NUL
ARENA_MIN_BODIES = 20.0     # une arène de moins de 20 longueurs de corps est une pièce, pas un monde
CROSSINGS_MAX = 4.0         # au-delà, l'entité fait le tour et l'espace cesse d'être une ressource
MOUTH_MIN_TICKS = 4.0       # hérité de G2
REAL_FOREST_M2_PER_STEM = (10.0, 20.0)   # 500-1000 tiges/ha : une forêt réelle, pour référence
# La forêt est VOLONTAIREMENT groupée (processus de Thomas) : ses voisins sont donc plus proches
# qu'un semis de Poisson. Facteur MESURÉ sur le monde servi (Clark-Evans 0,749) — sans lui l'audit
# surestime l'écart libre de 33 %, et déclare praticable une forêt qui ne l'est pas.
CLUSTERING = 0.749


def life_ticks(p: WorldPreset, vx: float) -> float:
    """Durée d'une vie SANS manger, à la vitesse vx — la jauge qui se vide le plus vite décide."""
    loco = p.speed_cost * vx * vx
    drains = [p.energy_drain + loco * 0.5, (p.thirst_drain or 0.0) + loco * 0.5] \
        if p.thirst_drain is not None else [p.energy_drain + loco]
    worst = max(d for d in drains if d > 0)
    return p.init_energy / worst


def audit(p: WorldPreset) -> list[tuple[bool, str, str]]:
    """Renvoie [(ok, titre, détail)] — chaque ligne oppose une constante du monde au corps servi."""
    vx = p.cheapest_vx if p.vx_fan and p.speed_cost > 0 else 0.6
    vx = min(max(vx, min(p.vx_fan)), max(p.vx_fan)) if p.vx_fan else vx
    v_cmd = p.kin_speed * vx / TICKS_PER_S                  # m/tick commandée
    v_real = v_cmd * p.terrain_factor                        # m/tick réellement parcourue
    life = life_ticks(p, vx)
    out: list[tuple[bool, str, str]] = []

    # --- géométrie de la forêt : espace libre entre deux voisins -------------------------------
    r_in, r_out = p.spawn_annulus_m
    area = math.pi * (r_out ** 2 - r_in ** 2)
    if p.forest_count > 0:
        m2_per_stem = area / p.forest_count
        # espacement moyen au plus proche voisin sous un semis de Poisson : 0.5/sqrt(densité)
        nn = 0.5 / math.sqrt(p.forest_count / area) * CLUSTERING
        # PIRE CAS, pas le cas moyen : avec des troncs de taille variable, deux GROS voisins laissent
        # moins d'espace que deux moyens. Un audit qui juge sur la moyenne déclare praticable un
        # monde où l'entité se coince régulièrement.
        r_max = 0.35 * (1.0 + p.forest_radius_var)
        gap = nn - 2.0 * r_max
        body_w = 2.0 * BODY_HALF_W
        # ⚠️ CRITÈRE CORRIGÉ (2026-07-28) : je testais le braquage à la CROISIÈRE, ce qui confond
        # « le corps peut passer » et « il peut passer SANS RALENTIR ». L'éventail de vitesse existe
        # justement pour ralentir en zone dense — un loup ne traverse pas un fourré au trot. Le
        # critère dur porte donc sur la vitesse la plus LENTE de l'éventail (peut-il passer, tout
        # court ?) ; le braquage à la croisière est rendu comme INFORMATION (doit-il ralentir ?).
        vx_slow = min(p.vx_fan) if p.vx_fan else vx
        turn_slow = (p.kin_speed * vx_slow) / p.kin_turn
        turn_cruise = (p.kin_speed * vx) / p.kin_turn
        must_slow = turn_cruise > gap
        out.append((turn_slow <= max(gap, 1e-6), "S1 MANŒUVRABILITÉ",
                    f"braquage {turn_slow:.2f} m au pas (croisière {turn_cruise:.2f}) vs écart libre "
                    f"{gap:.2f} m (centre-à-centre {nn:.2f} m) → "
                    f"{'passe au pas' if turn_slow <= gap else 'NE PASSE PAS même au pas : c est un fourré'}"
                    f"{', doit RALENTIR pour slalomer' if must_slow else ', passe même au trot'}"))
        out.append((gap >= body_w, "S5 DENSITÉ",
                    f"écart libre {gap:.2f} m vs largeur du corps {body_w:.2f} m | "
                    f"{m2_per_stem:.1f} m²/tige (forêt réelle {REAL_FOREST_M2_PER_STEM[0]:.0f}-"
                    f"{REAL_FOREST_M2_PER_STEM[1]:.0f}) → "
                    f"{'le corps passe' if gap >= body_w else 'le corps NE PASSE PAS : un modèle par RAYON le laisse traverser quand même'}"))

    # --- mobiles : sont-ils encore mobiles RELATIVEMENT au corps ? ------------------------------
    if p.prey_speed > 0:
        ratio = p.prey_speed / v_real
        out.append((ratio >= PREY_MIN_RATIO, "S2 MOBILES (proie)",
                    f"proie {p.prey_speed:.4f} m/tick = {ratio:.2f}x la croisière {v_real:.4f} "
                    f"(exigé >= {PREY_MIN_RATIO}) → "
                    f"{'la poursuite reste un problème' if ratio >= PREY_MIN_RATIO else 'la proie est QUASI IMMOBILE pour ce corps : la brique est inerte'}"))

    # --- horloges : se déclenchent-elles seulement une fois par vie ? ---------------------------
    out.append((p.regrow_ticks <= life, "S3 HORLOGE (repousse)",
                f"repousse {p.regrow_ticks} ticks vs vie {life:.0f} ticks → "
                f"{'la ressource se renouvelle' if p.regrow_ticks <= life else 'AUCUNE repousse dans une vie : le monde est un garde-manger qui se vide'}"))
    if p.water_puddle_period > 0:
        out.append((p.water_puddle_period <= life * 0.5, "S3 HORLOGE (flaque)",
                    f"cycle {p.water_puddle_period} ticks vs vie {life:.0f} → "
                    f"{'le rétrécissement est observable' if p.water_puddle_period <= life * 0.5 else 'à peine un cycle par vie : la variabilité est invisible'}"))

    # --- étendue : le monde est-il un espace, ou une pièce ? ------------------------------------
    diam = 2.0 * r_out
    bodies = diam / BODY_LEN_M
    crossings = (v_real * life) / diam
    out.append((bodies >= ARENA_MIN_BODIES, "S4 ÉTENDUE (taille)",
                f"arène {diam:.0f} m = {bodies:.0f} longueurs de corps (exigé >= {ARENA_MIN_BODIES:.0f})"))
    out.append((crossings <= CROSSINGS_MAX, "S4 ÉTENDUE (traversées)",
                f"{crossings:.1f} traversées par vie (max {CROSSINGS_MAX:.0f}) → "
                f"{'l espace reste une ressource' if crossings <= CROSSINGS_MAX else 'elle en fait le TOUR : ni mémoire ni exploration ne peuvent servir'}"))

    # --- perception et bouche ------------------------------------------------------------------
    out.append((p.retina_range_m / v_real >= 80, "S6 PERCEPTION",
                f"rétine {p.retina_range_m:.0f} m = {p.retina_range_m / v_real:.0f} ticks d'avance "
                f"(horizon planner 80)"))
    out.append((2.0 * p.eat_radius_m / v_real >= MOUTH_MIN_TICKS, "S7 BOUCHE",
                f"bouche traversée en {2.0 * p.eat_radius_m / v_real:.0f} ticks (min {MOUTH_MIN_TICKS:.0f})"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="foret_v1")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    p = FORET_V1 if a.preset == "foret_v1" else FORET_V1
    vx = p.cheapest_vx
    v_real = p.kin_speed * vx * p.terrain_factor / TICKS_PER_S
    print(f"MONDE {p.name} | corps kin_speed={p.kin_speed} croisière vx={vx:.2f} "
          f"→ {v_real:.4f} m/tick réels | vie sans manger {life_ticks(p, vx):.0f} ticks")
    print(f"corps MESURÉ : {BODY_LEN_M:.2f} m de long, {2 * BODY_HALF_W:.2f} m de large\n")
    rows = audit(p)
    for ok, title, detail in rows:
        print(f"{'✅' if ok else '❌'} {title:24s} {detail}")
    bad = [t for ok, t, _ in rows if not ok]
    print("\n" + "=" * 96)
    if bad:
        print(f"{len(bad)} échelle(s) INCOHÉRENTE(S) : {', '.join(bad)}")
        print("Aucune ne lève d'erreur à l'exécution — elles rendent la mécanique INERTE en silence.")
    else:
        print("Toutes les échelles sont cohérentes avec le corps servi.")
    return 1 if bad else 0


def selfcheck() -> int:
    import dataclasses
    # une horloge plus longue qu'une vie doit ÉCHOUER, plus courte doit passer
    slow = dataclasses.replace(FORET_V1, name="t", regrow_ticks=99999)
    fast = dataclasses.replace(FORET_V1, name="t", regrow_ticks=10)
    g = lambda pr, key: [ok for ok, t, _ in audit(pr) if t.startswith(key)][0]  # noqa: E731
    assert not g(slow, "S3 HORLOGE (repousse)") and g(fast, "S3 HORLOGE (repousse)")
    print("  [ok] S3 distingue une horloge qui se déclenche d'une qui ne se déclenche jamais")

    # une proie à 0.9x la croisière doit PASSER, à 0.1x échouer
    v = FORET_V1.kin_speed * FORET_V1.cheapest_vx * FORET_V1.terrain_factor / TICKS_PER_S
    ok9 = g(dataclasses.replace(FORET_V1, name="t", prey_speed=0.9 * v), "S2 MOBILES")
    ok1 = g(dataclasses.replace(FORET_V1, name="t", prey_speed=0.1 * v), "S2 MOBILES")
    assert ok9 and not ok1
    print("  [ok] S2 mesure la vitesse de la proie RELATIVEMENT au corps, pas dans l'absolu")

    # doubler le rayon de l'arène doit améliorer l'étendue et réduire les traversées
    # 22 m de rayon donnerait 19,8 longueurs de corps — JUSTE sous le seuil de 20. On prend donc
    # franchement plus grand, sinon le test mesure la marge du seuil et pas la sensibilité du critère.
    big = dataclasses.replace(FORET_V1, name="t", spawn_annulus_m=(3.0, 40.0))
    assert g(big, "S4 ÉTENDUE (taille)") and not g(FORET_V1, "S4 ÉTENDUE (taille)")
    print("  [ok] S4 réagit à l'étendue : l'arène servie échoue, une arène double passe")
    print("SELFCHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
