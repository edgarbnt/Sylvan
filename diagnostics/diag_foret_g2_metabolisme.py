"""G2 GRATUIT — RE-CALIBRATION MÉTABOLIQUE du monde-forêt (design_foret_complete.md §6quater C).

PÉRIMÈTRE. Aucun run, aucun Godot, aucun entraînement. Pure arithmétique, VALIDÉE contre les
chiffres déjà mesurés du monde courant. Objectif : dire quelles constantes atteignent la cible
« 10 à 30 ÉVÉNEMENTS par vie » (aujourd'hui 1 à 2), et ce que chacune coûte.

CE QUE LA SPEC DEMANDE (§6quater C, tranché) :
  « Ouvrir l'éventail de vitesse invalide drain, rayon de capture, portée rétine, durée d'épisode
    et densité de ressources — tous calés sur 0,011 m/tick. On re-calibre EXPLICITEMENT, avec une
    cible chiffrée : 10 à 30 événements par vie, et une survie ni saturée ni effondrée. »

────────────────────────────────────────────────────────────────────────────────────────────
LE CORPS EST EXACTEMENT ANALYTIQUE (lu dans sylvan_agent.gd `_kinematic_step`, pas supposé)
────────────────────────────────────────────────────────────────────────────────────────────
    vel      = forward * (kin_speed * cpg_command.x)        →  m/tick = kin_speed * vx / 60
    kin_yaw += kin_turn * cpg_command.y * delta             →  rad/tick = kin_turn * omega / 60
Contrôle : kin_turn=1,5 et omega=0,6 donnent 0,015 rad/tick — exactement la valeur que le preset
déclare avoir MESURÉE. C'est aussi, en une ligne, pourquoi le déplacement du WM est reconstructible
à R² 0,985 depuis la commande seule (§2.3) : il n'y a rien d'autre à modéliser dans ce corps.

────────────────────────────────────────────────────────────────────────────────────────────
LES QUATRE IDENTITÉS, ET LAQUELLE EST FRAGILE
────────────────────────────────────────────────────────────────────────────────────────────
(1) DEMANDE ....... evenements = (drain * steps - init_energy) / restore_par_item
    C'est le nombre de repas que la vie EXIGE. Le réservoir de départ est le terme que tout le
    monde oublie : drain*steps/restore vaut 3,75 sur le monde courant, mais la demande NETTE
    n'est que 1,25 — et le monde courant mesure 1,40 repas. La demande est donc bien ce qui LIE.
    ⇒ COROLLAIRE, et il corrige une phrase de la spec : la vitesse ne peut PAS à elle seule
      monter le nombre d'événements. Le plafond est métabolique. §2.13 propose « priorité à la
      vitesse » pour atteindre 10-30 événements ; la vitesse règle la JOIGNABILITÉ de ce nombre,
      pas le nombre lui-même. Les deux se composent, ils ne se substituent pas.

(2) PLANCHER ...... floor = init_energy / drain, à comparer à steps
    Combien de temps survit-on en ne faisant RIEN. Le monde courant : 2000 ticks pour une vie de
    3000, soit 67 %. C'est la cause mécanique de la survie SATURÉE que le projet a déjà payée
    (9 vies sur 10 au plafond → métrique aveugle). Une survie informative exige floor << steps.

(3) JOIGNABILITÉ .. v * steps >= evenements * metres_par_repas
    Le budget de déplacement d'une vie doit couvrir la distance réellement parcourue par repas.

(4) BOUCHE ........ ticks_dans_la_bouche = 2 * eat_radius / v
    Le test de capture tourne une fois par tick : trop vite, on traverse la bouche entre deux
    tests. C'est la contrainte que « ouvrir la vitesse » menace le plus directement.

⚠️ LE TERME FRAGILE EST `metres_par_repas`. Il n'est PAS dérivable : il contient le comportement
(recherche, détours, hésitations). On le CALIBRE donc sur le monde mesuré au lieu de le postuler —
c'est la leçon de bosquets_v2, où une arithmétique de balayage a été réfutée parce qu'elle chiffrait
le prix d'un comportement que l'entité n'a jamais eu. Ici : 1,40 repas mesurés sur un budget de
33 m donnent 23,6 m par repas, tout compris.

────────────────────────────────────────────────────────────────────────────────────────────
CE QUE CETTE SONDE PEUT ET NE PEUT PAS CONCLURE
────────────────────────────────────────────────────────────────────────────────────────────
Elle donne une condition NÉCESSAIRE : le monde doit EXIGER 10-30 repas, et le budget de
déplacement doit pouvoir les couvrir. Elle ne donne PAS la condition suffisante : que l'entité les
OBTIENNE. Cela se mesure en vies, après la collecte et le retrain — et ne peut pas se mesurer avant,
puisque tout changement de vitesse change la dynamique que le WM doit avoir apprise (§6quinquies E).

CLI :
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_foret_g2_metabolisme.py
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_foret_g2_metabolisme.py --selfcheck
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from sylvan.world import BOSQUETS_V2, WorldPreset  # noqa: E402

TICKS_PER_SEC = 60.0            # delta physique de Godot : m/tick = kin_speed * vx / 60
PLANNER_HORIZON = 80            # ticks d'imagination du planner (A5 : 0,88 m pour des cibles à 7,6 m)

# ⚠️ ÉCART NON RÉSOLU, SIGNALÉ PLUTÔT QUE TRANCHÉ EN SILENCE (trouvé par le selfcheck de cette sonde).
# La spec §2.13 et la docstring du preset déclarent tous deux une vitesse MESURÉE de 0,011 m/tick.
# Or la formule lue dans le code donne kin_speed * vx / 60, et le grid réellement servi par le
# planner (control/planning/command_planner.py:52) plafonne à vx=0,75, soit 0,8*0,75/60 = 0,0100
# m/tick. Reproduire 0,011 demanderait vx≈0,825, HORS du grid. 10 % d'écart, origine non tranchée.
# On calibre sur la MESURE — c'est elle qui a produit les 1,40 repas de référence, et apparier un
# budget analytique avec un comptage mesuré mélangerait deux mondes — et on garde l'écart VISIBLE.
MEASURED_M_PER_TICK = 0.011
VX_EFFECTIVE = MEASURED_M_PER_TICK * TICKS_PER_SEC / 0.8    # 0,825 — le vx qui REPRODUIT la mesure
VX_GRID_MAX = 0.75                                          # ce que le planner offre réellement

# ── LA CIBLE ET LES BORNES, PRÉ-ENREGISTRÉES ───────────────────────────────────────────────
EVENTS_MIN, EVENTS_MAX = 10.0, 30.0   # §2.13(b) / §6quater C
FLOOR_FRAC_MAX = 0.25                 # ne rien faire ne doit pas survivre plus d'1/4 de la vie
MOUTH_TICKS_MIN = 4.0                 # au moins 4 tests de capture en traversant la bouche
# MARGE de joignabilité. Exiger exactement 1,00x serait un fil du rasoir : `metres_par_repas` est le
# terme FRAGILE (il est calibré sur UN monde mesuré et contient du comportement). Un monde
# dimensionné au ras du budget n'a aucune réserve pour la moindre inefficacité de recherche.
REACH_MIN = 1.20

# ── LE MONDE MESURÉ, QUI SERT DE CALIBRAGE ────────────────────────────────────────────────
# Chiffres MESURÉS sur bosquets_v1/v2 (docstring du preset : « 1.40 meals », 55 % de vies pleines,
# 15 % au plancher de famine). C'est notre seul point d'ancrage empirique — tout ce qui suit s'y
# rapporte, et rien n'y est postulé.
MEASURED_MEALS = 1.40


def m_per_tick(p: WorldPreset, vx: float = VX_EFFECTIVE) -> float:
    return p.kin_speed * vx / TICKS_PER_SEC


def events_demanded(p: WorldPreset) -> float:
    """Identité (1) — le nombre de repas que la vie EXIGE, réservoir de départ déduit."""
    drain = p.energy_drain + (p.thirst_drain or 0.0)
    return max(0.0, drain * p.episode_steps - p.init_energy) / p.restore_per_item


def floor_frac(p: WorldPreset) -> float:
    """Identité (2) — fraction de la vie que l'on survit en ne faisant RIEN."""
    return p.starvation_floor_ticks / p.episode_steps


def travel_budget(p: WorldPreset, vx: float = VX_EFFECTIVE) -> float:
    return m_per_tick(p, vx) * p.episode_steps


def metres_per_meal(p: WorldPreset, meals: float, vx: float = VX_EFFECTIVE) -> float:
    """Le terme FRAGILE, calibré sur une mesure : budget de déplacement / repas réellement obtenus."""
    return travel_budget(p, vx) / meals


def report(p: WorldPreset, m_per_meal: float, vx: float = VX_EFFECTIVE) -> dict:
    v = m_per_tick(p, vx)
    ev = events_demanded(p)
    need = ev * m_per_meal
    have = travel_budget(p, vx)
    return {
        "name": p.name,
        "v": v,
        "events": ev,
        "floor_frac": floor_frac(p),
        "floor_ticks": p.starvation_floor_ticks,
        "travel_have": have,
        "travel_need": need,
        "reach": have / need if need > 0 else float("inf"),
        "mouth_ticks": 2.0 * p.eat_radius_m / v,
        "imagination_m": v * PLANNER_HORIZON,
        "reaction_ticks": p.retina_range_m / v,
        "m_per_meal": m_per_meal,
    }


def verdict(r: dict) -> tuple[str, list[str]]:
    fails = []
    if not EVENTS_MIN <= r["events"] <= EVENTS_MAX:
        fails.append(f"événements {r['events']:.1f} hors [{EVENTS_MIN:.0f}, {EVENTS_MAX:.0f}]")
    if r["floor_frac"] > FLOOR_FRAC_MAX:
        fails.append(f"plancher de famine {r['floor_frac'] * 100:.0f} % de la vie "
                     f"(> {FLOOR_FRAC_MAX * 100:.0f} % → survie saturée)")
    if r["reach"] < REACH_MIN - 1e-9:   # epsilon : un candidat dimensionné À la marge la respecte
        fails.append(f"joignabilité {r['reach']:.2f}x < {REACH_MIN:.2f}x requis "
                     f"({r['travel_have']:.0f} m disponibles / {r['travel_need']:.0f} m requis)")
    if r["mouth_ticks"] < MOUTH_TICKS_MIN:
        fails.append(f"bouche traversée en {r['mouth_ticks']:.1f} ticks (< {MOUTH_TICKS_MIN:.0f})")
    return ("PASS" if not fails else "ÉCHEC"), fails


def show(r: dict, tag: str = "") -> None:
    v, fails = verdict(r)
    print(f"\n  {r['name']}{tag}  →  {v}")
    print(f"    vitesse {r['v']:.4f} m/tick | événements EXIGÉS {r['events']:5.1f} "
          f"| plancher {r['floor_ticks']:.0f} ticks = {r['floor_frac'] * 100:.0f} % de la vie")
    print(f"    déplacement : {r['travel_have']:6.1f} m disponibles / {r['travel_need']:6.1f} m requis "
          f"({r['reach']:.2f}x) à {r['m_per_meal']:.1f} m par repas")
    print(f"    bouche {r['mouth_ticks']:5.1f} ticks | imagination du planner {r['imagination_m']:.2f} m "
          f"sur {PLANNER_HORIZON} ticks | rétine vue {r['reaction_ticks']:.0f} ticks à l'avance")
    for f in fails:
        print(f"    ✗ {f}")


def solve(base: WorldPreset, m_per_meal_base: float,
          target: float) -> list[tuple[str, WorldPreset, float]]:
    """Candidats atteignant `target` événements. Renvoie (étiquette, preset, mètres_par_repas).

    ⚠️ Le mètres_par_repas est rendu AVEC chaque candidat, parce qu'un levier de DENSITÉ le change :
    juger un monde trois fois plus dense avec le trajet-par-repas du monde clairsemé le ferait
    échouer sur une contrainte qu'il vient précisément de desserrer.

    Chaque levier est celui que la spec nomme. On les sépare pour que le coût de chacun soit
    lisible, au lieu de livrer un seul jeu de constantes dont on ne saurait pas ce qui agit.
    Tous sont dimensionnés pour REACH_MIN, jamais au ras du budget.
    """
    out: list[tuple[str, WorldPreset, float]] = []
    L, E0 = base.episode_steps, base.init_energy

    def kin_for(mpm: float, steps: int = 0) -> float:
        """kin_speed tel que le budget de déplacement couvre REACH_MIN fois le besoin.

        Arrondi VERS LE HAUT au centième : arrondir au plus proche rabote la marge pour laquelle
        le candidat vient d'être dimensionné, et le fait échouer sur son propre critère.
        """
        steps = steps or L
        exact = REACH_MIN * target * mpm / (VX_EFFECTIVE / TICKS_PER_SEC * steps)
        return math.ceil(exact * 100.0) / 100.0

    # Levier A — RESTORE plus petit : des repas plus petits, donc plus nombreux. Gratuit en calcul.
    #   target = (drain*L - E0)/restore  →  restore = (drain*L - E0)/target
    r_a = (base.energy_drain * L - E0) / target
    out.append(("A restore↓ seul (repas plus petits)", dataclasses.replace(
        base, name=f"foret_A_restore{target:.0f}", restore_per_item=round(r_a, 2)), m_per_meal_base))

    # Levier B — DRAIN plus fort : casse AUSSI la survie saturée, puisque le plancher est E0/drain.
    #   target = (drain*L - E0)/restore  →  drain = (target*restore + E0)/L
    d_b = (target * base.restore_per_item + E0) / L
    out.append(("B drain↑ seul (casse aussi la saturation)", dataclasses.replace(
        base, name=f"foret_B_drain{target:.0f}", energy_drain=round(d_b, 4)), m_per_meal_base))

    # Levier C — B + vitesse, la combinaison que la spec vise (§2.13). Le drain fixe le NOMBRE,
    # la vitesse rend ce nombre JOIGNABLE : c'est la composition, pas la substitution.
    out.append(("C drain↑ + vitesse↑ (la vitesse porte TOUT le trajet)", dataclasses.replace(
        base, name=f"foret_C_vitesse{target:.0f}", energy_drain=round(d_b, 4),
        kin_speed=kin_for(m_per_meal_base)), m_per_meal_base))

    # Levier D — C + densité : au lieu de tout demander à la vitesse, on raccourcit le trajet par
    # repas. On divise donc AUSSI le mètres_par_repas rendu, sinon on jugerait D sur le monde de C.
    mpm_d = m_per_meal_base / 3.0
    out.append(("D drain↑ + vitesse↑ + densité x3 (le trajet se partage)", dataclasses.replace(
        base, name=f"foret_D_dense{target:.0f}", energy_drain=round(d_b, 4),
        kin_speed=kin_for(mpm_d)), mpm_d))

    # Levier E — D + vie plus longue. Le seul levier qui coûte du CALCUL, linéairement (§2.13),
    # mais il est le seul à ne pas toucher au corps : il n'exige donc AUCUNE ré-exploration d'action.
    steps_e = 2 * L
    d_e = (target * base.restore_per_item + E0) / steps_e
    out.append((f"E drain↑ + vitesse↑ + densité x3 + vie x2 ({steps_e} ticks)", dataclasses.replace(
        base, name=f"foret_E_longue{target:.0f}", energy_drain=round(d_e, 4), episode_steps=steps_e,
        kin_speed=kin_for(mpm_d, steps_e)), mpm_d))
    return out


def selfcheck() -> int:
    p = BOSQUETS_V2
    assert abs(p.kin_speed * VX_GRID_MAX / TICKS_PER_SEC - 0.0100) < 1e-9
    print(f"  [ok] cinématique analytique au HAUT du grid servi : {p.kin_speed} x {VX_GRID_MAX} / 60 "
          f"= 0,0100 m/tick — contre {MEASURED_M_PER_TICK} DÉCLARÉ mesuré (écart 10 %, signalé)")

    assert abs(VX_EFFECTIVE - 0.825) < 1e-9 and VX_EFFECTIVE > VX_GRID_MAX
    print(f"  [ok] le vx qui reproduit la mesure vaut {VX_EFFECTIVE:.3f}, soit AU-DESSUS du grid "
          f"({VX_GRID_MAX}) — l'écart est porté à l'écran, pas absorbé")

    assert abs(p.kin_turn * 0.6 / TICKS_PER_SEC - 0.015) < 1e-9
    print("  [ok] rotation : kin_turn 1.5 x omega 0.6 / 60 = 0.0150 rad/tick = la valeur MESURÉE "
          "déclarée par le preset (là, formule et mesure coïncident)")

    assert abs(events_demanded(p) - 1.25) < 1e-9, events_demanded(p)
    assert abs(p.meals_needed - events_demanded(p)) < 1e-9   # même identité que le preset
    print(f"  [ok] demande du monde courant {events_demanded(p):.2f} repas — et 1,40 MESURÉS : "
          "c'est bien la demande qui lie, pas l'offre")

    assert abs(floor_frac(p) - 2000.0 / 3000.0) < 1e-9
    print(f"  [ok] plancher de famine {floor_frac(p) * 100:.0f} % de la vie → la survie DOIT saturer")

    assert abs(travel_budget(p) - 33.0) < 0.01, travel_budget(p)
    mpm = metres_per_meal(p, MEASURED_MEALS)
    assert abs(mpm - 23.57) < 0.05, mpm
    print(f"  [ok] terme fragile calibré sur la mesure : {mpm:.1f} m par repas "
          f"(budget {travel_budget(p):.0f} m / {MEASURED_MEALS} repas) — les 33 m de la spec")

    # une solution doit vraiment résoudre : on remet le candidat dans l'identité de départ
    cands = solve(p, mpm, 15.0)
    for _, cand, _ in cands:
        assert abs(events_demanded(cand) - 15.0) < 0.05, (cand.name, events_demanded(cand))
    print("  [ok] chaque candidat re-injecté dans l'identité (1) redonne bien 15 événements")

    # le levier densité doit être JUGÉ sur le trajet qu'il raccourcit, sinon il échoue à tort
    dense = [(c, m) for t, c, m in cands if "dense" in c.name][0]
    assert abs(dense[1] - mpm / 3.0) < 1e-9, dense[1]
    assert report(dense[0], dense[1])["reach"] >= REACH_MIN - 1e-9
    print(f"  [ok] le candidat densité est jugé à {dense[1]:.1f} m/repas (et non {mpm:.1f}) "
          "— sinon on le recalerait sur la contrainte qu'il desserre")

    r = report(p, mpm)
    v, fails = verdict(r)
    assert v == "ÉCHEC" and len(fails) == 3, (v, fails)
    print(f"  [ok] le monde COURANT échoue le gate sur {len(fails)} points, comme attendu — "
          "sinon la sonde ne discriminerait rien")
    print("       (dont la joignabilité à 1,12x : le monde d'aujourd'hui est déjà SANS marge, "
          "il ne tient sa cible que parce qu'elle vaut 1,25 repas)")
    print("SELFCHECK PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--targets", type=float, nargs="+", default=[15.0])
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()

    base = BOSQUETS_V2
    mpm = metres_per_meal(base, MEASURED_MEALS)
    print("=== RÉFÉRENCE — le monde courant, tel qu'il est MESURÉ ===")
    print(f"  vitesse de calibrage {MEASURED_M_PER_TICK} m/tick (MESURÉE ; la formule au haut du grid "
          f"servi donne {base.kin_speed * VX_GRID_MAX / TICKS_PER_SEC:.4f} — écart 10 % non résolu)")
    print(f"  calibrage du terme fragile : {travel_budget(base):.0f} m de budget / {MEASURED_MEALS} "
          f"repas mesurés = {mpm:.1f} m par repas (recherche et détours COMPRIS)")
    show(report(base, mpm), "  [le monde d'aujourd'hui]")
    print(f"\n  cible : {EVENTS_MIN:.0f}-{EVENTS_MAX:.0f} événements/vie | plancher <= "
          f"{FLOOR_FRAC_MAX * 100:.0f} % de la vie | bouche >= {MOUTH_TICKS_MIN:.0f} ticks")

    for t in a.targets:
        print(f"\n{'=' * 92}\n=== CANDIDATS POUR {t:.0f} ÉVÉNEMENTS PAR VIE ===")
        for tag, cand, mpm_c in solve(base, mpm, t):
            d = dataclasses.asdict(cand)
            b = dataclasses.asdict(base)
            diff = ", ".join(f"{k} {b[k]}→{d[k]}" for k in d
                             if k != "name" and d[k] != b[k]) or "aucun changement"
            if mpm_c != mpm:
                diff += f", trajet par repas {mpm:.1f}→{mpm_c:.1f} m"
            print(f"\n  ── {tag}\n     {diff}")
            show(report(cand, mpm_c), f"   (x{cand.kin_speed / base.kin_speed:.1f} vitesse)")
    print(f"\n{'=' * 92}")
    print("RAPPEL DE PORTÉE : condition NÉCESSAIRE seulement. Que l'entité OBTIENNE ces repas se")
    print("mesure en vies, après collecte et retrain — tout changement de vitesse change la")
    print("dynamique que le WM doit avoir apprise (§6quinquies E).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
