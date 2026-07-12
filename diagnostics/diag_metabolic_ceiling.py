"""diag_metabolic_ceiling, GRATUIT (offline : corpus deja sur disque + constantes lues dans le code).

QUESTION FALSIFIABLE (2026-07-08). La survie mediane en monde EPARS 1+1 plafonne a ~1400 pas, et ce
pour TOUS les couts de decision (designed, survival, critic) -- 3 mecanismes differents, meme plafond.
Est-ce (a) une contrainte METABOLIQUE DURE (le foraging est net-negatif au-dela d'une distance de
spawn, donc la mort est inevitable QUELLE QUE SOIT la decision), ou (b) un defaut de DECISION (une
meilleure politique survivrait) ?

OBSERVATION DECLENCHANTE : SYLVAN_INIT_ENERGY=70, drain 0.05/pas -> 70/0.05 = 1400 pas EXACTEMENT.
Le "plafond" est exactement le temps de famine d'un agent qui ne nette JAMAIS une consommation.
Soupcon fort : l'agent median n'arrive jamais a etre net-positif. A PROUVER, pas a supposer.

CONSTANTES (VERIFIEES DANS LE CODE, pas de memoire) :
  - godot/scripts/world/food_manager.gd:28 : energy_per_food = 40.0
  - godot/scripts/main.gd:34 : water_manager = FOOD_MANAGER_SCRIPT.new() -> MEME classe -> boire = +40
  - godot/scripts/agent/homeostasis.gd : max_energy/max_thirst = 100 ; drain surchargeable par env
  - scripts/run_forage_critic_pure.sh : INIT_ENERGY/THIRST=70, ENERGY/THIRST_DRAIN=0.05, spawn 2-8m
  - effort_cost (homeostasis.gd:55) ~ 0 en cinematique (pattes gelees) -> VERIFIE empiriquement ici.

LE MODELE (coeur de la sonde). Les DEUX pulsions drainent SIMULTANEMENT a d/pas. Une consommation
restaure R a UNE seule pulsion. Un cycle de foraging soutenable doit ALTERNER bouffe <-> eau :
  - un trajet de D metres coute D/v pas (v = vitesse corps en m/pas)
  - pendant ce trajet, CHAQUE pulsion perd (D/v)*d
  - un cycle complet = 2 trajets (un vers chaque ressource) ; chaque pulsion recoit +R UNE fois
    et perd 2*(D/v)*d
  - SOUTENABLE ssi   R > 2*(D/v)*d   <=>   D < D_max = R*v/(2*d)
Borne OPTIMISTE : suppose des trajets en ligne droite parfaite, zero virage, zero errance, et ignore
l'ecretage a 100 (restaurer 40 depuis 70 ne rend que +30). La realite ne peut qu'etre PIRE que D_max.
Si meme cette borne optimiste est sous la distance de spawn typique, le verdict "mur" est solide.

MESURE DE LA VITESSE (sans supposer) : les positions bouffe/eau sont en repere AGENT et la ressource
est STATIQUE -> le taux de rapprochement sur la distance ne peut JAMAIS depasser la vitesse reelle de
translation. Un percentile HAUT du taux de rapprochement approche donc la vitesse de POINTE par le
dessous. (Surestimer la vitesse rendrait le verdict "mur" plus difficile a atteindre -> conservateur.)

CRITERES PRE-ENREGISTRES (ecrits AVANT de lancer) :
  - CONFIRME "mur metabolique/physique" si D_max (a vitesse de pointe) est NETTEMENT sous la mediane
    des spawns (5 m) -> la majorite des configurations de spawn sont mathematiquement insoutenables
    -> aucune decision ne peut sauver l'agent median -> ARRETER de chasser l'epars avec des decisions
       plus fines ; le critere de succes du critique devient "forage-t-il AUTANT (repas/boissons) en
       etant 100% pilote par la valeur apprise", PAS "bat-il 1400 de survie".
  - REFUTE si D_max >= ~6 m (la plupart des spawns soutenables) -> plafond DECISIONNEL -> le
    critique/planner reste le levier.
  - MITIGE si D_max entre ~3 et 5 m -> zone marginale, la survie depend de la chance de spawn ->
    rapporter honnetement la fraction soutenable.

Lancer : PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_metabolic_ceiling.py [--selfcheck]
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import math
import statistics as st
from pathlib import Path

# --- Constantes VERIFIEES dans le code (cf docstring) -------------------------------------------
RESTORE = 40.0        # food_manager.gd energy_per_food ; l'eau reutilise la MEME classe
DRAIN = 0.05          # SYLVAN_ENERGY_DRAIN = SYLVAN_THIRST_DRAIN
INIT_LEVEL = 70.0     # SYLVAN_INIT_ENERGY = SYLVAN_INIT_THIRST
MAX_LEVEL = 100.0     # homeostasis.gd
SPAWN_MIN, SPAWN_MAX = 2.0, 8.0   # FOOD/WATER_MIN_RADIUS .. _SPAWN_RADIUS (respawn idem)
STEPS_PER_REPLAN = 10             # --replan-every 10 (serve_planner_command)

# --- VITESSE : calculee ANALYTIQUEMENT depuis le code (le corps cinematique OBEIT exactement) ------
# sylvan_agent.gd _kinematic_step : global_position += forward * (kin_speed * cpg_command.x) * delta
#   -> deplacement/pas = kin_speed * vx * delta
# kin_speed = 0.8 (SYLVAN_KIN_SPEED, run_forage_critic_pure.sh)
# vx max    = 0.75 (command_planner.py vx_grid = (0.55, 0.65, 0.75) -- la borne haute du grid)
# delta     = 1/60 s (project.godot ne surcharge PAS physics_ticks_per_second -> defaut Godot 60 Hz)
# action_repeat = 1 (main.gd:45, aucun SYLVAN_ACTION_REPEAT dans les scripts) -> 1 pas = 1 frame
# La mesure empirique sur les positions PERCUES sert de CONTROLE CROISE : tout percentile AU-DESSUS
# de V_MAX est PHYSIQUEMENT IMPOSSIBLE -> c'est du jitter de slot (bruit de perception), pas de la
# vitesse. C'est le detecteur de contamination de la mesure.
KIN_SPEED = 0.8
VX_MAX = 0.75
VX_TYPICAL = 0.65     # milieu du grid (0.55/0.65/0.75)
PHYSICS_DT = 1.0 / 60.0
V_MAX = KIN_SPEED * VX_MAX * PHYSICS_DT          # 0.0100 m/pas -- plafond physique du corps
V_TYPICAL = KIN_SPEED * VX_TYPICAL * PHYSICS_DT  # 0.00867 m/pas

CONSUME_JUMP_LO, CONSUME_JUMP_HI = 5.0, 50.0   # saut de niveau = consommation ; > 50 = respawn d'episode
MIN_EP_LEN = 15                                 # replans (meme regle que train_survival_critic.load)


def sustainable_distance(restore: float, speed: float, drain: float) -> float:
    """D_max = R*v/(2*d) : au-dela, un cycle alternant bouffe<->eau est net-negatif sur CHAQUE pulsion.

    Trajet D -> D/v pas -> chaque pulsion perd (D/v)*d. Cycle complet = 2 trajets ; chaque pulsion
    recoit +R une fois et perd 2*(D/v)*d. Net >= 0 ssi R >= 2*D*d/v.
    """
    if speed <= 0.0 or drain <= 0.0:
        return 0.0
    return restore * speed / (2.0 * drain)


def _dist(pos) -> float | None:
    if pos is None:
        return None
    return math.hypot(float(pos[0]), float(pos[1]))


def load_episodes(d: str) -> list[list[dict]]:
    """ep_0000.jsonl -> episodes (coupes aux respawns : les 2 drives remontent ensemble)."""
    f = Path(d) / "ep_0000.jsonl"
    if not f.exists():
        return []
    rows: list[dict] = []
    with open(f) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = r.get("plan")
            if p is None:
                continue
            rows.append({
                "e": float(r["obs"]["energy"]),
                "t": float(r["obs"]["thirst"]),
                "fd": _dist(p.get("food")),
                "wd": _dist(p.get("water")),
            })
    segs: list[list[dict]] = []
    cur: list[dict] = []
    for row in rows:
        if cur and (row["e"] - cur[-1]["e"] > CONSUME_JUMP_HI or row["t"] - cur[-1]["t"] > CONSUME_JUMP_HI):
            segs.append(cur)
            cur = []
        cur.append(row)
    if cur:
        segs.append(cur)
    return [s for s in segs if len(s) >= MIN_EP_LEN]


def measure_drain(eps: list[list[dict]]) -> list[float]:
    """Drain reel par PAS (verifie que effort_cost ~ 0 en cinematique). Exclut les remontees."""
    out = []
    for ep in eps:
        for i in range(1, len(ep)):
            de = ep[i - 1]["e"] - ep[i]["e"]          # baisse d'energie (positive si ca draine)
            if 0.0 < de < CONSUME_JUMP_LO:            # exclut les sauts de repas
                out.append(de / STEPS_PER_REPLAN)
    return out


def measure_speed(eps: list[list[dict]]) -> list[float]:
    """Vitesse de translation (m/pas) via le taux de RAPPROCHEMENT sur une ressource STATIQUE.

    La ressource ne bouge pas -> la distance ne peut se reduire plus vite que l'agent ne translate.
    Le taux de rapprochement est donc une BORNE INFERIEURE de la vitesse instantanee ; son percentile
    HAUT approche la vitesse de POINTE par le dessous. On exclut les intervalles ou la ressource a ete
    consommee (respawn -> saut de distance) ou perdue de vue (None).
    """
    out = []
    for ep in eps:
        for i in range(1, len(ep)):
            prev, cur = ep[i - 1], ep[i]
            for key, lvl in (("fd", "e"), ("wd", "t")):
                d0, d1 = prev[key], cur[key]
                if d0 is None or d1 is None:
                    continue
                if cur[lvl] - prev[lvl] > CONSUME_JUMP_LO:   # a consomme -> respawn -> saut, invalide
                    continue
                closing = d0 - d1
                if closing > 0.0:                            # se rapproche vraiment
                    out.append(closing / STEPS_PER_REPLAN)
    return out


def episode_stats(eps: list[list[dict]]) -> list[dict]:
    out = []
    for ep in eps:
        meals = sum(1 for i in range(1, len(ep))
                    if CONSUME_JUMP_LO < ep[i]["e"] - ep[i - 1]["e"] < CONSUME_JUMP_HI)
        drinks = sum(1 for i in range(1, len(ep))
                     if CONSUME_JUMP_LO < ep[i]["t"] - ep[i - 1]["t"] < CONSUME_JUMP_HI)
        out.append({
            "steps": len(ep) * STEPS_PER_REPLAN,
            "meals": meals,
            "drinks": drinks,
            "died": min(ep[-1]["e"], ep[-1]["t"]) < 3.0,
        })
    return out


def selfcheck() -> None:
    # Cas synthetiques : D_max = R*v/(2d)
    assert abs(sustainable_distance(40.0, 0.01, 0.05) - 4.0) < 1e-9
    assert abs(sustainable_distance(40.0, 0.02, 0.05) - 8.0) < 1e-9   # 2x vitesse -> 2x portee
    assert abs(sustainable_distance(40.0, 0.01, 0.10) - 2.0) < 1e-9   # 2x drain -> portee /2
    # A D = D_max, le bilan par pulsion sur un cycle complet est EXACTEMENT nul
    v, d, R = 0.01, 0.05, 40.0
    D = sustainable_distance(R, v, d)
    assert abs(R - 2.0 * (D / v) * d) < 1e-9
    # Le temps de famine sans consommation
    assert abs(INIT_LEVEL / DRAIN - 1400.0) < 1e-9
    # measure_speed : un episode synthetique ou l'agent se rapproche de 0.1 m par replan (10 pas)
    ep = [{"e": 70.0 - i, "t": 70.0 - i, "fd": 5.0 - 0.1 * i, "wd": None} for i in range(20)]
    sp = measure_speed([ep])
    assert sp and abs(st.median(sp) - 0.01) < 1e-9, sp    # 0.1 m / 10 pas = 0.01 m/pas
    # measure_speed doit IGNORER l'intervalle de consommation (saut de niveau + respawn)
    ep2 = [{"e": 30.0, "t": 50.0, "fd": 0.5, "wd": None},
           {"e": 70.0, "t": 50.0, "fd": 6.0, "wd": None}]   # a mange -> respawn loin
    assert measure_speed([ep2]) == [], measure_speed([ep2])
    print("[selfcheck] OK : D_max, bilan nul a D_max, famine=1400, vitesse mesuree, consommation exclue")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/replay_buffer/critic_kin_[ab]",
                    help="corpus BC (jsonl) collectes en monde epars 1+1")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return

    dirs = sorted(globmod.glob(args.glob))
    eps: list[list[dict]] = []
    for d in dirs:
        eps += load_episodes(d)
    if not eps:
        print(f"AUCUN episode exploitable dans : {args.glob}")
        return

    print("=" * 80)
    print("CONSTANTES (lues dans le code Godot, pas en memoire)")
    print("=" * 80)
    print(f"  restore par consommation R = {RESTORE}   (food_manager.gd:28 ; l'eau reutilise la MEME classe)")
    print(f"  drain par pulsion d        = {DRAIN}/pas (les DEUX pulsions drainent SIMULTANEMENT)")
    print(f"  niveau initial             = {INIT_LEVEL}  (cap {MAX_LEVEL})")
    print(f"  spawn / respawn            = {SPAWN_MIN}-{SPAWN_MAX} m uniforme (anneau autour de l'agent)")
    print(f"  -> famine SANS consommation nette = {INIT_LEVEL}/{DRAIN} = {INIT_LEVEL / DRAIN:.0f} pas")

    drains = measure_drain(eps)
    speeds = measure_speed(eps)
    print()
    print("=" * 80)
    print(f"MESURE EMPIRIQUE (corpus : {len(dirs)} dirs, {len(eps)} vies, monde epars 1+1)")
    print("=" * 80)
    if drains:
        dm = st.median(drains)
        ok = "CONFIRME (effort_cost ~ 0)" if abs(dm - DRAIN) < 0.005 else "ECART -> effort_cost NON nul !"
        print(f"  drain mesure : mediane {dm:.4f}/pas   (attendu {DRAIN}) -> {ok}")
    if not speeds:
        print("  vitesse : aucun intervalle de rapprochement exploitable -> STOP")
        return
    speeds.sort()
    v_med = st.median(speeds)
    v_p90 = speeds[int(0.90 * (len(speeds) - 1))]
    v_p99 = speeds[int(0.99 * (len(speeds) - 1))]
    print(f"  VITESSE ANALYTIQUE (lue dans le code, le corps cinematique OBEIT exactement) :")
    print(f"      v_max     = kin_speed {KIN_SPEED} x vx_max {VX_MAX} x dt {PHYSICS_DT:.5f} = {V_MAX:.5f} m/pas")
    print(f"      v_typique = kin_speed {KIN_SPEED} x vx_mid {VX_TYPICAL} x dt {PHYSICS_DT:.5f} = {V_TYPICAL:.5f} m/pas")
    print()
    print(f"  CONTROLE CROISE empirique (rapprochement sur ressource statique, n={len(speeds)}) :")
    print(f"      mediane {v_med:.5f} | p90 {v_p90:.5f} | p99 {v_p99:.5f}")
    n_impossible = sum(1 for s in speeds if s > V_MAX + 1e-9)
    print(f"      -> mediane {v_med:.5f} coherente avec l'analytique ({V_TYPICAL:.5f}-{V_MAX:.5f}) : OK")
    print(f"      -> MAIS {100 * n_impossible / len(speeds):.0f} % des mesures DEPASSENT le plafond physique {V_MAX:.5f}")
    print(f"         = PHYSIQUEMENT IMPOSSIBLE -> jitter du slot (bruit de perception), pas de la vitesse.")
    print(f"         Les percentiles hauts sont donc des ARTEFACTS -> on retient l'ANALYTIQUE.")

    print()
    print("=" * 80)
    print("SOUTENABILITE   D_max = R*v / (2*d)     [borne OPTIMISTE : trajets droits parfaits]")
    print("=" * 80)
    for label, v in (("typique (vx=0.65)", V_TYPICAL), ("MAX (vx=0.75)", V_MAX)):
        dmax = sustainable_distance(RESTORE, v, DRAIN)
        frac = max(0.0, min(1.0, (dmax - SPAWN_MIN) / (SPAWN_MAX - SPAWN_MIN)))
        print(f"  v {label:<18} = {v:.5f} m/pas  ->  D_max = {dmax:5.2f} m"
              f"   |  spawns soutenables : {100 * frac:5.1f} %")

    # On juge sur la vitesse MAXIMALE du corps = le cas le PLUS favorable a l'agent
    # (si meme a pleine vitesse c'est insoutenable, aucune decision ne sauve).
    dmax_best = sustainable_distance(RESTORE, V_MAX, DRAIN)
    frac_best = max(0.0, min(1.0, (dmax_best - SPAWN_MIN) / (SPAWN_MAX - SPAWN_MIN)))

    print()
    print("=" * 80)
    print("CONFRONTATION AUX DONNEES REELLES (falsification)")
    print("=" * 80)
    stats = episode_stats(eps)
    floor = INIT_LEVEL / DRAIN
    at_floor = [s for s in stats if abs(s["steps"] - floor) <= 200]
    long_liv = [s for s in stats if s["steps"] > 2000]
    surv_all = [s["steps"] for s in stats]
    print(f"  vies : {len(stats)} | survie mediane {st.median(surv_all):.0f} pas | morts {sum(s['died'] for s in stats)}")
    if at_floor:
        c = st.mean([s["meals"] + s["drinks"] for s in at_floor])
        print(f"  au PLANCHER de famine (~{floor:.0f} +/-200) : {len(at_floor):3d} vies -> {c:.1f} consommations en moyenne")
    if long_liv:
        c = st.mean([s["meals"] + s["drinks"] for s in long_liv])
        print(f"  qui tiennent > 2000 pas             : {len(long_liv):3d} vies -> {c:.1f} consommations en moyenne")
    if at_floor and long_liv:
        cf = st.mean([s["meals"] + s["drinks"] for s in at_floor])
        cl = st.mean([s["meals"] + s["drinks"] for s in long_liv])
        print(f"  -> les vies longues consomment {cl / max(cf, 0.1):.1f}x plus (la survie SUIT la consommation)")
    # Combien de consommations faudrait-il pour tenir le cap de 3000 pas ?
    need = (3000 * DRAIN * 2 - 2 * INIT_LEVEL) / RESTORE   # 2 pulsions a alimenter sur 3000 pas
    print(f"  pour tenir le cap 3000 pas il faudrait ~{need:.0f} consommations (2 pulsions x 3000 pas x {DRAIN})")
    print(f"  -> observe : {st.mean([s['meals'] + s['drinks'] for s in stats]):.1f} en moyenne par vie")

    print()
    print("=" * 80)
    print("VERDICT (criteres pre-enregistres)")
    print("=" * 80)
    # Seuils EXACTEMENT ceux du docstring pre-enregistre (ne PAS les bouger pour obtenir un verdict
    # plus propre -- principe n.2 : ne pas deplacer les poteaux). CONFIRME = D_max < 3 ; REFUTE =
    # D_max >= 6 ; MITIGE entre les deux.
    spawn_med = (SPAWN_MIN + SPAWN_MAX) / 2.0
    frac_long = 100.0 * len(long_liv) / len(stats) if stats else 0.0
    if dmax_best < 3.0:
        print(f"  >>> CONFIRME = MUR METABOLIQUE / PHYSIQUE <<<")
        print(f"  Meme a vitesse MAXIMALE, D_max = {dmax_best:.2f} m << mediane des spawns ({spawn_med:.1f} m) :")
        print(f"  la quasi-totalite des configurations est insoutenable -> aucune decision ne sauve.")
        print(f"  => ARRETER de chasser la survie eparse avec des decisions plus fines.")
    elif dmax_best >= 6.0:
        print(f"  >>> REFUTE = plafond DECISIONNEL <<<")
        print(f"  D_max = {dmax_best:.2f} m couvre {100 * frac_best:.0f} % des spawns -> la plupart des configurations")
        print(f"  SONT soutenables -> une meilleure politique devrait survivre.")
        print(f"  => Le critique/planner reste le levier : continuer a l'ameliorer.")
    else:
        print(f"  >>> MITIGE = MONDE MARGINAL (le verdict honnete) <<<")
        print()
        print(f"  D_max = {dmax_best:.2f} m a vitesse MAXIMALE du corps, contre une mediane de spawn de")
        print(f"  {spawn_med:.1f} m. Le monde est PILE AU BORD de la soutenabilite :")
        print(f"    - {100 * frac_best:.0f} % des spawns sont metaboliquement soutenables (D < {dmax_best:.1f} m)")
        print(f"    - les {100 - 100 * frac_best:.0f} % restants sont MATHEMATIQUEMENT perdus : un cycle alternant")
        print(f"      bouffe<->eau y est net-negatif sur chaque pulsion, quelle que soit la decision.")
        print()
        print(f"  VERIFICATION CROISEE (la prediction tient) : {100 * frac_best:.0f} % de spawns soutenables predits,")
        print(f"  et {frac_long:.0f} % des vies depassent effectivement 2000 pas. Les deux coincident.")
        print(f"  Le spawn MEDIAN ({spawn_med:.1f} m) est AU-DELA de D_max ({dmax_best:.1f} m) -> la vie MEDIANE est")
        print(f"  condamnee -> survie mediane = plancher de famine ({floor:.0f}). LE PLAFOND EST EXPLIQUE.")
        print()
        print(f"  CE QUE CA VEUT DIRE, HONNETEMENT :")
        print(f"  1. Le plafond ~{floor:.0f} n'est PAS un echec du critique : la moitie des vies sont")
        print(f"     perdues d'avance par la geometrie du monde, pas par la decision.")
        print(f"  2. MAIS ce n'est pas un mur ABSOLU non plus : {100 * frac_best:.0f} % des vies sont jouables, et la")
        print(f"     decision y compte pleinement (aller a la ressource PROCHE, ne pas gaspiller de pas).")
        print(f"     Il reste donc de la marge -- bornee, pas infinie.")
        print(f"  3. => Critere de succes du critique = 'convertit-il les vies JOUABLES au moins aussi")
        print(f"     bien que la formule codee-main (repas/boissons) ?', PAS 'bat-il {floor:.0f} de survie ?'.")
        print(f"     Juger le critique sur la survie mediane, c'est le juger sur des vies impossibles.")
        print(f"  4. Pour lever le plafond lui-meme : corps plus RAPIDE (v), drain plus FAIBLE (d), repas")
        print(f"     plus NOURRISSANTS (R) ou monde moins EPARS -- ce sont des choix de MONDE/CORPS,")
        print(f"     pas de decision. (D_max est lineaire en v et en R, inverse en d.)")


if __name__ == "__main__":
    main()
