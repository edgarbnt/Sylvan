"""G11 — LE TRAJET PAR ÉVÉNEMENT SOUS UNE POLITIQUE : ce que le babillage ne pouvait pas dire.

PÉRIMÈTRE. Analyseur PUR : lit DEUX corpus déjà collectés (le harnais scripts/probe_foret_portee.sh
les produit) et rend un verdict. Aucun entraînement. Le seul coût est celui de la collecte du
harnais — bien moins qu'une vraie collecte, et infiniment plus qu'un babillage puisqu'ici une
POLITIQUE forage (le planner du WM actuel), donc des repas ARRIVENT.

POURQUOI (dry-run 2026-07-25). La calibration métabolique de foret_v1 tablait sur 5,4 m de trajet
par événement, dérivés de la GÉOMÉTRIE. Le dry-run a montré deux choses : le terrain fait chuter le
budget réel (facteur effectif médian 0,455, pas la moyenne d'arène 0,842), ET le babillage ne mange
JAMAIS — donc le terme décisif, le trajet RÉEL par repas, ne pouvait pas être mesuré. Une politique
qui forage le peut. C'est ce que fait cette sonde.

🚨 TROIS HONNÊTETÉS, chacune inscrite dans la mesure (retours du pair, 2026-07-25) :

  1. C'EST UNE BORNE HAUTE, PAS LE CHIFFRE FINAL. Le WM actuel a été entraîné sur l'ANCIEN monde
     (zéro forêt, zéro terrain, zéro occlusion, corps à kin_speed 0,8). Dans la forêt il est
     hors-distribution : l'occlusion vide le slot, l'entité erre, le trajet par repas GONFLE. Le WM
     ré-entraîné fera mieux — donc le trajet mesuré ici MAJORE le vrai, il ne le donne pas.

  2. L'ANCRE SÉPARE « LE WM EST OOD » DE « LE MONDE EST PLUS LONG ». On mesure le MÊME WM, la MÊME
     nourriture, la MÊME politique dans DEUX mondes : avec la forêt+terrain (FORÊT) et sans
     (ANCRE). La nourriture identique des deux côtés fait que l'OOD commun (types, proies mobiles,
     que le WM ne connaît pas non plus) se SIMPLIFIE dans le RATIO forêt/ancre. Ce ratio isole la
     contribution de la forêt+terrain ; l'absolu de l'ancre calibre la compétence du WM chez lui.

  3. LE CORPS EST CELUI DU WM, PAS L'ÉVENTAIL. On sert kin_speed 0,8 et la grille de vitesse
     d'origine (0,55-0,75), PAS l'éventail (kin 2,83, vx 0,25-1,0) : sinon on empilerait un
     SECOND OOD (le WM ne connaît pas non plus ce corps) et on ne saurait plus quelle dégradation
     vient d'où. Le trajet par repas est une grandeur GÉOMÉTRIQUE (mètres), au premier ordre
     indépendante de la vitesse du corps ; le facteur terrain l'est aussi (G4). Les mètres
     transfèrent ; la conversion mètres→énergie se fera in vivo, après le retrain, sur la vitesse
     de croisière que la politique CHOISIRA (inconnue avant).

CE QU'ON MESURE, PAR CONDITION (depuis le corpus, jamais un oracle) :
  * repas par vie          — meal_flags (remontée d'énergie, frontières d'épisode exclues) ;
  * longueur de trajet     — somme des pas du torse < seuil de téléport (exclut les respawns) ;
  * TRAJET PAR REPAS (m)   — le terme qui manquait à G2 ;
  * facteur terrain vécu   — médiane de vitesse_réalisée / (kin_speed·vx) sur les ticks mobiles.
    C'est la réponse à la nuance 1 du pair : une politique CONTOURNE les massifs là où le babillage
    les TRAVERSE, donc son facteur vécu est > 0,455. On l'oppose au 0,455 du babillage.

CRITÈRES PRÉ-ENREGISTRÉS (ce sont des critères de VALIDITÉ de la mesure, pas de succès du monde —
la barre de succès appartient au retrain, §6ter ; ici on vérifie qu'on a un chiffre fiable) :
  V1 LA POLITIQUE MANGE ..... repas/vie >= 2 DANS LES DEUX mondes. En dessous, le trajet par repas
                              est trop bruité pour rien en conclure (le babillage était à 0).
  V2 L'ANCRE EST SAINE ...... dans l'ANCRE (monde du WM), trajet par repas <= 15 m. Au-delà, le WM
                              forage déjà mal chez lui → sa mesure forêt ne dirait rien du monde.
  V3 LE FACTEUR VÉCU ........ facteur terrain de la politique en forêt RAPPORTÉ, comparé au 0,455
                              du babillage : la nuance 1 rendue chiffrée, pas affirmée.
Et un CHIFFRE RENDU, sans seuil (c'est une borne, pas un verdict) : le trajet par repas en forêt,
son ratio à l'ancre, et la re-lecture du budget/densité qui en découle.

CLI :
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_foret_g11_portee.py \
        <corpus_foret> <corpus_ancre>
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_foret_g11_portee.py --selfcheck
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))

from sylvan.critic_corpus import TELEPORT_M, load_bc_corpus, meal_flags  # noqa: E402
from sylvan.world import FORET_V1  # noqa: E402

KIN_SPEED_PROBE = 0.8            # le corps du WM (nuance 3), pas l'éventail
BABBLING_MEAN = 0.842            # facteur terrain MOYEN en babillage (dry-run) — l'agrégat de budget
BABBLING_MEDIAN = 0.455          # facteur terrain MÉDIAN en babillage — où le babillage passait son temps

MEALS_MIN = 2.0                  # V1 : sous ce nombre le trajet par repas n'est pas fiable
ANCHOR_MAX_TRAVEL = 15.0         # V2 : au-delà le WM forage déjà mal chez lui


def _torso_xz(run: str) -> torch.Tensor:
    """Positions (x, z) du torse, tick par tick, depuis le corpus. Lues via le loader canonique
    pour ne pas re-coder la lecture des deux dispositions de fichiers (ep_ vs episode_)."""
    import glob
    import gzip
    import json
    fichiers = [f for f in (os.path.join(run, "ep_0000.jsonl"), os.path.join(run, "ep_0000.jsonl.gz"))
                if os.path.exists(f)] or sorted(glob.glob(os.path.join(run, "episode_*.jsonl*")))
    xz = []
    for f in fichiers:
        op = gzip.open(f, "rt") if f.endswith(".gz") else open(f)
        with op as fh:
            for line in fh:
                if line.strip():
                    t = json.loads(line)["wm"]["torso0"]
                    xz.append([t[0], t[1]])
    return torch.tensor(xz, dtype=torch.float32)


def _path_length(xz: torch.Tensor) -> float:
    """Longueur de trajet RÉELLE = somme des pas, hors respawns.

    Un respawn téléporte le torse (> TELEPORT_M) ; l'inclure ajouterait la distance arène entière à
    chaque frontière et gonflerait le trajet de plusieurs mètres par vie. On additionne donc les pas
    strictement sous ce seuil — la même convention que les frontières d'épisode du loader.
    """
    if len(xz) < 2:
        return 0.0
    step = (xz[1:] - xz[:-1]).norm(dim=1)
    return float(step[step < TELEPORT_M].sum())


def _terrain_factor(run: str) -> tuple[float, float]:
    """Facteur terrain VÉCU sous la politique = vitesse_réalisée / (kin_speed·vx), (moyenne, médiane).

    Non circulaire (G4) : la vitesse réalisée est lue dans la proprioception (dims 1,3, sans lag),
    la commande vx dans le corpus ; leur rapport ne dépend que de la POSITION si le corps obéit.
    Restreint aux ticks mobiles (les fenêtres de settle fausseraient l'agrégat).

    🚨 LES DEUX STATISTIQUES DISENT DES CHOSES DIFFÉRENTES, et confondre les deux a failli me tromper.
    * La MOYENNE est l'agrégat du BUDGET : sol parcouru / ticks = kin·vx·moyenne(facteur). C'est elle
      qu'on met dans la re-lecture de portée. Vérifiée = 0,904 côté corpus, 0,907 côté log [terrain].
    * La MÉDIANE dit OÙ la politique passe son temps. Elle vaut 1,0 (clairières) sous une politique et
      0,455 en babillage : le babillage TRAVERSE les massifs, la politique les CONTOURNE (nuance 1 du
      pair, rendue chiffrée). Prise pour le budget, elle SUR-estimerait le sol parcouru de ~10 %.
    """
    obs, _e, cmd, _b = load_bc_corpus(run)
    speed = (obs[:, 1] ** 2 + obs[:, 3] ** 2).sqrt()
    vx = cmd[:, 0]
    mob = (speed > 1e-3) & (vx > 1e-3)
    if int(mob.sum()) < 20:
        return float("nan"), float("nan")
    fac = speed[mob] / (KIN_SPEED_PROBE * vx[mob])
    return float(fac.mean()), float(fac.median())


def measure(run: str) -> dict:
    """Toutes les grandeurs d'une condition, depuis son corpus."""
    obs, e, _cmd, bounds = load_bc_corpus(run)
    n_life = max(1, len(bounds) - 1)
    meals = int(meal_flags(e, bounds).sum())
    travel = _path_length(_torso_xz(run))
    tf_mean, tf_median = _terrain_factor(run)
    return {
        "run": run,
        "ticks": len(e),
        "lives": n_life,
        "meals": meals,
        "meals_per_life": meals / n_life,
        "travel_m": travel,
        "travel_per_meal": (travel / meals) if meals > 0 else float("inf"),
        "terrain_mean": tf_mean,       # agrégat du BUDGET
        "terrain_median": tf_median,   # où la politique passe son temps
    }


def _show(tag: str, m: dict) -> None:
    print(f"  {tag:<8} {m['lives']} vies, {m['ticks']} ticks | repas {m['meals']} "
          f"({m['meals_per_life']:.1f}/vie) | trajet {m['travel_m']:.1f} m | "
          f"TRAJET/REPAS {m['travel_per_meal']:.2f} m | facteur terrain moy {m['terrain_mean']:.3f} "
          f"(méd {m['terrain_median']:.3f})")


def render(forest: dict, anchor: dict) -> int:
    print("=" * 96)
    print("condition  vies/ticks           repas          trajet      TRAJET/REPAS   facteur terrain")
    _show("FORÊT", forest)
    _show("ANCRE", anchor)
    print("=" * 96)

    ok = True

    # V1 — la politique mange-t-elle assez pour que le trajet par repas veuille dire quelque chose ?
    v1 = forest["meals_per_life"] >= MEALS_MIN and anchor["meals_per_life"] >= MEALS_MIN
    ok &= v1
    print(f"{'✅' if v1 else '❌'} V1 LA POLITIQUE MANGE   forêt {forest['meals_per_life']:.1f}/vie, "
          f"ancre {anchor['meals_per_life']:.1f}/vie (exigé >= {MEALS_MIN:.0f} des deux côtés)")

    # V2 — l'ancre calibre-t-elle une compétence saine (sinon la mesure forêt ne dit rien) ?
    v2 = anchor["travel_per_meal"] <= ANCHOR_MAX_TRAVEL
    ok &= v2
    print(f"{'✅' if v2 else '❌'} V2 L'ANCRE EST SAINE    trajet/repas chez le WM = "
          f"{anchor['travel_per_meal']:.2f} m (exigé <= {ANCHOR_MAX_TRAVEL:.0f} m)")

    # V3 — le facteur terrain vécu, opposé au babillage (nuance 1 chiffrée). On juge sur la MOYENNE
    # (l'agrégat de budget) ; la médiane est reportée comme « où la politique passe son temps ».
    tf_mean, tf_med = forest["terrain_mean"], forest["terrain_median"]
    v3 = tf_mean == tf_mean  # non-NaN
    ok &= v3
    plus = "PLUS DOUX" if v3 and tf_mean > BABBLING_MEAN else "aussi dur"
    print(f"{'✅' if v3 else '❌'} V3 FACTEUR VÉCU         moyenne {tf_mean:.3f} vs babillage "
          f"{BABBLING_MEAN} ({plus}) | médiane {tf_med:.3f} vs babillage {BABBLING_MEDIAN} "
          "(la politique CONTOURNE, le babillage TRAVERSE)")

    print("=" * 96)
    if forest["meals"] > 0 and anchor["meals"] > 0:
        ratio = forest["travel_per_meal"] / anchor["travel_per_meal"]
        print("CHIFFRE RENDU (une BORNE HAUTE, le WM re-entraîné fera mieux — nuance 1) :")
        print(f"  trajet/repas en forêt       = {forest['travel_per_meal']:.2f} m")
        print(f"  trajet/repas dans l'ancre   = {anchor['travel_per_meal']:.2f} m "
              "(la compétence du WM chez lui)")
        print(f"  ratio forêt/ancre           = {ratio:.2f}x  (la part forêt+terrain, OOD commun "
              "simplifié — nuance 2)")
        # re-lecture budget/densité, au facteur terrain MOYEN vécu (pas la médiane, pas la moyenne
        # d'arène du babillage) et au CORPS DE PRODUCTION (kin 2,83 + éventail), puisque le trajet en
        # mètres transfère (nuance 3).
        tf_use = tf_mean if tf_mean == tf_mean else BABBLING_MEAN
        budget_trot = (FORET_V1.kin_speed * 0.60 * tf_use / 60.0) * FORET_V1.episode_steps
        print(f"  budget de trajet au trot    = {budget_trot:.1f} m/vie (kin {FORET_V1.kin_speed} x "
              f"vx 0,6 x facteur moyen {tf_use:.2f} x {FORET_V1.episode_steps} ticks)")
        if forest["travel_per_meal"] > 0:
            ev = budget_trot / forest["travel_per_meal"]
            print(f"  ⇒ événements/vie atteignables (borne BASSE, WM OOD) = {ev:.1f} "
                  f"(cible 10-30 ; le retrain remonte ce chiffre)")
    print("\nRAPPEL DE PORTÉE : V1-V3 valident que la MESURE est fiable, pas que le monde est bon. La")
    print("barre de succès appartient au retrain (le WM a besoin de COUVERTURE + contacts, pas de la")
    print("fréquence exacte des repas — nuance 3). Le calage fin du drain se fait in vivo, après.")
    return 0 if ok else 1


def selfcheck() -> int:
    # _path_length : additionne les pas, IGNORE un respawn (saut > TELEPORT_M)
    xz = torch.tensor([[0.0, 0.0], [0.0, 0.3], [0.0, 0.6], [5.0, 5.0], [5.0, 5.3]])
    got = _path_length(xz)
    assert abs(got - 0.9) < 1e-5, got            # 0.3+0.3 + (saut exclu) + 0.3
    print(f"  [ok] longueur de trajet {got:.2f} m — le respawn (saut 7 m) est exclu, pas additionné")

    # travel_per_meal et meals_per_life sur des grandeurs connues
    m = {"lives": 2, "meals": 8, "travel_m": 40.0}
    tpm = m["travel_m"] / m["meals"]
    assert abs(tpm - 5.0) < 1e-9 and abs(m["meals"] / m["lives"] - 4.0) < 1e-9
    print("  [ok] 40 m / 8 repas = 5,0 m par repas ; 8 repas / 2 vies = 4,0 repas/vie")

    # le ratio forêt/ancre simplifie un facteur OOD commun : si les deux trajets doublent, le ratio
    # est INCHANGÉ — c'est exactement ce qui isole la forêt de l'OOD partagé (nuance 2)
    forest = {"travel_per_meal": 12.0, "meals": 5}
    anchor = {"travel_per_meal": 6.0, "meals": 5}
    r1 = forest["travel_per_meal"] / anchor["travel_per_meal"]
    r2 = (forest["travel_per_meal"] * 2) / (anchor["travel_per_meal"] * 2)
    assert abs(r1 - r2) < 1e-9 and abs(r1 - 2.0) < 1e-9
    print(f"  [ok] ratio forêt/ancre = {r1:.1f}x, invariant si un OOD commun double les deux (nuance 2)")

    # V1 refuse un monde où la politique ne mange pas (le piège du babillage à 0 repas)
    famine = {"meals_per_life": 0.5, "travel_per_meal": float("inf")}
    assert not (famine["meals_per_life"] >= MEALS_MIN)
    print("  [ok] V1 refuse 0,5 repas/vie — le trajet par repas d'un quasi-jeûne n'est pas fiable")

    print("SELFCHECK PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("forest", nargs="?", help="corpus de la condition FORÊT (foret_v1 + forêt/terrain)")
    ap.add_argument("anchor", nargs="?", help="corpus de la condition ANCRE (même nourriture, sans forêt)")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    if not a.forest or not a.anchor:
        raise SystemExit("usage : diag_foret_g11_portee.py <corpus_foret> <corpus_ancre>")
    print(f"FORÊT = {a.forest}\nANCRE = {a.anchor}\n"
          f"(même WM, même nourriture, même politique ; corps du WM kin_speed={KIN_SPEED_PROBE})\n")
    return render(measure(a.forest), measure(a.anchor))


if __name__ == "__main__":
    raise SystemExit(main())
