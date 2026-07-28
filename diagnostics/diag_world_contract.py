"""VÉRIFICATEUR DE CONTRAT DE MONDE — ce qui a été DEMANDÉ vs ce qui a été SERVI.

POURQUOI (docs/design_outil_matrice_information.md §4). Le projet a déjà perdu du temps TROIS fois
sur un réglage qui semblait appliqué sans l'être. Un réglage silencieusement inactif ne produit pas
d'erreur : il produit un RÉSULTAT, et on l'interprète comme une propriété du monde. C'est le pire
mode de panne du projet, et il est entièrement automatisable — d'où cet outil.

CE QU'IL FAIT. Pour chaque clause du contrat : à gauche la valeur DEMANDÉE (variable d'environnement
si elle est posée, sinon le DÉFAUT DU CODE, lu dans le code et cité), à droite la valeur MESURÉE sur
le corpus servi, puis un verdict. Sortie != 0 s'il y a une divergence — utilisable en garde-fou avant
de lancer une collecte de plusieurs heures.

PRINCIPE (celui de diagnostics/guards.py, étendu du corps au MONDE) : on ne croit jamais une constante
déclarée, on la mesure. Les défauts cités ici viennent du code (`homeostasis.gd`, `food_manager.gd`,
`sylvan_agent.gd`) — les inventer ferait de cet outil un menteur de plus.

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_world_contract.py \
      data/replay_buffer/critic_bosq_ripe11 [--set SYLVAN_ENERGY_DRAIN=0.05 ...] [--preset-file f.env]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Callable

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # guards.py (idiome du repo)

from guards import measured_constants, scaffold_banner                  # noqa: E402
from sylvan.critic_corpus import RETINA_DIM, load_bc_corpus, meal_flags  # noqa: E402
from sylvan.info_matrix import (                                        # noqa: E402
    bush_brightness, food_mask, nearest_food, pick_palette, rays,
)

TICKS_PER_S = 60.0            # cadence physique de Godot headless (temps réel)
RETINA_RANGE_M = 10.0         # perception.gd : MAX_RANGE — depth du rayon = d / MAX_RANGE


@dataclass
class Served:
    """Le monde tel qu'il a été SERVI, mesuré — jamais déclaré."""
    const: dict                    # guards.measured_constants (implémentation canonique, non dupliquée)
    retina: torch.Tensor           # [N, 144]
    energy: torch.Tensor           # [N]
    health: torch.Tensor           # [N]
    bounds: list[int]
    ate: torch.Tensor              # [N] 1 au tick d'un repas
    cmd: torch.Tensor              # [N, 2] commandes (vx, ω) RÉELLEMENT exécutées
    proprio_dim: int               # 132, ou 133 quand le regard était actif
    moving: torch.Tensor           # [N] booléen : le corps AVANCE vraiment à ce tick
    thirst: torch.Tensor           # [N] — la 2e jauge, mesurée comme la 1re et non par une médiane


@dataclass(frozen=True)
class Clause:
    """Une clause du contrat : un réglage demandé, une mesure qui l'atteste."""
    env: str
    label: str
    default: float                 # DÉFAUT DU CODE quand la variable n'est pas posée
    default_src: str               # où ce défaut est écrit (pour pouvoir le re-vérifier)
    measure: Callable[[Served], float]
    mode: str                      # "approx" | "plafond" | "presence" | "compte"
    unit: str = ""
    tol: float = 0.15
    why: str = ""


# --------------------------------------------------------------------------------------------- #
# Mesures — chacune lit le corpus, aucune ne lit un réglage.
# --------------------------------------------------------------------------------------------- #

def _fit_drain(s: Served, series: torch.Tensor | None = None) -> tuple[float, float]:
    """Sépare le drain PASSIF du coût de LOCOMOTION par régression : conso/tick = D + k·vx².

    🚨 POURQUOI ON NE PEUT PLUS PRENDRE LA MÉDIANE. Depuis l'éventail de vitesse (§2.13), la baisse
    d'énergie par tick vaut D + k·vx² : la médiane des baisses mesure donc la SOMME. Opposée au
    drain DEMANDÉ (le passif seul), elle criait « servi sans être demandé » sur un monde parfaitement
    conforme — un faux positif qui aurait discrédité l'outil au moment précis où on lui demande de
    valider une collecte. La régression rend les DEUX termes, chacun comparable à son propre réglage.

    Renvoie (D, k), ou (médiane, nan) quand le corpus ne fait pas varier vx assez pour identifier k —
    on préfère dire « non mesurable » (verdict ⚠️) que rendre une pente ajustée sur du bruit.
    """
    src = s.energy if series is None else series
    d = (src[:-1] - src[1:])
    vx2 = s.cmd[:len(d), 0] ** 2
    garde = torch.ones(len(d), dtype=torch.bool)
    for b in s.bounds[1:-1]:                       # frontières d'épisode : la jauge y est remise à plein
        if 0 < b <= len(d):
            garde[b - 1] = False
    # 🚨 ET LES TICKS IMMOBILES, qui ne sont PAS un détail. Après un respawn le corps a une fenêtre de
    # stabilisation : la commande est déjà tirée et écrite dans le corpus, mais le corps ne bouge pas
    # encore, donc le drain passif est prélevé SANS coût de locomotion. Les garder mélange deux
    # régimes : mesuré ici, k tombait à 0,387 pour 0,556 servi (R² 0,48) — une conclusion « le réglage
    # n'a pas pris » sur un monde parfaitement conforme. Filtrés, D=0,1000 et k=0,5556 à R²=1,0000.
    garde &= s.moving[:len(d)]
    m = garde & (d > 0) & (d < 1.0)                # baisses seules : un repas est une HAUSSE
    if int(m.sum()) < 30:
        return float("nan"), float("nan")
    y, x = d[m].double(), vx2[m].double()
    if float(x.var()) < 1e-4:                      # vx quasi constant → k et D ne sont pas séparables
        return float(y.median()), float("nan")
    a = torch.stack([x, torch.ones_like(x)], dim=1)
    coef = torch.linalg.lstsq(a, y.unsqueeze(1)).solution.flatten()
    return float(coef[1]), float(coef[0])          # (intercept = D, pente = k)


def _m_drain_e(s: Served) -> float:
    return _fit_drain(s)[0]


def _m_speed_cost(s: Served) -> float:
    """k SUR LA SOMME DES DEUX JAUGES, parce que c'est ainsi que la constante est définie.

    🚨 CORRIGÉ LE 2026-07-28, et l'erreur était la mienne. Depuis que le coût de locomotion est
    réparti au prorata sur les deux jauges (correctif d'asymétrie de cette même session), l'énergie
    n'en paie plus que la MOITIÉ. Mesurer la pente sur l'énergie seule et l'opposer à speed_cost
    donnait donc un écart de 50 % — exactement, ce qui aurait dû me mettre la puce à l'oreille — sur
    un monde parfaitement conforme. Vérifié sur le corpus : énergie et soif rendent la MÊME pente
    (0,2003 chacune) et la somme rend 0,4006 pour 0,4444 demandé. La constante décrit un TOTAL ;
    on mesure donc le total. Changer la physique sans changer l'instrument qui la mesure, c'est
    fabriquer un faux positif au moment précis où l'outil doit valider une collecte.
    """
    if s.thirst.numel() != s.energy.numel():
        return _fit_drain(s)[1]                       # monde mono-pulsion : l'énergie EST le total
    return _fit_drain(s, s.energy + s.thirst)[1]


def _m_gaze(s: Served) -> float:
    """Le regard est SERVI ssi la proprioception porte sa dimension supplémentaire.

    C'est la preuve la plus dure disponible sur un corpus : l'angle de tête n'existe dans les données
    que si le drapeau était actif à la collecte. Un corpus à 132 avec SYLVAN_GAZE=1 demandé signifie
    que le drapeau n'a pas pris — et tout ce qui en dépend serait mesuré sur un monde sans regard.
    """
    return 1.0 if s.proprio_dim == 133 else 0.0


def _m_drain_t(s: Served) -> float:
    """Drain PASSIF de la soif, par la même régression que l'énergie — et pas par une médiane.

    La médiane des baisses mesure D + k·vx², donc elle gonfle avec la vitesse : elle rendait 0,0939
    pour 0,08 demandé (+17 %) pendant que l'énergie, régressée, tombait pile. Deux jauges
    rigoureusement symétriques dans le monde paraissaient donc asymétriques dans le rapport, ce qui
    est le pire des mensonges pour un outil dont le rôle est de garantir la symétrie de l'arbitrage.
    """
    if s.thirst.numel() != s.energy.numel():
        return float("nan")                           # pas de 2e jauge dans ce corpus
    return _fit_drain(s, s.thirst)[0]


def _m_restore_e(s: Served) -> float:
    """Restore ABSORBÉ, pas nominal : le plafond à 100 écrête (garde déjà documentée dans guards)."""
    return s.const["restore_e_absorbed"]


def _m_speed(s: Served) -> float:
    """Vitesse observée en m/s. `const['speed']` est la médiane par tick QUAND l'entité avance."""
    return s.const["speed"] * TICKS_PER_S


def _m_eat_distance(s: Served) -> float:
    """Distance à la proie AU TICK OÙ ELLE EST MANGÉE — la mesure directe du rayon de capture.

    Lue sur la rétine du tick PRÉCÉDENT : au tick du repas la baie a disparu de la scène.
    """
    idx = torch.nonzero(s.ate).flatten()
    idx = idx[idx > 0] - 1
    if not len(idx):
        return float("nan")
    depth, rgb = rays(s.retina[idx])
    is_food = food_mask(rgb)
    d = torch.where(is_food, depth, torch.full_like(depth, 9e9)).min(dim=1).values
    d = d[d < 9e8]
    return float(d.median()) * RETINA_RANGE_M if len(d) else float("nan")


def _m_n_types(s: Served) -> float:
    """Nombre d'apparences DISTINCTES réellement rendues sur les proies (arrondi au centième)."""
    rgb, valid = nearest_food(s.retina)
    rgb = rgb[valid]
    if not len(rgb):
        return float("nan")
    return float(len(torch.unique((rgb * 100).round(), dim=0)))


def _m_appearance_spread(s: Served) -> float:
    """Jitter d'apparence PAR INSTANCE = résidu à la palette de types, pas dispersion brute.

    ⚠️ MESURÉ en construisant l'outil : la dispersion brute des couleurs de proie crie « variation
    servie » dès que les TYPES sont actifs — quatre teintes exactes suffisent à la faire monter à
    0,136. Or ce sont deux mécanismes différents (`_n_types` vs `_appearance_var`) et les confondre
    ferait accuser un réglage fantôme là où le monde fait exactement ce qu'on lui a demandé. Le
    jitter, lui, est l'écart RÉSIDUEL à la couleur de type la plus proche.
    """
    _, _, stat = pick_palette(s.retina)
    return stat["ecart_median"]


def _m_ripe_cue(s: Served) -> float:
    """Variation de luminosité du buisson-marqueur : 0 = l'indice de maturité n'est PAS servi."""
    b, valid = bush_brightness(s.retina)
    return float(b[valid].std()) if int(valid.sum()) > 1 else 0.0


def _m_health_regen(s: Served) -> float:
    h = s.health
    rise = [float(h[i] - h[i - 1]) for i in range(1, len(h))
            if 0 < h[i] - h[i - 1] < 1.0 and i not in set(s.bounds)]
    return float(torch.tensor(rise).median()) if rise else 0.0


def _m_damage(s: Served) -> float:
    """Dégâts subis : 0 = aucun danger actif dans ce monde, quoi qu'en dise le réglage."""
    d = s.health[:-1] - s.health[1:]
    return float(d[d > 0.1].sum())


CONTRACT: list[Clause] = [
    Clause("SYLVAN_ENERGY_DRAIN", "drain d'énergie", 0.15, "homeostasis.gd passive_energy_drain",
           _m_drain_e, "approx", "/tick",
           why="le métabolisme fixe la PORTÉE ; une erreur ici a déjà rendu la portée 2x optimiste"),
    Clause("SYLVAN_THIRST_DRAIN", "drain de soif", 0.15, "homeostasis.gd passive_thirst_drain",
           _m_drain_t, "approx", "/tick",
           why="la symétrie des deux drains est la condition d'un arbitrage propre"),
    # 🚨 NOM CORRIGÉ (2026-07-25) : la clause interrogeait `SYLVAN_FOOD_ENERGY`, or Godot lit
    # `SYLVAN_<PREFIX>_ENERGY_PER` (food_manager.gd:259) et le preset émet `SYLVAN_FOOD_ENERGY_PER`.
    # La clause opposait donc TOUJOURS le défaut du code (40) à la mesure, quelle que soit la valeur
    # demandée : elle ne pouvait structurellement PAS vérifier ce réglage, et criait « au-dessus du
    # plafond » dès qu'on servait un repas plus gros. Un vérificateur qui lit la mauvaise variable est
    # exactement le mensonge que cet outil existe pour supprimer.
    Clause("SYLVAN_FOOD_ENERGY_PER", "restore d'un repas (absorbé)", 40.0, "food_manager.gd energy_per_food",
           _m_restore_e, "plafond", "pts", tol=0.02,
           why="l'absorbé est écrêté par le plafond 100 : il DOIT rester sous le nominal"),
    Clause("SYLVAN_KIN_SPEED", "vitesse du corps", 0.8, "sylvan_agent.gd kin_speed",
           _m_speed, "plafond", "m/s",
           why="mesurée médiane en mouvement ; au-dessus du demandé = ce n'est pas ce corps"),
    Clause("SYLVAN_EAT_RADIUS", "rayon de capture", 1.0, "food_manager.gd eat_radius",
           _m_eat_distance, "plafond", "m",
           why="distance RÉELLE au moment du repas — la seule preuve que la bouche est bien celle-là"),
    Clause("SYLVAN_FOOD_TYPES", "nombre de types de proie", 0.0, "food_manager.gd _n_types",
           _m_n_types, "compte", "apparences",
           why="un type invisible = l'entité meurt dessus sans jamais pouvoir l'apprendre"),
    Clause("SYLVAN_FOOD_APPEARANCE_VAR", "variation d'apparence", 0.0, "food_manager.gd _appearance_var",
           _m_appearance_spread, "presence", "écart-type RGB",
           why="c'est la variation que l'encodeur n'a jamais vue : savoir si elle est SERVIE"),
    Clause("SYLVAN_FOOD_RIPE_CUE", "indice de maturité (buisson)", 0.0, "food_manager.gd _ripe_cue",
           _m_ripe_cue, "presence", "écart-type",
           why="l'indice non-géométrique du monde périssable ; inactif, tout le chantier ne mesure rien"),
    Clause("SYLVAN_HEALTH_REGEN", "régénération de santé", 0.0, "homeostasis.gd health_regen",
           _m_health_regen, "approx", "/tick",
           why="rend la santé cyclique ; à 0 c'est un budget à sens unique"),
    Clause("SYLVAN_SPEED_COST", "coût de locomotion (k de k·vx²)", 0.0, "sylvan_agent.gd speed_cost_k",
           _m_speed_cost, "approx", "énergie/tick à vx=1", tol=0.20,
           why="sans lui la vitesse maximale domine toujours ; demandé mais non servi = un éventail "
               "de vitesse qui ne porte aucune décision, et rien ne le signalerait"),
    Clause("SYLVAN_GAZE", "regard indépendant (133ᵉ dim)", 0.0, "sylvan_agent.gd gaze_enabled",
           _m_gaze, "presence", "dim proprio",
           why="l'angle de tête n'existe dans les données que si le drapeau a pris ; un corpus à 132 "
               "collecté en croyant le regard actif est un corpus sans perception active"),
    Clause("SYLVAN_HAZARD_COUNT", "danger actif (dégâts subis)", 0.0, "hazard_manager.gd (opt-in)",
           _m_damage, "presence", "pts cumulés",
           why="un danger demandé mais jamais rencontré ne prouve RIEN sur l'évitement"),
]


# --------------------------------------------------------------------------------------------- #

def load_served(run: str) -> Served:
    """Deux passes assumées : `guards` pour les constantes du corps (implémentation canonique, on ne
    la duplique pas), `critic_corpus` pour la rétine. Le troisième champ (santé) n'est exposé par
    aucune des deux : lu ici, sans re-coder leur logique."""
    obs, energy, _cmds, bounds = load_bc_corpus(run)
    fichiers = [f for f in (os.path.join(run, "ep_0000.jsonl"), os.path.join(run, "ep_0000.jsonl.gz"))
                if os.path.exists(f)] or sorted(glob.glob(os.path.join(run, "episode_*.jsonl*")))
    health, thirst = [], []
    for f in fichiers:
        with (gzip.open(f, "rt") if f.endswith(".gz") else open(f)) as fh:
            for line in fh:
                if not line.strip():
                    continue
                o = json.loads(line)["obs"]
                health.append(o.get("health", float("nan")))
                thirst.append(o.get("thirst", float("nan")))
    # La tranche rétine commence après la proprio ; obs = proprio ++ rétine ++ énergie (277).
    p = obs.shape[1] - RETINA_DIM - 1
    # Le corps avance-t-il ? Vitesse RÉALISÉE = norme horizontale de la vitesse du torse (proprio
    # dims 1 et 3) — la grandeur sans lag que G4 a établie comme fiable.
    moving = (obs[:, 1] ** 2 + obs[:, 3] ** 2).sqrt() > 1e-3
    return Served(const=measured_constants(run), retina=obs[:, p:p + RETINA_DIM], energy=energy,
                  cmd=_cmds, proprio_dim=p, moving=moving,
                  health=torch.tensor(health, dtype=torch.float32), bounds=bounds,
                  thirst=torch.tensor(thirst, dtype=torch.float32),
                  ate=meal_flags(energy, bounds))


def judge(c: Clause, asked: float, served: float) -> tuple[str, str]:
    """-> (symbole, commentaire). Toute divergence est un 🚨 : ce n'est pas un avis, c'est un contrat."""
    if math.isnan(served):
        return "⚠️", "non mesurable sur ce corpus (l'événement ne s'y produit pas)"
    if c.mode == "presence":
        on_asked, on_served = asked > 0, served > 1e-3
        if on_asked == on_served:
            return "✅", "servi" if on_served else "inactif des deux côtés"
        return "🚨", ("DEMANDÉ mais NON SERVI — le réglage n'a pas pris" if on_asked
                      else "SERVI SANS ÊTRE DEMANDÉ — réglage fantôme")
    if c.mode == "compte":
        if round(served) == round(asked) or (asked == 0 and served <= 1):
            return "✅", "compte conforme"
        return "🚨", f"{round(served)} apparences rendues pour {round(asked)} demandées"
    if c.mode == "plafond":
        if served <= asked * (1 + c.tol):
            hint = " (mais l'entité n'en utilise qu'une fraction)" if served < 0.4 * asked else ""
            return "✅", "sous le plafond demandé" + hint
        return "🚨", f"AU-DESSUS du plafond demandé ({served:.4g} > {asked:.4g})"
    if asked == 0:
        return ("✅", "nul des deux côtés") if abs(served) < 1e-3 else \
               ("🚨", f"mesuré {served:.4g} alors que rien n'était demandé")
    err = abs(served - asked) / abs(asked)
    if err <= c.tol:
        return "✅", f"écart {100 * err:.0f} %"
    return "🚨", f"écart {100 * err:.0f} % — corriger la constante ou le harnais, pas la mesure"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("corpus")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="valeur DEMANDÉE (prioritaire sur l'environnement)")
    ap.add_argument("--preset-file", default=None, help="fichier KEY=VAL (un par ligne)")
    args = ap.parse_args()
    torch.set_num_threads(1)

    asked_env: dict[str, str] = {k: v for k, v in os.environ.items() if k.startswith("SYLVAN_")}
    if args.preset_file:
        with open(args.preset_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("export "):        # `lstrip("export ")` mangerait des lettres
                    line = line[len("export "):].strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    asked_env[k.strip()] = v.strip().strip('"\'')
    for kv in args.set:
        k, v = kv.split("=", 1)
        asked_env[k] = v

    # Les DEUX dispositions : un fichier BC unique, ou un fichier par épisode (collecte WM).
    if not (glob.glob(os.path.join(args.corpus, "ep_*.jsonl*"))
            or glob.glob(os.path.join(args.corpus, "episode_*.jsonl*"))):
        raise SystemExit(f"corpus introuvable : {args.corpus}")
    print(scaffold_banner())
    served = load_served(args.corpus)
    print(f"\nCONTRAT DE MONDE — {args.corpus} | {len(served.energy)} ticks | "
          f"{len(served.bounds) - 1} épisodes | {int(served.ate.sum())} repas")
    print(f"{'clause':<32} {'DEMANDÉ':>12} {'SERVI':>12}   verdict")
    print("-" * 96)

    breaches = 0
    for c in CONTRACT:
        raw = asked_env.get(c.env)
        asked = float(raw) if raw not in (None, "") else c.default
        origin = "" if raw not in (None, "") else "*"
        try:
            value = c.measure(served)
        except Exception as exc:                                  # une mesure cassée n'est pas un PASS
            print(f"{c.label:<32} {asked:>12.4g}{origin} {'—':>12}   ⚠️ mesure impossible : {exc}")
            continue
        sym, note = judge(c, asked, value)
        breaches += sym == "🚨"
        shown = "nan" if math.isnan(value) else f"{value:.4g}"
        print(f"{c.label:<32} {asked:>12.4g}{origin} {shown:>12}   {sym} {note}")
        if sym == "🚨":
            print(f"{'':<32} {'':>12}  {c.env} — {c.why}")

    print("-" * 96)
    print("  * = valeur non demandée : c'est le DÉFAUT DU CODE qui est opposé à la mesure")
    for c in CONTRACT:
        if asked_env.get(c.env) in (None, ""):
            print(f"      {c.env:<32} défaut {c.default:g} — {c.default_src}")
    if breaches:
        print(f"\n🚨 {breaches} clause(s) VIOLÉE(S) — le monde servi n'est pas le monde demandé.\n"
              f"   Ne rien conclure de ce corpus avant d'avoir tranché : réglage qui n'a pas pris,\n"
              f"   ou constante déclarée fausse ?")
    else:
        print("\n✅ contrat respecté — le monde servi est bien celui qui a été demandé")
    return 1 if breaches else 0


if __name__ == "__main__":
    sys.exit(main())
