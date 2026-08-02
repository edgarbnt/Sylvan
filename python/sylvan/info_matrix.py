"""MATRICE DE SURVIE DE L'INFORMATION — où, dans le pipeline, une information du monde MEURT.

POURQUOI CET OUTIL EXISTE (docs/design_outil_matrice_information.md).
Les trois vraies trouvailles du 2026-07-24 ont EXACTEMENT la même forme :
  * le type de la proie   : 83 % dans la rétine  ->  30 % après l'encodeur
  * « bouffe en vue »     : 0,556 à la profondeur 0  ->  0,160 à la profondeur 79
  * la position de l'objet: absente du latent, présente dans le slot
À chaque fois la découverte n'était pas « ça marche mal » mais « l'information disparaît ENTRE tel
étage et tel étage » — et à chaque fois il a fallu une heure de sondes ad-hoc. Dans une archi JEPA
c'est TOUJOURS la bonne question, puisque tout le principe repose sur « la représentation garde-t-elle
ce qui compte ? ». Un logger dit ce qu'un module a FAIT ; ceci dit où l'information MEURT.

CE QUE C'EST. Une matrice : LIGNES = propriétés du monde, COLONNES = étages du pipeline
(rétine brute -> encodeur -> latent d0 -> latent dH -> slot -> token du planner). Chaque case = la
part RÉCUPÉRABLE de l'information, mesurée par une sonde entraînée sur le train et lue en held-out :
R² pour un continu, précision + baseline majoritaire pour une catégorie. Toute chute entre deux
colonnes est une piste, et on sait immédiatement quel module accuser.

RÈGLES DE MESURE (elles viennent d'erreurs déjà payées — ne pas les refaire) :
  1. Held-out PAR ÉPISODE, jamais par tick : deux ticks voisins sont quasi identiques, un split
     aléatoire fuit massivement (`sylvan.critic_corpus` porte déjà la convention de frontières).
  2. TOUJOURS afficher la baseline (majorité / moyenne). 31 % de précision a déjà été pris pour un
     résultat alors que la majorité était à 44 %.
  3. Sonder en LINÉAIRE **et** en MLP. Conclure « l'info est absente » depuis une sonde linéaire
     seule est une faute de mesure. Nuance de ce projet : le MLP n'a JAMAIS battu franchement le
     linéaire — un écart soudain doit donc éveiller les soupçons, pas rassurer.
  4. WM GELÉ : on mesure ce qu'il CONTIENT déjà, on ne l'entraîne pas (CLAUDE.md §3).
  5. Déterminisme : seed fixé, torch mono-thread.

EXTENSIBLE : une nouvelle mécanique du monde = une `Property` de plus dans `PROPERTIES`. C'est ce qui
matérialise la règle « une sonde écrite EN MÊME TEMPS que la mécanique, jamais après ».
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import nn

from sylvan.critic_corpus import RETINA_DIM, auc as corpus_auc, token as planner_token
from sylvan.models.command_wm import CommandWorldModel

N_RAYS = 36                      # perception.gd : RETINA_RAYS
RAY_CHANNELS = 4                 # [depth_norm, R, G, B]
FOOD_CONE = 0.55                 # MÊME seuil que le slot (slot_head) — on ne juge pas avec un autre œil
BUSH_COS = 0.999                 # le buisson-marqueur n'est jamais que la couleur de base ÉCHELONNÉE
# 🚨 RESSERRÉ DE 0,99 À 0,999 (2026-07-25), sur mesure et pas par précaution. Le critère est un
# COSINUS, donc invariant d'échelle : un tronc vert foncé est « la couleur du buisson en plus
# sombre ». Mesuré sur le premier corpus forestier : cos(buisson, écorce de base) = 0,9855, et
# 14 100 rayons d'ARBRES passaient pour des buissons — la luminosité « de maturité » lisait en fait
# la teinte des troncs (31 verts distincts, écart-type 0,124 sur un monde où l'indice de maturité est
# ÉTEINT). C'est le problème du tronc-brun (§2bis) transposé au vert.
# Le buisson étant PAR CONSTRUCTION la couleur de base exactement échelonnée, son cosinus vaut 1,0 :
# 0,999 est la lecture fidèle de cette construction, pas un durcissement arbitraire. NON-RÉGRESSION
# VÉRIFIÉE sur deux corpus sans forêt (bosquets_v2 sans indice, bosquets_v4 AVEC indice) : nombre de
# ticks buisson-en-vue et écart-type de luminosité IDENTIQUES à 0,99 et 0,999.

# Couleur DÉCLARÉE du buisson-marqueur (food_manager.gd PATCH_BUSH_COLOR). Sa LUMINOSITÉ encode la
# maturité de sa baie (x1,0 fraîche -> x0,2 imminente) : la teinte est invariante, l'échelle porte
# l'indice. Déclarée ici, MESURÉE sur le corpus par `measure_palette` (esprit de diagnostics/guards.py).
BUSH_COLOR = torch.tensor([0.47, 0.93, 0.53])

# Palettes de TYPES déclarées par le monde (food_manager.gd TYPE_COLORS). Deux mondes ont été
# collectés : v7 « teinte » (typ31, quatre teintes distinctes) puis v7 « luminosité » (lum41, une
# seule teinte à quatre échelles). La seconde est COLINÉAIRE : toute règle qui normalise le RGB
# (le cosinus du slot) est structurellement aveugle à ces types-là. D'où l'appariement en RGB BRUT
# ci-dessous, exact pour les deux (la rétine rend l'albédo tel quel, sans atténuation — perception.gd).
PALETTES: dict[str, torch.Tensor] = {
    "teinte": torch.tensor([[0.90, 0.10, 0.10], [0.80, 0.60, 0.15],
                            [0.90, 0.10, 0.45], [0.85, 0.55, 0.35]]),
    "luminosite": torch.tensor([[0.900, 0.300, 0.200], [0.648, 0.216, 0.144],
                                [0.450, 0.150, 0.100], [0.288, 0.096, 0.064]]),
    # La palette SERVIE par le monde-forêt (sylvan.world.FORET_V1.food_type_hues), validée séparable
    # par G5. Sans elle, `pick_palette` retenait la plus proche palette connue (« teinte », qui en
    # diffère de 0,05 à 0,07 par canal) et rapportait l'écart comme un JITTER PAR INSTANCE : le
    # contrat accusait alors un « réglage fantôme » d'apparence variable sur un monde qui n'en
    # demandait aucune. Une palette absente du catalogue ne produit pas une erreur, elle produit une
    # fausse mesure — le mode de panne que ces outils existent pour supprimer.
    "foret_v1": torch.tensor([[0.90, 0.12, 0.10], [0.90, 0.55, 0.08],
                              [0.85, 0.10, 0.45], [0.80, 0.42, 0.42]]),
}


# --------------------------------------------------------------------------------------------- #
# Lecture de la rétine — la vérité-terrain est TOUJOURS calculée depuis ce que l'agent VOIT,
# jamais depuis un état caché du monde (sinon on mesurerait un oracle, pas une perception).
# --------------------------------------------------------------------------------------------- #

def rays(retina: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, 144] -> (depth [N, 36], rgb [N, 36, 3])."""
    r = retina.reshape(len(retina), N_RAYS, RAY_CHANNELS)
    return r[..., 0], r[..., 1:4]


def food_mask(rgb: torch.Tensor) -> torch.Tensor:
    """Rayons « nourriture » [N, 36] — MÊME critère que le slot (cône rouge, seuil 0,55)."""
    norm = rgb.norm(dim=-1)
    return (rgb[..., 0] / (norm + 1e-6) > FOOD_CONE) & (norm > 1e-3)


def nearest_food(retina: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rayon de la proie CIBLÉE (la plus proche) -> (rgb [N, 3], valide [N])."""
    depth, rgb = rays(retina)
    is_food = food_mask(rgb)
    d = torch.where(is_food, depth, torch.full_like(depth, 9e9))
    idx = d.argmin(dim=1)
    return rgb[torch.arange(len(rgb)), idx], is_food.any(dim=1)


def bush_brightness(retina: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Luminosité MOYENNE des rayons buisson [N] (= maturité), et validité (buisson en vue).

    MOYENNE et non plus-proche : c'est la mesure exacte du 2026-07-24 (« la même mesure sur le
    BUISSON donne +0,650 » là où elle donne −0,659 sur les rayons bouffe).
    """
    _, rgb = rays(retina)
    norm = rgb.norm(dim=-1)
    # 🚨 DIRECTION EXACTE, pas un cône. Le marqueur est PAR CONSTRUCTION la couleur de base
    # ÉCHELONNÉE : la maturité change son AMPLITUDE, jamais sa DIRECTION. Un seuil de cosinus, lui,
    # est invariant d'échelle et attrape donc tout ce qui est du même vert — mesuré sur le premier
    # corpus forestier : 14 100 rayons d'ARBRES passaient pour des buissons à 0,99, et 278 tenaient
    # encore à 0,999 (les teintes jitterées de G8 finissent par tomber dans le cône). La luminosité
    # « de maturité » lisait alors la teinte des troncs sur un monde où l'indice est ÉTEINT.
    # Aucun estimateur robuste ne rattrape ça (la médiane empire : les rayons d'arbre sont souvent
    # majoritaires dans un tick) — il faut le bon critère, pas une moyenne plus maligne.
    # NON-RÉGRESSION VÉRIFIÉE : sur bosquets_v2 (indice éteint) et bosquets_v4 (indice ACTIF), même
    # nombre de ticks buisson-en-vue et même écart-type qu'avec l'ancien masque, et l'échelle balaie
    # toujours 1,00 -> 0,20 quand l'indice est servi. La vraie mesure est préservée, le bruit part.
    ref = (BUSH_COLOR / BUSH_COLOR.norm() * 1000).round()
    direction = (rgb / (norm.unsqueeze(-1) + 1e-6) * 1000).round()
    is_bush = (direction == ref).all(dim=-1) & (norm > 1e-3)
    scale = norm / BUSH_COLOR.norm()                       # x1,0 fraîche -> x0,2 imminente
    n = is_bush.sum(dim=1)
    return (scale * is_bush).sum(dim=1) / n.clamp_min(1), n > 0


def measure_palette(retina: torch.Tensor, declared: torch.Tensor) -> dict:
    """Constantes MESURÉES vs DÉCLARÉES, appliqué à la palette (esprit `diagnostics/guards.py`).

    Renvoie l'écart médian entre les couleurs de proie RÉELLEMENT rendues et la palette déclarée, et
    la part de rayons-bouffe qu'aucune entrée n'explique. Une couverture basse = la ligne « type »
    étiquette du bruit (des troncs bruns passent le cône rouge, par exemple) : le dire, pas le taire.
    """
    rgb, valid = nearest_food(retina)
    rgb = rgb[valid]
    if not len(rgb):
        return {"n": 0, "ecart_median": float("nan"), "hors_palette": float("nan")}
    dist = torch.cdist(rgb, declared).min(dim=1).values
    return {"n": int(len(rgb)),
            "ecart_median": float(dist.median()),
            "hors_palette": float((dist > 0.05).float().mean())}


def pick_palette(retina: torch.Tensor) -> tuple[str, torch.Tensor, dict]:
    """Choisit la palette qui EXPLIQUE le corpus (on ne devine pas le monde d'un nom de dossier)."""
    scored = {k: measure_palette(retina, p) for k, p in PALETTES.items()}
    name = min(scored, key=lambda k: scored[k]["ecart_median"])
    return name, PALETTES[name], scored[name]


# --------------------------------------------------------------------------------------------- #
# Propriétés du monde = les LIGNES de la matrice.
# --------------------------------------------------------------------------------------------- #

@dataclass
class Sample:
    """Ce qui est disponible au tick t pour définir une vérité-terrain."""
    obs: torch.Tensor          # [N, obs_dim] observation réelle
    retina: torch.Tensor       # [N, 144] tranche rétine de cette observation
    energy: torch.Tensor       # [N]
    slot: torch.Tensor         # [N, 2] slot du WM VIVANT (x_droite, z_avant)
    palette: torch.Tensor      # [K, 3] palette de types retenue pour ce corpus
    meal: torch.Tensor         # [N] 1 s'il y a un repas dans les K ticks à venir (cible du critique)


@dataclass(frozen=True)
class Property:
    """Une ligne : une propriété du monde, sa vérité-terrain, et comment on la note."""
    key: str
    label: str
    kind: str                                                    # "cont" | "cat"
    extract: Callable[[Sample], tuple[torch.Tensor, torch.Tensor]]  # -> (valeurs, validité)
    n_classes: int = 0
    extra: Callable[[torch.Tensor, torch.Tensor], dict[str, float]] | None = None
    why: str = ""


def _p_type(s: Sample) -> tuple[torch.Tensor, torch.Tensor]:
    rgb, valid = nearest_food(s.retina)
    # Appariement en RGB BRUT : exact pour les deux palettes (la rétine rend l'albédo sans
    # atténuation), là où le cosinus est structurellement aveugle à la palette « luminosité ».
    return torch.cdist(rgb, s.palette).argmin(dim=1), valid


def _p_food_visible(s: Sample) -> tuple[torch.Tensor, torch.Tensor]:
    _, rgb = rays(s.retina)
    seen = food_mask(rgb).any(dim=1).float().unsqueeze(1)
    return seen, torch.ones(len(seen), dtype=torch.bool)


def _p_retina_full(s: Sample) -> tuple[torch.Tensor, torch.Tensor]:
    return s.retina, torch.ones(len(s.retina), dtype=torch.bool)


def _p_slot(s: Sample) -> tuple[torch.Tensor, torch.Tensor]:
    return s.slot, torch.ones(len(s.slot), dtype=torch.bool)


def _slot_extra(pred: torch.Tensor, truth: torch.Tensor) -> dict[str, float]:
    """La DISTANCE lue dans la prédiction du slot — c'est elle que `-min_dist` utilise, donc c'est
    elle qui décide. Un R² vectoriel honnête peut cacher une distance pire que la moyenne."""
    dp, dt = pred.norm(dim=-1), truth.norm(dim=-1)
    r2d = 1 - ((dp - dt) ** 2).sum() / ((dt - dt.mean()) ** 2).sum()
    return {"R² distance": float(r2d), "err. médiane (m)": float((pred - truth).norm(dim=-1).median())}


def _p_ripeness(s: Sample) -> tuple[torch.Tensor, torch.Tensor]:
    b, valid = bush_brightness(s.retina)
    return b.unsqueeze(1), valid


def _p_food_distance(s: Sample) -> tuple[torch.Tensor, torch.Tensor]:
    depth, rgb = rays(s.retina)
    is_food = food_mask(rgb)
    d = torch.where(is_food, depth, torch.full_like(depth, 9e9))
    return d.min(dim=1).values.unsqueeze(1), is_food.any(dim=1)


def _p_meal(s: Sample) -> tuple[torch.Tensor, torch.Tensor]:
    return s.meal.unsqueeze(1), torch.ones(len(s.meal), dtype=torch.bool)


def _meal_extra(pred: torch.Tensor, truth: torch.Tensor) -> dict[str, float]:
    """AUC en plus du R² : la cible est RARE et déséquilibrée, donc un R² proche de 0 peut cacher un
    classement parfaitement utile. C'est la métrique de `diag_critic_beyond_geometry`, reprise telle
    quelle (`critic_corpus.auc`) pour que les deux mesures restent comparables."""
    return {"AUC": corpus_auc(pred.squeeze(-1), truth.squeeze(-1))}


PROPERTIES: list[Property] = [
    Property("position", "position de la ressource (slot x,z)", "cont", _p_slot,
             extra=_slot_extra,
             why="ce que le coût du planner consomme : si elle meurt, la décision est aveugle"),
    Property("distance", "distance de la proie visée", "cont", _p_food_distance,
             why="le terme porteur du coût analytique (-min_dist)"),
    Property("type", "type / apparence de la proie visée", "cat", _p_type, n_classes=4,
             why="valeur ARBITRAIRE du monde v7 : sans lui, aucun critique ne peut apprendre la table"),
    Property("vue", "présence de bouffe en vue", "cont", _p_food_visible,
             why="le fait le plus élémentaire de la scène ; sa décroissance mesure la fidélité du rêve"),
    Property("maturite", "maturité (luminosité du buisson)", "cont", _p_ripeness,
             why="indice perceptible NON géométrique, invisible au slot par construction"),
    Property("retine", "rétine entière (la scène)", "cont", _p_retina_full,
             why="borne haute : tout ce que l'observation contient"),
    Property("repas", "repas à venir (cible du critique)", "cont", _p_meal, extra=_meal_extra,
             why="la seule ligne qui porte une CONSÉQUENCE et non une perception : c'est ce qu'un "
                 "critique doit prédire, et l'étage où elle meurt est l'étage qui le condamne"),
]
PROPERTY_BY_KEY = {p.key: p for p in PROPERTIES}


# --------------------------------------------------------------------------------------------- #
# Étages = les COLONNES.
# --------------------------------------------------------------------------------------------- #

@dataclass
class Stages:
    """Les colonnes, PLUS le graphe qui les relie.

    ⚠️ Le pipeline n'est PAS une chaîne unique : le slot est une BRANCHE SÉPARÉE, encodée par
    `slot_encoder` directement sur la rétine, jamais par l'encodeur du WM ni par le RSSM. Chaîner
    naïvement les colonnes de gauche à droite attribuerait au latent une chute qui appartient au
    slot — c'est-à-dire accuserait le mauvais module, exactement ce que l'outil doit empêcher.
    """
    names: list[str] = field(default_factory=list)
    reps: dict[str, torch.Tensor] = field(default_factory=dict)
    parent: dict[str, str] = field(default_factory=dict)      # colonne -> étage qui l'alimente

    @property
    def edges(self) -> list[tuple[str, str]]:
        return [(p, c) for c, p in self.parent.items()]


def column_offset(col: str, mode: str) -> int:
    """Décalage temporel de la VÉRITÉ-TERRAIN pour une colonne donnée.

    Deux questions DIFFÉRENTES, qu'il ne faut pas confondre — l'outil les sépare explicitement :
      * mode « predit » (défaut) : le latent rêvé à la profondeur d est jugé sur ce qui est VRAI à
        t+d. C'est la question JEPA — le rêve prédit-il encore le monde ? C'est aussi la convention
        des mesures historiques (« dégradation le long du rêve », 0,556 -> 0,160).
      * mode « percu » : toutes les colonnes sont jugées sur ce qui était vrai à t. C'est la question
        de la MÉMOIRE — la représentation garde-t-elle ce qu'elle a perçu ?
    À la profondeur 0 les deux coïncident, ce qui rend les deux lectures comparables à la racine.
    """
    if mode == "percu" or not col.startswith("latent d"):
        return 0
    return int(col.rsplit("d", 1)[1])


@torch.no_grad()
def sample_at(wm: CommandWorldModel, obs: torch.Tensor, energy: torch.Tensor,
              starts: torch.Tensor, offset: int, palette: torch.Tensor,
              meal: torch.Tensor) -> Sample:
    """L'état RÉEL du monde à t+offset — la vérité-terrain, jamais une sortie du modèle."""
    idx = starts + offset
    o = obs[idx]
    slot = torch.cat([wm.encode_slot(o[i:i + 4096]) for i in range(0, len(o), 4096)])
    return Sample(obs=o, retina=o[:, wm.proprio_dim:wm.proprio_dim + RETINA_DIM],
                  energy=energy[idx], slot=slot, palette=palette, meal=meal[idx])


@torch.no_grad()
def build_stages(wm: CommandWorldModel, obs: torch.Tensor, cmds: torch.Tensor,
                 starts: torch.Tensor, energy: torch.Tensor, depths: list[int],
                 batch: int = 256) -> Stages:
    """Extrait toutes les représentations aux MÊMES états — c'est ce qui rend les colonnes comparables.

    rétine brute -> encodeur -> latent rêvé à chaque profondeur demandée -> slot -> token du planner.
    Le rêve est celui du déploiement (`rollout_open_loop` sous les commandes RÉELLEMENT exécutées) :
    mesurer sur un rollout teacher-forced donnerait une matrice qui décrit un pipeline qu'on ne sert pas.

    ⚠️ SLOT OPTIONNEL (2026-08-02). Les colonnes « slot » et « token planner » ne sont produites que
    si le checkpoint porte RÉELLEMENT un canal-slot. Avant ce correctif l'appelant forçait
    `with_slot=True` : sur un WM sans slot, `slot_encoder` était à poids ALÉATOIRES et ces deux
    colonnes rendaient du bruit avec l'apparence d'une mesure. Une colonne absente se lit ; une
    colonne fausse se croit.
    """
    P = wm.proprio_dim
    horizon = max(2, max(depths) + 1)
    retina = obs[starts][:, P:P + RETINA_DIM]

    keep = torch.tensor(depths)
    enc, lat = [], []
    for i in range(0, len(starts), batch):
        idx = starts[i:i + batch]
        enc.append(wm.encoder(obs[idx]))
        seq = torch.stack([cmds[j:j + horizon] for j in idx])
        # On ne garde QUE les profondeurs demandées : tout le rollout ferait [N, 80, 128] en mémoire
        # pour rien (213 Mo à 5 000 états, et ça grandit avec le corpus).
        lat.append(wm.rollout_open_loop(obs[idx], seq)["predicted_latents"][:, keep])
    encoded = torch.cat(enc)
    latents = torch.cat(lat)                                  # [N, len(depths), latent_dim]

    st = Stages()
    st.names.append("rétine")
    st.reps["rétine"] = retina                                # racine : l'observation elle-même
    st.names.append("encodeur")
    st.reps["encodeur"] = encoded
    st.parent["encodeur"] = "rétine"
    prev = "encodeur"
    for k, d in enumerate(depths):
        name = f"latent d{d}"
        st.names.append(name)
        st.reps[name] = latents[:, k]
        st.parent[name] = prev                                # la chaîne du rêve
        prev = name
    if getattr(wm, "with_slot", False):
        slot = torch.cat([wm.encode_slot(obs[starts[i:i + 4096]])
                          for i in range(0, len(starts), 4096)])
        st.names.append("slot")
        st.reps["slot"] = slot
        st.parent["slot"] = "rétine"                          # BRANCHE SÉPARÉE (slot_encoder)
        st.names.append("token planner")
        st.reps["token planner"] = planner_token(energy[starts] / 100.0, slot)
        st.parent["token planner"] = "slot"
    return st


def sample_starts(bounds: list[int], stride: int, horizon: int,
                  frac_train: float = 0.7) -> tuple[torch.Tensor, torch.Tensor]:
    """Ticks échantillonnés + masque TRAIN, split PAR ÉPISODE (règle n°1).

    Un tick n'est retenu que si son rêve tient dans SON épisode : sinon on rêverait à travers un
    respawn, ce qui mélangerait deux vies dans une même trajectoire imaginée.
    """
    n_ep = len(bounds) - 1
    cut = bounds[max(1, int(round(frac_train * n_ep)))]
    starts, is_tr = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        for t in range(a, b - horizon - 1, stride):
            starts.append(t)
            is_tr.append(t < cut)
    return torch.tensor(starts), torch.tensor(is_tr)


def positional_split(n: int, frac_train: float = 0.7) -> torch.Tensor:
    """Split CHRONOLOGIQUE brut : les `frac_train` premières lignes en train.

    ⚠️ NE PAS utiliser pour rendre un verdict — la coupe tombe AU MILIEU d'un épisode, dont la tête
    est en train et la queue en held-out ; à stride 6 deux lignes voisines sont quasi identiques,
    donc ça fuit. Fourni UNIQUEMENT pour rejouer la convention des mesures historiques
    (`diag_latent_carries_type.py`) et pouvoir comparer des chiffres à des chiffres.
    """
    return torch.arange(n) < max(1, int(frac_train * n))


# --------------------------------------------------------------------------------------------- #
# Sondes — LINÉAIRE et MLP, toujours les deux (règle n°3).
# --------------------------------------------------------------------------------------------- #

def r2(pred: torch.Tensor, truth: torch.Tensor) -> float:
    ss_res = ((pred - truth) ** 2).sum()
    ss_tot = ((truth - truth.mean(0)) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def _standardise(x: torch.Tensor, tr: torch.Tensor) -> torch.Tensor:
    mu, sd = x[tr].mean(0), x[tr].std(0).clamp_min(1e-6)
    return (x - mu) / sd


RIDGE_LAMBDA = 1e-3      # sur des features standardisées, X'X ≈ n·I : 0,1 % de biais, et la stabilité


def probe_linear(xtr: torch.Tensor, ytr: torch.Tensor, xte: torch.Tensor) -> torch.Tensor:
    """Sonde linéaire = RIDGE, pas des moindres carrés nus.

    ⚠️ MESURÉ le 2026-07-24 en construisant cet outil : `torch.linalg.lstsq` sur une colonne
    RANG-DÉFICIENT (le token du planner porte un canal constant `connu`) rendait DEUX RÉSULTATS
    DIFFÉRENTS pour la même entrée d'un run à l'autre (+0,584 puis −1,178) — la révélation de rang
    de LAPACK tranche une quasi-égalité. Une case de la matrice qui bouge sans que rien ne bouge
    ruinerait l'usage NON-RÉGRESSION de l'outil. Le ridge rend le système défini, donc déterministe,
    et borne aussi l'extrapolation délirante d'un ajustement à 144 features corrélées.
    """
    xm, ym = xtr.mean(0), ytr.mean(0)
    xc = xtr - xm
    a = xc.T @ xc + RIDGE_LAMBDA * len(xtr) * torch.eye(xtr.shape[1])
    w = torch.linalg.solve(a, xc.T @ (ytr - ym))
    return (xte - xm) @ w + ym


def probe_regression(x: torch.Tensor, y: torch.Tensor, tr: torch.Tensor,
                     steps: int, hidden: int) -> dict[str, torch.Tensor]:
    """-> prédictions held-out des deux sondes (règle n°3 : toujours les deux)."""
    xs = _standardise(x, tr)
    xtr, ytr, xte = xs[tr], y[tr], xs[~tr]
    lin = probe_linear(xtr, ytr, xte)

    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(xtr.shape[1], hidden), nn.SiLU(),
                        nn.Linear(hidden, hidden), nn.SiLU(),
                        nn.Linear(hidden, ytr.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(steps):
        idx = torch.randperm(len(xtr))[:1024]
        opt.zero_grad()
        ((net(xtr[idx]) - ytr[idx]) ** 2).mean().backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        return {"lin": lin, "mlp": net(xte)}


def probe_classification(x: torch.Tensor, y: torch.Tensor, tr: torch.Tensor, n_classes: int,
                         steps: int, hidden: int) -> dict[str, torch.Tensor]:
    """-> classes prédites en held-out. Même recette (Adam, 512) pour le linéaire et le MLP, afin
    que l'écart entre les deux mesure l'EXPRESSIVITÉ et pas deux optimiseurs différents."""
    xs = _standardise(x, tr)
    xtr, ytr, xte = xs[tr], y[tr], xs[~tr]
    out = {}
    for name, net in (("lin", nn.Linear(xtr.shape[1], n_classes)),
                      ("mlp", nn.Sequential(nn.Linear(xtr.shape[1], hidden), nn.SiLU(),
                                            nn.Linear(hidden, hidden), nn.SiLU(),
                                            nn.Linear(hidden, n_classes)))):
        torch.manual_seed(0)
        for m in net.modules():                       # ré-init sous le seed : deux sondes reproductibles
            if isinstance(m, nn.Linear):
                m.reset_parameters()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        lossf = nn.CrossEntropyLoss()
        for _ in range(steps):
            i = torch.randperm(len(xtr))[:512]
            opt.zero_grad()
            lossf(net(xtr[i]), ytr[i]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            out[name] = net(xte).argmax(1)
    return out


@dataclass
class Cell:
    """Une case : la part récupérable, par les deux sondes, plus la baseline qui lui donne un sens."""
    lin: float
    mlp: float
    baseline: float
    n_test: int
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def best(self) -> float:
        return max(self.lin, self.mlp)


def measure_cell(prop: Property, x: torch.Tensor, y: torch.Tensor, tr: torch.Tensor,
                 steps: int, hidden: int) -> Cell:
    """Une propriété lue depuis UNE représentation. La baseline est calculée sur le TRAIN et jugée
    en held-out — une baseline calculée sur le held-out serait, elle aussi, une fuite."""
    te = ~tr
    if prop.kind == "cat":
        pred = probe_classification(x, y, tr, prop.n_classes, steps, hidden)
        counts = torch.bincount(y[tr], minlength=prop.n_classes)
        major = int(counts.argmax())
        base = float((y[te] == major).float().mean())
        return Cell(lin=float((pred["lin"] == y[te]).float().mean()),
                    mlp=float((pred["mlp"] == y[te]).float().mean()),
                    baseline=base, n_test=int(te.sum()))
    pred = probe_regression(x, y, tr, steps, hidden)
    base = float(r2(y[tr].mean(0).expand_as(y[te]), y[te]))
    extra = prop.extra(pred["lin"], y[te]) if prop.extra else {}
    return Cell(lin=r2(pred["lin"], y[te]), mlp=r2(pred["mlp"], y[te]),
                baseline=base, n_test=int(te.sum()), extra=extra)
