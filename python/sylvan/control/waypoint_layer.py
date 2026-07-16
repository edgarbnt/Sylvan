"""Étage WAYPOINT — « petit H-JEPA » v0, niveau HAUT à 2 niveaux (ÉCHAFAUDAGE DÉCLARÉ).

POURQUOI (verdict d'élimination 2026-07-16, commit 8a3d80a) : la fente MPC du niveau bas (33 arcs
myopes ~0.8 m, replan glouton) ne peut pas COMPOSER « contourne le gardien puis mange », quel que
soit le score — six juges codés-main éliminés. Le fix licencié (docs/recherche_hjepa_waypoint.md) :
un étage AU-DESSUS qui propose des SOUS-BUTS spatiaux (LeCun §4.7 : les actions du haut = « cibles
pour les états du niveau bas » ; proposeur = trou explicitement ouvert §6/§8.1) et les COMMIT
(TangentBug : l'échappée d'un minimum local est un CHANGEMENT DE MODE, pas un blend).

CE QUE FAIT CET ÉTAGE (spec = recherche §6) :
  - à chaque DÉCISION (spawn / waypoint atteint / timeout / changement de cible) : candidats =
    cible directe + anneau de `ring_n` waypoints autour de l'entité (R ≈ 2-3 m, positions au sol) ;
  - score ANALYTIQUE par candidat : ligne entité→wp dégagée de vert-proche + ligne wp→cible dégagée
    + longueur totale (monnaie : mètres ≡ pas à vitesse nominale — la « queue de survie » v0 ;
    v1 possible : _survival_extension avec les niveaux de drives). ZÉRO reconstruction du danger
    (pas de centre/rayon estimés — leçon des 6 échecs) : les rayons VERTS bruts de la rétine sont
    les obstacles, point (« cette direction est-elle verte-proche ? », style mur-vert) ;
  - COMMIT du gagnant jusqu'à atteinte (~0.9 m) ou timeout (~180 pas), avec HYSTÉRÉSIS (un wp ne
    détrône le direct que s'il le bat nettement — anti-dithering, et G0 structurel : monde plat →
    zéro vert → direct gagne toujours → comportement identique au vivant).

CE QUE CET ÉTAGE NE FAIT PAS : il ne touche PAS le niveau bas (command_planner.py inchangé — il
reçoit juste une cible via le mécanisme override du serveur) ; il ne pilote pas (vx, ω).

SUIVI DU WAYPOINT ENTRE DÉCISIONS : odométrie-par-COMMANDE — le corps cinématique obéit exactement
à (vx, ω) (sylvan_agent.gd:_kinematic_step), donc intégrer la commande émise suffit. Constantes
CALIBRÉES sur données réelles (diag_waypoint_deadreckon.py : err médiane 0.22 m @150 pas, PASS) :
k_fwd = KIN_SPEED·dt_eff = 0.8×0.018, k_yaw = KIN_TURN·dt_eff = 1.5×0.018, ω>0 = droite.

🚨 ÉCHAFAUDAGE (architecture.json) : proposeur (anneau) et scoreur (lignes-vertes analytiques)
codés-main — l'étape suivante est le CRITIQUE APPRIS qui note les waypoints (écarts larges à cet
étage → apprenable, HIQL fig.8), puis le remplacement du proposeur/scoreur (aspiration LeCun §4.7).
Opt-in serveur : SYLVAN_WAYPOINT=1 (défaut OFF = zéro régression).
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

N_RAY = 36
RETINA_RANGE_M = 10.0    # MAX_RANGE (perception.gd)


@dataclass
class WaypointConfig:
    ring_n: int = 8              # waypoints sur l'anneau (spec : 6-8)
    ring_radius: float = 2.5     # m autour de l'entité (spec : R ≈ 2-3)
    reach_m: float = 1.2         # seuil d'atteinte du waypoint. ⚠️ DOIT rester ≥ resource_reach
                                 # (1.0) du niveau bas : sous ce rayon le coût survie considère la
                                 # cible « atteinte » et cesse de tirer → G1 v0 : 8/11 timeouts
                                 # groupés à 1.05-1.40 m avec reach=0.9 (near-miss structurel).
    timeout_steps: int = 180     # pas Godot avant re-décision forcée (spec : 150-200)
    abort_patience: int = 2      # replans CONSÉCUTIFS de bascule de cible avant d'avorter le leg.
                                 # G1 v0 : 18/27 abandons à 10 pas = flicker d'égalité (scores
                                 # saturés au cap, bruit 1-replan mesuré 2026-07-04) — une vraie
                                 # bascule d'urgence PERSISTE, le bruit non.
    tangent_margin: float = 1.4  # m de dégagement perpendiculaire des candidats TANGENTS au-delà
                                 # des bords du nuage vert perçu (TangentBug : sous-but = extrémité
                                 # de l'obstacle SENTI ; géométrie capteur pure, zéro centre/rayon).
                                 # ≥ green_margin + 0.39 (bord létal derrière le pilier perçu).
    green_margin: float = 1.0    # ρ : marge autour d'un point vert PERÇU. ⚠️ Fait mesuré
                                 # (hazard_manager.gd) : piliers = centre + anneau à 0.7·r=0.91 m,
                                 # mais le disque de DÉGÂTS va à r=1.3 → le bord létal est 0.39 m
                                 # AU-DELÀ du pilier le plus externe ; + jitter slot/arc → 1.0.
    block_weight: float = 25.0   # W : mètres de trajet équivalents par mètre d'intrusion (décisif)
    hysteresis: float = 0.15     # un wp ne bat le direct que s'il coûte 15 % de moins (anti-dither)
    recheck_every: int = 1       # en mode direct, re-décision tous les K replans. G1 v1 (mesuré) :
                                 # avec K=5, la fenêtre aveugle de 50 ticks concentrait l'exposition
                                 # au vert (le bas ne voit PAS le vert ; arcs+jitter dévient de la
                                 # ligne notée) → K=1. L'hystérésis pro-direct + le commit (aucune
                                 # décision pendant un leg) préviennent déjà le dithering.
    k_fwd: float = 0.0144        # m/tick par unité de vx  (calibré : 0.8 × dt_eff 0.018)
    k_yaw: float = 0.027         # rad/tick par unité de ω (calibré : 1.5 × 0.018 ; ω>0 = droite)

    @classmethod
    def from_env(cls) -> "WaypointConfig":
        """Surcharges d'environnement SYLVAN_WP_* (mêmes clés que les champs, en majuscules)."""
        c = cls()
        for name, cast in (("ring_n", int), ("ring_radius", float), ("reach_m", float),
                           ("timeout_steps", int), ("green_margin", float), ("block_weight", float),
                           ("hysteresis", float), ("recheck_every", int),
                           ("abort_patience", int), ("tangent_margin", float),
                           ("k_fwd", float), ("k_yaw", float)):
            v = os.environ.get(f"SYLVAN_WP_{name.upper()}")
            if v is not None:
                setattr(c, name, cast(v))
        return c


def green_points(retina: list[float]) -> list[tuple[float, float]]:
    """Points-obstacles VERTS perçus, en ego (x_right, z_fwd) — critère mur-vert
    (command_planner.py : G>R, G>B, saturation>0.15, rayon qui touche). Aucune reconstruction."""
    pts: list[tuple[float, float]] = []
    for k in range(N_RAY):
        d, r, g, b = retina[4 * k:4 * k + 4]
        if d >= 0.999:
            continue
        if g > r and g > b and (max(r, g, b) - min(r, g, b)) > 0.15:
            bearing = 2.0 * math.pi * k / N_RAY
            pts.append((d * RETINA_RANGE_M * math.sin(bearing), d * RETINA_RANGE_M * math.cos(bearing)))
    return pts


def _seg_point_dist(ax: float, az: float, bx: float, bz: float, px: float, pz: float) -> float:
    """Distance du point P au segment [A, B]."""
    vx, vz = bx - ax, bz - az
    l2 = vx * vx + vz * vz
    if l2 < 1e-12:
        return math.hypot(px - ax, pz - az)
    t = max(0.0, min(1.0, ((px - ax) * vx + (pz - az) * vz) / l2))
    return math.hypot(px - (ax + t * vx), pz - (az + t * vz))


def _seg_intrusion(ax: float, az: float, bx: float, bz: float,
                   greens: list[tuple[float, float]], margin: float) -> float:
    """Intrusion max (m) du segment dans la marge d'un point vert : (ρ − dist)⁺, GRADUÉE
    (leçon détour : binaire = zéro gradient ; ici les candidats diffèrent de mètres → décisif)."""
    worst = 0.0
    for px, pz in greens:
        worst = max(worst, margin - _seg_point_dist(ax, az, bx, bz, px, pz))
    return max(0.0, worst)


def tangent_candidates(greens: list[tuple[float, float]],
                       margin: float) -> list[tuple[float, float]]:
    """Candidats TANGENTS (TangentBug, recherche §4) : un waypoint juste AU-DELÀ de chaque bord
    angulaire du nuage vert perçu, décalé perpendiculairement vers l'extérieur. Géométrie capteur
    PURE (les points verts bruts), zéro centre/rayon estimé. POURQUOI (G1 v0, mesuré) : depuis
    6-8 m, l'anneau autour de l'ENTITÉ ne dévie le 2ᵉ segment que de ~1 m au niveau du gardien
    (le segment reconverge vers la cible) → best_wp ≈ direct sur 22 décisions bloquées. Le via-point
    doit être posé À CÔTÉ DE L'OBSTACLE, pas à côté de l'entité.

    Bords angulaires = extrémités du plus grand TROU angulaire entre points verts consécutifs
    (robuste au wraparound, aucune hypothèse « un seul disque »)."""
    if not greens:
        return []
    if len(greens) == 1:
        gx, gz = greens[0]
        d = max(math.hypot(gx, gz), 1e-6)
        px, pz = -gz / d, gx / d                       # perpendiculaire unitaire
        return [(gx + margin * px, gz + margin * pz), (gx - margin * px, gz - margin * pz)]
    order = sorted(range(len(greens)),
                   key=lambda i: math.atan2(greens[i][0], greens[i][1]))
    bearings = [math.atan2(greens[i][0], greens[i][1]) for i in order]
    gaps = []
    for j in range(len(order)):
        nxt = (j + 1) % len(order)
        gap = bearings[nxt] - bearings[j]
        if nxt == 0:
            gap += 2.0 * math.pi
        gaps.append(gap)
    jmax = max(range(len(gaps)), key=lambda j: gaps[j])
    # le nuage s'étend du point APRÈS le plus grand trou (bord 1) au point AVANT (bord 2)
    edges = (order[(jmax + 1) % len(order)], order[jmax])
    out: list[tuple[float, float]] = []
    for k, side in zip(edges, (+1.0, -1.0)):
        gx, gz = greens[k]
        d = max(math.hypot(gx, gz), 1e-6)
        # perpendiculaire orientée vers l'EXTÉRIEUR du nuage. Convention : bearing atan2(x,z)
        # croissant = vers la DROITE ; perp gauche = (−gz, gx)/d, perp droite = (gz, −gx)/d.
        # Bord 1 (après le trou, côté gauche du nuage) → décale à GAUCHE ; bord 2 → à DROITE.
        px, pz = (-gz / d, gx / d) if side > 0 else (gz / d, -gx / d)
        out.append((gx + margin * px, gz + margin * pz))
    return out


def route_cost(wp: tuple[float, float], target: tuple[float, float],
               greens: list[tuple[float, float]], cfg: WaypointConfig) -> tuple[float, float]:
    """Coût (m équivalents) d'un trajet 0→wp→cible : longueur totale + W × intrusion-verte des
    2 legs. Retourne (coût, intrusion totale). Le candidat DIRECT = wp placé sur la cible."""
    leg1 = math.hypot(wp[0], wp[1])
    leg2 = math.hypot(target[0] - wp[0], target[1] - wp[1])
    intr = _seg_intrusion(0.0, 0.0, wp[0], wp[1], greens, cfg.green_margin)
    if leg2 > 1e-6:
        intr += _seg_intrusion(wp[0], wp[1], target[0], target[1], greens, cfg.green_margin)
    return leg1 + leg2 + cfg.block_weight * intr, intr


def candidate_features(wp: tuple[float, float], target: tuple[float, float],
                       greens: list[tuple[float, float]]) -> list[float]:
    """Features d'un candidat pour le CRITIQUE-WAYPOINT (docs/design_critique_waypoint.md).

    ⚠️ Ce featurizer est LE point de parité train=déploiement : l'entraînement l'importe d'ici.
    SYMÉTRIE MIROIR PAR CONSTRUCTION (leçon token |sin| : une symétrie connue s'IMPOSE) :
    canonicalisation — si wp_x < 0, miroir de TOUTES les x (wp, cible, verts) → le côté est aboli,
    la géométrie RELATIVE wp↔cible↔verts est préservée (contrairement à |sin| par objet, qui la
    perdrait : contourner à gauche d'un vert à droite ≠ à gauche d'un vert à gauche).
    d_vert_leg1/2 = distance BRUTE du vert perçu le plus proche à chaque segment, SANS marge : les
    constantes codées-main (green_margin 1.0 / bord létal 0.39) SORTENT des features — le critique
    apprend la distance létale de ses morts vécues, pas de la géométrie connue du monde."""
    if wp[0] < 0.0:
        wp = (-wp[0], wp[1])
        target = (-target[0], target[1])
        greens = [(-gx, gz) for gx, gz in greens]
    d_wp = math.hypot(wp[0], wp[1])
    d_tg = math.hypot(target[0], target[1])
    leg2 = math.hypot(target[0] - wp[0], target[1] - wp[1])
    dg1 = dg2 = 10.0
    for gx, gz in greens:
        dg1 = min(dg1, _seg_point_dist(0.0, 0.0, wp[0], wp[1], gx, gz))
        if leg2 > 1e-6:
            dg2 = min(dg2, _seg_point_dist(wp[0], wp[1], target[0], target[1], gx, gz))
    is_direct = 1.0 if leg2 < 1e-6 else 0.0
    return [min(d_wp, 10.0) / 10.0, wp[0] / (d_wp + 1e-6), wp[1] / (d_wp + 1e-6),
            min(d_tg, 10.0) / 10.0, target[0] / (d_tg + 1e-6), target[1] / (d_tg + 1e-6),
            min(d_wp + leg2, 20.0) / 20.0, dg1 / 10.0, dg2 / 10.0, is_direct]


WP_FEAT_DIM = 10


class WaypointLayer:
    """État du niveau haut : décision, commitment, odométrie du waypoint, événements.

    Contrat serveur (serve_planner_command) :
      - chaque tick : `tick(cmd)` avec la commande RETOURNÉE (celle que le corps exécute) ;
      - replan SANS commitment : `maybe_decide(target_id, target_pos, retina)` → dict décision ou
        None ; si un wp est commité, le serveur route le niveau bas dessus (override) ;
      - replan AVEC commitment : lire `wp` ; si `first_target` du plan ≠ `target_id` →
        `abort("target_change")` ; atteinte/timeout sont détectés ici (`consume_event()`).
    """

    def __init__(self, cfg: WaypointConfig | None = None) -> None:
        self.cfg = cfg or WaypointConfig.from_env()
        self.debug = os.environ.get("SYLVAN_WAYPOINT_DEBUG", "0") == "1"
        # === EXPLORATION À L'ÉTAGE WAYPOINT (docs/design_critique_waypoint.md) ===
        # COLLECTE SEULEMENT (défaut 0 = OFF, déploiement déterministe). Avec prob ε par décision,
        # commettre un candidat UNIFORME (y compris les mauvais : le critique doit voir « à travers
        # le vert = mort »). Sans elle, le corpus est AUTO-CONFIRMANT (leçon 2026-07-08) — le
        # critique n'apprendrait que les choix du scoreur analytique. Leçon Director : l'exploration
        # au MANAGER (varier les waypoints), jamais au worker (le bruit de commande comprimait les vies).
        self.explore_eps = max(0.0, float(os.environ.get("SYLVAN_WP_EXPLORE_EPS", "0")))
        self._rng = random.Random(int(os.environ.get("SYLVAN_WP_EXPLORE_SEED", "0")))
        # Log de décisions (SYLVAN_WP_LOG=dir) : 1 ligne jsonl par décision — tick global (clé de
        # jointure avec le flux BC pour drives + issue vécue), features PAR CANDIDAT (le featurizer
        # candidate_features = parité train/déploiement), coûts analytiques, choix, flag explore.
        _log_dir = os.environ.get("SYLVAN_WP_LOG")
        self._log_file = None
        if _log_dir:
            p = Path(_log_dir)
            p.mkdir(parents=True, exist_ok=True)
            self._log_file = open(p / "decisions.jsonl", "w", buffering=1)
        self._global_ticks = 0               # jamais remis à zéro (jointure sur le flux BC continu)
        # === ORACLE-SPRINT (SONDE Phase-2 monde v2, ÉCHAFAUDAGE DÉCLARÉ ET JETABLE) ===
        # Question du gate G-place : une politique santé/drive-consciente SIMPLE bat-elle la
        # géométrie pure quand la bouffe est AU CŒUR du danger ? Règle-cheat : cible bouffe BLOQUÉE
        # (intrusion directe > 0) ET santé > 60 ET énergie < 50 → SPRINT (direct malgré le vert).
        # = plafond atteignable ; si oracle ≥ analytique +4 repas → la place existe pour l'APPRIS.
        self.oracle_sprint = os.environ.get("SYLVAN_WP_ORACLE_SPRINT", "0") == "1"
        self._drives: tuple[float, float, float] | None = None    # (énergie, soif, santé) 0-100
        if self.oracle_sprint:
            print("[waypoint] ORACLE-SPRINT actif (sonde G-place, échafaudage) : bouffe bloquée + "
                  "santé>60 + énergie<50 → direct malgré le vert", flush=True)
        # === GARDE SANS-CIBLE (dette Phase 0 MESURÉE) ===
        # Sans cible, le bas croise TOUT DROIT (no_food_command) et personne ne regarde le vert →
        # vies noyées sans une seule décision (seed 3 : vie de 250 ticks toute collée au vert, 115
        # ticks sans cible). Ici : cible VIRTUELLE droit devant ; si sa ligne est intruse de vert →
        # commit du meilleur candidat évasif. ÉVITEMENT PUR, pas de recherche (≠ CHERCHER).
        # Opt-in SYLVAN_WP_GUARD=1 (défaut OFF), jugé sur re-run seed 3 : morts 5→≤2 attendu.
        self.guard_enable = os.environ.get("SYLVAN_WP_GUARD", "0") == "1"
        self.guard_lookahead = float(os.environ.get("SYLVAN_WP_GUARD_AHEAD", "4.0"))
        if self.guard_enable:
            print(f"[waypoint] GARDE SANS-CIBLE active : croisière + vert intrusant la ligne avant "
                  f"({self.guard_lookahead} m) → wp évasif", flush=True)
        # === CRITIQUE-DOULEUR (SYLVAN_WP_PAIN_CRITIC=ckpt, gates v2 passés : AUC 0.881, monotone) ===
        # Quand chargé, le scoreur remplace les termes verts CODÉS-MAIN (marges 1.0/1.4, W=25) par la
        # douleur APPRISE des morsures vécues : coût = longueur + κ·Q_douleur(candidat). κ = taux
        # d'échange pas/dégât (SYLVAN_WP_PAIN_KAPPA, défaut 100 ; ancre : 100 dégâts = mort ≈ vie
        # restante — constante d'échafaudage flaggée, jugée par l'A/B). Le vert reste un PERCEPT dans
        # les features (distances brutes) ; sa LÉTALITÉ est ce qui est appris.
        self.pain_critic = None
        self.pain_kappa_m = 0.0
        _pc = os.environ.get("SYLVAN_WP_PAIN_CRITIC")
        if _pc:
            import torch as _torch
            from scripts.train_waypoint_pain import PainCritic
            _ck = _torch.load(_pc, map_location="cpu", weights_only=False)
            self.pain_critic = PainCritic()
            self.pain_critic.load_state_dict(_ck["state_dict"])
            self.pain_critic.eval()
            _kappa = float(os.environ.get("SYLVAN_WP_PAIN_KAPPA", "100"))     # pas / dégât
            self.pain_kappa_m = _kappa * 0.02                                 # → mètres / dégât
            print(f"[waypoint] CRITIQUE-DOULEUR actif : {Path(_pc).name} (AUC_cv={_ck.get('auc_cv', 0):.3f}) "
                  f"κ={_kappa} pas/dégât → les marges vertes codées-main SORTENT du scoring", flush=True)
        # === CRITIQUE-SPRINT (SYLVAN_WP_SPRINT_CRITIC=ckpt — chantier IC+TC, docs/design_critique_sprint.md) ===
        # Forme D1 (owner 2026-07-16) : score(c) = leg1+leg2 + (W − g(s,c))·intrusion(c), g = W·p ∈ [0, W].
        # L'ANALYTIQUE COMPLET reste le socle (g=0 ⇒ bit-identique — leçon des 2 A/B perdus par
        # REMPLACEMENT) ; la correction ne peut qu'ADOUCIR la pénalité verte selon l'état (drives,
        # santé, douleur prédite) → elle n'apprend QUE la licence de sprint. Si son juge passe
        # (2×24 vies : ≥ géométrie +8 repas poolés ET morts ≤ +2), la règle-oracle ci-dessus MEURT.
        self.sprint_critic = None
        self._sprint_pain = None
        _sc = os.environ.get("SYLVAN_WP_SPRINT_CRITIC")
        if _sc:
            if self.pain_critic is not None or self.oracle_sprint:
                raise ValueError("SYLVAN_WP_SPRINT_CRITIC est exclusif de SYLVAN_WP_PAIN_CRITIC et "
                                 "SYLVAN_WP_ORACLE_SPRINT (modes de scoring concurrents)")
            import torch as _torch
            from scripts.train_sprint_critic import SprintCritic
            from scripts.train_waypoint_pain import PainCritic as _Pain
            _ck = _torch.load(_sc, map_location="cpu", weights_only=True)
            self.sprint_critic = SprintCritic()
            self.sprint_critic.load_state_dict(_ck["state_dict"])
            self.sprint_critic.eval()
            # la feature douleur vient du checkpoint bankée DANS le ckpt sprint (parité train/déploiement)
            _pk = _torch.load(_ck["pain_ckpt"], map_location="cpu", weights_only=True)
            self._sprint_pain = _Pain()
            self._sprint_pain.load_state_dict(_pk["state_dict"])
            self._sprint_pain.eval()
            print(f"[waypoint] CRITIQUE-SPRINT actif : {Path(_sc).name} (AUC_cv={_ck.get('auc_cv', 0):.3f}) "
                  f"— g=W·p module la pénalité verte, analytique intact "
                  f"(douleur : {Path(_ck['pain_ckpt']).name})", flush=True)
        if self.explore_eps > 0.0:
            print(f"[waypoint] EXPLORATION active : ε={self.explore_eps} (uniforme sur les candidats, "
                  f"collecte seulement) — corpus contrasté pour le critique-waypoint", flush=True)
        self.reset()

    def reset(self) -> None:
        self.wp: tuple[float, float] | None = None
        self.target_id: str | None = None
        self._target_at_decision: tuple[float, float] | None = None
        self.leg_steps = 0
        self._replans_since_decision = 0
        self._flip_streak = 0                    # replans consécutifs où l'arbitrage préfère l'AUTRE cible
        self._event: str | None = None           # "reached" | "timeout" (consommé par le serveur)
        # compteurs de trace (diagnostic G1 : LOCALISER, ne pas deviner)
        self.n_decisions = 0
        self.n_commits = 0
        self.n_reached = 0
        self.n_timeouts = 0
        self.n_aborts = 0

    # ------------------------------------------------------------------ odométrie / événements
    def active(self) -> bool:
        return self.wp is not None

    def tick(self, cmd: tuple[float, float]) -> None:
        """Dead-reckon le waypoint commité d'un tick sous la commande exécutée (calibrée PASS :
        0.22 m @150 pas). Détecte atteinte/timeout — l'événement force une re-décision au replan."""
        self._global_ticks += 1              # compte TOUS les ticks (clé de jointure du log décisions)
        if self.wp is None:
            return
        dfwd = self.cfg.k_fwd * cmd[0]
        dyaw = self.cfg.k_yaw * cmd[1]
        vx_, vz_ = self.wp[0], self.wp[1] - dfwd
        c, s = math.cos(dyaw), math.sin(dyaw)
        self.wp = (c * vx_ - s * vz_, s * vx_ + c * vz_)
        self.leg_steps += 1
        if math.hypot(self.wp[0], self.wp[1]) < self.cfg.reach_m:
            self.n_reached += 1
            self._clear("reached")
        elif self.leg_steps > self.cfg.timeout_steps:
            self.n_timeouts += 1
            self._clear("timeout")

    def abort(self, reason: str) -> None:
        self.n_aborts += 1
        self._clear(reason)

    def note_first_target(self, first_target: str | None) -> None:
        """Pendant un leg : l'arbitrage du plan préfère-t-il l'AUTRE ressource ? On n'avorte que si
        la bascule PERSISTE abort_patience replans — G1 v0 : 18/27 abandons à 10 pas = bruit
        d'égalité 1-replan (scores saturés au cap) ; une vraie urgence, elle, persiste."""
        if self.wp is None or first_target is None:
            return
        if first_target != self.target_id:
            self._flip_streak += 1
            if self._flip_streak >= self.cfg.abort_patience:
                self.abort("target_change")
        else:
            self._flip_streak = 0

    def consume_event(self) -> str | None:
        ev, self._event = self._event, None
        return ev

    def _clear(self, event: str) -> None:
        if self.debug and self.wp is not None:
            print(f"[waypoint] leg END ({event}) après {self.leg_steps} pas "
                  f"(wp restant à {math.hypot(self.wp[0], self.wp[1]):.2f} m)", flush=True)
        self.wp = None
        self.target_id = None
        self.leg_steps = 0
        self._flip_streak = 0
        self._event = event

    # ------------------------------------------------------------------ décision
    def maybe_decide(self, target_id: str, target_pos: tuple[float, float],
                     retina: list[float],
                     drives: tuple[float, float, float] | None = None) -> dict | None:
        """À appeler à chaque replan SANS commitment. Décide si : première fois / cible changée /
        cible téléportée (respawn) / re-check périodique. Retourne le dict décision, ou None."""
        if drives is not None:
            self._drives = drives            # (énergie, soif, santé) — consommé par l'oracle-sprint
        self._replans_since_decision += 1
        jumped = (self._target_at_decision is not None and self.target_id == target_id
                  and math.hypot(target_pos[0] - self._target_at_decision[0],
                                 target_pos[1] - self._target_at_decision[1]) > 1.5)
        due = (self._target_at_decision is None or target_id != self.target_id or jumped
               or self._replans_since_decision >= self.cfg.recheck_every)
        if not due:
            return None
        return self.decide(target_id, target_pos, retina)

    def maybe_guard(self, retina: list[float]) -> dict | None:
        """À appeler au replan quand RIEN n'est visible (aucune cible) et pas de leg en cours.
        Cible virtuelle droit devant : ligne intruse de vert → décision évasive (commit tangent/anneau
        via decide, target_id='guard'). Ligne dégagée → None (croisière normale, zéro churn)."""
        if not self.guard_enable or self.active():
            return None
        greens = green_points(retina)
        if not greens:
            return None
        virtual = (0.0, self.guard_lookahead)
        _, intr = route_cost(virtual, virtual, greens, self.cfg)
        if intr <= 0.0:
            return None
        return self.decide("guard", virtual, retina)

    def decide(self, target_id: str, target_pos: tuple[float, float],
               retina: list[float]) -> dict:
        """Candidats direct + anneau, score lignes-vertes, hystérésis pro-direct, commit éventuel."""
        cfg = self.cfg
        greens = green_points(retina)
        # candidats INDEXÉS : 0 = DIRECT (wp = la cible), puis l'anneau, puis les TANGENTS (posés
        # au-delà des bords du nuage vert — seuls capables de dégager le 2ᵉ segment quand la cible
        # est loin derrière le gardien ; G1 v0 : best_wp≈direct sur toutes les décisions bloquées).
        cands: list[tuple[float, float]] = [target_pos]
        cands += [(cfg.ring_radius * math.sin(2.0 * math.pi * i / cfg.ring_n),
                   cfg.ring_radius * math.cos(2.0 * math.pi * i / cfg.ring_n))
                  for i in range(cfg.ring_n)]
        cands += tangent_candidates(greens, cfg.tangent_margin)
        feats = [candidate_features(w, target_pos, greens) for w in cands]
        if self.pain_critic is not None:
            # MODE DOULEUR APPRISE : coût = longueur + κ·Q_douleur — zéro marge verte codée-main.
            import torch as _torch
            with _torch.no_grad():
                pain = self.pain_critic.pain(_torch.tensor(feats, dtype=_torch.float32)) * 100.0
            scored = []
            for i, w in enumerate(cands):
                length = math.hypot(w[0], w[1]) + math.hypot(target_pos[0] - w[0], target_pos[1] - w[1])
                scored.append((length + self.pain_kappa_m * float(pain[i]), float(pain[i])))
        else:
            scored = [route_cost(w, target_pos, greens, cfg) for w in cands]
            # CORRECTION SPRINT (IC+TC) : adoucit la pénalité verte selon l'état — seuls les
            # candidats à intrusion>0 bougent, jamais en aggravation ; l'intrusion GÉOMÉTRIQUE
            # (scored[i][1]) reste intacte pour l'aval. Sans drives connus ou cible non-ressource
            # (guard) : g=0 ⇒ analytique pur.
            if (self.sprint_critic is not None and self._drives is not None and greens
                    and target_id in ("food", "water")):
                import torch as _torch
                from scripts.train_sprint_critic import sprint_inputs
                with _torch.no_grad():
                    _pain = self._sprint_pain.pain(_torch.tensor(feats, dtype=_torch.float32))
                    _p = self.sprint_critic.p(sprint_inputs(feats, self._drives, _pain.tolist()))
                scored = [(c - cfg.block_weight * float(_p[i]) * intr, intr)
                          for i, (c, intr) in enumerate(scored)]
        cost_direct, intr_direct = scored[0]
        best_i = min(range(1, len(cands)), key=lambda i: scored[i][0])
        # HYSTÉRÉSIS pro-direct : un détour ne s'engage que s'il bat nettement la ligne droite.
        chosen = best_i if scored[best_i][0] < cost_direct * (1.0 - cfg.hysteresis) else 0
        # EXPLORATION (collecte seulement, ε=0 en déploiement) : candidat UNIFORME, y compris les
        # mauvais — le critique doit voir « à travers le vert = mort » (corpus contrasté, anti
        # boucle auto-confirmante). L'hystérésis est court-circuitée : c'est le but.
        explored = self.explore_eps > 0.0 and self._rng.random() < self.explore_eps
        if explored:
            chosen = self._rng.randrange(len(cands))
        # ORACLE-SPRINT (sonde G-place, échafaudage) : bouffe bloquée + affamé + en bonne santé →
        # SPRINT direct malgré le vert. Le plafond qu'une valeur santé/drive-consciente peut viser.
        # v2 de la sonde (diagnostic seed-2 : 2 noyades à engouffrements=0 — la règle sprintait à
        # travers une zone MI-CHEMIN vers une bouffe LOINTAINE = traversée complète 223 ticks = mort,
        # alors qu'un détour existait). Sprint SEULEMENT si la cible est PROCHE (<3 m) : la douleur
        # est courte par construction (bouffe engouffrée ~55 ticks), la traversée lointaine reste au
        # scoreur normal (détour). Distinction sprint-rentable / traversée-suicidaire.
        if (self.oracle_sprint and not explored and target_id == "food"
                and self._drives is not None and intr_direct > 0.0
                and math.hypot(target_pos[0], target_pos[1]) < 3.0
                and self._drives[2] > 60.0 and self._drives[0] < 50.0):
            chosen = 0
        commit = chosen != 0
        self.n_decisions += 1
        self._replans_since_decision = 0
        self._target_at_decision = target_pos
        self.target_id = target_id
        self.leg_steps = 0
        self._flip_streak = 0
        self.wp = cands[chosen] if commit else None
        if commit:
            self.n_commits += 1
        rec = {"choice": "waypoint" if commit else "direct", "target": target_id,
               "target_pos": (round(target_pos[0], 2), round(target_pos[1], 2)),
               "wp": (round(self.wp[0], 2), round(self.wp[1], 2)) if commit else None,
               "cost_direct": round(cost_direct, 2), "cost_best_wp": round(scored[best_i][0], 2),
               "intr_direct": round(intr_direct, 2), "greens": len(greens), "explore": explored}
        if self._log_file is not None:
            # champs ADDITIFS drives/intr (2026-07-16) : les corpus post-Phase-A n'ont plus besoin
            # ni de la jointure tick (drives) ni de la reconstruction costs−longueur (intrusion).
            # ⚠️ en mode pain-critic, s[1] est la douleur, pas l'intrusion (piège documenté).
            self._log_file.write(json.dumps({
                "tick": self._global_ticks, "target": target_id, "chosen": chosen,
                "explore": explored, "n_greens": len(greens),
                "costs": [round(s[0], 3) for s in scored],
                "feats": [[round(v, 4) for v in f] for f in feats],
                "drives": [round(v, 2) for v in self._drives] if self._drives is not None else None,
                "intr": [round(s[1], 4) for s in scored],
            }) + "\n")
        if self.debug:
            print(f"[waypoint] DÉCISION {rec}", flush=True)
        return rec
