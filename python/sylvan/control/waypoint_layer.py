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

import math
import os
from dataclasses import dataclass

N_RAY = 36
RETINA_RANGE_M = 10.0    # MAX_RANGE (perception.gd)


@dataclass
class WaypointConfig:
    ring_n: int = 8              # waypoints sur l'anneau (spec : 6-8)
    ring_radius: float = 2.5     # m autour de l'entité (spec : R ≈ 2-3)
    reach_m: float = 0.9         # seuil d'atteinte du waypoint
    timeout_steps: int = 180     # pas Godot avant re-décision forcée (spec : 150-200)
    green_margin: float = 0.8    # ρ : marge de sécurité autour d'un point vert perçu (m)
    block_weight: float = 25.0   # W : mètres de trajet équivalents par mètre d'intrusion (décisif)
    hysteresis: float = 0.15     # un wp ne bat le direct que s'il coûte 15 % de moins (anti-dither)
    recheck_every: int = 5       # en mode direct, re-décision périodique tous les K replans
    k_fwd: float = 0.0144        # m/tick par unité de vx  (calibré : 0.8 × dt_eff 0.018)
    k_yaw: float = 0.027         # rad/tick par unité de ω (calibré : 1.5 × 0.018 ; ω>0 = droite)

    @classmethod
    def from_env(cls) -> "WaypointConfig":
        """Surcharges d'environnement SYLVAN_WP_* (mêmes clés que les champs, en majuscules)."""
        c = cls()
        for name, cast in (("ring_n", int), ("ring_radius", float), ("reach_m", float),
                           ("timeout_steps", int), ("green_margin", float), ("block_weight", float),
                           ("hysteresis", float), ("recheck_every", int),
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
        self.reset()

    def reset(self) -> None:
        self.wp: tuple[float, float] | None = None
        self.target_id: str | None = None
        self._target_at_decision: tuple[float, float] | None = None
        self.leg_steps = 0
        self._replans_since_decision = 0
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
        self._event = event

    # ------------------------------------------------------------------ décision
    def maybe_decide(self, target_id: str, target_pos: tuple[float, float],
                     retina: list[float]) -> dict | None:
        """À appeler à chaque replan SANS commitment. Décide si : première fois / cible changée /
        cible téléportée (respawn) / re-check périodique. Retourne le dict décision, ou None."""
        self._replans_since_decision += 1
        jumped = (self._target_at_decision is not None and self.target_id == target_id
                  and math.hypot(target_pos[0] - self._target_at_decision[0],
                                 target_pos[1] - self._target_at_decision[1]) > 1.5)
        due = (self._target_at_decision is None or target_id != self.target_id or jumped
               or self._replans_since_decision >= self.cfg.recheck_every)
        if not due:
            return None
        return self.decide(target_id, target_pos, retina)

    def decide(self, target_id: str, target_pos: tuple[float, float],
               retina: list[float]) -> dict:
        """Candidats direct + anneau, score lignes-vertes, hystérésis pro-direct, commit éventuel."""
        cfg = self.cfg
        greens = green_points(retina)
        cost_direct, intr_direct = route_cost(target_pos, target_pos, greens, cfg)
        best_wp: tuple[float, float] | None = None
        best_cost = float("inf")
        for i in range(cfg.ring_n):
            th = 2.0 * math.pi * i / cfg.ring_n
            wp = (cfg.ring_radius * math.sin(th), cfg.ring_radius * math.cos(th))
            cost, _ = route_cost(wp, target_pos, greens, cfg)
            if cost < best_cost:
                best_cost, best_wp = cost, wp
        # HYSTÉRÉSIS pro-direct : un détour ne s'engage que s'il bat nettement la ligne droite.
        commit = best_wp is not None and best_cost < cost_direct * (1.0 - cfg.hysteresis)
        self.n_decisions += 1
        self._replans_since_decision = 0
        self._target_at_decision = target_pos
        self.target_id = target_id
        self.leg_steps = 0
        self.wp = best_wp if commit else None
        if commit:
            self.n_commits += 1
        rec = {"choice": "waypoint" if commit else "direct", "target": target_id,
               "target_pos": (round(target_pos[0], 2), round(target_pos[1], 2)),
               "wp": (round(self.wp[0], 2), round(self.wp[1], 2)) if commit else None,
               "cost_direct": round(cost_direct, 2), "cost_best_wp": round(best_cost, 2),
               "intr_direct": round(intr_direct, 2), "greens": len(greens)}
        if self.debug:
            print(f"[waypoint] DÉCISION {rec}", flush=True)
        return rec
