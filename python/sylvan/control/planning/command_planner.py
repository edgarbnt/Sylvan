"""Phase 5: Mode-2 planner over the command-space world model (CommandWorldModel).

NOT a neural network — a search. At each replan it dreams N candidate command
sequences forward in the (frozen) world model, scores each by how close the imagined
trajectory gets to the food, and returns the first command of the best candidate.
Godot executes it via the frozen CPG+residual base and sends a fresh real observation,
so the plan is recomputed from reality (receding-horizon MPC) and the WM error never
compounds beyond `horizon` — the grounded guard validated in Phase 4.

Design choices:
  * Candidates are piecewise-constant command SEQUENCES: the fine CONSTANT grid PLUS a
    2-SEGMENT grid ("turn for L1 steps, then cruise"). The 2-segment set lets the EFFICIENT
    motor plan ("orient, then go straight") EMERGE from the cost. Both segments are
    ≥40 steps so each stays in the WM's trained 40-80-step in-distribution regime.
  * The command space is 2-D, so we search an exhaustive GRID (deterministic).
  * Scoring uses the STRONG signal (predicted body-frame displacement, ~13% error) plus
    the t=0 radar to locate the food, and deliberately AVOIDS trusting the dreamed radar
    (the WM's weakest head). Food is static, so its position fixed at replan time is valid
    over the horizon. The cost = distance-to-food + DISTANCE-GATED heading-alignment + energy
    + survival. The heading term (2026-06-18) was NOT in the original design ("no how-to hint"),
    but a free A→B azimuth diagnostic proved that pure distance-min has a ~flat/noisy turn
    gradient for rear/far targets → the planner never engaged the U-turn (A→B ~37%, foraging
    starved 9/12). Rewarding "point at the food" (gated to fade near it, so min_dist drives the
    terminal approach without orbiting) fixed it: A→B 88-94% at 2-4.5m, foraging survival 610→990.
    Food still lives ONLY in the planner cost → BLUEPRINT §14 (body/WM/CPG food-agnostic) intact.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import torch

from ...models.command_wm import DISPLACEMENT_SCALE, CommandWorldModel

NUM_SECTORS = 12
RADAR_MAX_RANGE = 10.0
SECTOR_SIZE = 2.0 * math.pi / NUM_SECTORS


@dataclass(slots=True)
class CommandPlanConfig:
    horizon: int = 120                      # lookahead where the WM stays accurate (~0.15m err @120);
                                            # split into two ≥40-step segments (both in-distribution)
    # Fine CONSTANT-command grid (the v1 candidates: a single held arc over the horizon).
    # vx band tracks the HEXAPOD's clean regime (~0.55-0.75, where it holds heading + turns both
    # ways); MUST stay within the WM-collection babbling range (SYLVAN_WM_VX_MIN/MAX) → in-distribution.
    vx_grid: tuple[float, ...] = (0.55, 0.65, 0.75)
    omega_grid: tuple[float, ...] = (-0.6, -0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6)
    # Coarser per-segment grid for the 2-SEGMENT candidates ("turn for segment_split, then cruise").
    seg_splits: tuple[int, ...] = (40,)     # turn-segment length(s) for the 2-segment candidates. NOTE: adding
                                            # 80 was TESTED (A→B azimuth diag 2026-06-18) and did NOT fix the
                                            # engagement tail — overall flat (37→40%) and it broke a front azimuth.
                                            # Root cause is the near-zero/noisy turn gradient (knife-edge), not the
                                            # candidate set. Kept at (40,) = original behaviour. seg 2 = horizon-split.
    seg_vx_grid: tuple[float, ...] = (0.6, 0.75)
    seg_omega_grid: tuple[float, ...] = (-0.6, -0.3, 0.0, 0.3, 0.6)
    heading_weight: float = 2.0             # A→B ENGAGEMENT fix (2026-06-18): reward the imagined trajectory for
                                            # POINTING AT the food (mean cos-bearing over the horizon). Without it the
                                            # turn gradient is ~flat for rear/far targets (min_dist identical for all
                                            # candidates that can't reach within the horizon) → knife-edge, never
                                            # engages the U-turn. Stays dominated by min_dist when a candidate actually
                                            # reaches food. Softens the planner's "no how-to hint" purity but food still
                                            # lives ONLY in the planner cost (BLUEPRINT §14 intact). Env: SYLVAN_PLANNER_HEADING_W.
    heading_far_gate: float = 2.0           # metres: the heading reward FADES to 0 within this distance of the food
                                            # (per-step align × clamp(dist/gate,0,1)). Far → full pull (engage the
                                            # U-turn); near → min_dist drives a clean straight-in approach. WITHOUT
                                            # this, a constant heading weight ORBITS during terminal approach (6m A→B
                                            # diag 2026-06-18: w=2 spiralled, dmin stuck 1-2m, never closed the last metre).
    energy_weight: float = 2.0              # tie-breaker for 5a; raised for foraging (5b)
    done_penalty: float = 3.0               # penalise candidates the WM predicts will fall
    reach_radius: float = 0.6               # within this predicted distance, treat food as reached
    no_food_command: tuple[float, float] = (0.7, 0.0)  # straight ahead (clean hexapod cruise) when no food sensed
    # === 2ᵉ PULSION — arbitrage homéostatique émergent (2026-06-18, étage 1, WM inchangé) ===
    # Actif UNIQUEMENT quand un radar EAU est fourni. Le chemin mono-ressource (bouffe seule) reste
    # le coût validé ci-dessus (A→B + foraging préservés). Ici la priorité ÉMERGE de l'urgence :
    # on minimise l'inconfort futur prédit Σ_r urgency(level_r_fin), urgency = (1-level)^exp (convexe →
    # une ressource critique domine). L'attraction/cap vers chaque ressource est pondérée par SON urgence
    # courante → quand on a soif, l'eau attire ; rassasié, son terme s'éteint. Dynamique des niveaux =
    # ANALYTIQUE (drain linéaire + refill au contact), donc le WM n'a PAS besoin de prédire la soif.
    urgency_weight: float = 6.0             # poids du terme d'inconfort futur (doit dominer). Env: SYLVAN_PLANNER_URGENCY_W
    dist_weight: float = 1.0               # poids de l'attraction-proximité (le -min_dist validé A→B), pondérée par urgence
    urgency_exp: float = 2.0               # convexité de l'urgence (2 = le critique coûte bien plus que le modéré)
    resource_drain: float = 0.0016         # drain normalisé/pas (≈ homeostasis passive_drain 0.15 / max 100)
    resource_restore: float = 0.5          # refill normalisé au contact (≈ energy_per_food 40-50 / 100)
    resource_reach: float = 1.0            # mètres : distance de capture imaginée (≈ eat/drink_radius)
    survival_weight: float = 300.0         # FORESIGHT de survie consciente du TRAJET — BANKÉ (validé multi-seed : multi-pulsions
                                            # survie médiane ~2000→~2300, 3/3 seeds positifs). N'agit QU'EN multi-pulsions (le
                                            # single-drive plan_wm_slot/single-resource est INTACT). Pénalise un candidat si une
                                            # ressource passerait SOUS zéro AVANT qu'on l'atteigne depuis la position imaginée en fin
                                            # de rollout. Mettre SYLVAN_PLANNER_SURVIVAL_W=0 pour désactiver. Env: SYLVAN_PLANNER_SURVIVAL_W
    nominal_speed: float = 0.02            # m/pas d'approche imaginée (calibre temps-pour-atteindre). Env: SYLVAN_PLANNER_SPEED
    # === COÛT SURVIE refill-aware (Mode-2, gate B0 2026-07-02) — remplace le coût designed multi-ressource ===
    # Score = PAS-VÉCUS SIMULÉS (le BUT lui-même, pas un proxy) : phase 1 = rollout WM (drain+refill au contact,
    # mort-des-drives persistante), phase 2 = extension analytique PAR ALTERNANCE au-delà de l'horizon WM
    # (aller à une ressource, refill, alterner ; on simule les DEUX ordres bouffe/eau-d'abord et on garde le max —
    # exactement le plan_rollout validé par diag_slot2_value_arbitration, 0.90-0.96 sur les morts-décision).
    # Tie-break quand tout le monde survit au cap : la MARGE d'arrivée (pire niveau à l'arrivée) → gradient lisse
    # partout, pas de knife-edge → engagement/committment émergent (anti-hésitation H0). ÉCHAFAUDAGE FLAGGÉ :
    # la continuation alternée + drain/refill analytiques restent codés-main (3ᵉ verrou) ; la version pure = tête
    # drive-dynamics APPRISE. Env: SYLVAN_PLANNER_COST=survival|designed.
    cost_mode: str = "designed"            # "survival" active le coût ci-dessus (multi-ressource seulement)
    surv_horizon: float = 3000.0           # cap de la simulation (pas planner ≈ pas Godot ≈ cap épisode)
    surv_margin_weight: float = 200.0      # poids du tie-break marge (unités: pas). Env: SYLVAN_PLANNER_SURV_MARGIN_W
    surv_turn_rate: float = 0.015          # rad/pas de virage imaginé phase-2 (hexapode ~25-50°/s ≈ 0.015-0.03
                                            # rad/pas à 30 Hz — prendre le bas = prudent). Env: SYLVAN_PLANNER_TURN_RATE


def food_xz_from_radar(radar: list[float] | torch.Tensor) -> tuple[float, float] | None:
    """Reconstruct the nearest food's position in the CURRENT body frame from the egocentric
    radar. Returns (x_right, z_fwd) in metres, or None if no food in range.

    Matches perception.gd::food_radar: sector = floor((bearing+PI)/sector_size) % 12, bearing
    measured as atan2(dir·right, dir·fwd) with 0 = straight ahead; proximity = 1 - dist/range.
    Body-frame axes here match the WM displacement convention: +z = forward, +x = right.
    """
    vals = [float(v) for v in radar]
    n = len(vals)
    if n == 0:
        return None
    # A1: infer sector count from the radar length → handles the WM's 12-sector radar AND the
    # finer 36-sector localisation radar. Finer n → finer bearing (±5° at 36 vs ±15° at 12).
    sector_size = 2.0 * math.pi / n
    best = max(range(n), key=lambda s: vals[s])
    if vals[best] <= 0.0:
        return None
    # Sector center bearing (inverse of the floor-binning).
    bearing = (best + 0.5) * sector_size - math.pi
    dist = (1.0 - vals[best]) * RADAR_MAX_RANGE
    return (dist * math.sin(bearing), dist * math.cos(bearing))  # (x_right, z_fwd)


def _survival_extension(
    df_end: torch.Tensor,
    dw_end: torch.Tensor,
    e_end: torch.Tensor,
    t_end: torch.Tensor,
    alive: torch.Tensor,
    steps_p1: torch.Tensor,
    dist_fw: float,
    drain: float,
    restore: float,
    spd: float,
    cap: float,
    margin_w: float,
    turn_f: torch.Tensor | None = None,
    turn_w: torch.Tensor | None = None,
) -> torch.Tensor:
    """Phase 2 du coût survie : depuis la FIN du rollout WM de chaque candidat (distances df/dw aux
    ressources, niveaux e/t, masque vivant, pas déjà vécus), simule analytiquement la suite « aller à
    une ressource → refill → alterner » jusqu'à mort ou `cap` pas. Les DEUX ordres (bouffe d'abord /
    eau d'abord) sont simulés, on garde le meilleur (= plan_rollout du gate B0). Score [n] =
    pas-vécus simulés + margin_w × niveau-à-la-PREMIÈRE-arrivée.

    Leçons de la sonde post-KILL (diag_survcost_omega_gradient, 2026-07-03) :
    - la marge = PREMIÈRE arrivée SEULEMENT (le plan ne contrôle que son premier leg ; un min sur
      toute l'alternance convergeait vers le régime permanent → score PLAT à distance, std=0.000) ;
    - le premier leg paie le TEMPS DE VIRAGE (turn_f/turn_w = pas nécessaires pour se tourner vers
      la ressource depuis le cap de fin d'arc — tourner coûte du temps, le temps coûte de la survie).
    Tout est en pas-planner (≈ pas Godot) et niveaux normalisés [0,1]."""
    drain = max(drain, 1e-9)
    spd = max(spd, 1e-6)
    leg_fw = max(dist_fw, 0.0) / spd                    # pas de trajet entre les deux ressources
    max_legs = int(cap / max(leg_fw, 1.0)) + 2
    zeros = torch.zeros_like(df_end)
    turn_f = zeros if turn_f is None else turn_f
    turn_w = zeros if turn_w is None else turn_w

    def sim(first_is_food: bool) -> torch.Tensor:
        e, t = e_end.clone(), t_end.clone()
        live = alive.clone()
        time = steps_p1.clone()
        margin = torch.zeros_like(e)                    # niveau à la 1ʳᵉ arrivée (0 si jamais atteinte)
        dist = (df_end if first_is_food else dw_end).clone()
        extra = (turn_f if first_is_food else turn_w).clone()   # virage du 1er leg seulement
        target_food = first_is_food
        for leg in range(max_legs):
            travel = dist / spd + extra
            t_die = torch.minimum(e, t).clamp(min=0.0) / drain   # pas avant que le pire drive meure
            died = (t_die < travel) & (live > 0.5)
            time = time + live * torch.minimum(t_die, travel)
            e = (e - travel * drain).clamp(min=0.0)
            t = (t - travel * drain).clamp(min=0.0)
            if leg == 0:
                arrive = torch.minimum(e, t)                     # niveau à l'arrivée (pré-refill)
                margin = torch.where((~died) & (live > 0.5), arrive, margin)
            live = live * (~died).float()
            if target_food:
                e = torch.where(live > 0.5, (e + restore).clamp(max=1.0), e)
            else:
                t = torch.where(live > 0.5, (t + restore).clamp(max=1.0), t)
            target_food = not target_food
            dist = torch.full_like(dist, max(dist_fw, 0.0))
            extra = zeros
            if not bool((live > 0.5).any()):
                break
        time = time.clamp(max=cap)
        return time + margin_w * margin

    return torch.maximum(sim(True), sim(False))


class CommandPlanner:
    def __init__(
        self,
        world_model: CommandWorldModel,
        cfg: CommandPlanConfig | None = None,
        *,
        device: str = "cpu",
    ) -> None:
        self.cfg = cfg or CommandPlanConfig()
        _hw = os.environ.get("SYLVAN_PLANNER_HEADING_W")  # cheap A→B tuning without re-editing
        if _hw not in (None, ""):
            self.cfg.heading_weight = float(_hw)
        _uw = os.environ.get("SYLVAN_PLANNER_URGENCY_W")  # cheap arbitration tuning (2ᵉ pulsion)
        if _uw not in (None, ""):
            self.cfg.urgency_weight = float(_uw)
        _sw = os.environ.get("SYLVAN_PLANNER_SURVIVAL_W")  # foresight de survie consciente du trajet (anti-myopie)
        if _sw not in (None, ""):
            self.cfg.survival_weight = float(_sw)
        _sp = os.environ.get("SYLVAN_PLANNER_SPEED")
        if _sp not in (None, ""):
            self.cfg.nominal_speed = float(_sp)
        _cm = os.environ.get("SYLVAN_PLANNER_COST")  # "survival" = coût pas-vécus simulés (gate B0)
        if _cm not in (None, ""):
            self.cfg.cost_mode = _cm
        _dr = os.environ.get("SYLVAN_PLANNER_DRAIN")  # drain normalisé/pas — caler sur le régime réel
        if _dr not in (None, ""):                     # (éco de vie 0.05 → 0.0005 ; défaut historique 0.0016)
            self.cfg.resource_drain = float(_dr)
        _rs = os.environ.get("SYLVAN_PLANNER_RESTORE")  # refill normalisé (energy_per_food 40 → 0.4)
        if _rs not in (None, ""):
            self.cfg.resource_restore = float(_rs)
        _sh = os.environ.get("SYLVAN_PLANNER_SURV_H")
        if _sh not in (None, ""):
            self.cfg.surv_horizon = float(_sh)
        _sm = os.environ.get("SYLVAN_PLANNER_SURV_MARGIN_W")
        if _sm not in (None, ""):
            self.cfg.surv_margin_weight = float(_sm)
        _tr = os.environ.get("SYLVAN_PLANNER_TURN_RATE")
        if _tr not in (None, ""):
            self.cfg.surv_turn_rate = float(_tr)
        if self.cfg.cost_mode == "survival":
            print(f"[planner-cmd] COÛT SURVIE actif (multi-ressource) : score = pas-vécus simulés, "
                  f"drain={self.cfg.resource_drain} restore={self.cfg.resource_restore} "
                  f"cap={self.cfg.surv_horizon:.0f} margin_w={self.cfg.surv_margin_weight:.0f}")
        self.device = torch.device(device)
        self.world_model = world_model.to(self.device).eval()
        for p in self.world_model.parameters():
            p.requires_grad_(False)
        self._cmd_seqs = self._build_candidates()  # [N, H, 2] piecewise-constant command sequences

    def _build_candidates(self) -> torch.Tensor:
        cfg = self.cfg
        h = cfg.horizon
        seqs: list[list[list[float]]] = []
        # (a) Fine CONSTANT-command candidates: one held arc over the whole horizon.
        for vx in cfg.vx_grid:
            for om in cfg.omega_grid:
                seqs.append([[vx, om]] * h)
        # (b) 2-SEGMENT candidates: "turn/manoeuvre for `split` steps, then cruise". Lets the
        #     beeline (orient → go straight) emerge from the unchanged cost. Skip degenerate
        #     same-command pairs (already covered by the constant grid). Multiple split lengths so
        #     a LARGER reorientation can be represented (engages rear/side targets).
        seen: set[tuple] = set()
        for split in cfg.seg_splits:
            if split <= 0 or split >= h:
                continue
            for vx1 in cfg.seg_vx_grid:
                for om1 in cfg.seg_omega_grid:
                    for vx2 in cfg.seg_vx_grid:
                        for om2 in cfg.seg_omega_grid:
                            if (vx1, om1) == (vx2, om2):
                                continue
                            key = (split, vx1, om1, vx2, om2)
                            if key in seen:
                                continue
                            seen.add(key)
                            seqs.append([[vx1, om1]] * split + [[vx2, om2]] * (h - split))
        return torch.tensor(seqs, dtype=torch.float32, device=self.device)  # [N, H, 2]

    @torch.no_grad()
    def plan(
        self,
        obs: torch.Tensor,
        radar: list[float],
        water_radar: list[float] | None = None,
        energy: float | None = None,
        thirst: float | None = None,
        override_pos: bool = False,
        food_override: tuple[float, float] | None = None,
        water_override: tuple[float, float] | None = None,
        slot_belief: "list[float] | None" = None,
        debug_scores: bool = False,
    ) -> dict[str, object]:
        """obs [obs_dim] = proprio ++ radar ++ energy(normalised) (the WM's encoder input).
        radar = the REAL 12-sector food radar. water_radar/energy/thirst are the 2ᵉ-pulsion planner-only
        inputs (None on single-resource runs → the original validated food-only cost).
        RÉTINE (étage 1, override_pos=True) : la position food/water vient de la TÊTE de perception APPRISE
        (food_override/water_override en frame agent, None = pas vue) au lieu de l'oracle food_xz_from_radar.
        Returns the chosen command + diagnostics."""
        cfg = self.cfg
        if override_pos:
            food = food_override
            water = water_override
        else:
            food = food_xz_from_radar(radar)
            water = food_xz_from_radar(water_radar) if water_radar else None

        n = self._cmd_seqs.shape[0]
        h = cfg.horizon

        # ── WM-SLOT (object-centric PUR, 2026-06-25) : la perception ET la permanence de l'objet vivent DANS le
        #    WM (out["slot"]). Le WM perçoit l'objet via son slot_encoder (rétine de l'obs) et le transporte par sa
        #    PROPRE displacement → coordonnée ego par pas. PLUS de coordonnée codée-main ni de boucle trigo de pose
        #    (l'échafaudage est dissous). Single-resource (l'eau reste gérée par le chemin multi-ressource ci-dessous). ──
        if getattr(self.world_model, "with_slot", False) and water is None:
            obs0 = obs.to(self.device).reshape(1, -1).expand(n, -1).contiguous()
            # MÉMOIRE SPATIALE (Task 3): when slot_belief is provided, override the t0 slot with the
            # server's persisted belief (dead-reckoned across replans). When None → encode from obs0 as before.
            if slot_belief is not None:
                _s0 = torch.tensor(slot_belief, dtype=torch.float32).reshape(1, 2)
                out = self.world_model.rollout_open_loop(obs0, self._cmd_seqs, slot0=_s0)
            else:
                out = self.world_model.rollout_open_loop(obs0, self._cmd_seqs)
            slot = out["slot"]                                    # [n, h, 2] (x_right, z_fwd) ego par pas
            dist = torch.linalg.vector_norm(slot, dim=-1)         # [n, h]
            min_dist = dist.min(dim=1).values                     # [n]
            done_prob = torch.sigmoid(out["predicted_done_logits"])
            energy_pred = out["predicted_next_obs"][..., -1].clamp(0.0, 1.0)
            cos_brg = slot[..., 1] / (dist + 1e-6)                # z_fwd / dist = cos(bearing)
            far_gate = (dist / cfg.heading_far_gate).clamp(max=1.0)
            mean_align = (cos_brg * far_gate).mean(dim=1)         # [n]
            alive = torch.ones(n, device=self.device)
            survival_pen = torch.zeros(n, device=self.device)
            for t in range(h):
                survival_pen = survival_pen + alive * done_prob[:, t]
                alive = alive * (1.0 - done_prob[:, t])
            score = (
                -min_dist
                + cfg.heading_weight * mean_align
                + cfg.energy_weight * energy_pred[:, -1]
                - cfg.done_penalty * survival_pen
            )
            best = int(torch.argmax(score).item())
            vx, om = (float(v) for v in self._cmd_seqs[best, 0])
            fx, fz = float(slot[best, 0, 0]), float(slot[best, 0, 1])
            return {
                "command": (vx, om),
                "food": (fx, fz),
                "food_dist": math.hypot(fx, fz),
                "pred_min_dist": float(min_dist[best]),
                "reason": "plan_wm_slot",
            }

        # ── SINGLE-RESOURCE (no water sensed): the original VALIDATED cost, untouched (A→B + foraging). ──
        if water is None:
            if food is None:
                vx, om = cfg.no_food_command
                return {"command": (vx, om), "food": None, "reason": "no_food"}
            obs0 = obs.to(self.device).reshape(1, -1).expand(n, -1).contiguous()
            out = self.world_model.rollout_open_loop(obs0, self._cmd_seqs)
            disp = out["predicted_displacement"] / DISPLACEMENT_SCALE
            done_prob = torch.sigmoid(out["predicted_done_logits"])
            energy_pred = out["predicted_next_obs"][..., -1].clamp(0.0, 1.0)
            fx, fz = food
            x = torch.zeros(n, device=self.device)
            z = torch.zeros(n, device=self.device)
            yaw = torch.zeros(n, device=self.device)
            min_dist = torch.full((n,), float("inf"), device=self.device)
            align_sum = torch.zeros(n, device=self.device)
            alive = torch.ones(n, device=self.device)
            survival_pen = torch.zeros(n, device=self.device)
            for t in range(h):
                d_fwd, d_lat, d_yaw = disp[:, t, 0], disp[:, t, 1], disp[:, t, 2]
                s, c = torch.sin(yaw), torch.cos(yaw)
                x = x + d_fwd * s + d_lat * c
                z = z + d_fwd * c - d_lat * s
                yaw = yaw + d_yaw
                dist = torch.sqrt((x - fx) ** 2 + (z - fz) ** 2)
                min_dist = torch.minimum(min_dist, dist)
                cos_brg = ((fx - x) * torch.sin(yaw) + (fz - z) * torch.cos(yaw)) / (dist + 1e-6)
                far_gate = (dist / cfg.heading_far_gate).clamp(max=1.0)
                align_sum = align_sum + cos_brg * far_gate
                survival_pen = survival_pen + alive * done_prob[:, t]
                alive = alive * (1.0 - done_prob[:, t])
            mean_align = align_sum / float(h)
            score = (
                -min_dist
                + cfg.heading_weight * mean_align
                + cfg.energy_weight * energy_pred[:, -1]
                - cfg.done_penalty * survival_pen
            )
            best = int(torch.argmax(score).item())
            vx, om = (float(v) for v in self._cmd_seqs[best, 0])
            return {
                "command": (vx, om),
                "food": (fx, fz),
                "food_dist": math.hypot(fx, fz),
                "pred_min_dist": float(min_dist[best]),
                "reason": "plan",
            }

        # ── MULTI-RESOURCE: emergent homeostatic ARBITRATION (food optional + water). ──
        # Imagine each candidate; track energy/thirst ANALYTICALLY (linear drain + refill on reach).
        # Score = minimise predicted future discomfort Σ urgency(level_end), + urgency-weighted heading
        # toward each resource (engages the turn toward whatever is URGENT), - fall penalty. The priority
        # (go to food vs water) EMERGES from the convex urgency + geometry — no fixed preference.
        obs0 = obs.to(self.device).reshape(1, -1).expand(n, -1).contiguous()
        out = self.world_model.rollout_open_loop(obs0, self._cmd_seqs)
        disp = out["predicted_displacement"] / DISPLACEMENT_SCALE
        done_prob = torch.sigmoid(out["predicted_done_logits"])

        # SLOT (perception promue, 2026-06-25) : si le WM porte un slot object-centric, localiser la BOUFFE
        # via le slot APPRIS (out["slot"][:,0] = position ego perçue à t0, identique pour tous les candidats)
        # au lieu de l'oracle radar `food`. L'EAU reste planner-only (étage 1, pas dans le WM). Backward-compat :
        # WM sans slot → garde l'oracle `food`. Le single-drive (plan_wm_slot / single-resource) n'est PAS touché.
        # SONDE-INTERVENTION (2026-07-04, cause racine f96991c) : le slot est HORS-DISTRIBUTION en
        # multi-ressource (positions bouffe fantômes, méd 2.3-4.3 m en 1+1) → SYLVAN_MULTI_FOOD_SLOT=0
        # débranche cet override et la bouffe retombe sur le MÊME pipeline que l'eau (EMA radar,
        # ~0.85 m). Flag de SONDE/échafaudage — le fix pur = slot multi-ressource ré-entraîné.
        if (getattr(self.world_model, "with_slot", False) and "slot" in out
                and os.environ.get("SYLVAN_MULTI_FOOD_SLOT", "1") != "0"):
            food = (float(out["slot"][0, 0, 0]), float(out["slot"][0, 0, 1]))

        e0 = float(obs[-1]) if energy is None else float(energy)
        t0 = 1.0 if thirst is None else float(thirst)
        e_lvl = torch.full((n,), max(0.0, min(1.0, e0)), device=self.device)
        t_lvl = torch.full((n,), max(0.0, min(1.0, t0)), device=self.device)

        x = torch.zeros(n, device=self.device)
        z = torch.zeros(n, device=self.device)
        yaw = torch.zeros(n, device=self.device)
        align_sum = torch.zeros(n, device=self.device)
        alive = torch.ones(n, device=self.device)
        survival_pen = torch.zeros(n, device=self.device)
        min_df = torch.full((n,), float("inf"), device=self.device)
        min_dw = torch.full((n,), float("inf"), device=self.device)
        has_food = food is not None
        fx, fz = food if has_food else (0.0, 0.0)
        wx, wz = water
        # COÛT SURVIE (gate B0) : actif si demandé ET bouffe visible (sans bouffe, pas d'alternance
        # simulable → on retombe sur le coût designed validé, qui pousse déjà vers l'eau/explore).
        surv_mode = cfg.cost_mode == "survival" and has_food
        drive_alive = torch.ones(n, device=self.device)     # mort-des-drives PERSISTANTE (pas de résurrection)
        steps_alive = torch.zeros(n, device=self.device)

        def _urg(level: torch.Tensor) -> torch.Tensor:
            return (1.0 - level).clamp(min=0.0) ** cfg.urgency_exp

        for t in range(h):
            d_fwd, d_lat, d_yaw = disp[:, t, 0], disp[:, t, 1], disp[:, t, 2]
            s, c = torch.sin(yaw), torch.cos(yaw)
            x = x + d_fwd * s + d_lat * c
            z = z + d_fwd * c - d_lat * s
            yaw = yaw + d_yaw
            # passive drain (both resources)
            e_lvl = (e_lvl - cfg.resource_drain).clamp(min=0.0)
            t_lvl = (t_lvl - cfg.resource_drain).clamp(min=0.0)
            if surv_mode:
                drive_alive = drive_alive * ((e_lvl > 0.0) & (t_lvl > 0.0)).float()
                steps_alive = steps_alive + drive_alive
            if has_food:
                df = torch.sqrt((x - fx) ** 2 + (z - fz) ** 2)
                min_df = torch.minimum(min_df, df)
                reached_f = (df < cfg.resource_reach).float()
                e_lvl = (e_lvl + reached_f * cfg.resource_restore).clamp(max=1.0)
                cos_f = ((fx - x) * torch.sin(yaw) + (fz - z) * torch.cos(yaw)) / (df + 1e-6)
                gate_f = (df / cfg.heading_far_gate).clamp(max=1.0)
                align_sum = align_sum + _urg(e_lvl) * cos_f * gate_f
            dw = torch.sqrt((x - wx) ** 2 + (z - wz) ** 2)
            min_dw = torch.minimum(min_dw, dw)
            reached_w = (dw < cfg.resource_reach).float()
            t_lvl = (t_lvl + reached_w * cfg.resource_restore).clamp(max=1.0)
            cos_w = ((wx - x) * torch.sin(yaw) + (wz - z) * torch.cos(yaw)) / (dw + 1e-6)
            gate_w = (dw / cfg.heading_far_gate).clamp(max=1.0)
            align_sum = align_sum + _urg(t_lvl) * cos_w * gate_w
            survival_pen = survival_pen + alive * done_prob[:, t]
            alive = alive * (1.0 - done_prob[:, t])

        if surv_mode:
            # ── COÛT SURVIE refill-aware (gate B0) : score = pas-vécus simulés (phase 1 WM + phase 2
            #    alternance analytique), × probabilité de ne pas chuter (la chute est aussi une mort),
            #    + marge d'arrivée en tie-break. Remplace ENTIÈREMENT le mélange designed
            #    urgence/attract/heading/survival_weight (les poids à tuner disparaissent). ──
            df_end = torch.sqrt((x - fx) ** 2 + (z - fz) ** 2)
            dw_end = torch.sqrt((x - wx) ** 2 + (z - wz) ** 2)
            dist_fw = math.hypot(fx - wx, fz - wz)
            # Temps de virage du 1er leg : bearing de la ressource vu du CAP DE FIN D'ARC (sonde
            # post-KILL : sans lui, un candidat dos-à-la-cible mais 10 cm plus près battait celui
            # qui s'était tourné vers elle — l'orientation gagnée en phase 1 était jetée).
            bear_f = torch.atan2((fx - x) * torch.cos(yaw) - (fz - z) * torch.sin(yaw),
                                 (fx - x) * torch.sin(yaw) + (fz - z) * torch.cos(yaw))
            bear_w = torch.atan2((wx - x) * torch.cos(yaw) - (wz - z) * torch.sin(yaw),
                                 (wx - x) * torch.sin(yaw) + (wz - z) * torch.cos(yaw))
            rate = max(cfg.surv_turn_rate, 1e-6)
            score = _survival_extension(
                df_end, dw_end, e_lvl, t_lvl, drive_alive, steps_alive,
                dist_fw, cfg.resource_drain, cfg.resource_restore, cfg.nominal_speed,
                cfg.surv_horizon, cfg.surv_margin_weight,
                turn_f=bear_f.abs() / rate, turn_w=bear_w.abs() / rate,
            ) * (1.0 - survival_pen.clamp(0.0, 1.0))
            best = int(torch.argmax(score).item())
            vx, om = (float(v) for v in self._cmd_seqs[best, 0])
            out_d: dict[str, object] = {
                "command": (vx, om),
                "food": (fx, fz),
                "water": (wx, wz),
                "energy0": e0,
                "thirst0": t0,
                "pred_min_food": float(min_df[best]),
                "pred_min_water": float(min_dw[best]),
                "pred_steps_alive": float(score[best]),
                "reason": "plan_multi_surv",
            }
            if debug_scores:  # sondes offline (diag_survcost_omega_gradient) : score par candidat
                out_d["scores"] = score.tolist()
                out_d["cand_cmd0"] = self._cmd_seqs[:, 0, :].tolist()
            return out_d

        discomfort = _urg(e_lvl) + _urg(t_lvl)  # predicted future discomfort at horizon end
        # Urgency-weighted proximity gradient = the VALIDATED A→B -min_dist attraction, but scaled by each
        # resource's CURRENT urgency → strong smooth pull toward the urgent resource (engages the WM's weak
        # right-turn side too), while a satisfied resource (urg→0) stops attracting. Keeps arbitration emergent.
        ue0 = (1.0 - max(0.0, min(1.0, e0))) ** cfg.urgency_exp
        ut0 = (1.0 - max(0.0, min(1.0, t0))) ** cfg.urgency_exp
        attract = ut0 * min_dw + (ue0 * min_df if has_food else 0.0)
        # FORESIGHT de survie consciente du TRAJET (anti-myopie, défaut OFF). Pour chaque ressource : de combien le
        # niveau passerait SOUS zéro le temps de l'atteindre depuis la position imaginée en fin de rollout
        # (deficit = relu(dist/vitesse × drain − niveau_fin), en unités de niveau). Pénalise les candidats qui laissent
        # une ressource devenir fatalement inatteignable → l'agent y va AU BON MOMENT (tôt si loin). 0 → inchangé.
        if cfg.survival_weight != 0.0:
            spd = max(cfg.nominal_speed, 1e-4)
            df_end = torch.sqrt((x - fx) ** 2 + (z - fz) ** 2)
            dw_end = torch.sqrt((x - wx) ** 2 + (z - wz) ** 2)
            deficit = torch.relu(dw_end / spd * cfg.resource_drain - t_lvl)
            if has_food:
                deficit = deficit + torch.relu(df_end / spd * cfg.resource_drain - e_lvl)
            survival_def = cfg.survival_weight * deficit
        else:
            survival_def = 0.0
        score = (
            -cfg.urgency_weight * discomfort
            - cfg.dist_weight * attract
            + cfg.heading_weight * (align_sum / float(h))
            - cfg.done_penalty * survival_pen
            - survival_def
        )
        best = int(torch.argmax(score).item())
        vx, om = (float(v) for v in self._cmd_seqs[best, 0])
        out_d = {
            "command": (vx, om),
            "food": (fx, fz) if has_food else None,
            "water": (wx, wz),
            "energy0": e0,
            "thirst0": t0,
            "pred_min_food": float(min_df[best]) if has_food else None,
            "pred_min_water": float(min_dw[best]),
            "reason": "plan_multi",
        }
        if debug_scores:
            out_d["scores"] = score.tolist()
            out_d["cand_cmd0"] = self._cmd_seqs[:, 0, :].tolist()
        return out_d

    def _retina_mirror_map(self, obs_dim: int):
        """Carte miroir gauche↔droite de l'obs WM-rétine (277 = proprio132 ++ rétine144(36×4) ++ énergie1).
        Réutilise la carte proprio de ppo/symmetry ; rétine = inverse l'ordre azimutal des rayons (ray k ↔
        (n−k)%n, validé : bouffe-droite miroitée → bouffe-gauche) ; énergie inchangée. Mise en cache."""
        cached = getattr(self, "_mir_cache", None)
        if cached is not None and cached[0] == obs_dim:
            return cached[1], cached[2]
        from ..ppo.symmetry import _build_proprio_maps
        pperm, psign = _build_proprio_maps()
        pd = self.world_model.proprio_dim
        perm = list(range(obs_dim)); sign = [1.0] * obs_dim
        for i in range(min(pd, len(pperm))):
            perm[i] = pperm[i]; sign[i] = psign[i]
        n_ray = (obs_dim - pd - 1) // 4
        for k in range(n_ray):
            src = (n_ray - k) % n_ray
            for j in range(4):
                perm[pd + 4 * k + j] = pd + 4 * src + j
        P = torch.tensor(perm, device=self.device); S = torch.tensor(sign, dtype=torch.float32, device=self.device)
        self._mir_cache = (obs_dim, P, S)
        return P, S

    @torch.no_grad()
    def plan_latent(self, obs: torch.Tensor, value_head, *, energy: float | None = None,
                    orient_head=None) -> dict[str, object]:
        """🅑-PUR (planning DANS LE LATENT, 2026-06-19) — score les candidats par la VALEUR du latent RÊVÉ
        (tête apprise V = 'va manger bientôt'), MOYENNÉE sur l'horizon, − pénalité de chute. AUCUNE coordonnée :
        ne touche NI food_xz, NI radar, NI min_dist, NI heading. La bouffe n'existe que dans ce que le WM a
        APPRIS à percevoir (rétine→latent) et ce que la tête de valeur en LIT. C'est le coût JEPA-pur.
        `obs` [obs_dim] = proprio ++ rétine ++ énergie (l'entrée encodeur du WM)."""
        n = self._cmd_seqs.shape[0]
        h = self.cfg.horizon
        obs0 = obs.to(self.device).reshape(1, -1).expand(n, -1).contiguous()
        out = self.world_model.rollout_open_loop(obs0, self._cmd_seqs)
        # LOGIT (pré-sigmoid, pas la proba qui SATURE), agrégé sur l'horizon. AGRÉGAT DÉPEND DE LA VALUE :
        #   • value TEACHER-FORCED (value_head_food) → MAX (le PIC où le rêve touche ; .mean récompensait l'orbite,
        #     rang 0.75 vs 0.38). • value RÊVE-MULTIPAS (value_head_food_dream, 2026-06-21) → MEAN bat MAX (calibrée
        #     à toutes les profondeurs, la moyenne DÉBRUITE : close rang 0.08 vs 0.33). Switch SYLVAN_VALUE_AGG.
        _agg = os.environ.get("SYLVAN_VALUE_AGG", "max")

        def _aggregate(logits):  # logits [N, H] → [N]
            return logits.mean(dim=1) if _agg == "mean" else logits.max(dim=1).values

        Vlogit = value_head.logit(out["predicted_latents"])                 # [N, H]
        L = _aggregate(Vlogit)                                              # [N] — LIT le latent rêvé, rien d'autre
        # ENGAGE (2026-06-21) — « quelque chose d'engageant est-il perçu/imaginable ? » = pic de proba-repas du
        # MEILLEUR futur (max sur candidats × horizon). Gate offline diag_search_trigger : front-proche ~0.98 vs
        # derrière ~0.17 (trou net) → seuil fiable pour déclencher la PERCEPTION ACTIVE (CHERCHER) côté serveur.
        # JEPA-pur : ne lit que la value du latent rêvé, aucune coordonnée.
        engage = float(torch.sigmoid(Vlogit).max())
        # SYMÉTRISATION gauche↔droite (fix de l'ASYMÉTRIE du WM, validé 2026-06-19 : le rêve tourne ~1.4× plus
        # fort à gauche qu'à droite → sans ça l'agent dérive à gauche et meurt). On évalue CHAQUE candidat aussi
        # dans le monde MIROIR (obs miroitée + omega négé) et on moyenne → le biais directionnel du WM s'annule,
        # seul le signal 'où est la bouffe' survit. (Toujours JEPA-pur : aucune coordonnée ; la symétrie est une
        # propriété du corps, pas une position de ressource.)
        try:
            P, S = self._retina_mirror_map(obs.shape[-1])
            obs_mir = (obs.to(self.device)[P] * S).reshape(1, -1).expand(n, -1).contiguous()
            seqs_mir = self._cmd_seqs.clone(); seqs_mir[..., 1] = -seqs_mir[..., 1]
            out_m = self.world_model.rollout_open_loop(obs_mir, seqs_mir)
            L = 0.5 * (L + _aggregate(value_head.logit(out_m["predicted_latents"])))
        except Exception:
            pass
        # TERME DE CAP LATENT (2026-06-21) — l'analogue JEPA-pur du heading_weight coordonnées : récompense le
        # rêve qui finit ORIENTÉ vers la cible (orient_head.ahead lit le bearing PERÇU dans le latent, +1=devant).
        # Comble le trou de credit-assignment de la value-de-proximité pour les cibles ARRIÈRE (orienter ne réduit
        # pas la proximité dans l'horizon → value plate → jamais de demi-tour). GATÉ par (1−V) : s'éteint quand un
        # repas devient atteignable (V haut) → la proximité pilote le close (équivalent de heading_far_gate).
        # AUCUNE coordonnée : ahead vient du latent (perception apprise). Off par défaut (SYLVAN_ORIENT_W=0).
        _ow = float(os.environ.get("SYLVAN_ORIENT_W", "0.0"))
        if orient_head is not None and _ow > 0.0:
            ahead = orient_head.ahead(out["predicted_latents"])           # [N, H] ∈ [-1,1], +1 = cible devant
            val = torch.sigmoid(value_head.logit(out["predicted_latents"]))  # [N, H] proximité (gate)
            orient_term = (ahead * (1.0 - val)).mean(dim=1)               # [N] : cap moyen, atténué près du repas
            L = L + _ow * orient_term
        done_prob = torch.sigmoid(out["predicted_done_logits"])
        alive = torch.ones(n, device=self.device)
        survival_pen = torch.zeros(n, device=self.device)
        for t in range(h):
            survival_pen = survival_pen + alive * done_prob[:, t]
            alive = alive * (1.0 - done_prob[:, t])
        score = L - self.cfg.done_penalty * survival_pen               # valeur future + cap latent, AUCUNE coordonnée
        # GARDE-FOU 🅑 : ce chemin ne doit JAMAIS lire de position de ressource. (Vérif structurelle : aucune
        # des variables food/water/min_dist/heading n'est référencée ici — le score ne dépend que de V et done.)
        best = int(torch.argmax(score).item())
        vx, om = (float(v) for v in self._cmd_seqs[best, 0])
        return {"command": (vx, om), "reason": "plan_latent", "best_value": float(score[best]),
                "engage": engage}
