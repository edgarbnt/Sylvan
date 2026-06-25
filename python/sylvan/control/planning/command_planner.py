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

        discomfort = _urg(e_lvl) + _urg(t_lvl)  # predicted future discomfort at horizon end
        # Urgency-weighted proximity gradient = the VALIDATED A→B -min_dist attraction, but scaled by each
        # resource's CURRENT urgency → strong smooth pull toward the urgent resource (engages the WM's weak
        # right-turn side too), while a satisfied resource (urg→0) stops attracting. Keeps arbitration emergent.
        ue0 = (1.0 - max(0.0, min(1.0, e0))) ** cfg.urgency_exp
        ut0 = (1.0 - max(0.0, min(1.0, t0))) ** cfg.urgency_exp
        attract = ut0 * min_dw + (ue0 * min_df if has_food else 0.0)
        score = (
            -cfg.urgency_weight * discomfort
            - cfg.dist_weight * attract
            + cfg.heading_weight * (align_sum / float(h))
            - cfg.done_penalty * survival_pen
        )
        best = int(torch.argmax(score).item())
        vx, om = (float(v) for v in self._cmd_seqs[best, 0])
        return {
            "command": (vx, om),
            "food": (fx, fz) if has_food else None,
            "water": (wx, wz),
            "energy0": e0,
            "thirst0": t0,
            "pred_min_food": float(min_df[best]) if has_food else None,
            "pred_min_water": float(min_dw[best]),
            "reason": "plan_multi",
        }

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
