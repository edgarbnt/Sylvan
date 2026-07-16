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
from ...models.perception_head import RETINA_DIM

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
    surv_discount: float = 0.0             # ⚠️ NE PAS ACTIVER — négatif informatif (2026-07-04) : la survie
                                            # escomptée Σγ^t TÉLESCOPE à 1/(1−γ) pour tout plan immortel (vérifié :
                                            # 3 candidats → 2000.0 exact) → aveugle entre plans sûrs, ne résout PAS
                                            # la saturation (67-90% des replans ont les 2 ordres au cap, écarts 1-6
                                            # pts = bruit). Le vrai fix = RISQUE sous bruit MESURÉ : mort DOUCE par
                                            # leg (p_death = f(marge d'arrivée / σ_jitter)) → Π(1−p) discrimine les
                                            # plans sûrs par leurs marges, principiel. À designer à tête reposée.
    surv_turn_rate: float = 0.015          # rad/pas de virage imaginé phase-2 (hexapode ~25-50°/s ≈ 0.015-0.03
                                            # rad/pas à 30 Hz — prendre le bas = prudent). Env: SYLVAN_PLANNER_TURN_RATE
    # === ÉCHAFAUDAGE DE DONNÉES far-target (2026-07-06, RETIRABLE — cf docs/orbite_far_target_pur.md) ===
    # PROBLÈME : cible >5 m au-delà de l'horizon WM → dans le coût survie, tous les candidats survivent
    # (score saturé au cap) → argmax pique ω MAX → pivote sans translater → ORBITE (mesuré diag_orbit_scoring :
    # spread min_df <0.5 m, cause (a) : aucun candidat n'approche). FIX = ré-injecter dans le score survie le
    # terme de CAP `align_sum` (DÉJÀ calculé mais inutilisé ici) : récompense la trajectoire RÊVÉE qui POINTE
    # vers la ressource urgente (cos-bearing moyen, urgency-weighted, near-faded) → le candidat qui reoriente
    # gagne, la replanification glissante accumule le beeline (doc §5). MÊME mécanisme que heading_weight
    # (branche designed, validé A→B) — récompense l'OUTCOME (pointe-vers-bouffe), PAS l'ω brut (≠ le hack
    # omega-injection retiré le 2026-07-06). SCAFFOLD : amorce le corpus de poursuites lointaines pour le
    # critique appris, puis RETIRÉ (SYLVAN_PLANNER_FAR_ALIGN=0). Env: SYLVAN_PLANNER_FAR_ALIGN, SYLVAN_PLANNER_ALIGN_GAIN.
    far_align: bool = False                # échafaudage OFF par défaut (boucle finale = pure)
    # MODE du terme de cap (peaufinage efficacité 2026-07-06) : "mean" = cos-bearing MOYEN sur l'horizon
    # (récompense « rester pointé » → SPIRALE de tracking, l'agent re-vise à chaque replan) ; "end" = cap
    # moyen sur le DERNIER QUART de l'horizon (récompense « finir aligné+avancé » → favorise « tourne tôt puis
    # COMMIT droit » vs la spirale). Env: SYLVAN_PLANNER_ALIGN_MODE=mean|end.
    align_mode: str = "mean"
    align_gain: float = 60.0               # poids du terme de cap dans le score survie (unités ≈ pas ;
                                            # align_sum ∈ [−h, h], survie ∈ [0, cap] → ~60 tranche les ex-aequo far)
    # Candidats PIVOT (pur, élargit la recherche — doc §6) : virage SERRÉ court (vx=min bande, ω max) puis
    # cruise droit → reorient bref sans arc large, in-distribution (vx≈0 vrai-pivot serait HORS bande WM
    # 0.55-0.75 → rêve non fiable, hors scope). Env: SYLVAN_PLANNER_PIVOT.
    pivot: bool = False
    pivot_splits: tuple[int, ...] = (15, 25)   # longueurs du segment de reorient bref
    pivot_vx: float = 0.55                     # min de la bande babbling WM (le plus lent in-distribution)


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
    gamma: float = 0.0,
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
        # MODE ESCOMPTÉ (gamma>0) : valeur = Σ γ^t·vivant — jamais saturée (arriver tôt vaut
        # toujours plus), queue infinie γ^t/(1−γ) pour qui tient l'alternance → le CAP et la MARGE
        # deviennent inutiles (2 boutons en moins). Phase 1 ≈ vécue d'un bloc depuis t=0.
        if gamma > 0.0:
            val = (1.0 - gamma ** steps_p1.clamp(min=0.0)) / (1.0 - gamma) * alive.clamp(max=1.0)
            val = val + steps_p1 * (1.0 - alive)        # morts phase 1 : approx linéaire (négligeable)
        dist = (df_end if first_is_food else dw_end).clone()
        extra = (turn_f if first_is_food else turn_w).clone()   # virage du 1er leg seulement
        target_food = first_is_food
        for leg in range(max_legs):
            travel = dist / spd + extra
            t_die = torch.minimum(e, t).clamp(min=0.0) / drain   # pas avant que le pire drive meure
            died = (t_die < travel) & (live > 0.5)
            lived = torch.minimum(t_die, travel)
            if gamma > 0.0:
                val = val + live * (gamma ** time) * (1.0 - gamma ** lived.clamp(min=0.0)) / (1.0 - gamma)
            time = time + live * lived
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
        if gamma > 0.0:
            return val + live * (gamma ** time) / (1.0 - gamma)  # queue infinie des survivants
        time = time.clamp(max=cap)
        return time + margin_w * margin

    return sim(True), sim(False)          # (bouffe-d'abord, eau-d'abord) — le choix d'ordre
                                          # appartient au caller (règle d'incumbent, committment)


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
        _sd = os.environ.get("SYLVAN_PLANNER_SURV_DISCOUNT")
        if _sd not in (None, ""):
            self.cfg.surv_discount = float(_sd)
        _sm = os.environ.get("SYLVAN_PLANNER_SURV_MARGIN_W")
        if _sm not in (None, ""):
            self.cfg.surv_margin_weight = float(_sm)
        _tr = os.environ.get("SYLVAN_PLANNER_TURN_RATE")
        if _tr not in (None, ""):
            self.cfg.surv_turn_rate = float(_tr)
        _fa = os.environ.get("SYLVAN_PLANNER_FAR_ALIGN")  # échafaudage de cap far-target (RETIRABLE)
        if _fa not in (None, ""):
            self.cfg.far_align = _fa not in ("0", "false", "False")
        _ag = os.environ.get("SYLVAN_PLANNER_ALIGN_GAIN")
        if _ag not in (None, ""):
            self.cfg.align_gain = float(_ag)
        _am = os.environ.get("SYLVAN_PLANNER_ALIGN_MODE")  # mean (défaut) | end (anti-spirale)
        if _am not in (None, ""):
            self.cfg.align_mode = _am
        _pv = os.environ.get("SYLVAN_PLANNER_PIVOT")  # candidats pivot in-band (pur)
        if _pv not in (None, ""):
            self.cfg.pivot = _pv not in ("0", "false", "False")
        if self.cfg.cost_mode == "survival":
            print(f"[planner-cmd] COÛT SURVIE actif (multi-ressource) : score = pas-vécus simulés, "
                  f"drain={self.cfg.resource_drain} restore={self.cfg.resource_restore} "
                  f"cap={self.cfg.surv_horizon:.0f} margin_w={self.cfg.surv_margin_weight:.0f}")
        # CRITIQUE APPRIS (2026-07-05, Phase B) : remplace la queue analytique (alternance+drain)
        # quand SYLVAN_PLANNER_COST=critic. Gates offline passés : AUC .995, non-saturation .66,
        # swap .95 (vs hasard pour la valeur plate B0). Chargé une fois, gelé.
        self._critic = None
        _cp = os.environ.get("SYLVAN_PLANNER_CRITIC", "data/checkpoints/survival_critic_kin/critic_best.pt")
        # λ = poids de la CORRECTION apprise (mode "residual"). Réglable car il porte un RISQUE réel :
        # l'inné tranche entre candidats un écart d'action de ~1e-5, alors que l'erreur d'un réseau est
        # ~2e-4 (mesuré, diag_critic_aggregation) → une correction à pleine échelle peut NOYER le
        # classement fin que l'inné réussit, et donc casser ce qui marche. Le gate gratuit
        # diag_residual_lambda.py mesure exactement ça AVANT tout forage. Ne pas monter λ sans lui.
        self.critic_lambda = float(os.environ.get("SYLVAN_CRITIC_LAMBDA", "1.0"))
        if self.cfg.cost_mode in ("critic", "residual"):
            from scripts.train_survival_critic import SurvivalCritic
            _ck = torch.load(_cp, map_location="cpu", weights_only=False)
            self._critic = SurvivalCritic()
            self._critic.load_state_dict(_ck["state_dict"])
            self._critic.eval()
            for prm in self._critic.parameters():
                prm.requires_grad_(False)
            _lab = _ck.get("labels", "mc")
            if self.cfg.cost_mode == "residual":
                if _lab != "residual":
                    raise ValueError(
                        f"cost_mode=residual attend un critique entraîné en --labels residual, "
                        f"mais {_cp} porte labels={_lab!r}. Brancher un critique de VALEUR comme une "
                        f"CORRECTION serait un train≠déploiement silencieux (il prédit des pas-vécus "
                        f"absolus, pas un écart) → refusé.")
                print(f"[planner-cmd] NOTE = INNÉ + λ×CORRECTION APPRISE : {_cp} "
                      f"(λ={self.critic_lambda}, R² inné {_ck.get('r2_innate'):.3f} → "
                      f"{_ck.get('r2_corrected'):.3f} sur vies jamais vues)")
            else:
                print(f"[planner-cmd] CRITIQUE APPRIS actif : {_cp} (AUC {_ck.get('auc'):.3f}, "
                      f"non-sat {_ck.get('nonsat_ratio'):.2f}, swap {_ck.get('swap'):.2f}) — queue analytique remplacée")
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
        # (c) PIVOT candidates (pur, doc §6) : virage SERRÉ court (vx=min bande, ω extrême) puis cruise
        #     droit. Laisse la reorient s'exprimer à coût-translation minimal in-distribution → le beeline
        #     émerge via la replanification glissante. Actif seulement si cfg.pivot (élargit la recherche).
        if cfg.pivot:
            for plen in cfg.pivot_splits:
                if plen <= 0 or plen >= h:
                    continue
                for om1 in (-0.6, -0.45, 0.45, 0.6):
                    seqs.append([[cfg.pivot_vx, om1]] * plen + [[0.7, 0.0]] * (h - plen))
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
        slots_belief: "list[list[float] | None] | None" = None,
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
        # FIX v3 (2026-07-05) : les SOUVENIRS d'une ressource éclipsée deviennent le SLOT INITIAL du
        # rêve (slots0) → transportés PAR CANDIDAT comme la perception fraîche. (v1 = fantômes hors-vue ;
        # v2 = souvenir statique identique pour tous les candidats → gradient mort, sans-cible 76%.)
        _slots0 = None
        _vis = None
        _ret0 = None   # rétine t0 (144) — consommée aussi par le mur-vert (surv_mode) ; None si pas multi-slot
        if (getattr(self.world_model, "slot_resources", 1) > 1
                and os.environ.get("SYLVAN_MULTI_SLOT2", "1") != "0"):
            _ret0 = obs[self.world_model.proprio_dim:self.world_model.proprio_dim + RETINA_DIM].to(self.device)
            with torch.no_grad():
                _vis = self.world_model.slot_encoder.visibility(_ret0)
                _pos0 = self.world_model.slot_encoder.positions(_ret0)      # [R, 2]
            _slots0 = _pos0.clone()
            if slots_belief is not None:
                for _k in range(_slots0.shape[0]):
                    if float(_vis[_k]) <= 1e-3 and _k < len(slots_belief) and slots_belief[_k] is not None:
                        _slots0[_k, 0] = float(slots_belief[_k][0])
                        _slots0[_k, 1] = float(slots_belief[_k][1])
        out = self.world_model.rollout_open_loop(obs0, self._cmd_seqs, slots0=_slots0)
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
        if "slots" in out and os.environ.get("SYLVAN_MULTI_SLOT2", "1") != "0":
            # SLOT-2 (chantier pureté 2026-07-04) : bouffe ET eau lues du WM (slots requêtés-couleur)
            # → l'EAU QUITTE l'oracle radar-EMA. food/water_idx = assignation label-free du ckpt slot.
            # GATE DE VISIBILITÉ (leçon 1+1 : ressource hors de vue → le readout géométrique sortait un
            # fantôme ~10 m direction-bruit → replans poubelle). Invisible = ABSENTE (pas hallucinée),
            # SYMÉTRIQUE entre bouffe et eau (2026-07-07 : l'ancienne béquille "eau garde sa dernière
            # position, jamais None" est retirée — has_water gère son absence exactement comme has_food ;
            # A/B a montré qu'elle nuisait aux deux chemins vivants, survival +29% / critic +7x repas
            # sans elle). Le vrai comportement « ressource hors de vue » = CHERCHER, prochaine feature.
            fi = int(getattr(self.world_model, "food_idx", 0) or 0)
            wi = getattr(self.world_model, "water_idx", None)
            vis = _vis
            # Gate 3 états (mémoire spatiale 2026-07-04) : VISIBLE = lecture fraîche ;
            # ÉCLIPSÉE = souvenir dead-reckoné (MultiSlotMemory, ego-motion apprise — l'objet-
            # permanence ENTRE les replans, « retourner où j'ai vu » émerge du coût existant) ;
            # JAMAIS-VUE = absente, SYMÉTRIQUE bouffe/eau (les deux → None, pas hallucinées).
            def _bel(k: int) -> tuple[float, float] | None:
                if slots_belief is not None and k < len(slots_belief) and slots_belief[k] is not None:
                    return (float(slots_belief[k][0]), float(slots_belief[k][1]))
                return None
            if float(vis[fi]) > 1e-3 or _bel(fi) is not None:
                food = (float(out["slots"][0, 0, fi, 0]), float(out["slots"][0, 0, fi, 1]))
            else:
                food = None                          # jamais vue → absente (pas hallucinée)
            if wi is not None and (float(vis[int(wi)]) > 1e-3 or _bel(int(wi)) is not None):
                water = (float(out["slots"][0, 0, int(wi), 0]), float(out["slots"][0, 0, int(wi), 1]))
            else:
                water = None                         # jamais vue → absente (pas hallucinée), symétrique à la bouffe
        elif (getattr(self.world_model, "with_slot", False) and "slot" in out
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
        align_f = torch.zeros(n, device=self.device)   # cap-vers-bouffe (échafaudage far-target, branche survie)
        align_w = torch.zeros(n, device=self.device)   # cap-vers-eau
        align_f_late = torch.zeros(n, device=self.device)  # cap sur le DERNIER QUART (mode "end", anti-spirale)
        align_w_late = torch.zeros(n, device=self.device)
        late0 = int(0.75 * h)                            # seuil du dernier quart de l'horizon
        alive = torch.ones(n, device=self.device)
        survival_pen = torch.zeros(n, device=self.device)
        min_df = torch.full((n,), float("inf"), device=self.device)
        min_dw = torch.full((n,), float("inf"), device=self.device)
        has_food = food is not None
        has_water = water is not None                       # NIVEAU 1 : l'eau peut être absente, symétrique à la bouffe
        fx, fz = food if has_food else (0.0, 0.0)
        wx, wz = water if has_water else (0.0, 0.0)
        # PHASE 1 (suivi analytique des pulsions dans le rêve : drain/refill, drive_alive, steps_alive).
        # Exige les DEUX ressources : l'alternance food↔eau simulée n'a pas de sens sinon. Alimente le
        # coût survie ET le diagnostic sf/sw de la branche critique → garder "critic" ici (le retirer
        # laisserait drive_alive/steps_alive non initialisés et le diagnostic sf/sw serait faux).
        surv_mode = cfg.cost_mode in ("survival", "critic", "residual") and has_food and has_water
        # ⭐ DÉCOUVERTE 2026-07-08 : le critique ne décidait que ~3% des replans en monde ÉPARS.
        # `critic_mode` était gaté sur has_food AND has_water (via surv_mode) → il ne s'activait QUE
        # si les DEUX ressources étaient visibles SIMULTANÉMENT, ce qui est rare en épars. Les ~97%
        # restants retombaient SILENCIEUSEMENT sur le coût `designed` codé-main. La « boucle 100%
        # pure » était donc, en épars, pilotée par la formule codée-main la majorité du temps.
        # Le token drive-symétrique du critique porte pourtant DÉJÀ un flag « connu=0 » pour une
        # ressource absente (train_survival_critic.py:token()) : il a été ENTRAÎNÉ avec ce cas et n'a
        # jamais eu besoin des deux ressources à la fois. C'était un TROU DE GATING, pas une limite.
        # ⚠️ MAIS le lever DÉGRADE le forage (mesuré, 2 graines) : consommations 41/28 (gate d'origine)
        # → 20/12 (critique à 100%), et ce MALGRÉ 4 corrections principielles et vérifiées (corpus
        # apparié au déploiement, symétrie miroir imposée, exploration en collecte, pic de valeur
        # hors-axe résorbé). Diagnostic : l'horizon du rêve (~0.8 m) est minuscule devant les distances
        # (2-8 m) → les 33 candidats sont quasi ex-aequo (marge relative 0.003-0.005) → on demande au
        # critique de RANGER des options presque identiques, alors qu'il a été entraîné à PRÉDIRE une
        # valeur (MSE sur retours Monte-Carlo). Deux tâches différentes. Le coût designed s'en sort via
        # son terme de cap, qui donne une poussée directionnelle cohérente et cumulative entre replans.
        # → OPT-IN (défaut = gate d'origine) : la découverte reste reproductible sans imposer la
        # régression. Piste suivante : changer l'OBJECTIF (apprendre à CLASSER — préférences/TD — au
        # lieu de prédire une moyenne), pas les données. Voir diagnostics/diag_critic_landscape.py.
        critic_mode = cfg.cost_mode == "critic" and self._critic is not None and "slots" in out
        if os.environ.get("SYLVAN_CRITIC_ALWAYS", "0") == "0":
            critic_mode = critic_mode and has_food and has_water
        lvl_traj: list[torch.Tensor] = []
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
            if critic_mode:
                # e_lvl/t_lvl sont drainés SANS condition juste au-dessus → la trajectoire de niveaux
                # est valide même quand une seule ressource est connue (le refill de la ressource
                # absente ne s'applique simplement pas : sa pulsion draine, ce qui est la vérité).
                lvl_traj.append(torch.stack([e_lvl, t_lvl], dim=-1))       # [n, 2] par pas
            if has_food:
                df = torch.sqrt((x - fx) ** 2 + (z - fz) ** 2)
                min_df = torch.minimum(min_df, df)
                reached_f = (df < cfg.resource_reach).float()
                e_lvl = (e_lvl + reached_f * cfg.resource_restore).clamp(max=1.0)
                cos_f = ((fx - x) * torch.sin(yaw) + (fz - z) * torch.cos(yaw)) / (df + 1e-6)
                gate_f = (df / cfg.heading_far_gate).clamp(max=1.0)
                align_sum = align_sum + _urg(e_lvl) * cos_f * gate_f
                align_f = align_f + _urg(e_lvl) * cos_f * gate_f
                if t >= late0:
                    align_f_late = align_f_late + _urg(e_lvl) * cos_f * gate_f
            if has_water:                                # eau absente → pas de refill/attraction (la soif draine → mort)
                dw = torch.sqrt((x - wx) ** 2 + (z - wz) ** 2)
                min_dw = torch.minimum(min_dw, dw)
                reached_w = (dw < cfg.resource_reach).float()
                t_lvl = (t_lvl + reached_w * cfg.resource_restore).clamp(max=1.0)
                cos_w = ((wx - x) * torch.sin(yaw) + (wz - z) * torch.cos(yaw)) / (dw + 1e-6)
                gate_w = (dw / cfg.heading_far_gate).clamp(max=1.0)
                align_sum = align_sum + _urg(t_lvl) * cos_w * gate_w
                align_w = align_w + _urg(t_lvl) * cos_w * gate_w
                if t >= late0:
                    align_w_late = align_w_late + _urg(t_lvl) * cos_w * gate_w
            survival_pen = survival_pen + alive * done_prob[:, t]
            alive = alive * (1.0 - done_prob[:, t])

        if critic_mode and lvl_traj:
            # ── CRITIQUE APPRIS (Phase B) : score = moyenne sur l'horizon de V(état rêvé), où
            #    l'état rêvé = [niveaux simulés (drain/refill phase-1, dette analytique restante),
            #    coords slots TRANSPORTÉES par le WM]. La queue alternance+drain codée-main DISPARAÎT
            #    — le risque long-terme est ce que le critique a APPRIS de ses morts vécues.
            lv = torch.stack(lvl_traj, dim=1)                       # [n, h, 2]
            sl = out["slots"]                                       # [n, h, R, 2]
            fi = int(getattr(self.world_model, "food_idx", 0) or 0)
            wi_ = getattr(self.world_model, "water_idx", None)
            wi_ = int(wi_) if wi_ is not None else fi
            pos = torch.stack([sl[:, :, fi, :], sl[:, :, wi_, :]], dim=2)   # [n, h, 2, 2]
            # MÊME GATE 3-ÉTATS que la lecture slot-2 (bug v1 = trajectoires BRUTES → en épars,
            # ressource hors-vue = fantôme ~10 m → le critique évaluait des états imaginaires →
            # 0 repas/8 vies). VISIBLE → trajectoire rêvée ; ÉCLIPSÉE → souvenir (statique, approx) ;
            # JAMAIS-VUE → token « connu=0 » — le critique s'est ENTRAÎNÉ avec ce cas.
            known = torch.zeros(pos.shape[0], pos.shape[1], 2, device=pos.device)
            for j, k in enumerate((fi, wi_)):
                if float(vis[k]) > 1e-3 or _bel(k) is not None:
                    known[:, :, j] = 1.0            # visible OU souvenir : coords déjà dans les slots
                                                    # (t0=souvenir via slots0, transporté par candidat)
            d = pos.norm(dim=-1).clamp(min=1e-6)                    # [n, h, 2]
            # TOKEN — DOIT rester BYTE-IDENTIQUE à train_survival_critic.token() (train ≠ déploiement
            # sinon). SYMÉTRIE MIROIR PAR CONSTRUCTION (2026-07-08) : |sin(bearing)|, pas sin signé.
            # Gauche/droite est une symétrie exacte du monde → la VALEUR ne peut en dépendre. Avec le
            # sin signé, le critique avait appris un écart miroir jusqu'à 0.13 sur des états physiquement
            # identiques, ce qui plaçait son optimum de valeur HORS-AXE (~30°) → le planner gardait la
            # ressource de biais → ORBITE, dernier mètre jamais fermé. Voir le docstring de token().
            toks = torch.stack([lv, torch.where(known > 0.5, d.clamp(max=10.0) / 10.0, torch.ones_like(d)),
                                pos[..., 0].abs() / d * known, pos[..., 1] / d * known,
                                known], dim=-1)                     # [n, h, 2, TOK=5]
            if os.environ.get("SYLVAN_CRITIC_ORACLE") == "1":
                # ── SONDE « PLAFOND DE LA DISTILLATION » (2026-07-08, GRATUITE : zéro entraînement) ──
                # On remplace la valeur APPRISE par la valeur ANALYTIQUE (le coût survie codé-main, qui
                # EST déjà une fonction de valeur : « pas-vécus simulés »), calculée sur les MÊMES états
                # rêvés et consommée EXACTEMENT comme le critique (moyenne sur l'horizon).
                # QUESTION : si le critique copiait PARFAITEMENT le professeur analytique, foragerait-il
                # bien ? Ce n'est PAS évident : le planner utilise le critique en MOYENNANT sa note le
                # long du futur imaginé, alors que le coût survie, quand il travaille seul, score chaque
                # candidat depuis la FIN de son rêve. La façon de CONSOMMER la valeur diffère.
                #   - forage BON  → le plafond est atteignable → distiller le critique a du sens.
                #   - forage MAUVAIS → même une valeur PARFAITE échoue dans cette fente → le défaut est
                #     la façon dont le PLANNER interroge le critique, pas la valeur ni les données.
                #     → la distillation serait inutile, et on l'aura su sans l'entraîner.
                n_, h_ = lv.shape[0], lv.shape[1]
                pf, pw = pos[:, :, 0, :], pos[:, :, 1, :]              # [n, h, 2]
                df_ = pf.norm(dim=-1).reshape(-1)
                dw_ = pw.norm(dim=-1).reshape(-1)
                # temps de virage : bearing de chaque ressource dans le repère de l'état rêvé
                bf_ = torch.atan2(pf[..., 0], pf[..., 1]).abs().reshape(-1)
                bw_ = torch.atan2(pw[..., 0], pw[..., 1]).abs().reshape(-1)
                rate_ = max(cfg.surv_turn_rate, 1e-6)
                # ressource inconnue → la mettre HORS DE PORTÉE (le coût la traitera comme inatteignable),
                # cohérent avec le token « connu=0 » du critique appris.
                far = torch.full_like(df_, 1e4)
                kf = known[:, :, 0].reshape(-1) > 0.5
                kw = known[:, :, 1].reshape(-1) > 0.5
                df_ = torch.where(kf, df_, far)
                dw_ = torch.where(kw, dw_, far)
                s_f, s_w = _survival_extension(
                    df_, dw_, lv[:, :, 0].reshape(-1), lv[:, :, 1].reshape(-1),
                    torch.ones_like(df_), torch.zeros_like(df_),          # valeur DEPUIS cet état (frais)
                    float(torch.linalg.vector_norm(pf - pw, dim=-1).mean()),
                    cfg.resource_drain, cfg.resource_restore, cfg.nominal_speed,
                    cfg.surv_horizon, cfg.surv_margin_weight,
                    turn_f=bf_ / rate_, turn_w=bw_ / rate_, gamma=0.0,
                )
                vmap = torch.maximum(s_f, s_w).reshape(n_, h_) / cfg.surv_horizon   # ~[0,1] comme V
            else:
                with torch.no_grad():
                    vmap = self._critic.value(toks.reshape(-1, 2, toks.shape[-1])).reshape(lv.shape[0], lv.shape[1])
            score = vmap.mean(dim=1) * (1.0 - survival_pen.clamp(0.0, 1.0))
            best = int(torch.argmax(score).item())
            vx, om = (float(v) for v in self._cmd_seqs[best, 0])
            out_d: dict[str, object] = {
                "command": (vx, om),
                "food": (fx, fz) if has_food else None,
                "water": (wx, wz) if has_water else None,
                "energy0": e0, "thirst0": t0,
                "pred_min_food": float(min_df[best]) if has_food else None,
                "pred_min_water": float(min_dw[best]) if has_water else None,
                "reason": "plan_multi_critic",
            }
            if has_food and has_water:
                # DIAGNOSTIC SEUL (n'influence PAS `best`) : les scores analytiques par-ordre (sf/sw),
                # calculés en parallèle, pour que le corpus collecté en mode critique reste utilisable
                # par le gate offline NON-SATURATION (train_survival_critic.py) — sinon ce gate est
                # incalculable sur son propre corpus (ratio = NaN).
                # GATÉ sur has_food/has_water (2026-07-08, suite à la levée du gate du critique) :
                # sans ça, fx/fz ou wx/wz seraient des placeholders (0,0) quand une ressource est
                # inconnue → sf/sw calculés vers une position bidon, ce qui POLLUERAIT le gate.
                # Absence de "order_scores" = replan à perception partielle : attendu, simplement
                # non exploitable par ce gate-là.
                # ⚠️ N'inclut PAS le bonus far_align (contrairement à surv_mode ci-dessous) : sans
                # conséquence tant que la collecte se fait à FAR_ALIGN=0 (défaut boucle pure).
                df_end = torch.sqrt((x - fx) ** 2 + (z - fz) ** 2)
                dw_end = torch.sqrt((x - wx) ** 2 + (z - wz) ** 2)
                dist_fw = math.hypot(fx - wx, fz - wz)
                bear_f = torch.atan2((fx - x) * torch.cos(yaw) - (fz - z) * torch.sin(yaw),
                                     (fx - x) * torch.sin(yaw) + (fz - z) * torch.cos(yaw))
                bear_w = torch.atan2((wx - x) * torch.cos(yaw) - (wz - z) * torch.sin(yaw),
                                     (wx - x) * torch.sin(yaw) + (wz - z) * torch.cos(yaw))
                rate = max(cfg.surv_turn_rate, 1e-6)
                s_food_diag, s_water_diag = _survival_extension(
                    df_end, dw_end, e_lvl, t_lvl, drive_alive, steps_alive,
                    dist_fw, cfg.resource_drain, cfg.resource_restore, cfg.nominal_speed,
                    cfg.surv_horizon, cfg.surv_margin_weight,
                    turn_f=bear_f.abs() / rate, turn_w=bear_w.abs() / rate,
                    gamma=(1.0 - cfg.resource_drain) if cfg.surv_discount > 0 else 0.0,
                )
                fall = 1.0 - survival_pen.clamp(0.0, 1.0)
                out_d["order_scores"] = [float((s_food_diag * fall).max()),
                                         float((s_water_diag * fall).max())]
            if debug_scores:  # sondes offline (diag_critic_landscape) : le score PAR CANDIDAT
                out_d["scores"] = score.tolist()
                out_d["cand_cmd0"] = self._cmd_seqs[:, 0, :].tolist()
                out_d["min_df"] = min_df.tolist()   # distance MIN à la bouffe atteinte dans le rêve
                out_d["min_dw"] = min_dw.tolist()   # idem eau → mesure si l'argmax ferme sur la cible
                # VALEUR BRUTE PAR PAS [n, h] + facteur de chute, AVANT l'agrégation de la ligne 725.
                # Permet à diag_critic_aggregation de rejouer d'AUTRES agrégations (terminale, queue,
                # escomptée) sur les MÊMES rêves, sans ré-entraîner ni re-rouler le WM.
                out_d["vmap"] = vmap.tolist()
                out_d["fall"] = (1.0 - survival_pen.clamp(0.0, 1.0)).tolist()
            return out_d

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
            # ── QUEUE CONSCIENTE DU DANGER — ÉCHAFAUDAGE DIAGNOSTIC (2026-07-16) ─────────────────
            # La queue extrapole le trajet fin-de-rêve → ressource EN LIGNE DROITE. Si cette ligne
            # croise le danger PERÇU (slot vert, t0), le vrai trajet devra DÉTOURER → on rallonge la
            # distance effective. Donne le gradient « finis À CÔTÉ du danger » (ligne dégagée = trajet
            # court = gagne) que la répulsion seule ne peut pas produire : répulsion + attraction =
            # MINIMUM LOCAL au rebord (mesuré : repas 3-7 vs 15, à TOUS les poids et horizons — le
            # candidat qui glisse sur le côté n'était jamais récompensé). Classique champs-de-potentiel.
            # OPT-IN SYLVAN_HAZARD_DETOUR=0 (mètres de détour ; défaut OFF). ⚠️ r_h (étendue de la
            # zone) est un réglage main — la version APPRISE devra l'apprendre du vécu. df_eff/dw_eff
            # SÉPARÉS : le token du critique doit rester sur la distance BRUTE (train=déploiement).
            df_eff, dw_eff = df_end, dw_end
            _haz_i = getattr(self.world_model, "hazard_idx", None)
            _detour = float(os.environ.get("SYLVAN_HAZARD_DETOUR", "0"))
            if (_detour > 0.0 and _haz_i is not None and _slots0 is not None
                    and _vis is not None and float(_vis[int(_haz_i)]) > 1e-3):
                hx = float(_slots0[int(_haz_i), 0])
                hz = float(_slots0[int(_haz_i), 1])
                # CORRECTION DE BIAIS (2026-07-16, diagnostiquée sur données) : le slot lit « le vert
                # le plus proche » = le PILIER le plus proche = le BORD de la zone côté entité (anneau
                # à 0.7·r + rayon pilier ≈ 1.0 m du centre) — PAS le centre. Sans correction, le disque
                # protégé est décalé de ~1 m vers l'entité : la moitié ARRIÈRE de la vraie zone reste
                # nue (entrée 64% = aveugle, 6 morts malgré le gradient) et la répulsion bloque les
                # couloirs d'approche sains (repas 1). On prolonge donc le point perçu de CENTER_SHIFT
                # dans sa propre direction pour estimer le centre. ⚠️ Utilise la structure CONNUE de
                # l'échafaudage (géométrie des piliers) — la version APPRISE devra l'apprendre du vécu.
                _shift = float(os.environ.get("SYLVAN_HAZARD_CENTER_SHIFT", "1.0"))
                _pn = max(math.hypot(hx, hz), 1e-6)
                hx = hx * (1.0 + _shift / _pn)
                hz = hz * (1.0 + _shift / _pn)
                r_h = float(os.environ.get("SYLVAN_HAZARD_AVOID_RADIUS", "1.3"))

                def _cut_depth(qx: float, qz: float) -> torch.Tensor:
                    # PROFONDEUR DE COUPE [n] : de combien le segment fin-de-rêve (x,z) → ressource
                    # (qx,qz) s'enfonce dans le disque danger. GRADUÉE, pas binaire — leçon du 1er
                    # essai : à distance, TOUS les arcs de 0.8 m croisent la ligne (aucun ne peut se
                    # décaler de r_h en un arc) → un terme binaire est CONSTANT → zéro gradient. La
                    # profondeur, elle, diminue CONTINÛMENT quand l'arc se décale latéralement (~0.3 m
                    # de coupe en moins pour 0.5 m d'offset à 3 m) → préférence latérale cohérente à
                    # CHAQUE replan → s'intègre en contournement, comme le beeline s'intègre en droite.
                    vx_, vz_ = qx - x, qz - z
                    l2 = (vx_ ** 2 + vz_ ** 2).clamp(min=1e-9)
                    t_ = (((hx - x) * vx_ + (hz - z) * vz_) / l2).clamp(0.0, 1.0)
                    d_line = ((x + t_ * vx_ - hx) ** 2 + (z + t_ * vz_ - hz) ** 2).sqrt()
                    return (r_h - d_line).clamp(min=0.0)

                # _detour = ÉCHELLE (m de trajet en plus PAR m de coupe ; ~2-3 : contourner coûte
                # environ le double du décalage à gagner). Avant : montant forfaitaire binaire.
                _depth_f = _cut_depth(fx, fz)
                _depth_w = _cut_depth(wx, wz)
                df_eff = df_end + _depth_f * _detour
                dw_eff = dw_end + _depth_w * _detour
            else:
                _depth_f = _depth_w = None
            s_food, s_water = _survival_extension(
                df_eff, dw_eff, e_lvl, t_lvl, drive_alive, steps_alive,
                dist_fw, cfg.resource_drain, cfg.resource_restore, cfg.nominal_speed,
                cfg.surv_horizon, cfg.surv_margin_weight,
                turn_f=bear_f.abs() / rate, turn_w=bear_w.abs() / rate,
                gamma=(1.0 - cfg.resource_drain) if cfg.surv_discount > 0 else 0.0,
            )
            fall = 1.0 - survival_pen.clamp(0.0, 1.0)
            s_food = s_food * fall
            s_water = s_water * fall
            ic_only = torch.maximum(s_food, s_water).clone()   # l'inné SEUL (diagnostic λ)
            corr = torch.zeros_like(s_food)
            if cfg.cost_mode == "residual" and self._critic is not None:
                # ── NOTE = INNÉ + λ × CORRECTION APPRISE (2026-07-15) ──────────────────────────
                # L'inné (s_food/s_water) est EXACT : il tranche sans bruit l'écart minuscule entre
                # candidats. Mais il est OPTIMISTE ×1.7 (il rêve un trajet droit à vitesse nominale ;
                # la réalité erre et hésite). La correction apprend ce manque-à-vivre sur le VÉCU —
                # c'est le seul apport possible d'un critique ici. Voir residual_labels().
                # TOKEN — DOIT rester byte-identique à train_survival_critic.token() : même repère ego
                # (px = latéral, pz = avant), même |sin| (symétrie miroir imposée), même dist/10 capée.
                px_f = (fx - x) * torch.cos(yaw) - (fz - z) * torch.sin(yaw)
                pz_f = (fx - x) * torch.sin(yaw) + (fz - z) * torch.cos(yaw)
                px_w = (wx - x) * torch.cos(yaw) - (wz - z) * torch.sin(yaw)
                pz_w = (wx - x) * torch.sin(yaw) + (wz - z) * torch.cos(yaw)
                tok = torch.stack([
                    torch.stack([e_lvl, df_end.clamp(max=10.0) / 10.0, px_f.abs() / (df_end + 1e-6),
                                 pz_f / (df_end + 1e-6), torch.ones_like(e_lvl)], dim=-1),
                    torch.stack([t_lvl, dw_end.clamp(max=10.0) / 10.0, px_w.abs() / (dw_end + 1e-6),
                                 pz_w / (dw_end + 1e-6), torch.ones_like(t_lvl)], dim=-1),
                ], dim=1)                                          # [n, 2, TOK] — surv_mode ⇒ connu=1
                with torch.no_grad():
                    corr = self._critic.value(tok) * cfg.surv_horizon * self.critic_lambda
                # La correction est une propriété de l'ÉTAT ATTEINT, pas de l'ORDRE de visite : elle
                # s'ajoute aux deux ordres à l'identique et ne déplace donc PAS le choix bouffe/eau.
                # LIMITE ASSUMÉE et flaggée (§2) : le corpus ne porte aucun label d'ordre.
                s_food = s_food + corr
                s_water = s_water + corr

            # ── MUR-VERT — ÉCHAFAUDAGE (2026-07-16, redesign post-trace) : « ne marche pas vers le
            #    vert proche ». La famille ligne-vs-disque-estimé est morte au diagnostic (trace 88
            #    replans) : centre estimé depuis UN point perçu = ±1 m d'erreur pour une zone de 1.3 m,
            #    et gradient 6-37 pas vs spreads 300-500 → le terme suggérait sans jamais décider.
            #    ICI : zéro reconstruction — la rétine dit déjà QUELS secteurs sont verts et à quelle
            #    distance. On pénalise un candidat dont le DÉPLACEMENT pointe vers un secteur vert-
            #    proche (pondéré proximité²), au poids qui DÉCIDE (centaines de pas). C'est la forme
            #    exacte qu'une valeur apprise consommerait. OPT-IN SYLVAN_HAZARD_GREENWALL=0 (pas). ──
            _gw = float(os.environ.get("SYLVAN_HAZARD_GREENWALL", "0"))
            if _gw > 0.0 and _ret0 is not None:
                _r = _ret0.reshape(-1, 4)                                  # [36, 4] depth,R,G,B (t0)
                _is_green = (_r[:, 2] > _r[:, 1]) & (_r[:, 2] > _r[:, 3]) & \
                            ((_r[:, 1:4].amax(-1) - _r[:, 1:4].amin(-1)) > 0.15)
                _green_prox = _is_green.float() * (1.0 - _r[:, 0]).clamp(min=0.0) ** 2   # [36]
                if float(_green_prox.max()) > 1e-6:
                    n_ray = _green_prox.shape[0]
                    # bearing du DÉPLACEMENT de chaque candidat (frame t0, convention rétine :
                    # secteur k à 2πk/n, 0 = devant, positif vers la droite → atan2(x_right, z_fwd))
                    _beta = torch.atan2(x, z)                              # [n]
                    _sec = ((torch.round(_beta / (2.0 * math.pi / n_ray)).long()) % n_ray)
                    # lissage aux voisins : viser JUSTE À CÔTÉ d'un secteur vert compte à moitié
                    _gp = torch.maximum(_green_prox,
                                        0.5 * torch.maximum(_green_prox.roll(1), _green_prox.roll(-1)))
                    penalty_gw = _gw * _gp[_sec]                           # [n] en pas-de-survie
                    s_food = s_food - penalty_gw
                    s_water = s_water - penalty_gw

            # ── ÉVITE LE DANGER — ÉCHAFAUDAGE DIAGNOSTIC (2026-07-15), terme codé-main « fuis le vert » ──
            # BUT : PROUVER que le monde est soluble par la PERCEPTION (une entité qui VOIT le danger et le
            # contourne fait-elle tomber les morts-danger EN continuant à manger ?) → plafond atteignable,
            # AVANT d'investir dans la version APPRISE (critique-résidu). 🚨 ÉCHAFAUDAGE déclaré
            # (architecture.json), À RETIRER : la vraie évitement doit être APPRISE, pas codée (PRINCIPE §3).
            # OPT-IN, SYLVAN_HAZARD_AVOID=0 par défaut = OFF (zéro régression). Pénalise un candidat dont le
            # rêve s'APPROCHE du slot-danger (transporté par le WM), gaté par la visibilité du danger à t0
            # (pas de fantôme hors-vue). Comme corr, se soustrait aux DEUX ordres → ne biaise pas bouffe/eau.
            _haz_i = getattr(self.world_model, "hazard_idx", None)
            _haz_w = float(os.environ.get("SYLVAN_HAZARD_AVOID", "0"))
            if (_haz_w > 0.0 and _haz_i is not None and "slots" in out
                    and _vis is not None and float(_vis[int(_haz_i)]) > 1e-3):
                reach = float(os.environ.get("SYLVAN_HAZARD_AVOID_REACH", "1.0"))
                d_haz = out["slots"][:, :, int(_haz_i), :].norm(dim=-1)          # [n, h] dist au danger le long du rêve
                intrusion = (reach - d_haz.min(dim=1).values).clamp(min=0.0)      # [n] combien on entre dans la marge
                penalty = _haz_w * intrusion                                      # en pas-de-survie
                s_food = s_food - penalty
                s_water = s_water - penalty
            if cfg.far_align:
                # ÉCHAFAUDAGE far-target (RETIRABLE, doc §5/§7) : quand la cible est LOIN, tous les candidats
                # survivent (score saturé au cap) → départage vers celui dont la trajectoire RÊVÉE pointe la
                # ressource urgente (align_f/align_w = cos-bearing moyen, urgency-weighted, near-faded ∈ [−h,h]).
                # Restaure un cap consistant hors-horizon → la replanification glissante accumule le beeline.
                # Récompense l'OUTCOME (pointe-vers-bouffe), PAS l'ω brut. RETIRER (far_align=False) → boucle pure.
                # MODE "end" (anti-spirale) : cap sur le DERNIER QUART seulement, re-échelonné (×h/n_late ≈ 4) pour
                # garder la même plage que le sum-sur-l'horizon → favorise « tourne tôt puis COMMIT droit » (finit
                # aligné+avancé) plutôt que la spirale de tracking que récompense le cap MOYEN.
                if cfg.align_mode == "end":
                    n_late = max(h - late0, 1)
                    af = align_f_late * (h / n_late)
                    aw = align_w_late * (h / n_late)
                else:
                    af, aw = align_f, align_w
                s_food = s_food + cfg.align_gain * af
                s_water = s_water + cfg.align_gain * aw
            # COMMITTMENT (chantier 2026-07-04 soir) : près des égalités, le jitter des slots
            # (p90 0.7-1.5 m/replan) ré-ordonne les 2 plans en transit SANS changement des drives
            # (24/27 abandons-en-vue → autre ressource, 26/27 sans croisement d'urgence) → flicker
            # 44-46% + abandons 36-41%. Règle : le challenger ne détrône l'ordre INCUMBENT que s'il
            # le bat de plus de δ = bruit de score induit par le jitter (~75 pas pour 1.5 m à
            # 0.02 m/pas ; calibration initiale, jugée par le gate abandons<15% sans perte de survie).
            best_f, best_w = float(s_food.max()), float(s_water.max())
            delta = float(os.environ.get("SYLVAN_PLANNER_COMMIT_DELTA", "0.0"))
            inc = getattr(self, "_incumbent_target", None)
            if inc == "food":
                target_first = "food" if best_f >= best_w - delta else "water"
            elif inc == "water":
                target_first = "water" if best_w >= best_f - delta else "food"
            else:
                target_first = "food" if best_f >= best_w else "water"
            self._incumbent_target = target_first
            score = s_food if target_first == "food" else s_water
            best = int(torch.argmax(score).item())
            vx, om = (float(v) for v in self._cmd_seqs[best, 0])
            # TRACE DIAGNOSTIC (SYLVAN_HAZARD_DEBUG=1, lecture seule) : le détour PILOTE-t-il vraiment
            # l'argmax ? choisi≈min → le terme steer (chercher l'échec en aval : exécution/bascule) ;
            # choisi≫min → un autre terme l'écrase (suspect : margin_w=200 vs gradient ~40-160 pas).
            if _depth_f is not None and os.environ.get("SYLVAN_HAZARD_DEBUG", "0") == "1":
                _dch = _depth_f if target_first == "food" else _depth_w
                print(f"[hazdbg] tgt={target_first} depth_chosen={float(_dch[best]):.2f} "
                      f"min={float(_dch.min()):.2f} max={float(_dch.max()):.2f} "
                      f"haz=({hx:.1f},{hz:.1f}) d_haz={_pn + _shift:.1f} om={om:+.2f} "
                      f"score_spread={float(score.max() - score.min()):.0f}", flush=True)
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
                # INSTRUMENTATION COMMITTMENT (2026-07-04, post-KILL incumbent) : les 2 scores
                # d'ordre + le choix, loggés par le serveur → mesurer la VRAIE distribution des
                # écarts/bruit près des égalités avant tout nouveau fix (le δ=75 était deviné).
                "order_scores": [best_f, best_w],
                "first_target": target_first,
            }
            if debug_scores:  # sondes offline (diag_survcost_omega_gradient, diag_orbit_scoring) : par candidat
                out_d["scores"] = score.tolist()
                out_d["cand_cmd0"] = self._cmd_seqs[:, 0, :].tolist()
                out_d["min_df"] = min_df.tolist()   # distance MIN a la bouffe atteinte dans le reve
                out_d["df_end"] = df_end.tolist()   # distance a la bouffe en FIN de reve
                out_d["ic"] = ic_only.tolist()      # l'INNÉ seul, par candidat (gate λ)
                out_d["corr"] = corr.tolist()       # la CORRECTION apprise, par candidat (gate λ)
                out_d["min_dw"] = min_dw.tolist()
            return out_d

        discomfort = _urg(e_lvl) + _urg(t_lvl)  # predicted future discomfort at horizon end
        # Urgency-weighted proximity gradient = the VALIDATED A→B -min_dist attraction, but scaled by each
        # resource's CURRENT urgency → strong smooth pull toward the urgent resource (engages the WM's weak
        # right-turn side too), while a satisfied resource (urg→0) stops attracting. Keeps arbitration emergent.
        ue0 = (1.0 - max(0.0, min(1.0, e0))) ** cfg.urgency_exp
        ut0 = (1.0 - max(0.0, min(1.0, t0))) ** cfg.urgency_exp
        attract = (ut0 * min_dw if has_water else 0.0) + (ue0 * min_df if has_food else 0.0)
        # FORESIGHT de survie consciente du TRAJET (anti-myopie, défaut OFF). Pour chaque ressource : de combien le
        # niveau passerait SOUS zéro le temps de l'atteindre depuis la position imaginée en fin de rollout
        # (deficit = relu(dist/vitesse × drain − niveau_fin), en unités de niveau). Pénalise les candidats qui laissent
        # une ressource devenir fatalement inatteignable → l'agent y va AU BON MOMENT (tôt si loin). 0 → inchangé.
        if cfg.survival_weight != 0.0:
            spd = max(cfg.nominal_speed, 1e-4)
            deficit = torch.zeros(n, device=self.device)
            if has_water:
                dw_end = torch.sqrt((x - wx) ** 2 + (z - wz) ** 2)
                deficit = deficit + torch.relu(dw_end / spd * cfg.resource_drain - t_lvl)
            if has_food:
                df_end = torch.sqrt((x - fx) ** 2 + (z - fz) ** 2)
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
            "water": (wx, wz) if has_water else None,
            "energy0": e0,
            "thirst0": t0,
            "pred_min_food": float(min_df[best]) if has_food else None,
            "pred_min_water": float(min_dw[best]) if has_water else None,
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
