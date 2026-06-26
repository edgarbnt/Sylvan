"""Phase 5: serve the command-space planner + frozen residual as ONE server.

Godot sends {proprio, vision=REAL food radar, energy} each tick. The server:
  1. every K ticks, re-plans (CommandPlanner over the CommandWorldModel) → command (vx, omega);
  2. runs the frozen residual2 policy on [proprio ++ command-in-vision-slot] → 12-D joint action;
  3. returns {"action":[12], "command":[vx, omega]}.
Godot applies the action AND set_cpg_command(vx, omega) (the CPG steers by construction in-engine).
This keeps ONE network round-trip per tick and the existing wire protocol (+ a "command" field).

Usage (from python/):
    python -m scripts.serve_planner_command \
        --wm ../data/checkpoints/wm_command_hex_v1/wm_best.pt \
        --residual ../data/checkpoints/hexapod_v2/policy_best.pt \
        --host 127.0.0.1 --port 6051 --replan-every 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socketserver
import threading
from pathlib import Path

# ---------------------------------------------------------------------------
# OCCLUSION MASK — MÉMOIRE SPATIALE (Task 4)
# ---------------------------------------------------------------------------
# Simule un CÔNE FRONTAL de perception : les rayons hors du cône sont mis à
# zéro (depth→1.0, RGB→0.0), ce qui efface la saillance dans le slot_encoder
# et force SlotMemory à maintenir l'objet par dead-reckoning (mémoire pure).
#
# NOTE HONNÊTETÉ : occluder la rétine côté serveur présente une rétine
# hors-distribution au WM (entraîné sur 360°).  C'est une approximation
# acceptable pour la GATE (l'objet disparaît correctement de la perception),
# mais un cône frontal « de production » nécessiterait un retrain WM sur
# données avec cone — travail différé, hors scope Task 4.
# ---------------------------------------------------------------------------
_N_RAYS = 36          # rayons 0..35, espacement 10° (ray 0 = avant)
_CHANNELS = 4         # [depth, R, G, B] par rayon → 36×4 = 144 floats

# Lire la valeur au démarrage du process (une seule fois) pour éviter les
# appels os.environ dans la boucle chaude.
_OCCLUDE_FOV_DEG: float = float(os.environ.get("SYLVAN_OCCLUDE_FOV_DEG", "360.0"))


def occlude_retina(retina: list[float], fov_deg: float) -> list[float]:
    """Retourne une COPIE de la rétine (144 floats) dans laquelle tous les
    rayons dont la distance angulaire par rapport à l'avant (ray 0) dépasse
    ``fov_deg / 2`` sont mis à zéro :
        depth  → 1.0  (objet à l'infini)
        R,G,B  → 0.0  (aucune saturation → saillance 0 dans slot_encoder)

    ``fov_deg >= 360`` → identité exacte (aucun changement, non-régression).

    Le rayon k est à l'angle ``k × 10°`` ; distance angulaire par rapport à
    l'avant = ``min(k*10, 360 - k*10)`` degrés.
    L'entrée n'est PAS mutée (copie défensive).
    """
    if fov_deg >= 360.0:
        return list(retina)          # identité — non-régression byte-identique
    half = fov_deg / 2.0
    out = list(retina)               # copie
    for k in range(_N_RAYS):
        angle = k * 10.0
        dist = min(angle, 360.0 - angle)
        if dist > half:
            base = k * _CHANNELS
            out[base] = 1.0          # depth = loin (rien à voir)
            out[base + 1] = 0.0      # R
            out[base + 2] = 0.0      # G
            out[base + 3] = 0.0      # B
    return out

import torch

from sylvan.config import SylvanConfig
from sylvan.control.planning.command_planner import CommandPlanner, CommandPlanConfig
from sylvan.control.ppo.policy import GaussianActorCritic
from sylvan.models.command_wm import CommandWorldModel
from sylvan.models.perception_head import RetinaPerceptionHead, RETINA_DIM
from sylvan.training.checkpointing import load_checkpoint
from sylvan.hud.live_writer import write_live

VISION_DIM = 12  # the residual's vision slot: [vx, omega, 0*10] in CPG mode (matches training)


class _PlannerService:
    def __init__(self, wm_ckpt: Path, residual_ckpt: Path, cfg: CommandPlanConfig, replan_every: int,
                 retina_head_ckpt: Path | None = None, value_head_ckpt: Path | None = None,
                 slot_head_ckpt: Path | None = None,
                 egomotion_head_ckpt: Path | None = None,
                 use_slot_memory: bool = False) -> None:
        import os as _os
        torch.set_num_threads(int(_os.environ.get("SYLVAN_PLANNER_THREADS", "4")))  # rollout latent batché →
        config = SylvanConfig()                                                      # plus de threads = plus de FPS
        payload = torch.load(wm_ckpt, map_location="cpu", weights_only=False)
        meta = payload["meta"]
        self.proprio_dim = meta["proprio_dim"]
        # RÉTINE étage 2 : WM-rétine si l'obs = proprio ++ rétine(144) ++ énergie (277) au lieu de +radar(12).
        self.wm_uses_retina = (meta["obs_dim"] == meta["proprio_dim"] + RETINA_DIM + 1)
        wm = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
                               predictor_arch=meta.get("predictor_arch", "shallow"),
                               with_slot=meta.get("with_slot", False),
                               slot_resources=meta.get("slot_resources", 1))
        wm.load_state_dict(payload["model"])
        wm.eval()
        if meta.get("with_slot", False):
            print(f"[planner-cmd] WM OBJECT-CENTRIC : out['slot'] actif (slot appris dans le WM) → "
                  f"la perception+permanence vient du WM, plus de coordonnée codée-main dans le planner")
        self.planner = CommandPlanner(wm, cfg)
        self.action_dim = config.env.action_dim  # 18 (hexapod); used by the TCP fallbacks below
        self.residual = GaussianActorCritic(
            obs_dim=config.env.policy_input_dim,
            hidden_dim=config.controller.hidden_dim,
            action_dim=config.env.action_dim,
        )
        load_checkpoint(Path(residual_ckpt), self.residual)
        self.residual.eval()
        for p in self.residual.parameters():
            p.requires_grad_(False)
        self.replan_every = max(1, replan_every)
        self._lock = threading.Lock()
        self._cmd = cfg.no_food_command
        self._ticks = 0
        # Smooth the food direction across ticks: the egocentric radar's sector jitters as the gait
        # sways the torso heading ±a few degrees/step → the reconstructed food bearing flips → the
        # planner flip-flops its turn and never commits. An EMA over the radar steadies the target so
        # the agent COMMITS to a turn. (Perception fix — does NOT touch the base/motor.)
        self._radar_ema: list[float] | None = None
        self._water_ema: list[float] | None = None  # 2ᵉ pulsion: EMA du radar eau (localisation planner)
        self.radar_alpha = 0.8
        # RÉTINE étage 1 : tête de perception APPRISE (rayons couleur bruts → position food/eau), REMPLACE
        # l'oracle food_xz_from_radar quand chargée. WM inchangé (il encode toujours le radar pour la
        # dynamique) ; seule la LOCALISATION de la ressource devient apprise. EMA sur les positions sorties.
        self.retina_head: RetinaPerceptionHead | None = None
        self._retina_n_res = 1
        if retina_head_ckpt is not None:
            hck = torch.load(retina_head_ckpt, map_location="cpu", weights_only=False)
            self._retina_n_res = int(hck.get("n_resources", 1))
            self.retina_head = RetinaPerceptionHead(n_resources=self._retina_n_res)
            self.retina_head.load_state_dict(hck["state_dict"])
            self.retina_head.eval()
            print(f"[planner-cmd] RETINA HEAD = {retina_head_ckpt.name} (n_res={self._retina_n_res}, "
                  f"heldout_mae={hck.get('heldout_mae_m', '?')}) → oracle food_xz REMPLACÉ par la perception apprise")
        # SLOT object-centric AUTO-SUPERVISÉ (chantier WM pur) — drop-in de retina_head MAIS entraîné SANS label
        # de position (consistance de transport + saillance perceptuelle, cf train_slot_head). Prioritaire s'il est
        # chargé. Même interface .locate() → la perception devient pleinement label-free (zéro oracle dans la boucle).
        self.slot_head = None
        if slot_head_ckpt is not None:
            from sylvan.models.slot_head import load_slot_head
            self.slot_head = load_slot_head(str(slot_head_ckpt))
            self._retina_n_res = self.slot_head.n_resources
            sck = torch.load(slot_head_ckpt, map_location="cpu", weights_only=False)
            print(f"[planner-cmd] SLOT HEAD (auto-supervisé, label-free) = {slot_head_ckpt.name} "
                  f"(bearing_mae={sck.get('heldout_bearing_deg', '?')}°, pos_mae={sck.get('heldout_mae_m', '?')}m) "
                  f"→ perception PURE (remplace retina_head)")
        # localizer effectif = slot_head (pur) en priorité, sinon retina_head (supervisé)
        self.localizer = self.slot_head if self.slot_head is not None else self.retina_head
        self._food_pos_ema: tuple[float, float] | None = None
        self._water_pos_ema: tuple[float, float] | None = None
        # MÉMOIRE SPATIALE (Task 3) — OPT-IN : uniquement quand --egomotion-head + --slot-memory + WM with_slot.
        # Quand inactif (défaut), le code path est BYTE-IDENTIQUE à aujourd'hui (non-régression).
        self.slot_memory = None
        self._slot_belief: list[float] | None = None
        if use_slot_memory and egomotion_head_ckpt is not None and wm.with_slot:
            from sylvan.models.egomotion_head import load_egomotion_head
            from sylvan.control.slot_memory import SlotMemory
            ego_head = load_egomotion_head(str(egomotion_head_ckpt))
            # slot_encoder = le SelfSupervisedSlotHead déjà chargé dans le WM
            slot_enc = self.planner.world_model.slot_encoder
            self.slot_memory = SlotMemory(ego_head, slot_enc)
            print(f"[planner-cmd] MÉMOIRE SPATIALE active : egomotion={egomotion_head_ckpt.name} "
                  f"threshold={self.slot_memory.salience_threshold} — belief persistant entre replans")
        elif use_slot_memory:
            print("[planner-cmd] AVERTISSEMENT : --slot-memory demandé mais ignoré "
                  f"(egomotion_head={'OK' if egomotion_head_ckpt else 'MANQUANT'}, "
                  f"wm.with_slot={wm.with_slot})")
        self.hud_enable = os.environ.get("SYLVAN_HUD") == "1"
        self.hud_ts = 0
        self.hud_path = os.environ.get("SYLVAN_HUD_PATH", "data/hud/live.json")
        self._last_min_dist = float("nan")
        # ---------------------------------------------------------------------------
        # BC LOGGER (Task 2) — OPT-IN via SYLVAN_BC_LOG=<dir>
        # ---------------------------------------------------------------------------
        # Quand défini : logge une ligne JSONL par pas dans <dir>/ep_XXXX.jsonl.
        # Rotation automatique à chaque reset() → un fichier = un épisode (contrat Task 3).
        # Quand absent : aucun code path actif — non-régression byte-identique.
        # ---------------------------------------------------------------------------
        _bc_dir = os.environ.get("SYLVAN_BC_LOG")
        self._bc_log_dir: Path | None = Path(_bc_dir) if _bc_dir else None
        self._bc_episode: int = -1          # incrémenté à chaque reset() → ep_0000, ep_0001, …
        self._bc_file = None                # file handle ouvert ; None = pas encore d'épisode
        if self._bc_log_dir is not None:
            self._bc_log_dir.mkdir(parents=True, exist_ok=True)
            print(f"[planner-cmd] BC LOGGER actif : {self._bc_log_dir} → ep_XXXX.jsonl / épisode",
                  flush=True)
        self.pos_alpha = float(os.environ.get("SYLVAN_RETINA_POS_ALPHA", "0.0"))  # 0 = position brute (défaut)
        # 🅑-PUR : coût-VALEUR latent (planifier DANS le latent, AUCUNE coordonnée). Quand une tête de valeur
        # est chargée, le serveur route vers planner.plan_latent — la bouffe n'existe QUE dans ce que le WM a
        # appris à percevoir (rétine→latent) + ce que la tête en LIT. Débranche TOTALEMENT food_xz/radar/min_dist.
        self.value_head = None
        if value_head_ckpt is not None:
            from sylvan.models.value_head import load_value_head
            self.value_head = load_value_head(str(value_head_ckpt))
            print(f"[planner-cmd] 🅑 VALUE HEAD = {Path(value_head_ckpt).name} → COÛT-VALEUR LATENT actif "
                  f"(coordonnées DÉBRANCHÉES : plan_latent, aucune position de ressource).")
        # === CHERCHER — perception active (2026-06-21) ===
        # Capacité GÉNÉRALE (pas un patch bouffe-derrière) : quand la value latente de la pulsion active est
        # PLATE (rien d'engageant perçu/imaginable, engage < τ ; gate offline front-proche ~0.98 vs derrière
        # ~0.17), l'entité EXPLORE (scan du cap puis errance) au lieu de suivre un argmax de paysage plat,
        # jusqu'à ce qu'une cible entre dans le cône avant (latent fort) → handoff AUTO au mode-avant latent.
        # JEPA-pur : déclencheur lu DANS le latent, approche planifiée DANS le latent ; seul l'acte d'explorer
        # est un réflexe substrat (comme le CPG). C'est l'étape 'chercher' du north-star. Drive-agnostique
        # (déclenché par engage de la pulsion active). Off par défaut → non-régression.
        self.search_enable = os.environ.get("SYLVAN_SEARCH_ENABLE", "0") == "1"
        self.search_tau = float(os.environ.get("SYLVAN_SEARCH_TAU", "0.5"))        # seuil engage (trou 0.33–0.73)
        self.search_vx = float(os.environ.get("SYLVAN_SEARCH_VX", "0.55"))         # vx du scan (régime propre min)
        self.search_omega = float(os.environ.get("SYLVAN_SEARCH_OMEGA", "0.6"))    # ω du scan (sens fixe = balaye)
        self.search_scan = int(os.environ.get("SYLVAN_SEARCH_SCAN", "10"))         # replans de scan (balaye le cap)
        self.search_wander = int(os.environ.get("SYLVAN_SEARCH_WANDER", "3"))      # replans d'errance (couvre l'espace)
        self.search_patience = int(os.environ.get("SYLVAN_SEARCH_PATIENCE", "1"))  # replans bas-engage avant de chercher
        self.search_log = os.environ.get("SYLVAN_SEARCH_LOG", "0") == "1"
        self._low_engage = 0
        self._search_t = 0
        self._searching = False
        if self.search_enable:
            print(f"[planner-cmd] CHERCHER actif : τ={self.search_tau} scan={self.search_scan}×(vx{self.search_vx},"
                  f"ω{self.search_omega}) wander={self.search_wander} patience={self.search_patience}")
        print(f"[planner-cmd] WM={wm_ckpt.name} residual={residual_ckpt.name} | replan_every={self.replan_every} "
              f"| horizon={cfg.horizon} grid={len(cfg.vx_grid)}x{len(cfg.omega_grid)}")
        if _OCCLUDE_FOV_DEG < 360.0:
            print(f"[planner-cmd] OCCLUSION MASK actif : SYLVAN_OCCLUDE_FOV_DEG={_OCCLUDE_FOV_DEG}° "
                  f"(cone frontal ±{_OCCLUDE_FOV_DEG/2:.0f}°, rayons hors-cone zeroed → SlotMemory requis)")
        else:
            print(f"[planner-cmd] OCCLUSION MASK inactif : SYLVAN_OCCLUDE_FOV_DEG={_OCCLUDE_FOV_DEG}° "
                  f"(perception 360° intacte, non-régression)")

    @torch.no_grad()
    def predict_full(self, payload: dict) -> dict:
        proprio = payload.get("proprio")
        if not isinstance(proprio, list):
            raise TypeError("request must contain proprio")
        radar = list(payload.get("vision") or [])
        fine = list(payload.get("vision_fine") or [])
        retina = list(payload.get("retina") or [])  # RÉTINE étage 1 : rayons couleur bruts (144)
        # OCCLUSION MASK (Task 4) : appliquer UNE SEULE FOIS ici, avant tout usage downstream
        # (localisation, wm_obs, SlotMemory re-grounding).  FOV >= 360 → identité (non-régression).
        retina = occlude_retina(retina, _OCCLUDE_FOV_DEG)
        energy = float(payload.get("energy", 0.0))
        # 2ᵉ pulsion (planner-only, HORS WM): radar eau + niveau de soif. Absents → run mono-ressource
        # → thirst plein (pas de pression) → coût identique à avant.
        water_fine = list(payload.get("vision_water") or [])
        thirst = float(payload.get("thirst", 100.0))
        with self._lock:
            # MÉMOIRE SPATIALE (Task 3) : mise à jour du belief par tick (dead-reckon + re-ground si saillant).
            # Doit s'exécuter AVANT le bloc de replan pour que belief soit à jour quand le planner est appelé.
            # Quand slot_memory est None (défaut) : cette section est absente → non-régression totale.
            if self.slot_memory is not None and len(retina) > 0:
                self._slot_belief = self.slot_memory.update(proprio, retina)
            # A1: localise food from the FINER radar when Godot sends it (±5° vs ±15°); the WM still
            # encodes the trained 12-sector radar. Smooth whichever radar drives localisation.
            loc = fine if fine else radar
            if loc:
                if self._radar_ema is None or len(self._radar_ema) != len(loc):
                    self._radar_ema = list(loc)
                else:
                    a = self.radar_alpha
                    self._radar_ema = [a * e + (1.0 - a) * r for e, r in zip(self._radar_ema, loc)]
            if water_fine:
                if self._water_ema is None or len(self._water_ema) != len(water_fine):
                    self._water_ema = list(water_fine)
                else:
                    a = self.radar_alpha
                    self._water_ema = [a * e + (1.0 - a) * r for e, r in zip(self._water_ema, water_fine)]
            # RÉTINE étage 1 : localiser food/eau via la TÊTE APPRISE (rayons bruts), EMA sur les positions.
            food_pos = water_pos = None
            if self.localizer is not None and len(retina) == RETINA_DIM:
                locs = self.localizer.locate(torch.tensor(retina, dtype=torch.float32))
                food_pos = self._ema_pos("_food_pos_ema", locs[0])
                if self._retina_n_res > 1:
                    water_pos = self._ema_pos("_water_pos_ema", locs[1])
            if self._ticks % self.replan_every == 0 and self._radar_ema is not None and (
                    (self.wm_uses_retina and len(retina) == RETINA_DIM) or
                    (not self.wm_uses_retina and len(radar) == VISION_DIM)):
                # RÉTINE étage 2 : si le WM est un WM-rétine (obs_dim 277), son encodeur voit les RAYONS
                # BRUTS (proprio ++ retina ++ énergie) ; sinon il voit le radar 12 (proprio ++ radar ++ énergie).
                if self.wm_uses_retina:
                    wm_obs = torch.tensor(proprio + retina + [energy / 100.0], dtype=torch.float32)
                else:
                    wm_obs = torch.tensor(proprio + radar + [energy / 100.0], dtype=torch.float32)
                if self.value_head is not None:
                    # 🅑-PUR : planifier DANS LE LATENT — score-valeur sur les latents RÊVÉS, AUCUNE coordonnée
                    # (food_pos / radar / min_dist / heading TOTALEMENT débranchés). C'est le coût JEPA-pur :
                    # la bouffe n'existe que dans ce que le WM perçoit (rétine→latent) + ce que la tête en lit.
                    res = self.planner.plan_latent(wm_obs, self.value_head, energy=energy / 100.0)
                    # CHERCHER (perception active) : si rien d'engageant n'est perçu, explorer au lieu de suivre
                    # un argmax plat ; handoff auto au mode-avant latent dès qu'une cible entre dans le cône avant.
                    self._cmd = self._apply_search(res) if self.search_enable else res["command"]
                elif self.localizer is not None:
                    # LOCALISATION = perception APPRISE (slot pur si chargé, sinon retina_head). water override seulement
                    # si la tête gère l'eau ; sinon on garde l'EMA radar eau (2ᵉ pulsion non encore rétinisée).
                    plan_res = self.planner.plan(
                        wm_obs, self._radar_ema,
                        water_radar=None if self._retina_n_res > 1 else self._water_ema,
                        energy=energy / 100.0, thirst=thirst / 100.0,
                        override_pos=True, food_override=food_pos,
                        water_override=water_pos if self._retina_n_res > 1 else (
                            None),
                    )
                    self._cmd = plan_res["command"]
                    self._last_min_dist = float(plan_res.get("pred_min_dist", float("nan")))
                else:
                    # food LOCALISED from the fine EMA (oracle) — or plan_wm_slot when wm.with_slot=True.
                    # MÉMOIRE SPATIALE (Task 3) : passer slot_belief quand actif → override slot t0 du WM.
                    # Quand slot_memory est None → slot_belief=None → comportement byte-identique à avant.
                    self._cmd = self.planner.plan(
                        wm_obs, self._radar_ema,
                        water_radar=self._water_ema,
                        energy=energy / 100.0, thirst=thirst / 100.0,
                        slot_belief=self._slot_belief,
                    )["command"]
            self._ticks += 1
            vx, om = self._cmd
            if self.hud_enable:
                self.hud_ts += 1
                fields = {"command": [float(vx), float(om)], "energy": float(energy), "thirst": float(thirst)}
                if food_pos is not None:
                    fields["bearing"] = math.degrees(math.atan2(float(food_pos[0]), float(food_pos[1])))
                if self._last_min_dist == self._last_min_dist:  # pas NaN
                    fields["min_dist"] = self._last_min_dist
                try:
                    write_live(self.hud_path, ts=self.hud_ts, episode=0, step=self._ticks, fields=fields)
                except Exception:
                    pass
            # BC LOGGER (Task 2) : logge (obs, cmd) en JSONL pour l'entraînement BC (Task 3).
            # Contrat : {"obs":{"proprio":[132], "energy":float, "thirst":float},
            #            "wm" :{"retina0":[144], "cmd":[vx, omega]}}
            # Non-régression : quand _bc_log_dir is None → bloc absent, comportement identique.
            if self._bc_log_dir is not None and len(retina) == RETINA_DIM:
                if self._bc_file is None:
                    # sécurité : premier predict avant le premier reset() (rare)
                    self._bc_episode += 1
                    ep_path = self._bc_log_dir / f"ep_{self._bc_episode:04d}.jsonl"
                    self._bc_file = open(ep_path, "w", buffering=1)   # line-buffered
                    print(f"[planner-cmd] BC → {ep_path.name} (auto-open)", flush=True)
                line = json.dumps({
                    "obs": {
                        "proprio": proprio,
                        "energy":  float(energy),
                        "thirst":  float(thirst),
                    },
                    "wm": {
                        "retina0": retina,
                        "cmd":     [float(vx), float(om)],
                    },
                })
                self._bc_file.write(line + "\n")
            vision = [float(vx), float(om)] + [0.0] * (VISION_DIM - 2)
            res_in = torch.tensor(proprio + vision, dtype=torch.float32).unsqueeze(0)
            action = self.residual.mean(res_in)[0]
        action = torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        return {"action": [float(v) for v in action.tolist()], "command": [float(vx), float(om)]}

    def _ema_pos(self, attr: str, new: tuple[float, float] | None) -> tuple[float, float] | None:
        """Lissage de la position (frame agent) sortie par la tête. La tête est STABLE quand la ressource
        est visible (≠ radar jittery) → par défaut PAS de lissage (pos_alpha=0, position brute) : EMA-er une
        position relative MÉLANGE quand la « + proche » bascule entre 2 pastilles → cible fantôme. None
        (non vue) → garde la dernière estimation. SYLVAN_RETINA_POS_ALPHA pour réactiver un lissage."""
        cur = getattr(self, attr)
        if new is None:
            return cur
        a = self.pos_alpha
        ema = new if (cur is None or a <= 0.0) else (a * cur[0] + (1 - a) * new[0], a * cur[1] + (1 - a) * new[1])
        setattr(self, attr, ema)
        return ema

    def _apply_search(self, res: dict) -> tuple[float, float]:
        """Perception active (CHERCHER) : si rien d'engageant n'est perçu (engage < τ), explorer — scan du cap
        puis errance, cycliquement — jusqu'au handoff. `res` = sortie de plan_latent ('command' + 'engage').
        État persistant entre replans (le serveur garde le lock). Drive-agnostique (engage = pulsion active)."""
        engage = float(res.get("engage", 1.0))
        approach = res["command"]
        if engage >= self.search_tau:                  # cible engageable perçue → APPROCHER, reset
            if self._searching and self.search_log:
                print(f"[planner-cmd] CHERCHER→APPROCHE (engage={engage:.2f})", flush=True)
            self._low_engage = 0; self._search_t = 0; self._searching = False
            return approach
        self._low_engage += 1
        if self._low_engage <= self.search_patience:   # hystérésis : ignorer un creux transitoire
            return approach
        if not self._searching and self.search_log:
            print(f"[planner-cmd] APPROCHE→CHERCHER (engage={engage:.2f})", flush=True)
        self._searching = True
        cycle = max(1, self.search_scan + self.search_wander)
        phase = self._search_t % cycle
        self._search_t += 1
        if phase < self.search_scan:
            return (self.search_vx, self.search_omega)  # scan : balaye le cap (sens fixe, undirected)
        return (0.7, 0.0)                               # errance : avance vers un nouveau point de vue

    def reset(self) -> None:
        with self._lock:
            self._ticks = 0
            self._cmd = self.planner.cfg.no_food_command
            self._radar_ema = None
            self._water_ema = None
            self._food_pos_ema = None
            self._water_pos_ema = None
            self._low_engage = 0
            self._search_t = 0
            self._searching = False
            # MÉMOIRE SPATIALE (Task 3) : réinitialiser le belief entre épisodes
            if self.slot_memory is not None:
                self.slot_memory.reset()
                self._slot_belief = None
            # BC LOGGER (Task 2) : rotation de fichier à chaque reset d'épisode.
            # Ferme l'épisode courant et ouvre le suivant → ep_0000.jsonl, ep_0001.jsonl, …
            if self._bc_log_dir is not None:
                if self._bc_file is not None:
                    self._bc_file.close()
                self._bc_episode += 1
                ep_path = self._bc_log_dir / f"ep_{self._bc_episode:04d}.jsonl"
                self._bc_file = open(ep_path, "w", buffering=1)   # line-buffered
                print(f"[planner-cmd] BC → {ep_path.name}", flush=True)


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while True:
            raw = self.rfile.readline()
            if not raw:
                return
            # Robustness: a malformed payload or a transient inference error must NOT crash the
            # handler thread (that left orphan servers all over the place). Catch, log, send a SAFE
            # fallback command, keep serving.
            try:
                payload = json.loads(raw.decode("utf-8"))
                if payload.get("reset"):
                    self.server.service.reset()
                    resp = {"action": [0.0] * self.server.service.action_dim,
                            "command": list(self.server.service._cmd)}
                else:
                    resp = self.server.service.predict_full(payload)
            except Exception as exc:  # noqa: BLE001 — deliberately broad: never kill the server
                print(f"[planner-cmd] request error: {exc!r} — sending safe fallback", flush=True)
                resp = {"action": [0.0] * self.server.service.action_dim,
                        "command": [0.5, 0.0], "error": str(exc)}
            self.wfile.write(json.dumps(resp).encode("utf-8") + b"\n")
            self.wfile.flush()


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, addr, service):
        super().__init__(addr, _Handler)
        self.service = service


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", required=True)
    ap.add_argument("--residual", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6051)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--replan-every", type=int, default=10)
    ap.add_argument("--energy-weight", type=float, default=2.0)
    ap.add_argument("--retina-head", default=None, help="checkpoint tête de perception apprise (étage 1) → "
                    "remplace l'oracle food_xz par la localisation depuis les rayons couleur bruts")
    ap.add_argument("--value-head", default=None, help="🅑-PUR : checkpoint tête de VALEUR (V=latent) → coût-valeur "
                    "latent, coordonnées DÉBRANCHÉES (plan_latent). Exclut --retina-head (pas de localisation).")
    ap.add_argument("--slot-head", default=None, help="SLOT object-centric AUTO-SUPERVISÉ (label-free) → drop-in pur "
                    "de --retina-head (perception sans aucun label de position ; prioritaire s'il est fourni).")
    ap.add_argument("--egomotion-head", default=None, help="MÉMOIRE SPATIALE (Task 3) : checkpoint EgomotionHead "
                    "(proprio→dyaw,dfwd,dlat) pour dead-reckoner le belief entre replans. Requis avec --slot-memory.")
    ap.add_argument("--slot-memory", action="store_true", default=False,
                    help="MÉMOIRE SPATIALE (Task 3) : activer la persistance inter-replans du slot (OPT-IN). "
                    "Requiert --egomotion-head + WM with_slot. Défaut OFF → comportement byte-identique.")
    args = ap.parse_args()

    cfg = CommandPlanConfig(horizon=args.horizon, energy_weight=args.energy_weight)
    service = _PlannerService(Path(args.wm), Path(args.residual), cfg, args.replan_every,
                             retina_head_ckpt=Path(args.retina_head) if args.retina_head else None,
                             value_head_ckpt=Path(args.value_head) if args.value_head else None,
                             slot_head_ckpt=Path(args.slot_head) if args.slot_head else None,
                             egomotion_head_ckpt=Path(args.egomotion_head) if args.egomotion_head else None,
                             use_slot_memory=args.slot_memory)
    server = _Server((args.host, args.port), service)
    print(f"[planner-cmd] serving on {args.host}:{args.port} — Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
