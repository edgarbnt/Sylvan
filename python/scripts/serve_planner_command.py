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
import socketserver
import threading
from pathlib import Path

import torch

from sylvan.config import SylvanConfig
from sylvan.control.planning.command_planner import CommandPlanner, CommandPlanConfig
from sylvan.control.ppo.policy import GaussianActorCritic
from sylvan.models.command_wm import CommandWorldModel
from sylvan.models.perception_head import RetinaPerceptionHead, RETINA_DIM
from sylvan.training.checkpointing import load_checkpoint

VISION_DIM = 12  # the residual's vision slot: [vx, omega, 0*10] in CPG mode (matches training)


class _PlannerService:
    def __init__(self, wm_ckpt: Path, residual_ckpt: Path, cfg: CommandPlanConfig, replan_every: int,
                 retina_head_ckpt: Path | None = None) -> None:
        torch.set_num_threads(4)
        config = SylvanConfig()
        payload = torch.load(wm_ckpt, map_location="cpu", weights_only=False)
        meta = payload["meta"]
        self.proprio_dim = meta["proprio_dim"]
        # RÉTINE étage 2 : WM-rétine si l'obs = proprio ++ rétine(144) ++ énergie (277) au lieu de +radar(12).
        self.wm_uses_retina = (meta["obs_dim"] == meta["proprio_dim"] + RETINA_DIM + 1)
        wm = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
                               predictor_arch=meta.get("predictor_arch", "shallow"))
        wm.load_state_dict(payload["model"])
        wm.eval()
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
        self._food_pos_ema: tuple[float, float] | None = None
        self._water_pos_ema: tuple[float, float] | None = None
        import os
        self.pos_alpha = float(os.environ.get("SYLVAN_RETINA_POS_ALPHA", "0.0"))  # 0 = position brute (défaut)
        print(f"[planner-cmd] WM={wm_ckpt.name} residual={residual_ckpt.name} | replan_every={self.replan_every} "
              f"| horizon={cfg.horizon} grid={len(cfg.vx_grid)}x{len(cfg.omega_grid)}")

    @torch.no_grad()
    def predict_full(self, payload: dict) -> dict:
        proprio = payload.get("proprio")
        if not isinstance(proprio, list):
            raise TypeError("request must contain proprio")
        radar = list(payload.get("vision") or [])
        fine = list(payload.get("vision_fine") or [])
        retina = list(payload.get("retina") or [])  # RÉTINE étage 1 : rayons couleur bruts (144)
        energy = float(payload.get("energy", 0.0))
        # 2ᵉ pulsion (planner-only, HORS WM): radar eau + niveau de soif. Absents → run mono-ressource
        # → thirst plein (pas de pression) → coût identique à avant.
        water_fine = list(payload.get("vision_water") or [])
        thirst = float(payload.get("thirst", 100.0))
        with self._lock:
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
            if self.retina_head is not None and len(retina) == RETINA_DIM:
                locs = self.retina_head.locate(torch.tensor(retina, dtype=torch.float32))
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
                if self.retina_head is not None:
                    # LOCALISATION = perception APPRISE (oracle food_xz débranché). water override seulement
                    # si la tête gère l'eau ; sinon on garde l'EMA radar eau (2ᵉ pulsion non encore rétinisée).
                    self._cmd = self.planner.plan(
                        wm_obs, self._radar_ema,
                        water_radar=None if self._retina_n_res > 1 else self._water_ema,
                        energy=energy / 100.0, thirst=thirst / 100.0,
                        override_pos=True, food_override=food_pos,
                        water_override=water_pos if self._retina_n_res > 1 else (
                            None),
                    )["command"]
                else:
                    # food LOCALISED from the fine EMA (oracle).
                    self._cmd = self.planner.plan(
                        wm_obs, self._radar_ema,
                        water_radar=self._water_ema,
                        energy=energy / 100.0, thirst=thirst / 100.0,
                    )["command"]
            self._ticks += 1
            vx, om = self._cmd
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

    def reset(self) -> None:
        with self._lock:
            self._ticks = 0
            self._cmd = self.planner.cfg.no_food_command
            self._radar_ema = None
            self._water_ema = None
            self._food_pos_ema = None
            self._water_pos_ema = None


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
    args = ap.parse_args()

    cfg = CommandPlanConfig(horizon=args.horizon, energy_weight=args.energy_weight)
    service = _PlannerService(Path(args.wm), Path(args.residual), cfg, args.replan_every,
                             retina_head_ckpt=Path(args.retina_head) if args.retina_head else None)
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
