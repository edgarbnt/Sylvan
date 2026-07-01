"""Mode-1 : serve the BC (DriveSymmetricPolicy) + frozen residual as ONE TCP server.

Godot sends {proprio, retina, energy, thirst} each tick. The server:
  1. every K ticks, re-queries the BC policy → command (vx, omega) ;
  2. runs the frozen residual policy on [proprio ++ command-in-vision-slot] → 18-D joint action ;
  3. returns {"action":[18], "command":[vx, omega]}.

NO WorldModel, NO CommandPlanner : perception = rétine directe via build_tokens.
Contrat TCP IDENTIQUE à serve_planner_command.py (non-régression côté Godot).

Usage (depuis la racine) :
    PYTHONPATH=python env_pytorch_3.12/bin/python -m scripts.serve_mode1 \\
        --residual data/checkpoints/hexapod_v2/policy_best.pt \\
        --bc-policy data/checkpoints/mode1_bc/policy.pt \\
        --host 127.0.0.1 --port 6052 --replan-every 10
"""

from __future__ import annotations

import argparse
import json
import os
import socketserver
import threading
from pathlib import Path

import torch

from sylvan.config import SylvanConfig
from sylvan.control.mode1.policy import DriveSymmetricPolicy, map_action  # noqa: F401 (map_action exported)
from sylvan.control.mode1.obs import build_tokens
from sylvan.control.mode1.residual import residual_action
from sylvan.control.ppo.policy import GaussianActorCritic
from sylvan.training.checkpointing import load_checkpoint

# Commande neutre par défaut (centre du régime propre hexapode)
_DEFAULT_CMD: tuple[float, float] = (0.65, 0.0)


class _Mode1Service:
    """Service Mode-1 : BC policy → (vx,ω) → résidu gelé → action[18]."""

    def __init__(self, residual_ckpt: Path, bc_policy_ckpt: Path, replan_every: int) -> None:
        torch.set_num_threads(int(os.environ.get("SYLVAN_PLANNER_THREADS", "4")))
        config = SylvanConfig()
        self.action_dim = config.env.action_dim  # 18

        # ------------------------------------------------------------------ #
        # Résidu gelé (hexapod_v2) — IDENTIQUE à serve_planner_command.py
        # ------------------------------------------------------------------ #
        self.residual = GaussianActorCritic(
            obs_dim=config.env.policy_input_dim,
            hidden_dim=config.controller.hidden_dim,
            action_dim=config.env.action_dim,
        )
        load_checkpoint(residual_ckpt, self.residual)
        self.residual.eval()
        for p in self.residual.parameters():
            p.requires_grad_(False)

        # ------------------------------------------------------------------ #
        # Politique BC Mode-1 (DriveSymmetricPolicy)
        # ------------------------------------------------------------------ #
        _ck = torch.load(bc_policy_ckpt, map_location="cpu", weights_only=False)
        self._pol = DriveSymmetricPolicy(proprio_dim=_ck["meta"]["proprio_dim"])
        self._pol.load_state_dict(_ck["model"])
        self._pol.eval()
        for p in self._pol.parameters():
            p.requires_grad_(False)
        _n_drives = _ck["meta"].get("n_drives", "?")
        _n_epochs = _ck["meta"].get("epochs_trained", "?")
        print(
            f"[serve-mode1] BC policy = {bc_policy_ckpt.name} "
            f"(proprio_dim={_ck['meta']['proprio_dim']}, n_drives={_n_drives}, "
            f"epochs={_n_epochs})",
            flush=True,
        )

        self.replan_every = max(1, replan_every)
        self._lock = threading.Lock()
        self._cmd: tuple[float, float] = _DEFAULT_CMD
        self._ticks: int = 0
        print(
            f"[serve-mode1] residual={residual_ckpt.name} | replan_every={self.replan_every}",
            flush=True,
        )

    @torch.no_grad()
    def predict_full(self, payload: dict) -> dict:
        if not isinstance(payload.get("proprio"), list):
            raise TypeError("request must contain proprio (list)")

        proprio = payload["proprio"]  # list[132]

        with self._lock:
            # Re-query the BC policy every K ticks (tenir la commande entre replans,
            # même comportement que serve_planner_command.py)
            if self._ticks % self.replan_every == 0:
                p_t, tok_t, _meta = build_tokens(payload)        # proprio[132], tokens[D,38]
                cmd_t = self._pol.act(p_t.unsqueeze(0), tok_t.unsqueeze(0))[0]  # (vx,ω) bornées
                self._cmd = (float(cmd_t[0]), float(cmd_t[1]))

            self._ticks += 1
            vx, om = self._cmd
            # Obs-résidu PARTAGÉE (helper unique — byte-identique à serve_mode1_collect / planner).
            action = residual_action(self.residual, proprio, vx, om)

        return {"action": action, "command": [float(vx), float(om)]}

    def reset(self) -> None:
        with self._lock:
            self._ticks = 0
            self._cmd = _DEFAULT_CMD


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while True:
            raw = self.rfile.readline()
            if not raw:
                return
            try:
                payload = json.loads(raw.decode("utf-8"))
                if payload.get("reset"):
                    self.server.service.reset()
                    resp = {
                        "action": [0.0] * self.server.service.action_dim,
                        "command": list(self.server.service._cmd),
                    }
                else:
                    resp = self.server.service.predict_full(payload)
            except Exception as exc:  # noqa: BLE001
                print(f"[serve-mode1] request error: {exc!r} — sending safe fallback", flush=True)
                resp = {
                    "action": [0.0] * self.server.service.action_dim,
                    "command": list(_DEFAULT_CMD),
                    "error": str(exc),
                }
            self.wfile.write(json.dumps(resp).encode("utf-8") + b"\n")
            self.wfile.flush()


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, addr, service):
        super().__init__(addr, _Handler)
        self.service = service


def main() -> None:
    ap = argparse.ArgumentParser(description="Sylvan Mode-1 TCP server (BC policy + frozen residual)")
    ap.add_argument("--residual", default="data/checkpoints/hexapod_v2/policy_best.pt",
                    help="Checkpoint du résidu gelé (hexapod_v2)")
    ap.add_argument("--bc-policy", default="data/checkpoints/mode1_bc/policy.pt",
                    help="Checkpoint de la politique BC Mode-1")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "6052")))
    ap.add_argument("--replan-every", type=int, default=10,
                    help="Nombre de ticks entre deux requêtes à la politique BC")
    args = ap.parse_args()

    service = _Mode1Service(
        residual_ckpt=Path(args.residual),
        bc_policy_ckpt=Path(args.bc_policy),
        replan_every=args.replan_every,
    )
    server = _Server((args.host, args.port), service)
    print(f"[serve-mode1] serving on {args.host}:{args.port} — Ctrl-C to stop", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
