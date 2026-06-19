"""Serve the J0 PPO policy DETERMINISTICALLY (mean action) for visual inspection.

Loads `GaussianActorCritic` from a J0 checkpoint (default
`data/checkpoints/ppo_j0/policy_latest.pt`) and serves the MEAN action — no
exploration noise — over the same TCP wire protocol Godot already speaks
({"proprio":[...]} -> {"action":[...]}). This lets you watch exactly what the
policy has learned, decoupled from training (separate port, ephemeral training
port = 0 never clashes). The world model is NOT involved (J0 is grounded).

Usage (from the python/ directory):
    python3 -m scripts.serve_ppo_visual --port 6007
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sylvan.config import SylvanConfig
from sylvan.constants import CHECKPOINTS_DIR
from sylvan.control.policy_server import _PolicyRequestHandler, _PolicyTCPServer
from sylvan.control.ppo.policy import GaussianActorCritic
from sylvan.training.checkpointing import load_checkpoint


class _DeterministicService:
    """Serves the policy MEAN (the learned behaviour, no Gaussian sampling)."""

    def __init__(self, policy: GaussianActorCritic) -> None:
        self.policy = policy.eval()
        for p in self.policy.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def predict(self, payload: dict[str, object]) -> list[float]:
        proprio = payload.get("proprio")
        if not isinstance(proprio, list):
            raise TypeError("Policy request must contain a proprio list")
        vision = payload.get("vision") or []  # food radar; appended AFTER proprio
        t = torch.tensor(proprio + list(vision), dtype=torch.float32).unsqueeze(0)
        action = self.policy.mean(t)[0]
        action = torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        return [float(v) for v in action.detach().cpu().tolist()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve the J0 PPO policy deterministically.")
    ap.add_argument(
        "--checkpoint",
        default=str(CHECKPOINTS_DIR / "ppo_j0" / "policy_latest.pt"),
        help="Path to a J0 GaussianActorCritic checkpoint.",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6007)
    args = ap.parse_args()

    config = SylvanConfig()
    policy = GaussianActorCritic(
        obs_dim=config.env.policy_input_dim,  # proprio ++ vision
        hidden_dim=config.controller.hidden_dim,
        action_dim=config.env.action_dim,
    )
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"[J0-visual] checkpoint not found: {ckpt} — train J0 first.")
    payload = load_checkpoint(ckpt, policy)
    print(
        f"[J0-visual] loaded {ckpt} | epoch={payload.get('epoch')} "
        f"metrics={payload.get('metrics')}"
    )

    service = _DeterministicService(policy)
    server = _PolicyTCPServer((args.host, args.port), _PolicyRequestHandler, inference_service=service)
    print(f"[J0-visual] serving MEAN action on {args.host}:{args.port} — Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
