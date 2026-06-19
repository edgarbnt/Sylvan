"""Serve the J1b WM planner for visual inspection (parallels serve_ppo_visual.py).

Loads world_model_v1 + the J0 policy, builds the planner, and serves it over the
TCP policy protocol so Godot can be pointed at it. Use a separate port from any
training/J0 server.

Usage (from python/):
    python3 -m scripts.serve_planner_visual \
        --world-model ../data/checkpoints/wm_v1/world_model_v1.pt \
        --policy ../data/checkpoints/ppo_j0/policy_best.pt --port 6008
Then point Godot at 127.0.0.1:6008 (e.g. SYLVAN_POLICY_PORT=6008 ... like run_visual_j0.sh,
with SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sylvan.config import SylvanConfig
from sylvan.constants import CHECKPOINTS_DIR
from sylvan.control.planning.planner_server import serve_planner
from sylvan.control.planning.wm_planner import PlanConfig
from sylvan.control.ppo.policy import GaussianActorCritic
from sylvan.models.world_model import WorldModelV0
from sylvan.training.checkpointing import load_checkpoint


def load_planner_models(config, *, world_model_ckpt: Path, policy_ckpt: Path, device: str = "cpu"):
    """Build + load the WorldModelV0 and the J0 GaussianActorCritic (proprio-input)."""
    world_model = WorldModelV0(
        obs_dim=config.env.wm_obs_dim,           # [proprio ++ vision ++ energy] — food/energy-aware
        proprio_dim=config.env.proprio_dim,
        action_dim=config.env.action_dim,
        metrics_dim=config.env.metrics_dim,
        hidden_dim=config.train.hidden_dim,
        latent_dim=config.train.latent_dim,
    ).to(device)
    policy = GaussianActorCritic(
        obs_dim=config.env.policy_input_dim,     # 106 = proprio ++ vision (the J0 policy's true input)
        hidden_dim=config.controller.hidden_dim,
        action_dim=config.env.action_dim,
    ).to(device)
    load_checkpoint(Path(world_model_ckpt), world_model)
    load_checkpoint(Path(policy_ckpt), policy)
    world_model.eval()
    policy.eval()
    return world_model, policy


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve the J1b WM planner for visual inspection.")
    ap.add_argument("--world-model", default=str(CHECKPOINTS_DIR / "wm_v1" / "world_model_v1.pt"))
    ap.add_argument("--policy", default=str(CHECKPOINTS_DIR / "ppo_j0" / "policy_best.pt"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6008)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--num-samples", type=int, default=64)
    ap.add_argument("--done-penalty", type=float, default=5.0)
    ap.add_argument("--energy-weight", type=float, default=3.0,
                    help="Intrinsic hunger cost weight — higher = stronger food-seeking.")
    ap.add_argument("--proposal-std-scale", type=float, default=1.0,
                    help="Spread of the random-shooting candidates around the policy mean. >1 lets the "
                         "planner consider actions (e.g. LEFT turns) the biased J0 proposal never suggests.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    config = SylvanConfig()
    for name, p in (("world model", args.world_model), ("policy", args.policy)):
        if not Path(p).exists():
            raise SystemExit(f"[planner-visual] {name} not found: {p}")
    world_model, policy = load_planner_models(
        config, world_model_ckpt=args.world_model, policy_ckpt=args.policy
    )
    cfg = PlanConfig(horizon=args.horizon, num_samples=args.num_samples, done_penalty=args.done_penalty,
                     energy_weight=args.energy_weight, proposal_std_scale=args.proposal_std_scale)
    print(f"[planner-visual] horizon={cfg.horizon} samples={cfg.num_samples} done_penalty={cfg.done_penalty} "
          f"energy_weight={cfg.energy_weight} proposal_std_scale={cfg.proposal_std_scale}")
    with serve_planner(world_model, policy, cfg, host=args.host, port=args.port, seed=args.seed) as srv:
        print(f"[planner-visual] serving planner on {srv['host']}:{srv['port']} — Ctrl-C to stop")
        try:
            import threading
            threading.Event().wait()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
