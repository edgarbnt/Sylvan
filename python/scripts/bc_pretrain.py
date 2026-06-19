"""Behavior-clone the GOOD gait (CPG + residual7) into a GaussianActorCritic, as the init for a
fully-learned PPO finetune. The from-scratch learned policy can't DISCOVER walking with ~8 envs
(no Isaac 4096-env exploration); cloning our working controller is the SOTA bridge (IFM/SynLoco/
MPC-Net/DeepMimic, deep-research #4). The clone reproduces residual7's applied joint targets, so in
learned mode (action_scale=1.0, blend=1.0) the policy walks immediately → PPO then refines + makes
turning agile (free of the CPG ceiling) via command curriculum.

Data: godot/data/replay_buffer/bc_data/*.jsonl with (obs.proprio + obs.vision, applied).
Usage: cd python && ../env_pytorch_3.12/bin/python -m scripts.bc_pretrain \
         --data ../godot/data/replay_buffer/bc_data --out ../data/checkpoints/ppo_bc_init/policy_best.pt
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch

from sylvan.control.ppo.policy import GaussianActorCritic
from sylvan.training.checkpointing import save_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser(description="Behavior-clone the CPG+residual gait into a policy.")
    ap.add_argument("--data", required=True, help="Dir of bc_data JSONL episodes.")
    ap.add_argument("--out", required=True, help="Output checkpoint path (policy_best.pt).")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=128)
    args = ap.parse_args()

    X, Y = [], []
    for fp in sorted(glob.glob(str(Path(args.data) / "*.jsonl"))):
        with open(fp) as f:
            for line in f:
                d = json.loads(line)
                applied = d.get("applied")
                obs = d.get("obs", {})
                if not applied or "proprio" not in obs:
                    continue
                X.append(list(obs["proprio"]) + list(obs.get("vision", [])))
                Y.append(applied)
    if not X:
        raise SystemExit("No (obs, applied) pairs found — run run_bc_collect.sh first.")
    X = torch.tensor(X, dtype=torch.float32)
    # tanh actor saturates at ±1 → clamp targets just inside so the MSE is learnable.
    Y = torch.tensor(Y, dtype=torch.float32).clamp(-0.999, 0.999)
    obs_dim, act_dim = X.shape[1], Y.shape[1]
    print(f"[bc] {X.shape[0]} samples | obs_dim={obs_dim} act_dim={act_dim}")

    model = GaussianActorCritic(obs_dim=obs_dim, hidden_dim=args.hidden_dim, action_dim=act_dim)
    opt = torch.optim.Adam(model.actor.parameters(), lr=args.lr)
    n = X.shape[0]
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i : i + args.batch_size]
            pred = model.mean(X[idx])
            loss = torch.nn.functional.mse_loss(pred, Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"[bc] epoch {epoch:02d} | mse={tot / n:.5f}", flush=True)

    out = Path(args.out)
    save_checkpoint(destination=out, model=model, optimizer=opt, epoch=args.epochs,
                    metrics={"bc_mse": tot / n})
    print(f"[bc] saved → {out}")


if __name__ == "__main__":
    main()
