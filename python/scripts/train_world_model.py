"""J1a.2 — retrain WorldModelV0 on the clean grounded rollouts from J0.

The dormant world model was fit on the OLD action-space (pre-recentring) and on
the old failing behaviour. This refits it on the new clean data (collect_wm_data)
so it can become a FAITHFUL predictor — the prerequisite, gated in J1a.3, before
any planning. Reuses the existing training stack verbatim (train_epoch /
validate_epoch / ReplaySequenceDataset / WorldModelV0); no change to loops/losses.

Usage (from python/):
    python3 -m scripts.train_world_model --data-runs "wm_v1_data*" \
        --val-run wm_v1_data_val --epochs 40 --run-prefix wm_v1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sylvan.config import SylvanConfig
from sylvan.constants import CHECKPOINTS_DIR, REPLAY_BUFFER_DIR
from sylvan.buffer.dataset import ReplaySequenceDataset, collate_sequence_samples
from sylvan.models.world_model import WorldModelV0
from sylvan.training.checkpointing import save_checkpoint
from sylvan.training.loops import train_epoch, validate_epoch


def _resolve_runs(patterns: list[str]) -> list[Path]:
    """Resolve each pattern as an existing dir, else as a glob under the replay
    buffer. Keep only dirs that contain episode files."""
    resolved: list[Path] = []
    for pat in patterns:
        p = Path(pat)
        candidates = [p] if p.is_dir() else sorted(REPLAY_BUFFER_DIR.glob(pat))
        for c in candidates:
            if c.is_dir() and any(c.glob("*.jsonl")) and c not in resolved:
                resolved.append(c)
    return resolved


def main() -> None:
    ap = argparse.ArgumentParser(description="J1a.2 retrain WorldModelV0 on grounded data.")
    ap.add_argument("--data-runs", nargs="+", required=True,
                    help="Run names / glob patterns (relative to the replay buffer) for TRAIN data.")
    ap.add_argument("--val-run", nargs="+", required=True,
                    help="Held-out run name(s) for validation (excluded from train).")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--run-prefix", default="wm_v1")
    ap.add_argument("--device", default="cpu")
    # J1a re-balancing knobs: make the WM EXPRESS the fall signal it provably holds.
    ap.add_argument("--done-weight", type=float, default=1.0, help="Overall weight on the done BCE.")
    ap.add_argument("--metrics-weight", type=float, default=1.0, help="Weight on the metrics (incl. height) loss.")
    ap.add_argument("--done-pos-weight", type=float, default=None,
                    help="BCE pos_weight on the rare done=1 class (counters fall imbalance).")
    args = ap.parse_args()

    loss_kwargs = {"done_weight": args.done_weight, "metrics_weight": args.metrics_weight}
    if args.done_pos_weight is not None:
        loss_kwargs["done_pos_weight"] = args.done_pos_weight

    config = SylvanConfig()
    device = torch.device(args.device)

    val_dirs = [Path(v) if Path(v).is_dir() else REPLAY_BUFFER_DIR / v for v in args.val_run]
    for d in val_dirs:
        if not d.is_dir():
            raise SystemExit(f"[wm-train] val run not found: {d}")
    val_resolved = {d.resolve() for d in val_dirs}
    train_dirs = [d for d in _resolve_runs(args.data_runs) if d.resolve() not in val_resolved]
    if not train_dirs:
        raise SystemExit(f"[wm-train] no train runs matched {args.data_runs}")

    seq_len = config.train.sequence_length
    train_ds = ReplaySequenceDataset(train_dirs, seq_len)
    val_ds = ReplaySequenceDataset(val_dirs, seq_len)
    if len(train_ds) == 0:
        raise SystemExit(f"[wm-train] no sequences of length {seq_len} in train runs "
                         f"(episodes too short?). dirs={[d.name for d in train_dirs]}")
    print(f"[wm-train] train_dirs={[d.name for d in train_dirs]} val={[d.name for d in val_dirs]} | "
          f"train_seqs={len(train_ds)} val_seqs={len(val_ds)} seq_len={seq_len}")

    train_loader = DataLoader(train_ds, batch_size=config.train.batch_size, shuffle=True,
                              collate_fn=collate_sequence_samples)
    val_loader = DataLoader(val_ds, batch_size=config.train.batch_size, shuffle=False,
                            collate_fn=collate_sequence_samples)

    model = WorldModelV0(
        obs_dim=config.env.wm_obs_dim,          # [proprio ++ vision ++ energy] — food/energy-aware
        proprio_dim=config.env.proprio_dim,
        action_dim=config.env.action_dim,
        metrics_dim=config.env.metrics_dim,
        hidden_dim=config.train.hidden_dim,
        latent_dim=config.train.latent_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)

    ckpt_dir = CHECKPOINTS_DIR / args.run_prefix
    best_val = float("inf")
    print(f"[wm-train] loss_kwargs={loss_kwargs}")
    for epoch in range(args.epochs):
        tr = train_epoch(model, train_loader, optimizer, device, loss_kwargs=loss_kwargs)
        va = validate_epoch(model, val_loader, device, loss_kwargs=loss_kwargs) if len(val_ds) else {}
        val_loss = va.get("loss", float("nan"))
        save_checkpoint(destination=ckpt_dir / "world_model_latest.pt", model=model,
                        optimizer=optimizer, epoch=epoch, metrics={"train_loss": tr["loss"], "val_loss": val_loss})
        is_best = len(val_ds) > 0 and val_loss < best_val
        if is_best:
            best_val = val_loss
            save_checkpoint(destination=ckpt_dir / "world_model_v1.pt", model=model,
                            optimizer=optimizer, epoch=epoch, metrics={"train_loss": tr["loss"], "val_loss": val_loss})
        print(
            "[wm-train] epoch %d | train_loss=%.4f (proprio=%.4f vision=%.4f energy=%.4f "
            "reward=%.4f done=%.4f) | val_loss=%.4f (energy=%.4f)%s" % (
                epoch, tr["loss"], tr["proprio_loss"], tr["vision_loss"], tr["energy_loss"],
                tr["reward_loss"], tr["done_loss"], val_loss, va.get("energy_loss", float("nan")),
                "  <- best" if is_best else "",
            ),
            flush=True,
        )

    print(f"[wm-train] DONE | best val_loss={best_val:.4f} -> {ckpt_dir / 'world_model_v1.pt'}")


if __name__ == "__main__":
    main()
