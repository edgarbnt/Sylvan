"""J1a.3 — the WM fidelity + fall-anticipation GATE.

Runs two controller-INDEPENDENT checks on a held-out validation run (so it works
with the J0 GaussianActorCritic, which is NOT the latent LocomotionController that
evaluate_transfer's policy-rollout arm expects — see plan R4):

  1. Fidelity (forced-action): open-loop the WM fed the REAL actions, compare
     predicted vs real per-step reward (reward-head + dynamics error) and matched-
     horizon return. Reuses transfer._forced_action_rewards.
  2. Fall anticipation: evaluation/fall_prediction.evaluate_fall_prediction.

Writes data/reports/wm_v1_transfer_report.json and prints PASS/FAIL. Thresholds
are uncalibrated — the FIRST run calibrates them; do not treat a fail as final
until the numbers are read and the thresholds frozen.

Usage (from python/):
    python3 -m scripts.eval_world_model_v1 \
        --world-model data/checkpoints/wm_v1/world_model_v1.pt \
        --val-run wm_v1_data_val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sylvan.config import SylvanConfig
from sylvan.constants import REPLAY_BUFFER_DIR, REPORTS_DIR
from sylvan.buffer.reader import iter_episodes
from sylvan.evaluation.fall_prediction import evaluate_fall_prediction
from sylvan.evaluation.transfer import _forced_action_rewards
from sylvan.models.world_model import WorldModelV0
from sylvan.training.checkpointing import load_checkpoint

# Uncalibrated fidelity thresholds (tighter than transfer.py's 1.5/0.20 since the
# data is now clean). Calibrate on the first measured run, then freeze.
FIDELITY_RATIO_MAX = 1.3
FIDELITY_REWARD_MAE_MAX = 0.15


@torch.no_grad()
def _forced_fidelity(world_model, episodes, *, horizon, device="cpu") -> dict[str, float]:
    """Controller-independent fidelity: forced-action reward MAE + matched-horizon
    return ratio (predicted vs real)."""
    episodes = [ep for ep in episodes if len(ep) >= 1]
    n = len(episodes)
    if n == 0:
        return {"num_episodes": 0}
    pdim = len(episodes[0][0].obs.proprio)
    adim = len(episodes[0][0].action)
    s0 = torch.tensor([ep[0].obs.proprio for ep in episodes], dtype=torch.float32, device=device)
    actions_seq = torch.zeros((horizon, n, adim), dtype=torch.float32, device=device)
    matched = []
    for i, ep in enumerate(episodes):
        hk = min(horizon, len(ep))
        matched.append(hk)
        for t in range(hk):
            actions_seq[t, i] = torch.tensor(ep[t].action, dtype=torch.float32, device=device)
    forced = _forced_action_rewards(world_model, s0, actions_seq)  # [H, N]

    pred_returns, real_returns, reward_errs = [], [], []
    for i, ep in enumerate(episodes):
        hk = matched[i]
        real_seq = [float(ep[t].reward) for t in range(hk)]
        pred_returns.append(float(forced[:hk, i].sum().item()))
        real_returns.append(sum(real_seq))
        for t in range(hk):
            reward_errs.append(abs(float(forced[t, i].item()) - real_seq[t]))
    mean_pred = sum(pred_returns) / n
    mean_real = sum(real_returns) / n
    reward_mae = sum(reward_errs) / len(reward_errs) if reward_errs else float("nan")
    ratio = mean_pred / (abs(mean_real) + 1e-6)
    return {
        "num_episodes": n,
        "mean_matched_horizon": sum(matched) / n,
        "mean_forced_return": mean_pred,
        "mean_real_return": mean_real,
        "forced_return_ratio": ratio,
        "forced_reward_mae": reward_mae,
        "fidelity_pass": bool(ratio <= FIDELITY_RATIO_MAX and reward_mae <= FIDELITY_REWARD_MAE_MAX),
        "thresholds": {"ratio_max": FIDELITY_RATIO_MAX, "reward_mae_max": FIDELITY_REWARD_MAE_MAX, "calibrated": False},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="J1a.3 WM fidelity + fall-anticipation gate.")
    ap.add_argument("--world-model", required=True)
    ap.add_argument("--val-run", nargs="+", required=True,
                    help="Held-out run name(s)/path(s). Pass a survivors set AND a falls set so the "
                         "fall-anticipation AUROC has both classes.")
    ap.add_argument("--horizon", type=int, default=None, help="Default = config.controller.imagined_horizon.")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    config = SylvanConfig()
    horizon = args.horizon or config.controller.imagined_horizon
    device = torch.device(args.device)

    val_dirs = [Path(v) if Path(v).is_dir() else REPLAY_BUFFER_DIR / v for v in args.val_run]
    for d in val_dirs:
        if not d.is_dir():
            raise SystemExit(f"[wm-eval] val run not found: {d}")
    wm_path = Path(args.world_model)
    if not wm_path.exists():
        raise SystemExit(f"[wm-eval] world model not found: {wm_path}")

    world_model = WorldModelV0(
        obs_dim=config.env.wm_obs_dim,
        proprio_dim=config.env.proprio_dim,
        action_dim=config.env.action_dim,
        metrics_dim=config.env.metrics_dim,
        hidden_dim=config.train.hidden_dim,
        latent_dim=config.train.latent_dim,
    ).to(device)
    load_checkpoint(wm_path, world_model)
    world_model.eval()
    for p in world_model.parameters():
        p.requires_grad_(False)

    episodes = [ep for d in val_dirs for ep in iter_episodes(d)]
    fidelity = _forced_fidelity(world_model, episodes, horizon=horizon, device=str(device))
    falls = evaluate_fall_prediction(world_model, episodes, horizon=horizon, device=str(device))

    gate_pass = bool(fidelity.get("fidelity_pass") and falls.get("fall_pass"))
    report = {
        "world_model": str(wm_path),
        "val_run": [str(d) for d in val_dirs],
        "horizon": horizon,
        "fidelity": fidelity,
        "fall_prediction": falls,
        "J1a_gate_pass": gate_pass,
        "note": "Thresholds uncalibrated — read the numbers and freeze before gating J1b.",
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "wm_v1_transfer_report.json"
    out.write_text(json.dumps(report, indent=2))

    print(f"[wm-eval] val={[d.name for d in val_dirs]} | episodes={fidelity.get('num_episodes')} "
          f"(falls={falls.get('num_fall_episodes')}, survive={falls.get('num_survive_episodes')}) horizon={horizon}")
    print(f"[wm-eval] FIDELITY  forced_return_ratio={fidelity.get('forced_return_ratio'):.3f} "
          f"reward_mae={fidelity.get('forced_reward_mae'):.4f} -> {'PASS' if fidelity.get('fidelity_pass') else 'fail'}")
    auroc = falls.get("done_auroc")
    print(f"[wm-eval] FALL      done_auroc={auroc if auroc is None else round(auroc,3)} "
          f"gap={falls.get('done_prob_gap')} "
          f"pred_drop={falls.get('mean_pred_height_drop_falls')} real_drop={falls.get('mean_real_height_drop_falls')} "
          f"collapse={falls.get('height_predicts_collapse')} -> {'PASS' if falls.get('fall_pass') else 'fail'}")
    print(f"[wm-eval] J1a GATE: {'PASS ✅' if gate_pass else 'FAIL (read numbers, calibrate, maybe collect more falls)'}")
    print(f"[wm-eval] report -> {out}")


if __name__ == "__main__":
    main()
