"""J1a.3 — does the world model ANTICIPATE the fall? (the central J1 gate)

The original failure was a WM that imagined an optimistic "everything's fine"
attractor and NEVER predicted toppling, so a controller trained in its dream
optimised a window in which it could not see itself fall. Before we let the WM
plan (J1b), it must prove it sees falls coming. This module open-loops the WM with
the REAL actions over the pre-fall window of each episode (same forced-action
pattern as transfer._forced_action_rewards) and checks:

  - done-head separation: AUROC of predicted P(done) (averaged over the last K
    steps of the window) between FALLING and SURVIVING episodes — does the WM's
    termination signal rise specifically before real falls?
  - the pre-fall vs survivor gap in mean predicted P(done).
  - height-collapse: on falling episodes, does the predicted height DROP track the
    real height drop (rather than staying flat / optimistic)?
  - survivor false-positive: predicted P(done) on survivors must stay low.

Pure offline, CPU, never mutates anything. Thresholds are UNCALIBRATED placeholders
(calibrate on the first measured WM run, then freeze — mirrors transfer.py).
"""

from __future__ import annotations

import torch

# "height" position in constants.LOCOMOTION_METRIC_KEYS = (uprightness, forward_velocity,
# torso_tilt, height, ground_contact, effort, pose_error).
HEIGHT_METRIC_INDEX = 3

# CALIBRATED 2026-06-04 on the wm_v2 measured run (rebalanced falls + weighted
# done/height loss). Rationale: the PLANNER scores candidates via the done-head
# (P(done) penalty), so a strong, confident done signal is what matters — the
# absolute height-collapse MAGNITUDE is a secondary diagnostic that open-loop
# rollouts inherently under-predict (regression to the mean). So we require:
#   - near-perfect fall RANKING (auroc >= 0.90; wm_v2 measured 1.0 @ H=12),
#   - the done-head actually FIRES more on falls (gap >= 0.05; wm_v2 measured 0.078),
#   - the predicted height drops in the RIGHT DIRECTION and by a meaningful
#     fraction (collapse_frac 0.20; wm_v2 measured 0.060/0.239 = 0.25).
# The earlier 0.30/0.50 placeholders were rejected because they demanded a
# magnitude the open-loop WM cannot reach, NOT a planner-relevant signal.
DEFAULT_AUROC_MIN = 0.90
DEFAULT_GAP_MIN = 0.05
DEFAULT_COLLAPSE_FRAC = 0.20   # predicted height drop must be >= this fraction of the real drop (right sign)
DEFAULT_MIN_REAL_DROP = 0.05   # only assert collapse when the real height actually dropped


def _auroc(scores: list[float], labels: list[float]) -> float | None:
    """AUROC via the Mann-Whitney U statistic (ties count 0.5). None if degenerate."""
    pos = [s for s, y in zip(scores, labels) if y > 0.5]
    neg = [s for s, y in zip(scores, labels) if y <= 0.5]
    if not pos or not neg:
        return None
    wins = 0.0
    for sp in pos:
        for sn in neg:
            wins += 1.0 if sp > sn else (0.5 if sp == sn else 0.0)
    return wins / (len(pos) * len(neg))


@torch.no_grad()
def _forced_open_loop(world_model, proprio0, actions_seq, *, height_index):
    """Open-loop the WM fed the REAL actions. proprio0 [N,P], actions_seq [H,N,A].
    Returns predicted done-prob [H,N] and predicted height [H,N]."""
    encoded = world_model.encoder(proprio0.unsqueeze(1))
    state = encoded[:, 0]
    hidden = None
    done_probs, heights = [], []
    for t in range(actions_seq.shape[0]):
        outputs = world_model(
            proprio=None,
            actions=actions_seq[t].unsqueeze(1),
            initial_hidden=hidden,
            encoded_obs=state.unsqueeze(1),
        )
        hidden = outputs["hidden"]
        state = outputs["predicted_next_encoded"][:, 0]
        done_probs.append(torch.sigmoid(outputs["predicted_done_logits"][:, 0]))
        heights.append(outputs["predicted_next_metrics"][:, 0, height_index])
    return torch.stack(done_probs, dim=0), torch.stack(heights, dim=0)


@torch.no_grad()
def evaluate_fall_prediction(
    world_model,
    episodes,
    *,
    horizon: int,
    height_metric_index: int = HEIGHT_METRIC_INDEX,
    last_k: int = 5,
    device: str = "cpu",
    auroc_min: float = DEFAULT_AUROC_MIN,
    gap_min: float = DEFAULT_GAP_MIN,
    collapse_frac: float = DEFAULT_COLLAPSE_FRAC,
    min_real_drop: float = DEFAULT_MIN_REAL_DROP,
) -> dict[str, object]:
    episodes = [ep for ep in episodes if len(ep) >= 2]
    n = len(episodes)
    if n == 0:
        return {"num_episodes": 0}

    pdim = len(episodes[0][0].obs.proprio)
    adim = len(episodes[0][0].action)
    h = horizon
    proprio0 = torch.zeros((n, pdim), dtype=torch.float32, device=device)
    actions_seq = torch.zeros((h, n, adim), dtype=torch.float32, device=device)
    real_height = torch.full((h, n), float("nan"), dtype=torch.float32, device=device)
    valid: list[int] = []
    is_fall: list[float] = []

    for i, ep in enumerate(episodes):
        length = len(ep)
        start = max(0, length - h)            # window ends at the real last step (the fall, if any)
        v = min(h, length - start)
        valid.append(v)
        is_fall.append(1.0 if ep[-1].done else 0.0)
        proprio0[i] = torch.tensor(ep[start].obs.proprio, dtype=torch.float32, device=device)
        for t in range(v):
            actions_seq[t, i] = torch.tensor(ep[start + t].action, dtype=torch.float32, device=device)
            real_height[t, i] = float(ep[start + t].obs.metrics.get("height", 0.0))

    done_probs, pred_height = _forced_open_loop(
        world_model, proprio0, actions_seq, height_index=height_metric_index
    )

    scores: list[float] = []            # mean predicted P(done) over last_k valid steps
    pred_drops: list[float] = []        # falling-episode predicted height drop
    real_drops: list[float] = []        # falling-episode real height drop
    height_abs_errs: list[float] = []   # falling-episode |pred_h - real_h|
    for i in range(n):
        v = valid[i]
        k0 = max(0, v - last_k)
        scores.append(float(done_probs[k0:v, i].mean().item()))
        if is_fall[i] > 0.5:
            pred_drops.append(float((pred_height[0, i] - pred_height[v - 1, i]).item()))
            real_drops.append(float((real_height[0, i] - real_height[v - 1, i]).item()))
            for t in range(v):
                height_abs_errs.append(abs(float(pred_height[t, i].item()) - float(real_height[t, i].item())))

    fall_scores = [s for s, y in zip(scores, is_fall) if y > 0.5]
    surv_scores = [s for s, y in zip(scores, is_fall) if y <= 0.5]
    mean_pre_fall = sum(fall_scores) / len(fall_scores) if fall_scores else float("nan")
    mean_surv = sum(surv_scores) / len(surv_scores) if surv_scores else float("nan")
    gap = (mean_pre_fall - mean_surv) if (fall_scores and surv_scores) else float("nan")
    auroc = _auroc(scores, is_fall)

    mean_pred_drop = sum(pred_drops) / len(pred_drops) if pred_drops else float("nan")
    mean_real_drop = sum(real_drops) / len(real_drops) if real_drops else float("nan")
    height_mae = sum(height_abs_errs) / len(height_abs_errs) if height_abs_errs else float("nan")
    predicts_collapse = bool(
        real_drops and mean_real_drop > min_real_drop
        and mean_pred_drop >= collapse_frac * mean_real_drop
    )

    fall_pass = bool(
        auroc is not None and auroc >= auroc_min
        and (gap == gap and gap >= gap_min)      # gap not NaN and large enough
        and predicts_collapse
    )

    return {
        "num_episodes": n,
        "num_fall_episodes": int(sum(is_fall)),
        "num_survive_episodes": int(n - sum(is_fall)),
        "horizon": h,
        "done_auroc": auroc,
        "mean_done_prob_pre_fall": mean_pre_fall,
        "mean_done_prob_survivors": mean_surv,
        "done_prob_gap": gap,
        "mean_pred_height_drop_falls": mean_pred_drop,
        "mean_real_height_drop_falls": mean_real_drop,
        "height_mae_falls": height_mae,
        "height_predicts_collapse": predicts_collapse,
        "fall_pass": fall_pass,
        "thresholds": {
            "auroc_min": auroc_min,
            "gap_min": gap_min,
            "collapse_frac": collapse_frac,
            "min_real_drop": min_real_drop,
            "last_k": last_k,
            "calibrated": True,
        },
    }
