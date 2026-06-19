"""Honest imagination->reality transfer metric (milestone J1).

The controller is trained to maximise *imagined* return inside the world model.
If the world model is even slightly wrong, the actor can "win" in a dream that
never transfers to reality (model-exploitation) — observed empirically as an
actor loss pinned at the imagined upright-return cap while the real agent falls
100%. This module measures that gap directly, the guard-rail the loop was
missing (see BLUEPRINT.md §8-9).

Methodology (the subtle part):
  - Start from the REAL initial state s0 = ep[0].obs.proprio of each validation
    episode (the policy that produced those episodes is the one we evaluate).
  - POLICY rollout, noise=0: roll the world model Hk = min(H, len(ep)) steps
    under controller.act and sum the predicted per-step rewards -> imagined
    return over the MATCHED horizon Hk.
  - REAL return: sum(real reward) over the SAME first Hk steps. Matched horizon
    is essential — never compare a 30-step imagined return to a full-episode
    real return.
  - FORCED-ACTION rollout: replay the REAL actions through the world model and
    compare predicted vs real reward per step (reward-head + dynamics fidelity),
    isolating model error from policy-divergence error.

Pure offline; CPU; never mutates anything. Reuses imagine_rollout, the world
model, and iter_episodes.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..buffer.reader import iter_episodes
from ..control.imagined_rollouts import imagine_rollout

# Uncalibrated placeholder thresholds. j1_pass is reported but MUST NOT gate
# promotion until calibrated against a measured run (premature gating would
# recreate the "identical rows" freeze).
DEFAULT_RETURN_ERROR_RATIO_MAX = 1.5   # imagined/real within this factor
DEFAULT_REWARD_MAE_MAX = 0.20          # per-step reward-head MAE acceptable


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@torch.no_grad()
def _forced_action_rewards(world_model, proprio_batch, actions_seq):
    """Open-loop world-model rollout fed the REAL actions; returns predicted
    per-step reward [H, N]. The latents are the model's own predictions (so this
    captures combined reward-head + dynamics error), only the actions are real."""
    encoded = world_model.encoder(proprio_batch.unsqueeze(1))
    state = encoded[:, 0]
    hidden = None
    rewards = []
    for t in range(actions_seq.shape[0]):
        outputs = world_model(
            proprio=None,
            actions=actions_seq[t].unsqueeze(1),
            initial_hidden=hidden,
            encoded_obs=state.unsqueeze(1),
        )
        hidden = outputs["hidden"]
        state = outputs["predicted_next_encoded"][:, 0]
        rewards.append(outputs["predicted_reward"][:, 0])
    return torch.stack(rewards, dim=0)


@torch.no_grad()
def evaluate_transfer(
    controller,
    world_model,
    episodes,
    *,
    horizon: int,
    discount: float,
    device: str = "cpu",
    return_error_ratio_max: float = DEFAULT_RETURN_ERROR_RATIO_MAX,
    reward_mae_max: float = DEFAULT_REWARD_MAE_MAX,
) -> dict[str, object]:
    """Compute the imagination->reality transfer digest for a set of episodes."""
    episodes = [ep for ep in episodes if len(ep) >= 1]
    n = len(episodes)
    if n == 0:
        return {"num_episodes": 0}

    s0 = torch.tensor(
        [ep[0].obs.proprio for ep in episodes], dtype=torch.float32, device=device
    )
    action_dim = len(episodes[0][0].action)

    # POLICY rollout (deterministic — noise=0). metrics is unused by imagine_rollout
    # but required by its signature, so pass a placeholder.
    _, _, _, pol_rewards = imagine_rollout(
        controller,
        world_model,
        proprio=s0,
        metrics=torch.zeros((n, 1), dtype=torch.float32, device=device),
        horizon=horizon,
        discount=discount,
        action_noise=0.0,
        return_rewards=True,
    )  # [H, N]

    # FORCED-ACTION rollout from the real actions (zero-padded past each episode).
    actions_seq = torch.zeros((horizon, n, action_dim), dtype=torch.float32, device=device)
    matched = []
    for i, ep in enumerate(episodes):
        hk = min(horizon, len(ep))
        matched.append(hk)
        for t in range(hk):
            actions_seq[t, i] = torch.tensor(ep[t].action, dtype=torch.float32, device=device)
    forced_rewards = _forced_action_rewards(world_model, s0, actions_seq)  # [H, N]

    imagined_returns: list[float] = []
    real_returns: list[float] = []
    abs_return_errors: list[float] = []
    reward_abs_errors: list[float] = []
    # per-step accumulators (averaged across episodes still valid at step t)
    step_imag = [0.0] * horizon
    step_real = [0.0] * horizon
    step_forced = [0.0] * horizon
    step_count = [0] * horizon

    for i, ep in enumerate(episodes):
        hk = matched[i]
        imagined = float(pol_rewards[:hk, i].sum().item())
        real_seq = [float(ep[t].reward) for t in range(hk)]
        real = sum(real_seq)
        imagined_returns.append(imagined)
        real_returns.append(real)
        abs_return_errors.append(abs(imagined - real))
        for t in range(hk):
            reward_abs_errors.append(abs(float(forced_rewards[t, i].item()) - real_seq[t]))
            step_imag[t] += float(pol_rewards[t, i].item())
            step_real[t] += real_seq[t]
            step_forced[t] += float(forced_rewards[t, i].item())
            step_count[t] += 1

    mean_imagined = _mean(imagined_returns)
    mean_real = _mean(real_returns)
    mean_abs_err = _mean(abs_return_errors)
    reward_mae = _mean(reward_abs_errors)
    ratio = mean_imagined / (abs(mean_real) + 1e-6)

    per_step = [
        {
            "t": t,
            "imagined_reward": step_imag[t] / step_count[t],
            "real_reward": step_real[t] / step_count[t],
            "forced_reward": step_forced[t] / step_count[t],
        }
        for t in range(horizon)
        if step_count[t] > 0
    ]

    j1_pass = (ratio <= return_error_ratio_max) and (reward_mae <= reward_mae_max)

    return {
        "num_episodes": n,
        "horizon": horizon,
        "mean_matched_horizon": _mean([float(h) for h in matched]),
        "mean_imagined_return": mean_imagined,
        "mean_real_return": mean_real,
        "mean_abs_return_error": mean_abs_err,
        "return_error_ratio": ratio,
        "per_step_reward_mae": reward_mae,
        "per_step": per_step,
        "j1_pass": bool(j1_pass),
        "thresholds": {
            "return_error_ratio_max": return_error_ratio_max,
            "reward_mae_max": reward_mae_max,
            "calibrated": False,
        },
    }


def load_models(config, *, world_model_ckpt: Path, controller_ckpt: Path, device: str = "cpu"):
    """Build + load the world model and controller (mirrors policy_server.py:52-71)."""
    from ..models.world_model import WorldModelV0
    from ..control.controller import LocomotionController
    from ..training.checkpointing import load_checkpoint

    env, train = config.env, config.train
    world_model = WorldModelV0(
        proprio_dim=env.proprio_dim,
        action_dim=env.action_dim,
        metrics_dim=env.metrics_dim,
        hidden_dim=train.hidden_dim,
        latent_dim=train.latent_dim,
    ).to(device)
    controller = LocomotionController(
        input_dim=train.latent_dim,
        hidden_dim=config.controller.hidden_dim,
        action_dim=env.action_dim,
    ).to(device)
    load_checkpoint(world_model_ckpt, world_model)
    load_checkpoint(controller_ckpt, controller)
    world_model.eval()
    controller.eval()
    for p in world_model.parameters():
        p.requires_grad_(False)
    for p in controller.parameters():
        p.requires_grad_(False)
    return world_model, controller


def evaluate_transfer_from_checkpoints(
    config,
    *,
    validation_run_dir: Path,
    world_model_ckpt: Path,
    controller_ckpt: Path,
    horizon: int | None = None,
    device: str = "cpu",
) -> dict[str, object]:
    """Convenience: load checkpoints + episodes, then run evaluate_transfer."""
    horizon = horizon or config.controller.imagined_horizon
    world_model, controller = load_models(
        config, world_model_ckpt=world_model_ckpt, controller_ckpt=controller_ckpt, device=device
    )
    episodes = iter_episodes(Path(validation_run_dir))
    return evaluate_transfer(
        controller,
        world_model,
        episodes,
        horizon=horizon,
        discount=config.controller.discount,
        device=device,
    )
