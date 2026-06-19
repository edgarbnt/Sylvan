"""Turn real Godot JSONL rollouts into a PPO training batch (J0).

The behavior policy that COLLECTED the rollout is passed in; `old_log_prob` and
`values` are recomputed from it on the stored (proprio, action) pairs. This is
exact (the stored action IS the sampled action, Godot exploration noise = 0) and
needs no fragile ordering join and no schema change.

GAE(λ) uses the REAL rewards with correct terminal handling: a `done` (fall) zeroes
the bootstrap (genuinely terminal); a `truncated` (time-limit) bootstraps with
γ·V(next_obs) (the episode was cut artificially, not ended).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ...buffer.reader import iter_episodes
from .symmetry import mirror_obs, mirror_action


@dataclass(slots=True)
class RolloutBatch:
    obs: torch.Tensor          # [N, obs_dim]
    actions: torch.Tensor      # [N, action_dim]
    old_log_prob: torch.Tensor # [N]
    advantages: torch.Tensor   # [N] (normalised once over the whole batch)
    returns: torch.Tensor      # [N]
    values: torch.Tensor       # [N] (behavior-policy value)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    *,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE(λ) for ONE episode. `dones[t]` is the terminal (fall) flag; the only
    possibly-true entry is the last step. `last_value` = V(next_obs) when the
    episode was truncated (time-limit), else 0 (fall) — it is only used when the
    last step is non-terminal."""
    horizon = rewards.shape[0]
    advantages = torch.zeros(horizon, dtype=torch.float32)
    last_adv = torch.zeros((), dtype=torch.float32)
    for t in reversed(range(horizon)):
        nonterminal = 0.0 if bool(dones[t]) else 1.0
        next_value = last_value if t == horizon - 1 else values[t + 1]
        delta = rewards[t] + gamma * nonterminal * next_value - values[t]
        last_adv = delta + gamma * lam * nonterminal * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns


def build_rollout_batch(
    run_dir: Path,
    behavior_policy,
    *,
    gamma: float,
    lam: float,
    device: str = "cpu",
    mirror_augment: bool = False,
) -> tuple[RolloutBatch | None, dict[str, float]]:
    episodes = [ep for ep in iter_episodes(Path(run_dir)) if ep]
    obs_chunks, act_chunks, olp_chunks, adv_chunks, ret_chunks, val_chunks = ([] for _ in range(6))

    for ep in episodes:
        # Network input = [proprio ++ vision] (food radar appended last, matching the
        # collection servers and the warm-start zero-pad). vision is [] when perception is off.
        obs = torch.tensor([t.obs.proprio + t.obs.vision for t in ep], dtype=torch.float32, device=device)
        acts = torch.tensor([t.action for t in ep], dtype=torch.float32, device=device)
        rews = torch.tensor([t.reward for t in ep], dtype=torch.float32, device=device)
        dones = torch.tensor([1.0 if t.done else 0.0 for t in ep], dtype=torch.float32, device=device)

        with torch.no_grad():
            log_prob, _, values = behavior_policy.evaluate_actions(obs, acts)
            last = ep[-1]
            if last.truncated and not last.done:
                last_next = torch.tensor(last.next_obs.proprio + last.next_obs.vision, dtype=torch.float32, device=device).unsqueeze(0)
                last_value = behavior_policy.value(last_next)[0]
            else:
                last_value = torch.zeros((), dtype=torch.float32, device=device)

        adv, ret = compute_gae(rews, values, dones, last_value, gamma=gamma, lam=lam)
        obs_chunks.append(obs)
        act_chunks.append(acts)
        olp_chunks.append(log_prob)
        adv_chunks.append(adv)
        ret_chunks.append(ret)
        val_chunks.append(values)

    num_transitions = int(sum(c.shape[0] for c in obs_chunks))
    stats = {
        "num_episodes": float(len(episodes)),
        "num_transitions": float(num_transitions),
    }
    if num_transitions == 0:
        return None, stats

    advantages = torch.cat(adv_chunks)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # normalise once
    batch = RolloutBatch(
        obs=torch.cat(obs_chunks),
        actions=torch.cat(act_chunks),
        old_log_prob=torch.cat(olp_chunks),
        advantages=advantages,
        returns=torch.cat(ret_chunks),
        values=torch.cat(val_chunks),
    )

    if mirror_augment:
        # HARD symmetry: append the left-right mirror of every transition. The locomotion reward
        # is chirality-invariant (forward speed in the body frame, uprightness, effort), so the
        # mirrored sample shares the same advantage/return; only old_log_prob is recomputed under
        # the SAME behavior snapshot on (mirror_obs, mirror_action). Duplicating leaves the
        # already-normalised advantages' mean/std unchanged. Unlike the soft equivariance penalty,
        # this makes symmetry a property of the DATA so no asymmetric gait attractor (the dragging
        # "kickstand" leg) can be preferred.
        obs_m = mirror_obs(batch.obs)
        act_m = mirror_action(batch.actions)
        with torch.no_grad():
            olp_m, _, val_m = behavior_policy.evaluate_actions(obs_m, act_m)
        batch = RolloutBatch(
            obs=torch.cat([batch.obs, obs_m]),
            actions=torch.cat([batch.actions, act_m]),
            old_log_prob=torch.cat([batch.old_log_prob, olp_m]),
            advantages=torch.cat([batch.advantages, batch.advantages]),
            returns=torch.cat([batch.returns, batch.returns]),
            values=torch.cat([batch.values, val_m]),
        )
        stats["mirror_augmented"] = 1.0

    return batch, stats
