"""PPO update step (J0): clipped surrogate + value loss + entropy bonus.

Operates on a RolloutBatch whose `old_log_prob` came from the FROZEN behavior
snapshot (never the live policy), so the importance ratio is meaningful and the
clip actually engages. Health signals (approx_kl, clip_frac, mean_std) are returned
so a recurrence of the "loss pinned / exploration collapsed" pathology is visible.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .symmetry import mirror_obs, mirror_action


@dataclass(slots=True)
class PPOConfig:
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    epochs: int = 10
    minibatch_size: int = 256
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    # Mirror-symmetry (equivariance) regularizer: enforce pi(mirror(obs)) == mirror(pi(obs)).
    # 0 = off (default). Set > 0 to kill the learned left-right gait drift on the symmetric body.
    sym_coef: float = 0.0
    # Value-invariance regularizer: enforce V(mirror(obs)) == V(obs). The actor term alone
    # leaves the critic asymmetric, which re-injects the gait drift through the advantages
    # (see observation 233). < 0 = mirror sym_coef (on whenever sym_coef > 0); >= 0 = explicit.
    sym_v_coef: float = -1.0


def ppo_update(policy, optimizer, batch, cfg: PPOConfig | None = None) -> dict[str, float]:
    cfg = cfg or PPOConfig()
    n = batch.obs.shape[0]
    pl_sum = vl_sum = ent_sum = kl_sum = clip_sum = sym_sum = symv_sum = 0.0
    n_minibatches = 0
    epochs_run = 0

    for _epoch in range(cfg.epochs):
        epochs_run += 1
        perm = torch.randperm(n)
        epoch_kls = []
        for start in range(0, n, cfg.minibatch_size):
            idx = perm[start : start + cfg.minibatch_size]
            log_prob, entropy, value = policy.evaluate_actions(batch.obs[idx], batch.actions[idx])
            log_ratio = log_prob - batch.old_log_prob[idx]
            ratio = log_ratio.exp()
            adv = batch.advantages[idx]

            surr1 = ratio * adv
            surr2 = ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value, batch.returns[idx])
            entropy_mean = entropy.mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_mean

            sym_val = 0.0
            symv_val = 0.0
            if cfg.sym_coef > 0.0:
                obs_mb = batch.obs[idx]
                obs_mir = mirror_obs(obs_mb)  # one mirror forward, reused for actor + critic
                # Actor equivariance: the mirrored observation yields the mirrored mean action.
                mean = policy.mean(obs_mb)
                mean_mir = policy.mean(obs_mir)
                sym_loss = F.mse_loss(mirror_action(mean), mean_mir)
                loss = loss + cfg.sym_coef * sym_loss
                sym_val = float(sym_loss.item())
                # Value invariance: V is a scalar return estimate, unchanged by a left-right
                # reflection. Target is the (stop-grad) value already computed above for obs_mb.
                v_coef = cfg.sym_v_coef if cfg.sym_v_coef >= 0.0 else cfg.sym_coef
                if v_coef > 0.0:
                    sym_loss_v = F.mse_loss(policy.value(obs_mir), value.detach())
                    loss = loss + v_coef * sym_loss_v
                    symv_val = float(sym_loss_v.item())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean().clamp_min(0.0).item()  # Schulman estimator
                clip_frac = ((ratio - 1.0).abs() > cfg.clip).float().mean().item()
            pl_sum += float(policy_loss.item())
            vl_sum += float(value_loss.item())
            ent_sum += float(entropy_mean.item())
            kl_sum += approx_kl
            clip_sum += clip_frac
            sym_sum += sym_val
            symv_sum += symv_val
            epoch_kls.append(approx_kl)
            n_minibatches += 1

        if epoch_kls and (sum(epoch_kls) / len(epoch_kls)) > cfg.target_kl:
            break  # early-stop epochs to avoid a destructive update

    denom = max(1, n_minibatches)
    return {
        "policy_loss": pl_sum / denom,
        "value_loss": vl_sum / denom,
        "entropy": ent_sum / denom,
        "approx_kl": kl_sum / denom,
        "clip_frac": clip_sum / denom,
        "sym_loss": sym_sum / denom,
        "sym_loss_v": symv_sum / denom,
        "mean_std": policy.mean_std(),
        "epochs_run": float(epochs_run),
    }
