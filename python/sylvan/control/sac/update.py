"""SAC learner: twin-Q Bellman update, squashed-actor update, auto-tuned temperature.

Standard SAC (Haarnoja et al. 2018 v2): clipped double-Q targets with the entropy bonus,
a reparameterised actor loss, and an auto-tuned temperature alpha that drives the policy
entropy toward target_entropy = -action_dim. Targets are tracked by Polyak (soft) update.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from .models import SacActor, TwinQ


class SacLearner:
    def __init__(
        self,
        *,
        actor: SacActor,
        critic: TwinQ,
        action_dim: int,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        alpha_init: float = 0.2,
        alpha_min: float = 0.02,
        fixed_alpha: float | None = None,
        target_entropy: float | None = None,
        device: str = "cpu",
    ) -> None:
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.target_critic = copy.deepcopy(critic).to(device)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)
        self.gamma = gamma
        self.tau = tau
        self.device = device
        # Floor on log_alpha: the tanh-squashed entropy estimate sits well above
        # target_entropy=-action_dim, so auto-tuning drives alpha toward 0 → the entropy
        # regularizer vanishes → the actor greedily chases an inflating Q off the stability
        # boundary (the run1 collapse: alpha 0.2→0.004, falls→100%). The floor keeps a
        # minimum exploration/regularization pressure so the gait stays alive.
        self._log_alpha_min = float(torch.log(torch.tensor(alpha_min)))
        self.target_entropy = float(target_entropy if target_entropy is not None else -action_dim)

        # FIXED alpha: with a warm-started saturating actor the tanh squash inflates the
        # entropy estimate (H~8 instead of ~0), so the soft-Q entropy bonus (alpha*H) dwarfs
        # the task reward → SAC maximises randomness (jitter-in-place) instead of the task
        # (the run3 failure). A small constant alpha scaled to the reward keeps entropy a
        # minor regularizer and lets the task dominate. None → auto-tune (with the floor).
        self.fixed_alpha = fixed_alpha
        # BC anchor reference: a frozen snapshot of the warm-started (good-gait) actor. While the
        # critic is still weak/extrapolating, a decaying MSE anchor keeps the policy's actions IN
        # the data distribution (near the known-good gait, where Q is accurate) so the actor can
        # IMPROVE turning without the reparam gradient exploiting critic errors off-distribution
        # and collapsing the gait. Set by the trainer via set_bc_reference(); decayed to 0.
        self.bc_ref: SacActor | None = None
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        if fixed_alpha is None:
            self.log_alpha = torch.tensor(float(torch.log(torch.tensor(alpha_init))), requires_grad=True, device=device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        else:
            self.log_alpha = torch.tensor(float(torch.log(torch.tensor(fixed_alpha))), device=device)
            self.alpha_opt = None

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def set_bc_reference(self, ref_actor: SacActor) -> None:
        self.bc_ref = copy.deepcopy(ref_actor).eval()
        for p in self.bc_ref.parameters():
            p.requires_grad_(False)

    def _soft_update(self) -> None:
        with torch.no_grad():
            for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.mul_(1.0 - self.tau).add_(self.tau * p)

    def mc_warmup(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Critic warmup by SUPERVISED regression onto the Monte-Carlo return-to-go. Gives the
        critic the accurate value of the (frozen, good-gait) collection policy in one pass —
        unlike 1-step Bellman, which propagates value far too slowly across a 400-step horizon.
        Also syncs the target net so the later Bellman phase starts consistent."""
        obs, act, mc = batch["obs"], batch["actions"], batch["mc"]
        q1, q2 = self.critic(obs, act)
        loss = nn.functional.mse_loss(q1, mc) + nn.functional.mse_loss(q2, mc)
        self.critic_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()
        with torch.no_grad():
            for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.copy_(p)  # hard-sync during warmup
        return {"critic_loss": float(loss), "actor_loss": 0.0, "alpha": float(self.alpha),
                "entropy": 0.0, "q_mean": float(q1.mean()), "bc_loss": 0.0}

    def update(self, batch: dict[str, torch.Tensor], *, critic_only: bool = False,
               bc_coef: float = 0.0) -> dict[str, float]:
        obs, act = batch["obs"], batch["actions"]
        rew, nxt, done = batch["rewards"], batch["next_obs"], batch["dones"]
        alpha = self.alpha.detach()

        # --- Critic (clipped double-Q Bellman target) ---
        with torch.no_grad():
            next_a, next_logp = self.actor.sample(nxt)
            q_targ = self.target_critic.q_min(nxt, next_a) - alpha * next_logp
            target = rew + self.gamma * (1.0 - done) * q_targ
        q1, q2 = self.critic(obs, act)
        critic_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # CRITIC WARMUP: with a from-scratch critic and a warm-started (good-walker) actor,
        # the untrained critic can't tell walking from standing (Q≈flat), so the early actor
        # gradient destroys the good gait before the critic catches up → standstill collapse.
        # critic_only trains Q^π for the FROZEN good actor first; actor updates start from a
        # correct critic. (Target still tracks via the soft update below.)
        if critic_only:
            self._soft_update()
            return {"critic_loss": float(critic_loss), "actor_loss": 0.0,
                    "alpha": float(self.alpha), "entropy": 0.0, "q_mean": float(q1.mean())}

        # --- Actor (reparameterised, maximise Q + entropy, + decaying BC anchor) ---
        new_a, logp = self.actor.sample(obs)
        q_new = self.critic.q_min(obs, new_a)
        actor_loss = (alpha * logp - q_new).mean()
        bc_loss = torch.zeros((), device=self.device)
        if bc_coef > 0.0 and self.bc_ref is not None:
            mean, _ = self.actor.forward(obs)
            with torch.no_grad():
                ref_mean, _ = self.bc_ref.forward(obs)
            # anchor the squashed deterministic action toward the good-gait reference
            bc_loss = nn.functional.mse_loss(torch.tanh(mean), torch.tanh(ref_mean))
            actor_loss = actor_loss + bc_coef * bc_loss
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # --- Temperature (drive entropy toward target) — skipped when alpha is fixed ---
        if self.fixed_alpha is None:
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            with torch.no_grad():
                self.log_alpha.clamp_(min=self._log_alpha_min)

        self._soft_update()

        return {
            "critic_loss": float(critic_loss),
            "actor_loss": float(actor_loss),
            "alpha": float(self.alpha),
            "entropy": float(-logp.mean()),
            "q_mean": float(q_new.mean()),
            "bc_loss": float(bc_loss),
        }
