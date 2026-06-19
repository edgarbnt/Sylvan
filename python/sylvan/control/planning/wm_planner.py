"""J1b — grounded short-horizon planner (Mode 2) over the verified world model.

At each real step the planner imagines N candidate futures in the world model,
seeded by the J0 reactive policy (the amortized Mode-1 habit), scores them by
predicted discounted reward (minus a termination penalty, plus a terminal value
bootstrap), and returns the FIRST action of the best candidate. Godot sends the
REAL proprio every tick, so the plan is recomputed from reality each step and the
WM error never compounds beyond `horizon` — this is the "grounded" guard against
the old 100%-imagination failure.

Proposal = policy mean fed the WM's predicted proprio at each dream step, plus
per-step Gaussian noise (random shooting). Candidate 0 is the NOISELESS policy-mean
rollout; if no noisy candidate beats it, the planner executes the policy mean —
so it can never do worse than the reactive J0 baseline by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..ppo.policy import GaussianActorCritic


@dataclass(slots=True)
class PlanConfig:
    horizon: int = 12              # << the 30-step imagined-training horizon; plan only where the WM is trustworthy
    num_samples: int = 64          # random-shooting candidates (candidate 0 = noiseless policy mean)
    discount: float = 0.97
    proposal_std_scale: float = 1.0   # scale on the policy's own std for candidate spread
    done_penalty: float = 5.0      # cost subtracted per step weighted by predicted P(done)
    energy_weight: float = 3.0     # INTRINSIC HUNGER COST: reward predicted high energy (normalised [0,1]).
                                   # This is what makes the planner NAVIGATE — among candidate futures it
                                   # prefers the one that keeps energy high = reaches food. Set well above
                                   # the per-step reward scale so food-seeking dominates balance-only reward.
    use_terminal_value: bool = True
    cem_iters: int = 0             # 0 = pure random shooting (CEM not implemented yet — chosen default)


class WMPlanner:
    def __init__(
        self,
        world_model,
        policy: GaussianActorCritic,
        cfg: PlanConfig | None = None,
        *,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        if cfg is None:
            cfg = PlanConfig()
        if cfg.cem_iters > 0:
            raise NotImplementedError("CEM refinement not implemented; use cem_iters=0 (random shooting).")
        self.device = torch.device(device)
        self.world_model = world_model.to(self.device).eval()
        self.policy = policy.to(self.device).eval()
        for p in self.world_model.parameters():
            p.requires_grad_(False)
        for p in self.policy.parameters():
            p.requires_grad_(False)
        self.cfg = cfg
        # The policy consumes [proprio ++ vision] = obs minus the trailing energy dim.
        self.policy_input_dim = self.world_model.obs_dim - 1
        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(int(seed))

    @torch.no_grad()
    def plan(self, obs: torch.Tensor) -> torch.Tensor:
        """obs [1, O] = [proprio ++ vision ++ energy] (real current state) -> chosen action [A].

        The full observation seeds the (food/energy-aware) encoder, so the dream knows where
        food is and how hungry the agent is. Candidates that keep predicted energy HIGH (reach
        food) score higher via the intrinsic hunger term — this is what produces navigation.
        The policy proposal is fed the [proprio ++ vision] slice (its 106-d input)."""
        cfg = self.cfg
        n = cfg.num_samples
        policy_dim = self.policy_input_dim
        o0 = obs.to(self.device).reshape(1, -1).expand(n, -1).contiguous()       # [N,O]

        state = self.world_model.encoder(o0.unsqueeze(1))[:, 0]                  # [N,L]
        hidden = None
        std = (self.policy._std() * cfg.proposal_std_scale).to(self.device)      # [A]
        cur_policy_in = o0[:, :policy_dim]                                       # [N,106] proprio++vision
        score = torch.zeros(n, device=self.device)
        alive = torch.ones(n, device=self.device)
        first_action: torch.Tensor | None = None

        for t in range(cfg.horizon):
            mean = self.policy.mean(cur_policy_in)                               # [N,A]
            eps = torch.randn(mean.shape, generator=self.generator, device=self.device)
            eps[0] = 0.0                                                         # candidate 0 = noiseless policy mean
            action = (mean + std * eps).clamp(-1.0, 1.0)                         # [N,A]
            if t == 0:
                first_action = action

            outputs = self.world_model(
                obs=None,
                actions=action.unsqueeze(1),
                initial_hidden=hidden,
                encoded_obs=state.unsqueeze(1),
            )
            hidden = outputs["hidden"]
            state = outputs["predicted_next_encoded"][:, 0]
            reward = outputs["predicted_reward"][:, 0]                           # [N]
            done_prob = torch.sigmoid(outputs["predicted_done_logits"][:, 0])    # [N]
            next_obs = outputs["predicted_next_obs"][:, 0]                       # [N,O]
            energy = next_obs[:, -1].clamp(0.0, 1.0)                             # [N] predicted energy (normalised)
            # Reward predicted reward + HIGH energy (low hunger), minus death risk.
            step_value = reward + cfg.energy_weight * energy - cfg.done_penalty * done_prob
            score = score + (cfg.discount ** t) * alive * step_value
            alive = alive * (1.0 - done_prob)                                    # soft survival weighting
            cur_policy_in = next_obs[:, :policy_dim]                            # [N,106] habit feedback

        if cfg.use_terminal_value:
            score = score + (cfg.discount ** cfg.horizon) * alive * self.policy.value(cur_policy_in)

        assert first_action is not None
        best = int(torch.argmax(score).item())
        # Safety: never worse than reactive J0 — fall back to the noiseless mean
        # (candidate 0) unless a noisy candidate strictly beats it.
        if score[best].item() <= score[0].item() + 1e-6:
            best = 0
        action = torch.nan_to_num(first_action[best], nan=0.0, posinf=1.0, neginf=-1.0)
        return action.clamp(-1.0, 1.0)
