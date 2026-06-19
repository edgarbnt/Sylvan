"""Persistent off-policy replay buffer, filled from the Godot JSONL rollouts.

A circular buffer of preallocated tensors. Each iteration the trainer collects fresh
episodes into a run_dir and calls `ingest`, which reads them with the same
`iter_episodes` the PPO path uses and appends (obs, action, reward, next_obs, done)
tuples. SAC then samples uniform minibatches across the WHOLE buffer (many reuses per
transition) — that reuse is the entire point versus on-policy PPO.

`done` here is the genuine-terminal (fall) flag: it zeroes the Bellman bootstrap. A
time-limit `truncated` is NOT terminal (the episode was cut artificially), so its
transition is stored with done=0 and bootstraps normally.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ...buffer.reader import iter_episodes


class ReplayBuffer:
    def __init__(self, *, capacity: int, obs_dim: int, action_dim: int,
                 gamma: float = 0.99, device: str = "cpu") -> None:
        self.capacity = int(capacity)
        self.device = device
        self.gamma = gamma
        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.float32, device=device)
        # Monte-Carlo discounted return-to-go per transition — a fast, accurate value target for
        # the critic warmup (1-step Bellman can't propagate value across a 400-step horizon in a
        # few thousand updates; regressing Q onto the empirical return does it in one pass).
        self.mc = torch.zeros(capacity, dtype=torch.float32, device=device)
        self._ptr = 0
        self._full = False

    def __len__(self) -> int:
        return self.capacity if self._full else self._ptr

    def _push_arrays(self, obs, acts, rews, nxt, dones, mc) -> None:
        n = obs.shape[0]
        if n == 0:
            return
        idx = (torch.arange(n) + self._ptr) % self.capacity
        self.obs[idx] = obs
        self.actions[idx] = acts
        self.rewards[idx] = rews
        self.next_obs[idx] = nxt
        self.dones[idx] = dones
        self.mc[idx] = mc
        self._ptr = int((self._ptr + n) % self.capacity)
        if self._ptr < n:
            self._full = True

    @staticmethod
    def _returns_to_go(rews: torch.Tensor, gamma: float) -> torch.Tensor:
        g = torch.zeros_like(rews)
        acc = 0.0
        for t in range(rews.shape[0] - 1, -1, -1):
            acc = float(rews[t]) + gamma * acc
            g[t] = acc
        return g

    def ingest(self, run_dir: Path) -> int:
        """Read all episodes under run_dir, append their transitions. Returns count added."""
        added = 0
        for ep in iter_episodes(Path(run_dir)):
            if not ep:
                continue
            obs = torch.tensor([t.obs.proprio + t.obs.vision for t in ep], dtype=torch.float32)
            acts = torch.tensor([t.action for t in ep], dtype=torch.float32)
            rews = torch.tensor([t.reward for t in ep], dtype=torch.float32)
            nxt = torch.tensor([t.next_obs.proprio + t.next_obs.vision for t in ep], dtype=torch.float32)
            # fall = genuine terminal (zero bootstrap); time-limit truncation is NOT terminal.
            dones = torch.tensor([1.0 if (t.done and not t.truncated) else 0.0 for t in ep], dtype=torch.float32)
            mc = self._returns_to_go(rews, self.gamma)
            self._push_arrays(
                obs.to(self.device), acts.to(self.device), rews.to(self.device),
                nxt.to(self.device), dones.to(self.device), mc.to(self.device),
            )
            added += obs.shape[0]
        return added

    def sample(self, batch_size: int, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        n = len(self)
        idx = torch.randint(0, n, (batch_size,), generator=generator, device=self.device)
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
            "mc": self.mc[idx],
        }
