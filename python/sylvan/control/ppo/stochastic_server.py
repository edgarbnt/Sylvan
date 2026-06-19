"""Stochastic TCP policy server for PPO collection (J0).

Mirrors `serve_policy_controller` but (a) SAMPLES the action from the Gaussian
policy (PPO needs the log-prob of the sampled action) and (b) operates directly on
proprio with NO world-model encoder. Reuses the exact TCP scaffolding so the wire
protocol Godot speaks ({"proprio":[...]} -> {"action":[...]}) is unchanged.

Godot's own exploration noise must be set to 0 (exploration_noise_*=0.0) so the
action written to the JSONL equals the action we sampled here — the keystone that
makes behavior log-prob recomputation exact.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

import torch

from ..policy_server import _PolicyRequestHandler, _PolicyTCPServer
from .policy import GaussianActorCritic


class _StochasticInferenceService:
    def __init__(self, policy: GaussianActorCritic, *, device: str = "cpu", seed: int | None = None) -> None:
        self.device = torch.device(device)
        self.policy = policy.to(self.device).eval()
        for p in self.policy.parameters():
            p.requires_grad_(False)
        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(int(seed))
        self._lock = threading.Lock()

    @torch.no_grad()
    def predict(self, payload: dict[str, object]) -> list[float]:
        proprio = payload.get("proprio")
        if not isinstance(proprio, list):
            raise TypeError("Policy request must contain a proprio list")
        vision = payload.get("vision") or []  # food radar; appended AFTER proprio
        t = torch.tensor(proprio + list(vision), dtype=torch.float32, device=self.device).unsqueeze(0)
        with self._lock:
            action, _ = self.policy.sample(t, generator=self.generator)
        action = torch.nan_to_num(action[0], nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        return [float(v) for v in action.detach().cpu().tolist()]


@contextmanager
def serve_stochastic_policy(
    policy: GaussianActorCritic, *, host: str, port: int, seed: int | None = None
) -> Iterator[dict[str, object]]:
    service = _StochasticInferenceService(policy, seed=seed)
    server = _PolicyTCPServer((host, port), _PolicyRequestHandler, inference_service=service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        bound_host, bound_port = server.server_address
        yield {"host": bound_host, "port": bound_port}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
