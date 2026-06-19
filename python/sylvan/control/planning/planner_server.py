"""Serve the WM planner over the existing TCP policy protocol (J1b).

Mirrors ppo/stochastic_server.py: same wire protocol ({"proprio":[...]} ->
{"action":[...]}) via the reused _PolicyTCPServer/_PolicyRequestHandler, so Godot
is untouched. The only difference is that the action comes from WMPlanner.plan
(short-horizon planning) instead of a single reactive forward pass.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

import torch

from ..policy_server import _PolicyRequestHandler, _PolicyTCPServer
from ..ppo.policy import GaussianActorCritic
from ...models.obs_assembly import assemble_wm_obs
from .wm_planner import PlanConfig, WMPlanner


class _PlannerInferenceService:
    def __init__(
        self,
        world_model,
        policy: GaussianActorCritic,
        cfg: PlanConfig | None = None,
        *,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        self.planner = WMPlanner(world_model, policy, cfg, device=device, seed=seed)
        self._lock = threading.Lock()

    @torch.no_grad()
    def predict(self, payload: dict[str, object]) -> list[float]:
        proprio = payload.get("proprio")
        if not isinstance(proprio, list):
            raise TypeError("Policy request must contain a proprio list")
        vision = payload.get("vision") or []          # food radar
        energy = float(payload.get("energy", 0.0))     # raw 0..max_energy; normalised in assemble_wm_obs
        obs = assemble_wm_obs(
            torch.tensor(proprio, dtype=torch.float32),
            torch.tensor(vision, dtype=torch.float32),
            torch.tensor(energy, dtype=torch.float32),
        ).unsqueeze(0)                                 # [1, O=proprio+vision+energy]
        with self._lock:  # serialise: the planner's RNG/state is not re-entrant
            action = self.planner.plan(obs)
        return [float(v) for v in action.detach().cpu().tolist()]


@contextmanager
def serve_planner(
    world_model,
    policy: GaussianActorCritic,
    cfg: PlanConfig | None = None,
    *,
    host: str,
    port: int,
    seed: int | None = None,
) -> Iterator[dict[str, object]]:
    service = _PlannerInferenceService(world_model, policy, cfg, seed=seed)
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
