"""Persistent STOCHASTIC PPO policy server for PARALLEL collection (one per worker).

Runs as its OWN process (own GIL), so N of these collect TRULY in parallel — unlike N
server THREADS inside the trainer, which the GIL serialises (the ~1.8x ceiling). The
trainer spawns a pool of these ONCE; torch is imported ONCE per server (persistent), not
per iteration. Each iteration the trainer writes the new behavior checkpoint and sends
SIGHUP; the server reloads it (while idle, between collections) and touches its ack-file.

Wire protocol is the one Godot already speaks ({"proprio":[...]} -> {"action":[...]}), and
the action is SAMPLED from the Gaussian (PPO needs the sampled action; Godot exploration
noise must be 0 so the stored action equals the sampled one).

Startup: binds (host, port=0 -> OS picks), writes the bound port to --port-file (atomic),
then serve_forever. SIGHUP -> reload --checkpoint -> write --ack-file.
"""

from __future__ import annotations

import argparse
import signal
import threading
from pathlib import Path

import torch

from sylvan.config import SylvanConfig
from sylvan.control.policy_server import _PolicyRequestHandler, _PolicyTCPServer
from sylvan.control.ppo.policy import GaussianActorCritic
from sylvan.training.checkpointing import load_checkpoint


class _StochasticService:
    """Samples the action from the Gaussian policy; hot-reloadable via reload()."""

    def __init__(self, policy: GaussianActorCritic, ckpt: Path, seed: int) -> None:
        self._freeze(policy)
        self.policy = policy
        self.ckpt = ckpt
        self.generator = torch.Generator()
        self.generator.manual_seed(int(seed))
        self._lock = threading.Lock()
        self.action_dim = int(policy.log_std.shape[0])  # for the handler's error-fallback width

    @staticmethod
    def _freeze(policy: GaussianActorCritic) -> None:
        policy.eval()
        for p in policy.parameters():
            p.requires_grad_(False)

    def reload(self) -> None:
        with self._lock:
            load_checkpoint(self.ckpt, self.policy)
            self._freeze(self.policy)

    @torch.no_grad()
    def predict(self, payload: dict[str, object]) -> list[float]:
        proprio = payload.get("proprio")
        if not isinstance(proprio, list):
            raise TypeError("Policy request must contain a proprio list")
        vision = payload.get("vision") or []  # food radar; appended AFTER proprio
        t = torch.tensor(proprio + list(vision), dtype=torch.float32).unsqueeze(0)
        with self._lock:
            action, _ = self.policy.sample(t, generator=self.generator)
        action = torch.nan_to_num(action[0], nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        return [float(v) for v in action.detach().cpu().tolist()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Persistent stochastic PPO collection server.")
    ap.add_argument("--checkpoint", required=True, help="Behavior checkpoint (reloaded on SIGHUP).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0, help="0 = OS picks (written to --port-file).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port-file", required=True)
    ap.add_argument("--ack-file", required=True)
    args = ap.parse_args()

    config = SylvanConfig()
    policy = GaussianActorCritic(
        obs_dim=config.env.policy_input_dim,  # proprio ++ vision
        hidden_dim=config.controller.hidden_dim,
        action_dim=config.env.action_dim,
    )
    ckpt = Path(args.checkpoint)
    load_checkpoint(ckpt, policy)
    service = _StochasticService(policy, ckpt, args.seed)

    def _on_sighup(_signum, _frame) -> None:
        # Runs in the main thread between requests (servers are idle between iterations).
        service.reload()
        Path(args.ack_file).write_text("ok")

    signal.signal(signal.SIGHUP, _on_sighup)

    server = _PolicyTCPServer(
        (args.host, args.port), _PolicyRequestHandler, inference_service=service
    )
    bound_port = int(server.server_address[1])
    port_file = Path(args.port_file)
    tmp = port_file.with_suffix(".tmp")
    tmp.write_text(str(bound_port))
    tmp.rename(port_file)  # atomic publish of the port

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
