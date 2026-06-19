"""Persistent STOCHASTIC SAC actor server for parallel collection (one per worker).

Identical wire protocol and SIGHUP-reload lifecycle to serve_ppo_collect, but serves the
SAC squashed-Gaussian actor. The action is SAMPLED (SAC's own entropy IS the exploration,
so Godot exploration noise must be 0 → the stored action equals the served action, which
SAC's off-policy update relies on).
"""

from __future__ import annotations

import argparse
import signal
import threading
from pathlib import Path

import torch

from sylvan.config import SylvanConfig
from sylvan.control.policy_server import _PolicyRequestHandler, _PolicyTCPServer
from sylvan.control.sac.models import SacActor
from sylvan.training.checkpointing import load_checkpoint


class _SacService:
    def __init__(self, actor: SacActor, ckpt: Path, seed: int, deterministic: bool) -> None:
        self._freeze(actor)
        self.actor = actor
        self.ckpt = ckpt
        self.deterministic = deterministic
        self.generator = torch.Generator()
        self.generator.manual_seed(int(seed))
        self._lock = threading.Lock()

    @staticmethod
    def _freeze(actor: SacActor) -> None:
        actor.eval()
        for p in actor.parameters():
            p.requires_grad_(False)

    def reload(self) -> None:
        with self._lock:
            load_checkpoint(self.ckpt, self.actor)
            self._freeze(self.actor)

    @torch.no_grad()
    def predict(self, payload: dict[str, object]) -> list[float]:
        proprio = payload.get("proprio")
        if not isinstance(proprio, list):
            raise TypeError("Policy request must contain a proprio list")
        vision = payload.get("vision") or []
        t = torch.tensor(proprio + list(vision), dtype=torch.float32).unsqueeze(0)
        with self._lock:
            action, _ = self.actor.sample(t, deterministic=self.deterministic, generator=self.generator)
        action = torch.nan_to_num(action[0], nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        return [float(v) for v in action.detach().cpu().tolist()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Persistent stochastic SAC collection server.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port-file", required=True)
    ap.add_argument("--ack-file", required=True)
    ap.add_argument("--deterministic", action="store_true", help="Serve the tanh(mean) (eval).")
    args = ap.parse_args()

    config = SylvanConfig()
    actor = SacActor(
        obs_dim=config.env.policy_input_dim,
        hidden_dim=config.controller.hidden_dim,
        action_dim=config.env.action_dim,
    )
    ckpt = Path(args.checkpoint)
    load_checkpoint(ckpt, actor)
    service = _SacService(actor, ckpt, args.seed, args.deterministic)

    def _on_sighup(_signum, _frame) -> None:
        service.reload()
        Path(args.ack_file).write_text("ok")

    signal.signal(signal.SIGHUP, _on_sighup)

    server = _PolicyTCPServer((args.host, args.port), _PolicyRequestHandler, inference_service=service)
    bound_port = int(server.server_address[1])
    port_file = Path(args.port_file)
    tmp = port_file.with_suffix(".tmp")
    tmp.write_text(str(bound_port))
    tmp.rename(port_file)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
