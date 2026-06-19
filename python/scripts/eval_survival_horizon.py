"""Long-horizon survival eval: run a trained policy for MANY steps to see the REAL
time-to-fall distribution (the training fall_rate is truncated at max_episode_steps=400,
so a 10% rate only means '90% survive 400 steps' — it says nothing about step 500+)."""

from __future__ import annotations

import argparse
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch

from sylvan.config import SylvanConfig
from sylvan.buffer.reader import iter_episodes
from sylvan.control.policy_server import _PolicyRequestHandler, _PolicyTCPServer
from sylvan.control.ppo.policy import GaussianActorCritic
from sylvan.orchestration.day_cycle import prepare_day_run, run_godot_day
from sylvan.training.checkpointing import load_checkpoint


class _MeanService:
    """Deterministic: serves the policy MEAN action (no exploration noise)."""

    def __init__(self, policy: GaussianActorCritic) -> None:
        policy.eval()
        for p in policy.parameters():
            p.requires_grad_(False)
        self.policy = policy

    @torch.no_grad()
    def predict(self, payload: dict[str, object]) -> list[float]:
        t = torch.tensor(payload["proprio"], dtype=torch.float32).unsqueeze(0)
        a = self.policy.mean(t)[0]
        a = torch.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        return [float(v) for v in a.tolist()]


@contextmanager
def _serve(policy, *, host: str) -> Iterator[dict]:
    server = _PolicyTCPServer((host, 0), _PolicyRequestHandler, inference_service=_MeanService(policy))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        h, p = server.server_address
        yield {"host": h, "port": p}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=2000)
    args = ap.parse_args()

    config = SylvanConfig()
    config.day.num_episodes = args.episodes
    config.env.max_episode_steps = args.max_steps

    policy = GaussianActorCritic(
        obs_dim=config.env.proprio_dim,
        hidden_dim=config.controller.hidden_dim,
        action_dim=config.env.action_dim,
    )
    load_checkpoint(Path(args.checkpoint), policy)

    run_dir = prepare_day_run(config, run_name="eval_survival")
    for f in Path(run_dir).glob("**/*.jsonl"):
        f.unlink()
    with _serve(policy, host=config.godot.policy_host) as srv:
        run_godot_day(
            config, run_dir,
            policy_server_host=srv["host"], policy_server_port=srv["port"],
            exploration_noise_initial=0.0, exploration_noise_final=0.0,
            collector_mode="policy_server",
        )
    episodes = [ep for ep in iter_episodes(Path(run_dir)) if ep]
    lens = sorted(len(ep) for ep in episodes)
    fell = [len(ep) for ep in episodes if ep[-1].done]
    survived = [len(ep) for ep in episodes if not ep[-1].done]
    print(f"\n=== SURVIVAL @ horizon {args.max_steps} ({len(episodes)} episodes) ===")
    print(f"lengths (sorted): {lens}")
    print(f"FELL before horizon: {len(fell)}/{len(episodes)}  -> time-to-fall steps: {sorted(fell)}")
    print(f"reached horizon (no fall): {len(survived)}/{len(episodes)}")
    if lens:
        print(f"mean length={sum(lens)/len(lens):.0f}  median={lens[len(lens)//2]}  "
              f"min={lens[0]}  max={lens[-1]}")


if __name__ == "__main__":
    main()
