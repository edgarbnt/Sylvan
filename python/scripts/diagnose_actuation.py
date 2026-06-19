"""Actuation authority diagnostic (J0 post-mortem).

J0 PPO ran 250 iters and NEVER reduced fall% below 100%; the trained policy
survives no longer than an untrained one. Two candidate root causes: (a) the
reward gives no gradient toward active correction, or (b) the actions have no
physical authority over the fall (a body/joint problem). This script tranches
between them: it measures fall-time for three controllers on the SAME body, no
crutch, over a long horizon.

  - ZERO    : action = 0 everywhere (do nothing)
  - RANDOM  : action ~ Uniform[-1, 1] (maximal motor thrashing)
  - J0      : the trained J0 mean policy

If all three fall at ~the same step, the actions have NO authority over the fall
=> physical/joint problem (no amount of RL fixes this). If RANDOM or J0 survive
markedly longer/shorter than ZERO, actions DO matter => the blocker is reward
shaping / exploration, not physics.

Usage (from python/):
    python3 -m scripts.diagnose_actuation --episodes 5 --max-steps 400
"""

from __future__ import annotations

import argparse
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch

from sylvan.config import SylvanConfig
from sylvan.constants import CHECKPOINTS_DIR
from sylvan.buffer.reader import iter_episodes
from sylvan.control.policy_server import _PolicyRequestHandler, _PolicyTCPServer
from sylvan.control.ppo.policy import GaussianActorCritic
from sylvan.orchestration.day_cycle import prepare_day_run, run_godot_day
from sylvan.training.checkpointing import load_checkpoint


class _FnService:
    """Serves whatever `fn(proprio_list) -> action_list` returns."""

    def __init__(self, fn) -> None:
        self.fn = fn

    def predict(self, payload: dict[str, object]) -> list[float]:
        proprio = payload.get("proprio")
        if not isinstance(proprio, list):
            raise TypeError("Policy request must contain a proprio list")
        return self.fn(proprio)


@contextmanager
def _serve(fn, *, host: str) -> Iterator[dict[str, object]]:
    server = _PolicyTCPServer((host, 0), _PolicyRequestHandler, inference_service=_FnService(fn))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        h, p = server.server_address
        yield {"host": h, "port": p}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _measure(config: SylvanConfig, fn, *, label: str) -> dict[str, float]:
    run_dir = prepare_day_run(config, run_name=f"diag_actuation_{label}")
    for f in Path(run_dir).glob("*.jsonl"):  # clear stale episodes (fixed run name)
        f.unlink()
    with _serve(fn, host=config.godot.policy_host) as srv:
        run_godot_day(
            config,
            run_dir,
            policy_server_host=srv["host"],
            policy_server_port=srv["port"],
            exploration_noise_initial=0.0,
            exploration_noise_final=0.0,
            collector_mode="policy_server",
        )
    episodes = [ep for ep in iter_episodes(Path(run_dir)) if ep]
    if not episodes:
        return {"label": label, "n": 0, "mean_len": 0.0, "fall_rate": 0.0}
    lengths = [len(ep) for ep in episodes]
    falls = sum(1 for ep in episodes if ep[-1].done)
    n = len(episodes)
    return {
        "label": label,
        "n": float(n),
        "mean_len": sum(lengths) / n,
        "min_len": float(min(lengths)),
        "max_len": float(max(lengths)),
        "fall_rate": falls / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Actuation authority diagnostic.")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--checkpoint", default=str(CHECKPOINTS_DIR / "ppo_j0" / "policy_latest.pt")
    )
    args = ap.parse_args()

    config = SylvanConfig()
    config.day.num_episodes = args.episodes
    config.env.max_episode_steps = args.max_steps

    action_dim = config.env.action_dim
    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)

    # J0 trained mean policy.
    policy = GaussianActorCritic(
        obs_dim=config.env.proprio_dim, hidden_dim=config.controller.hidden_dim, action_dim=action_dim
    )
    ckpt = Path(args.checkpoint)
    j0_ok = ckpt.exists()
    if j0_ok:
        payload = load_checkpoint(ckpt, policy)
        policy.eval()
        print(f"[diag] J0 checkpoint epoch={payload.get('epoch')} metrics={payload.get('metrics')}")
    else:
        print(f"[diag] WARNING: no J0 checkpoint at {ckpt} — skipping J0 arm.")

    def zero_fn(proprio):
        return [0.0] * action_dim

    def random_fn(proprio):
        a = torch.rand(action_dim, generator=rng) * 2.0 - 1.0
        return [float(v) for v in a.tolist()]

    @torch.no_grad()
    def j0_fn(proprio):
        t = torch.tensor(proprio, dtype=torch.float32).unsqueeze(0)
        a = policy.mean(t)[0].clamp(-1.0, 1.0)
        return [float(v) for v in a.tolist()]

    arms = [("ZERO", zero_fn), ("RANDOM", random_fn)]
    if j0_ok:
        arms.append(("J0", j0_fn))

    print(f"[diag] {args.episodes} episodes/arm, max_steps={args.max_steps}, no crutch\n")
    results = []
    for label, fn in arms:
        r = _measure(config, fn, label=label)
        results.append(r)
        print(
            f"[diag] {label:6s} | mean_len={r['mean_len']:6.1f} "
            f"(min={r.get('min_len',0):.0f} max={r.get('max_len',0):.0f}) "
            f"fall={r['fall_rate']*100:.0f}% (n={int(r['n'])})"
        )

    print("\n=== VERDICT ===")
    by = {r["label"]: r["mean_len"] for r in results}
    base = by.get("ZERO", 0.0)
    spread = max(by.values()) - min(by.values()) if by else 0.0
    print(f"ZERO mean_len = {base:.1f}")
    if spread < 0.15 * max(base, 1.0):
        print(
            "→ Tous les bras tombent quasi au meme pas (spread {:.1f} < 15%). "
            "Les ACTIONS N'ONT PAS d'autorite sur la chute => probleme PHYSIQUE/JOINTS.".format(spread)
        )
    else:
        print(
            "→ Spread {:.1f} significatif: les actions CHANGENT la chute => "
            "probleme de REWARD SHAPING / exploration, pas la physique.".format(spread)
        )
        for label, v in sorted(by.items(), key=lambda kv: -kv[1]):
            print(f"    {label:6s}: {v:.1f}")


if __name__ == "__main__":
    main()
