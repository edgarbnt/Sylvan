"""J1a.1 — collect clean grounded rollouts for world-model retraining.

Serves the WORKING J0 policy (default policy_best.pt) in Godot and writes real
(proprio, action, reward, next, done) episodes to the replay buffer. The world
model (J1a.2) is fit on these. ONE batch per invocation — call it several times
with different --run-prefix / --exploration-std / --perturbation-strength to build
a mix: a clean set, a higher-noise diverse set, and a PERTURBED set (off-balance
states + real falls, which the WM must learn to predict — the J1a.3 gate).

The action stored in the JSONL is whatever Godot finally applied; the WM learns
action->next regardless of how the action was sampled, so exploration here only
adds useful state diversity (no exactness constraint, unlike PPO collection).

Usage (from python/):
    python3 -m scripts.collect_wm_data --run-prefix wm_v1_data --episodes 40
    python3 -m scripts.collect_wm_data --run-prefix wm_v1_data_div --episodes 20 --exploration-std 0.15
    python3 -m scripts.collect_wm_data --run-prefix wm_v1_data_perturbed --episodes 20 --perturbation-strength 45
    python3 -m scripts.collect_wm_data --run-prefix wm_v1_data_val --episodes 12   # held-out validation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sylvan.config import SylvanConfig
from sylvan.constants import CHECKPOINTS_DIR
from sylvan.buffer.reader import iter_episodes
from sylvan.control.ppo.policy import GaussianActorCritic
from sylvan.control.ppo.stochastic_server import serve_stochastic_policy
from sylvan.orchestration.day_cycle import prepare_day_run, run_godot_day
from sylvan.training.checkpointing import load_checkpoint


def _digest(run_dir: Path) -> dict[str, float]:
    episodes = [ep for ep in iter_episodes(Path(run_dir)) if ep]
    if not episodes:
        return {"mean_return": 0.0, "mean_length": 0.0, "fall_rate": 0.0, "num_episodes": 0.0}
    returns = [sum(t.reward for t in ep) for ep in episodes]
    lengths = [len(ep) for ep in episodes]
    falls = sum(1 for ep in episodes if ep[-1].done)
    n = len(episodes)
    return {
        "mean_return": sum(returns) / n,
        "mean_length": sum(lengths) / n,
        "fall_rate": falls / n,
        "num_episodes": float(n),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="J1a.1 collect grounded rollouts for the WM.")
    ap.add_argument("--checkpoint", default=str(CHECKPOINTS_DIR / "ppo_j0" / "policy_best.pt"))
    ap.add_argument("--run-prefix", default="wm_v1_data")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--exploration-std", type=float, default=0.0,
                    help="Extra Godot-side exploration noise added on top of the policy's own std.")
    ap.add_argument("--perturbation-strength", type=float, default=0.0,
                    help="External-push impulse magnitude (N·s); >0 creates off-balance states + falls.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    config = SylvanConfig()
    config.day.num_episodes = args.episodes
    config.env.max_episode_steps = args.max_steps
    config.env.seed = args.seed

    policy = GaussianActorCritic(
        obs_dim=config.env.policy_input_dim,   # 106 = proprio(94) ++ vision(12); the quad J0 policy
        hidden_dim=config.controller.hidden_dim,   # consumes the radar too (stochastic_server concatenates it)
        action_dim=config.env.action_dim,
    )
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"[collect] checkpoint not found: {ckpt}")
    payload = load_checkpoint(ckpt, policy)
    policy.eval()
    print(f"[collect] policy {ckpt.name} epoch={payload.get('epoch')} | "
          f"episodes={args.episodes} max_steps={args.max_steps} "
          f"expl_std={args.exploration_std} perturb={args.perturbation_strength}")

    run_dir = prepare_day_run(config, run_name=args.run_prefix)
    with serve_stochastic_policy(
        policy, host=config.godot.policy_host, port=config.godot.policy_port, seed=args.seed
    ) as srv:
        run_godot_day(
            config,
            run_dir,
            policy_server_host=srv["host"],
            policy_server_port=srv["port"],
            exploration_noise_initial=args.exploration_std,
            exploration_noise_final=args.exploration_std,
            collector_mode="policy_server",
            perturbation_strength=args.perturbation_strength,
        )

    d = _digest(run_dir)
    print(f"[collect] DONE {run_dir} | episodes={int(d['num_episodes'])} "
          f"mean_return={d['mean_return']:.2f} mean_length={d['mean_length']:.1f} "
          f"fall_rate={d['fall_rate']*100:.0f}%")


if __name__ == "__main__":
    main()
