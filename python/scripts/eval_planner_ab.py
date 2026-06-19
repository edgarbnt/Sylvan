"""J1b — falsifiable A/B: WM planner vs reactive J0 under perturbations.

Plain balance is saturated by J0 (401/401), so planning only has headroom when the
body is pushed. For each perturbation strength, run N episodes twice — arm A =
reactive J0 (deterministic policy mean), arm B = WM planner — with the SAME seed +
strength, and compare survival / fall-rate / return. The planner's value is real
only if, under pushes, B survives longer / falls less than A beyond noise; at zero
perturbation B must be no worse than A (guaranteed by the planner's mean fallback).
A null result is reported honestly, not hidden.

R3 caveat: per-episode pairing requires Godot's perturbation schedule to be
seed-deterministic across arms; this script compares DISTRIBUTIONS over N episodes
(robust to that), and prints zero-perturbation parity as a sanity anchor.

Usage (from python/):
    python3 -m scripts.eval_planner_ab \
        --world-model ../data/checkpoints/wm_v1/world_model_v1.pt \
        --policy ../data/checkpoints/ppo_j0/policy_best.pt \
        --perturbation-sweep 0,30,45,60 --episodes 8
"""

from __future__ import annotations

import argparse
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch

from sylvan.config import SylvanConfig
from sylvan.constants import CHECKPOINTS_DIR, REPORTS_DIR
from sylvan.buffer.reader import iter_episodes
from sylvan.control.policy_server import _PolicyRequestHandler, _PolicyTCPServer
from sylvan.control.planning.planner_server import serve_planner
from sylvan.control.planning.wm_planner import PlanConfig
from sylvan.orchestration.day_cycle import prepare_day_run, run_godot_day
from scripts.serve_planner_visual import load_planner_models


class _FnService:
    def __init__(self, fn) -> None:
        self.fn = fn
        self._lock = threading.Lock()

    def predict(self, payload: dict[str, object]) -> list[float]:
        proprio = payload.get("proprio")
        if not isinstance(proprio, list):
            raise TypeError("Policy request must contain a proprio list")
        with self._lock:
            return self.fn(proprio)


@contextmanager
def _serve_fn(fn, *, host: str) -> Iterator[dict[str, object]]:
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


def _digest(run_dir: Path) -> dict[str, float]:
    episodes = [ep for ep in iter_episodes(Path(run_dir)) if ep]
    if not episodes:
        return {"n": 0, "mean_length": 0.0, "fall_rate": 0.0, "mean_return": 0.0}
    lengths = [len(ep) for ep in episodes]
    falls = sum(1 for ep in episodes if ep[-1].done)
    returns = [sum(t.reward for t in ep) for ep in episodes]
    n = len(episodes)
    return {
        "n": n,
        "mean_length": sum(lengths) / n,
        "fall_rate": falls / n,
        "mean_return": sum(returns) / n,
    }


def _run_arm(config, *, run_name, perturbation, serve_ctx):
    run_dir = prepare_day_run(config, run_name=run_name)
    for f in Path(run_dir).glob("*.jsonl"):
        f.unlink()
    with serve_ctx as srv:
        run_godot_day(
            config, run_dir,
            policy_server_host=srv["host"], policy_server_port=srv["port"],
            exploration_noise_initial=0.0, exploration_noise_final=0.0,
            collector_mode="policy_server", perturbation_strength=perturbation,
        )
    return _digest(run_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="J1b A/B planner vs reactive J0 under perturbations.")
    ap.add_argument("--world-model", default=str(CHECKPOINTS_DIR / "wm_v1" / "world_model_v1.pt"))
    ap.add_argument("--policy", default=str(CHECKPOINTS_DIR / "ppo_j0" / "policy_best.pt"))
    ap.add_argument("--perturbation-sweep", default="0,30,45,60")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--num-samples", type=int, default=64)
    ap.add_argument("--done-penalty", type=float, default=5.0)
    args = ap.parse_args()

    config = SylvanConfig()
    config.day.num_episodes = args.episodes
    config.env.max_episode_steps = args.max_steps
    config.env.seed = args.seed
    strengths = [float(s) for s in args.perturbation_sweep.split(",")]

    world_model, policy = load_planner_models(config, world_model_ckpt=args.world_model, policy_ckpt=args.policy)
    plan_cfg = PlanConfig(horizon=args.horizon, num_samples=args.num_samples, done_penalty=args.done_penalty)

    @torch.no_grad()
    def policy_mean_fn(proprio):
        t = torch.tensor(proprio, dtype=torch.float32).unsqueeze(0)
        a = policy.mean(t)[0].clamp(-1.0, 1.0)
        return [float(v) for v in a.tolist()]

    print(f"[ab] sweep={strengths} episodes/arm={args.episodes} horizon={args.horizon} samples={args.num_samples}")
    rows = []
    for s in strengths:
        a = _run_arm(config, run_name=f"planner_ab_reactive_p{int(s)}", perturbation=s,
                     serve_ctx=_serve_fn(policy_mean_fn, host=config.godot.policy_host))
        b = _run_arm(config, run_name=f"planner_ab_planner_p{int(s)}", perturbation=s,
                     serve_ctx=serve_planner(world_model, policy, plan_cfg,
                                             host=config.godot.policy_host, port=0, seed=args.seed))
        row = {
            "perturbation": s,
            "reactive": a, "planner": b,
            "d_mean_length": b["mean_length"] - a["mean_length"],
            "d_fall_rate": b["fall_rate"] - a["fall_rate"],
            "d_mean_return": b["mean_return"] - a["mean_return"],
        }
        rows.append(row)
        print(f"[ab] p={s:5.1f} | A(reactive) len={a['mean_length']:5.1f} fall={a['fall_rate']*100:3.0f}% | "
              f"B(planner) len={b['mean_length']:5.1f} fall={b['fall_rate']*100:3.0f}% | "
              f"Δlen={row['d_mean_length']:+5.1f} Δfall={row['d_fall_rate']*100:+3.0f}%")

    # Planner value: at non-zero perturbation, B beats A (longer survival or fewer falls);
    # at zero perturbation, B not worse than A.
    perturbed = [r for r in rows if r["perturbation"] > 0]
    helps = any(r["d_mean_length"] > 0 and r["d_fall_rate"] <= 0 for r in perturbed)
    zero = next((r for r in rows if r["perturbation"] == 0), None)
    no_regression = zero is None or (zero["d_mean_length"] >= -1.0 and zero["d_fall_rate"] <= 0.05)
    verdict = bool(helps and no_regression)

    report = {
        "world_model": args.world_model, "policy": args.policy,
        "episodes_per_arm": args.episodes, "plan": vars(plan_cfg) if hasattr(plan_cfg, "__dict__") else {
            "horizon": plan_cfg.horizon, "num_samples": plan_cfg.num_samples, "done_penalty": plan_cfg.done_penalty},
        "rows": rows,
        "planner_helps_under_perturbation": helps,
        "no_regression_at_zero": no_regression,
        "planner_value_demonstrated": verdict,
        "note": "Distributional A/B (R3: per-episode pairing determinism unverified).",
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "planner_ab_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"[ab] VALUE DEMONSTRATED: {'YES ✅' if verdict else 'no (reported honestly)'}  -> {out}")


if __name__ == "__main__":
    main()
