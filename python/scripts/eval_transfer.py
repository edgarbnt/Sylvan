"""CLI: measure imagination->reality transfer (J1) on a saved validation run.

Pure offline, CPU. Loads the trained controller + world model and compares, on
the REAL start states of a validation run, the imagined return the policy gets in
the world model against the real return it got in Godot (matched horizon), plus
the reward-head fidelity under forced real actions. See sylvan.evaluation.transfer.

Usage (from the python/ directory):
    python3 -m scripts.eval_transfer <validation_run_dir>
    python3 -m scripts.eval_transfer --prefix activebal_r5        # latest *_validation
    python3 -m scripts.eval_transfer --prefix activebal_r5 --horizon 30 --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sylvan.config import SylvanConfig
from sylvan.constants import CHECKPOINTS_DIR, REPLAY_BUFFER_DIR
from sylvan.evaluation.transfer import evaluate_transfer_from_checkpoints


def _find_latest_validation(prefix: str) -> Path | None:
    if not REPLAY_BUFFER_DIR.exists():
        return None
    best_cycle, best_path = -1, None
    for p in REPLAY_BUFFER_DIR.iterdir():
        if not p.is_dir() or not p.name.startswith(f"{prefix}_cycle_"):
            continue
        if not p.name.endswith("_validation"):
            continue
        try:
            cycle = int(p.name[len(f"{prefix}_cycle_"):].split("_")[0])
        except (ValueError, IndexError):
            continue
        if cycle > best_cycle:
            best_cycle, best_path = cycle, p
    return best_path


def _resolve_checkpoint(explicit: str | None, *stable_then_latest: str) -> Path:
    if explicit:
        return Path(explicit)
    for name in stable_then_latest:
        candidate = CHECKPOINTS_DIR / name
        if candidate.exists():
            return candidate
    return CHECKPOINTS_DIR / stable_then_latest[0]


def _print_digest(d: dict) -> None:
    if d.get("num_episodes", 0) == 0:
        print("(no episodes in run)")
        return
    print(f"\n=== J1 imagination->reality transfer ({d['num_episodes']} episodes, "
          f"horizon {d['horizon']}, mean matched {d['mean_matched_horizon']:.0f}) ===")
    print(f"  imagined return : {d['mean_imagined_return']:.2f}")
    print(f"  real return     : {d['mean_real_return']:.2f}")
    print(f"  |return error|  : {d['mean_abs_return_error']:.2f}   "
          f"(ratio imagined/real = {d['return_error_ratio']:.2f})")
    print(f"  reward-head MAE : {d['per_step_reward_mae']:.4f}   (forced real actions)")
    gap = d["mean_imagined_return"] - d["mean_real_return"]
    if gap > 0.5 * (abs(d["mean_real_return"]) + 1e-6):
        print(f"  >> MODEL-EXPLOITATION: the policy is rewarded {gap:.1f} more in the "
              f"dream than in reality.")
    print(f"  j1_pass={d['j1_pass']}  (thresholds UNCALIBRATED — informational only)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure imagination->reality transfer (J1).")
    ap.add_argument("run_dir", nargs="?", default=None, help="Validation replay-buffer run dir.")
    ap.add_argument("--prefix", default=None, help="Auto-find the latest *_validation for this prefix.")
    ap.add_argument("--horizon", type=int, default=None, help="Override imagined horizon (default: config).")
    ap.add_argument("--controller-checkpoint", default=None)
    ap.add_argument("--world-model-checkpoint", default=None)
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = ap.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.prefix:
        run_dir = _find_latest_validation(args.prefix)
        if run_dir is None:
            ap.error(f"no *_validation run found for prefix '{args.prefix}'")
    else:
        ap.error("pass a validation run_dir or a --prefix")

    config = SylvanConfig()
    wm_ckpt = _resolve_checkpoint(
        args.world_model_checkpoint, "world_model_v0.stable.pt", "world_model_v0.pt"
    )
    ctrl_ckpt = _resolve_checkpoint(
        args.controller_checkpoint, "controller_v0.stable.pt", "controller_v0.pt"
    )

    digest = evaluate_transfer_from_checkpoints(
        config,
        validation_run_dir=run_dir,
        world_model_ckpt=wm_ckpt,
        controller_ckpt=ctrl_ckpt,
        horizon=args.horizon,
    )
    if args.json:
        print(json.dumps({"run_dir": str(run_dir), **digest}, indent=2))
    else:
        print(f"run_dir   : {run_dir}")
        print(f"world_model: {wm_ckpt.name}   controller: {ctrl_ckpt.name}")
        _print_digest(digest)


if __name__ == "__main__":
    main()
