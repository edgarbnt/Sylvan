"""Behavioural diagnostic for a Sylvan replay-buffer run.

The aggregate metrics (mean uprightness/height) are a poor proxy for what the
agent actually *does*: a frozen agent and a balancing agent can score nearly the
same. This tool reads the raw transitions of a run directory and derives the
behavioural signals that the headless numbers normally hide, so the training
loop can be driven without watching the 3D view for every cycle:

  - FREEZE      : mean effort + fraction of near-zero-effort steps.
  - POSTURE     : final/min torso height + mean pose_error (curl / crouch / ball).
  - TOPPLE      : signed forward COM drift (face-plant vs sit-back) + lateral drift.
  - STEPPING    : single-foot-contact fraction + foot-contact switches + |fwd vel|.
  - SURVIVAL    : episode-length distribution + fall vs truncation breakdown.

Nothing here touches the simulator or the data contract: it is pure offline
analysis over the proprio (94, quad) / metrics fields already stored in the buffer.

Usage (from the python/ directory):
    python3 -m scripts.diagnose_run <run_dir>
    python3 -m scripts.diagnose_run --prefix activebal            # latest training cycle
    python3 -m scripts.diagnose_run --prefix activebal --validation
    python3 -m scripts.diagnose_run --prefix activebal --both     # train + validation
    python3 -m scripts.diagnose_run <run_dir> --json              # machine-readable
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from sylvan.buffer.reader import iter_episodes
from sylvan.constants import REPLAY_BUFFER_DIR, REPORTS_DIR

# COM (relative to the spawn root) lives at proprio[65:68] — see
# sylvan_agent.gd:_rebuild_proprioception (QUADRUPED: 7 + 9 bodies×6=54 + 4 foot contacts = 65,
# then COM at 65,66,67). Forward = +z so COM_Z is fore/aft drift.
COM_X, COM_Y, COM_Z = 65, 66, 67
FROZEN_EFFORT = 0.01      # below this an action is doing ~nothing
EARLY_TERM_STEPS = 80     # matches the acceptance gate's early-termination cutoff


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if pct <= 0:
        return s[0]
    if pct >= 100:
        return s[-1]
    idx = (len(s) - 1) * (pct / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] * (1.0 - (idx - lo)) + s[hi] * (idx - lo)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def behavior_digest(run_dir: Path) -> dict[str, object]:
    episodes = iter_episodes(run_dir)
    if not episodes:
        return {"run_dir": str(run_dir), "num_episodes": 0}

    lengths: list[float] = []
    returns: list[float] = []
    all_effort: list[float] = []
    frozen_steps = 0
    total_steps = 0
    single_contact_steps = 0
    contact_switches: list[float] = []
    final_heights: list[float] = []
    min_heights: list[float] = []
    final_pose_errors: list[float] = []
    final_com_fwd: list[float] = []
    final_com_lat: list[float] = []
    abs_fwd_vel: list[float] = []
    fall_count = 0
    trunc_count = 0
    early_term = 0

    for ep in episodes:
        lengths.append(float(len(ep)))
        ep_return = 0.0
        prev_contact = None
        switches = 0
        ep_min_h = math.inf
        for t in ep:
            m = t.obs.metrics
            ep_return += t.reward
            eff = float(m.get("effort", 0.0))
            all_effort.append(eff)
            if eff < FROZEN_EFFORT:
                frozen_steps += 1
            total_steps += 1
            gc = float(m.get("ground_contact", 0.0))
            if abs(gc - 0.5) < 1e-3:
                single_contact_steps += 1
            if prev_contact is not None and abs(gc - prev_contact) > 1e-3:
                switches += 1
            prev_contact = gc
            ep_min_h = min(ep_min_h, float(m.get("height", 0.0)))
            abs_fwd_vel.append(abs(float(m.get("forward_velocity", 0.0))))
        returns.append(ep_return)
        contact_switches.append(float(switches))
        last = ep[-1]
        final_heights.append(float(last.obs.metrics.get("height", 0.0)))
        min_heights.append(ep_min_h if ep_min_h != math.inf else 0.0)
        final_pose_errors.append(float(last.obs.metrics.get("pose_error", 0.0)))
        proprio = last.obs.proprio
        if len(proprio) > COM_Z:
            final_com_fwd.append(float(proprio[COM_Z]))   # +z forward, -z back
            final_com_lat.append(abs(float(proprio[COM_X])))
        if last.done:
            fall_count += 1
        if last.truncated:
            trunc_count += 1
        if len(ep) < EARLY_TERM_STEPS:
            early_term += 1

    n = len(episodes)
    frozen_frac = frozen_steps / max(1, total_steps)
    single_frac = single_contact_steps / max(1, total_steps)

    return {
        "run_dir": str(run_dir),
        "num_episodes": n,
        "survival": {
            "mean_length": _mean(lengths),
            "p10_length": _percentile(lengths, 10),
            "p50_length": _percentile(lengths, 50),
            "max_length": max(lengths),
            "fall_rate": fall_count / n,
            "trunc_rate": trunc_count / n,
            "early_term_rate": early_term / n,
        },
        "freeze": {
            "mean_effort": _mean(all_effort),
            "frozen_step_frac": frozen_frac,
        },
        "posture": {
            "mean_final_height": _mean(final_heights),
            "mean_min_height": _mean(min_heights),
            "mean_final_pose_error": _mean(final_pose_errors),
        },
        "topple": {
            "mean_final_com_forward": _mean(final_com_fwd),
            "mean_final_com_lateral": _mean(final_com_lat),
        },
        "stepping": {
            "single_contact_frac": single_frac,
            "mean_contact_switches": _mean(contact_switches),
            "mean_abs_forward_velocity": _mean(abs_fwd_vel),
        },
        "reward": {"mean_return": _mean(returns)},
    }


def _verdict(d: dict[str, object]) -> list[str]:
    """Heuristic flags — the headline behavioural read for the loop."""
    flags: list[str] = []
    if d.get("num_episodes", 0) == 0:
        return ["EMPTY: no episodes"]
    sv, fz, po, tp, st = d["survival"], d["freeze"], d["posture"], d["topple"], d["stepping"]
    if fz["frozen_step_frac"] > 0.6 and fz["mean_effort"] < 0.03:
        flags.append(f"FROZEN: {fz['frozen_step_frac']:.0%} near-zero-effort steps (mean {fz['mean_effort']:.3f})")
    if po["mean_min_height"] < 0.45:
        flags.append(f"CROUCH/BALL: min height drops to {po['mean_min_height']:.2f} (pose_err {po['mean_final_pose_error']:.2f})")
    if tp["mean_final_com_forward"] > 0.12:
        flags.append(f"FACE-PLANT lean: COM ends {tp['mean_final_com_forward']:+.2f} forward")
    elif tp["mean_final_com_forward"] < -0.12:
        flags.append(f"SIT-BACK lean: COM ends {tp['mean_final_com_forward']:+.2f} back")
    if st["single_contact_frac"] > 0.15 or st["mean_contact_switches"] > 4:
        flags.append(f"STEPPING signs: {st['single_contact_frac']:.0%} single-foot, {st['mean_contact_switches']:.1f} switches/ep")
    if sv["fall_rate"] > 0.5:
        flags.append(f"FALLS: {sv['fall_rate']:.0%} of episodes terminate on a fall")
    if not flags:
        flags.append("nominal: no degenerate pattern detected")
    return flags


def _print_digest(d: dict[str, object]) -> None:
    print(f"\n=== {d['run_dir']} ===")
    if d.get("num_episodes", 0) == 0:
        print("  (no episodes)")
        return
    sv, fz, po, tp, st, rw = d["survival"], d["freeze"], d["posture"], d["topple"], d["stepping"], d["reward"]
    print(f"  episodes={d['num_episodes']}  return={rw['mean_return']:.1f}")
    print(f"  SURVIVAL  len mean={sv['mean_length']:.0f} p10={sv['p10_length']:.0f} p50={sv['p50_length']:.0f} max={sv['max_length']:.0f}"
          f" | fall={sv['fall_rate']:.0%} trunc={sv['trunc_rate']:.0%} early={sv['early_term_rate']:.0%}")
    print(f"  FREEZE    mean_effort={fz['mean_effort']:.3f}  frozen_steps={fz['frozen_step_frac']:.0%}")
    print(f"  POSTURE   final_h={po['mean_final_height']:.2f} min_h={po['mean_min_height']:.2f} pose_err={po['mean_final_pose_error']:.2f}")
    print(f"  TOPPLE    com_fwd={tp['mean_final_com_forward']:+.2f} com_lat={tp['mean_final_com_lateral']:.2f}")
    print(f"  STEPPING  single_foot={st['single_contact_frac']:.0%} switches/ep={st['mean_contact_switches']:.1f} |fwd_vel|={st['mean_abs_forward_velocity']:.3f}")
    print("  VERDICT:")
    for f in _verdict(d):
        print(f"    - {f}")


def _list_cycles(prefix: str, validation: bool) -> list[tuple[int, Path]]:
    """All cycle dirs for a prefix, sorted by cycle number."""
    out: list[tuple[int, Path]] = []
    if not REPLAY_BUFFER_DIR.exists():
        return out
    for p in REPLAY_BUFFER_DIR.iterdir():
        if not p.is_dir() or not p.name.startswith(f"{prefix}_cycle_"):
            continue
        if p.name.endswith("_validation") != validation:
            continue
        try:
            cycle = int(p.name[len(f"{prefix}_cycle_"):].split("_")[0])
        except (ValueError, IndexError):
            continue
        out.append((cycle, p))
    return sorted(out, key=lambda x: x[0])


def _print_trend(prefix: str, validation: bool) -> None:
    cycles = _list_cycles(prefix, validation)
    tag = "validation" if validation else "training"
    if not cycles:
        print(f"(no {tag} cycles for prefix '{prefix}')")
        return
    print(f"\n=== TREND: {prefix} ({tag}) ===")
    print(f"  {'cyc':>4} {'len':>5} {'fall%':>6} {'effort':>7} {'min_h':>6} {'com_fwd':>8} {'1foot%':>7} {'return':>7}")
    for cycle, path in cycles:
        d = behavior_digest(path)
        if d.get("num_episodes", 0) == 0:
            print(f"  {cycle:>4}  (empty)")
            continue
        sv, fz, po, tp, st, rw = d["survival"], d["freeze"], d["posture"], d["topple"], d["stepping"], d["reward"]
        print(f"  {cycle:>4} {sv['mean_length']:>5.0f} {sv['fall_rate']*100:>5.0f}% "
              f"{fz['mean_effort']:>7.3f} {po['mean_min_height']:>6.2f} {tp['mean_final_com_forward']:>+8.2f} "
              f"{st['single_contact_frac']*100:>6.0f}% {rw['mean_return']:>7.1f}")


def _find_latest(prefix: str, validation: bool) -> Path | None:
    if not REPLAY_BUFFER_DIR.exists():
        return None
    best_cycle = -1
    best_path: Path | None = None
    for p in REPLAY_BUFFER_DIR.iterdir():
        if not p.is_dir() or not p.name.startswith(f"{prefix}_cycle_"):
            continue
        is_val = p.name.endswith("_validation")
        if is_val != validation:
            continue
        try:
            core = p.name[len(f"{prefix}_cycle_"):]
            cycle = int(core.split("_")[0])
        except (ValueError, IndexError):
            continue
        if cycle > best_cycle:
            best_cycle = cycle
            best_path = p
    return best_path


def _print_training_report() -> None:
    """Surface the J1 transfer metric + actor-frozen verdict from the saved cycle
    report (data/reports/phase2_cycle_report.json). Offline, read-only."""
    report_path = REPORTS_DIR / "phase2_cycle_report.json"
    if not report_path.exists():
        print(f"(no cycle report at {report_path})")
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(f"\n=== TRAINING REPORT: {report_path} ===")

    health = (report.get("controller") or {}).get("health")
    if health is None:
        print("  HEALTH    (not recorded)")
    elif health.get("actor_frozen"):
        print(f"  HEALTH    \033[91mACTOR FROZEN\033[0m — rel_span={health.get('rel_span'):.2e} "
              f"(first={health.get('first'):.3f} last={health.get('last'):.3f}); no gradient flowing.")
    else:
        print(f"  HEALTH    actor moving (rel_span={health.get('rel_span'):.2e}, "
              f"{health.get('n_epochs')} epochs)")

    transfer = (report.get("validation") or {}).get("transfer")
    if not transfer or not transfer.get("num_episodes"):
        print("  TRANSFER  (not recorded)")
        return
    print(f"  TRANSFER  imagined={transfer['mean_imagined_return']:.2f} "
          f"real={transfer['mean_real_return']:.2f} |err|={transfer['mean_abs_return_error']:.2f} "
          f"ratio={transfer['return_error_ratio']:.2f} reward_mae={transfer['per_step_reward_mae']:.4f}")
    gap = transfer["mean_imagined_return"] - transfer["mean_real_return"]
    if gap > 0.5 * (abs(transfer["mean_real_return"]) + 1e-6):
        print(f"  VERDICT   MODEL-EXPLOITATION: dream pays {gap:.1f} more than reality.")
    else:
        print("  VERDICT   imagination roughly transfers to reality.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Behavioural diagnostic for a Sylvan run.")
    ap.add_argument("run_dir", nargs="?", default=None, help="Replay-buffer run directory.")
    ap.add_argument("--prefix", default=None, help="Auto-find the latest cycle for this run prefix.")
    ap.add_argument("--validation", action="store_true", help="Use the _validation sibling run.")
    ap.add_argument("--both", action="store_true", help="Show both the training and validation runs.")
    ap.add_argument("--trend", action="store_true", help="With --prefix: one line per cycle (progress curve).")
    ap.add_argument("--training-report", action="store_true",
                    help="Show the J1 transfer metric + actor-frozen verdict from the latest cycle report.")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = ap.parse_args()

    if args.training_report:
        _print_training_report()
        return

    if args.trend:
        if not args.prefix:
            ap.error("--trend requires --prefix")
        _print_trend(args.prefix, validation=False)
        if args.both:
            _print_trend(args.prefix, validation=True)
        return

    targets: list[Path] = []
    if args.run_dir:
        targets.append(Path(args.run_dir))
    elif args.prefix:
        if args.both:
            for val in (False, True):
                p = _find_latest(args.prefix, val)
                if p:
                    targets.append(p)
        else:
            p = _find_latest(args.prefix, args.validation)
            if p:
                targets.append(p)
    if not targets:
        ap.error("no run found — pass a run_dir or a --prefix that matches an existing cycle")

    digests = [behavior_digest(t) for t in targets]
    if args.json:
        print(json.dumps({"runs": [{**d, "verdict": _verdict(d)} for d in digests]}, indent=2))
    else:
        for d in digests:
            _print_digest(d)


if __name__ == "__main__":
    main()
