"""Unified command to run Sylvan training cycles."""

import argparse
import signal
import sys
from pathlib import Path
from typing import Any

from sylvan.config import SylvanConfig
from sylvan.orchestration.run_cycle import run_cycle


class GracefulKiller:
    kill_now = False
    
    def __init__(self) -> None:
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum: int, frame: Any) -> None:
        print("\n[SYLVAN] Interruption received. Will finish current cycle then exit...")
        self.kill_now = True

def _decay_curriculum(cycle: int, span: int, initial: float) -> float:
    """Linear decay of a crutch value from `initial` (cycle 0) to 0 (cycle >= span)."""
    if span <= 0 or initial <= 0.0 or cycle >= span:
        return 0.0
    return float(initial) * (1.0 - cycle / float(span))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Sylvan Circadian Cycles.")
    parser.add_argument("--num-cycles", type=int, default=1, help="Number of day/night cycles to run.")
    parser.add_argument("--steps-per-day", type=int, default=None, help="Number of steps per episode during the day.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for night training.")
    parser.add_argument("--epochs-per-night", type=int, default=None, help="General fallback number of epochs for night training.")
    parser.add_argument("--wm-epochs", type=int, default=None, help="Number of epochs for World Model training.")
    parser.add_argument("--controller-epochs", type=int, default=None, help="Number of epochs for Controller training.")
    parser.add_argument("--run-name-prefix", type=str, default="sylvan_run", help="Prefix for the run directories.")
    # Balance curriculum (the "crutches"): linearly decayed over the first N cycles.
    parser.add_argument("--curriculum-cycles", type=int, default=12, help="Number of cycles over which the balance crutches decay to 0. Set 0 to disable.")
    parser.add_argument("--assist-initial", type=float, default=0.3, help="Initial gravity assist at cycle 0 (0=full gravity, 1=weightless). Secondary crutch; the torso restoring torque is the main stabiliser.")
    parser.add_argument("--reflex-initial", type=float, default=0.85, help="Initial balance-reflex strength at cycle 0 (0..1): joint-neutral bias + active upright torso torque.")
    # Perturbation curriculum (the OPPOSITE of the crutch): random shoves that
    # ramp UP over the run, forcing active balance / stepping recovery.
    parser.add_argument("--perturbation-max", type=float, default=0.0, help="Max random shove impulse (N.s) on the torso. 0 = OFF (default). Meaningful values are ~40-60 (5 is negligible); any balance crutch >=0.3 absorbs them, so use with little/no crutch.")
    parser.add_argument("--perturbation-cycles", type=int, default=10, help="Number of cycles (within this run) over which the shove ramps from gentle to --perturbation-max.")

    args = parser.parse_args()
    config = SylvanConfig()
    
    import os
    if args.steps_per_day is not None:
        config.env.max_episode_steps = args.steps_per_day
        os.environ["SYLVAN_STEPS_PER_DAY"] = str(args.steps_per_day)
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
        os.environ["SYLVAN_BATCH_SIZE"] = str(args.batch_size)
    if args.epochs_per_night is not None:
        config.train.epochs = args.epochs_per_night
        config.controller.epochs = args.epochs_per_night
        os.environ["SYLVAN_EPOCHS_PER_NIGHT"] = str(args.epochs_per_night)
    if args.wm_epochs is not None:
        config.train.epochs = args.wm_epochs
        os.environ["SYLVAN_WM_EPOCHS"] = str(args.wm_epochs)
    if args.controller_epochs is not None:
        config.controller.epochs = args.controller_epochs
        os.environ["SYLVAN_CONTROLLER_EPOCHS"] = str(args.controller_epochs)

    killer = GracefulKiller()
    
    # Auto-resume logic: find the highest existing cycle for the given prefix
    start_cycle = 0
    run_dir_base = config.paths.replay_buffer_dir
    if run_dir_base.exists():
        existing_cycles = []
        for path in run_dir_base.iterdir():
            if path.is_dir() and path.name.startswith(f"{args.run_name_prefix}_cycle_"):
                try:
                    cycle_num = int(path.name.split("_")[-1])
                    existing_cycles.append(cycle_num)
                except ValueError:
                    pass
        if existing_cycles:
            start_cycle = max(existing_cycles) + 1
            print(f"[SYLVAN] Found existing runs. Resuming at Cycle {start_cycle + 1}...")

    for i in range(args.num_cycles):
        cycle = start_cycle + i
        if killer.kill_now:
            print("[SYLVAN] Exiting gracefully as requested.")
            break
            
        print(f"\n{'='*40}")
        print(f"[{args.run_name_prefix}] Starting Cycle {cycle + 1} (Run {i + 1}/{args.num_cycles})")
        print(f"{'='*40}")
        
        run_name = f"{args.run_name_prefix}_cycle_{cycle:04d}"

        # Decay the balance crutches linearly to 0 over --curriculum-cycles, then
        # stay at 0 (pure emergence). Propagated to the Godot subprocesses (day
        # collection AND validation) via os.environ.
        assist_ratio = _decay_curriculum(cycle, args.curriculum_cycles, args.assist_initial)
        reflex_strength = _decay_curriculum(cycle, args.curriculum_cycles, args.reflex_initial)
        os.environ["SYLVAN_ASSIST_RATIO"] = f"{assist_ratio:.4f}"
        os.environ["SYLVAN_REFLEX_STRENGTH"] = f"{reflex_strength:.4f}"

        # Perturbation ramps UP over this run's cycles (gentle -> max), starting
        # non-zero. Keyed to the in-run index so a resume restarts gently (safe).
        if args.perturbation_max > 0.0 and args.perturbation_cycles > 0:
            perturbation = args.perturbation_max * min(1.0, (i + 1) / args.perturbation_cycles)
        else:
            perturbation = 0.0
        os.environ["SYLVAN_PERTURBATION_STRENGTH"] = f"{perturbation:.4f}"

        print(f"[SYLVAN] Curriculum @ cycle {cycle + 1}: gravity_assist={assist_ratio:.3f} reflex={reflex_strength:.3f} perturbation={perturbation:.2f}")

        try:
            result = run_cycle(config, run_name=run_name, collector="godot")
            print(f"\n[SYLVAN] Cycle {cycle + 1} complete. Report: {result['report_path']}")
        except Exception as e:
            print(f"\n[SYLVAN] Error during cycle {cycle + 1}: {e}")
            if killer.kill_now:
                 print("[SYLVAN] Exiting gracefully after error.")
                 break
            raise

if __name__ == "__main__":
    main()
