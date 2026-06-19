"""High-level Day -> Night -> Wake orchestration."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ..config import SylvanConfig
from ..control.policy_server import serve_policy_controller
from ..evaluation.locomotion_report import write_locomotion_report
from ..evaluation.metrics import summarize_run
from ..evaluation.prediction_report import write_prediction_report
from .day_cycle import collect_bootstrap_day, prepare_day_run, run_godot_day
from .night_cycle import run_controller_training, run_night_training


def _load_json_dict(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _coerce_float(payload: dict[str, object], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stable_acceptance_gate(summary: dict[str, float]) -> dict[str, object]:
    thresholds = {
        "min_p50_episode_length": 120.0,
        "min_p10_episode_length": 80.0,
        "max_early_termination_rate": 0.15,
        "min_mean_episode_return": 0.0,
    }
    checks = {
        "p50_episode_length": float(summary.get("p50_episode_length", 0.0))
        >= thresholds["min_p50_episode_length"],
        "p10_episode_length": float(summary.get("p10_episode_length", 0.0))
        >= thresholds["min_p10_episode_length"],
        "early_termination_rate": float(summary.get("early_termination_rate", 1.0))
        <= thresholds["max_early_termination_rate"],
        "mean_episode_return": float(summary.get("mean_episode_return", 0.0))
        >= thresholds["min_mean_episode_return"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": thresholds,
    }


def _validation_score(summary: dict[str, float]) -> float:
    return float(summary.get("mean_episode_return", 0.0))


def _atomic_write_text(path: Path, content: str) -> None:
    """Write text to `path` atomically (write to temp, then os.replace).

    Prevents partial writes that would leave a corrupted JSON file behind
    if the process is killed mid-write. Also sidesteps "Permission denied"
    on stale root-owned files because the temp file is created fresh.
    """
    import os as _os
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with _os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        _os.replace(tmp_path, path)
    except Exception:
        try:
            _os.unlink(tmp_path)
        except OSError:
            pass
        raise


def run_cycle(
    config: SylvanConfig,
    run_name: str | None = None,
    *,
    collector: str = "godot",
) -> dict[str, object]:
    run_dir = prepare_day_run(config, run_name=run_name)

    # Checkpoints and stable policy metadata.
    wm_checkpoint = config.paths.checkpoints_dir / "world_model_v0.best.pt"
    ctrl_checkpoint = config.paths.checkpoints_dir / "controller_v0.pt"
    stable_wm_checkpoint = config.paths.checkpoints_dir / "world_model_v0.stable.pt"
    stable_ctrl_checkpoint = config.paths.checkpoints_dir / "controller_v0.stable.pt"
    stable_metrics_path = config.paths.checkpoints_dir / "controller_v0.stable.metrics.json"

    active_world_model_checkpoint = wm_checkpoint
    active_controller_checkpoint = ctrl_checkpoint
    active_policy_source = "candidate"

    if stable_wm_checkpoint.exists() and stable_ctrl_checkpoint.exists():
        active_world_model_checkpoint = stable_wm_checkpoint
        active_controller_checkpoint = stable_ctrl_checkpoint
        active_policy_source = "stable"

    import os
    import sys
    import subprocess
    import json

    def _run_subprocess_json(cmd: list[str], env_overrides: dict[str, str] | None = None) -> dict[str, object]:
        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)
        
        process = subprocess.Popen(
            [sys.executable] + cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        last_line = ""
        if process.stdout:
            for line in process.stdout:
                print(line, end="", flush=True)
                if line.strip():
                    last_line = line.strip()
                    
        returncode = process.wait()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd)
            
        return json.loads(last_line)

    if collector == "godot":
        if active_world_model_checkpoint.exists() and active_controller_checkpoint.exists():
            print(
                "[Python] Using %s policy for day collection: %s"
                % (active_policy_source, active_controller_checkpoint)
            )
            day_result = _run_subprocess_json([
                "-m", "scripts.collect_day",
                "--run-name", run_dir.name,
                "--collector", "godot",
                "--controller-checkpoint", str(active_controller_checkpoint),
                "--world-model-checkpoint", str(active_world_model_checkpoint),
                "--exploration-noise-initial", str(config.controller.exploration_noise_initial),
                "--exploration-noise-final", str(config.controller.exploration_noise_final),
            ], env_overrides={
                "ROCR_VISIBLE_DEVICES": "",
                "HIP_VISIBLE_DEVICES": "",
                "CUDA_VISIBLE_DEVICES": "",
                "SYLVAN_POLICY_DEVICE": "cpu",
            })
            day_result["policy_source"] = active_policy_source
            day_result["policy_world_model_checkpoint"] = str(active_world_model_checkpoint)
            day_result["policy_controller_checkpoint"] = str(active_controller_checkpoint)
        else:
            print("[Python] No policy found. Starting with babbling collection.")
            day_result = _run_subprocess_json([
                "-m", "scripts.collect_day",
                "--run-name", run_dir.name,
                "--collector", "godot",
            ], env_overrides={
                "ROCR_VISIBLE_DEVICES": "",
                "HIP_VISIBLE_DEVICES": "",
                "CUDA_VISIBLE_DEVICES": "",
                "SYLVAN_POLICY_DEVICE": "cpu",
            })
    elif collector == "bootstrap":
        day_result = collect_bootstrap_day(config, run_dir)
    else:
        raise ValueError(f"Unsupported collector: {collector}")

    summary = summarize_run(run_dir)
    
    # Night training runs on the GPU!
    training_result = _run_subprocess_json([
        "-m", "scripts.train_night",
        str(run_dir),
    ])
    
    # Controller training runs on the GPU!
    controller_result = _run_subprocess_json([
        "-m", "scripts.train_controller",
        str(run_dir),
        str(training_result["best_checkpoint_path"]),
    ])
    
    validation_run_dir = prepare_day_run(config, run_name=f"{run_dir.name}_validation")
    
    # Validation runs on the CPU!
    _run_subprocess_json([
        "-m", "scripts.validate_in_godot",
        str(controller_result["checkpoint_path"]),
        str(training_result["best_checkpoint_path"]),
        "--run-name", validation_run_dir.name,
    ], env_overrides={
        "ROCR_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "CUDA_VISIBLE_DEVICES": "",
        "SYLVAN_POLICY_DEVICE": "cpu",
    })
    
    validation_result = {
        "run_dir": str(validation_run_dir),
        "collector": "godot",
        "collector_mode": "policy_server",
    }
    validation_report_path = config.paths.reports_dir / "phase2_validation_report.json"
    write_locomotion_report(validation_run_dir, validation_report_path)

    validation_summary = summarize_run(validation_run_dir)

    # --- Honest instrumentation (additive, never fatal): J1 imagination->reality
    # transfer + the "is the gradient flowing?" actor-frozen light. Wrapped so a
    # measurement bug can never break a training cycle. CPU. ---
    transfer_digest = None
    try:
        from ..evaluation.transfer import evaluate_transfer_from_checkpoints

        transfer_digest = evaluate_transfer_from_checkpoints(
            config,
            validation_run_dir=validation_run_dir,
            world_model_ckpt=Path(training_result["best_checkpoint_path"]),
            controller_ckpt=Path(controller_result["checkpoint_path"]),
        )
        if transfer_digest.get("num_episodes", 0):
            print(
                "[Python] J1 transfer | imagined=%.2f real=%.2f |err|=%.2f reward_mae=%.4f (ratio %.2f)"
                % (
                    transfer_digest["mean_imagined_return"],
                    transfer_digest["mean_real_return"],
                    transfer_digest["mean_abs_return_error"],
                    transfer_digest["per_step_reward_mae"],
                    transfer_digest["return_error_ratio"],
                )
            )
    except Exception as exc:  # noqa: BLE001 — instrumentation must never be fatal
        print("[Python] J1 transfer eval skipped: %s" % exc)

    health_signal = None
    try:
        from ..evaluation.training_health import actor_frozen_signal

        health_signal = actor_frozen_signal(controller_result.get("history", []))
        if health_signal.get("actor_frozen"):
            print(
                "\033[91m[Python] ⚠ ACTOR FROZEN | rel_span=%.2e (first=%.3f last=%.3f) "
                "— no balance gradient is flowing.\033[0m"
                % (health_signal["rel_span"], health_signal["first"], health_signal["last"])
            )
    except Exception as exc:  # noqa: BLE001 — instrumentation must never be fatal
        print("[Python] training-health check skipped: %s" % exc)

    stable_metrics = _load_json_dict(stable_metrics_path)
    previous_best_score = _coerce_float(stable_metrics, "best_score")
    previous_stale_count = int(stable_metrics.get("stale_count", 0) or 0)

    # Promotion guards against a frozen stable checkpoint (the "identical rows"
    # failure): mean_return is noisy and non-monotonic, so one lucky early peak
    # would otherwise permanently block every later candidate and collection
    # keeps replaying the same stale policy. SCORE_TOLERANCE promotes a
    # near-equal candidate (keeps data fresh); STALENESS_LIMIT force-promotes the
    # latest candidate after that many consecutive rejections (guaranteed escape).
    SCORE_TOLERANCE = 5.0
    STALENESS_LIMIT = 3

    # Reset best score if the objective version changed to avoid comparing apples to oranges
    stable_objective_version = stable_metrics.get("objective_version", "unknown")
    current_objective_version = controller_result.get("objective_version", "unknown")
    if stable_objective_version != current_objective_version:
        print(f"[Python] Objective version changed ({stable_objective_version} -> {current_objective_version}). Resetting previous best score.")
        previous_best_score = None
        previous_stale_count = 0
        
    score = _validation_score(validation_summary)
    gate = _stable_acceptance_gate(validation_summary)

    stable_promoted = False
    stable_reason = ""
    # Treat the stable set as "intact" only if all three artifacts are present AND writable.
    # Stale root-owned files (from a previous Docker run) must not block promotion.
    def _writable(path: Path) -> bool:
        if not path.exists():
            return True
        try:
            with path.open("a"):
                return True
        except OSError:
            return False

    if not _writable(stable_metrics_path):
        try:
            stable_metrics_path.unlink()
            print("[Python] Removed stale non-writable %s" % stable_metrics_path)
        except OSError as exc:
            print("[Python] Could not remove stale %s: %s" % (stable_metrics_path, exc))
    stable_exists = (
        stable_ctrl_checkpoint.exists()
        and stable_wm_checkpoint.exists()
        and stable_metrics_path.exists()
    )
    if not stable_exists:
        shutil.copy2(controller_result["checkpoint_path"], stable_ctrl_checkpoint)
        shutil.copy2(training_result["best_checkpoint_path"], stable_wm_checkpoint)
        stable_metrics = {
            "best_score": score,
            "score_name": "mean_episode_return",
            "objective_version": controller_result.get("objective_version", "unknown"),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "source_run_dir": str(validation_run_dir),
            "validation_summary": validation_summary,
            "acceptance_gate": gate,
            "bootstrap": True,
            "stale_count": 0,
        }
        _atomic_write_text(stable_metrics_path, json.dumps(stable_metrics, indent=2))
        stable_promoted = True
        stable_reason = "Bootstrapped first stable checkpoint (no prior stable existed)."
    else:
        # Promote when the candidate (a) clears the absolute gate, (b) is within
        # SCORE_TOLERANCE of the best score so far (noise-tolerant progress), or
        # (c) the stable checkpoint has gone stale for STALENESS_LIMIT cycles
        # (forced escape from a frozen optimum). Without (b)/(c) a single lucky
        # early peak freezes the run forever -> the "identical rows" failure.
        is_improvement = (
            previous_best_score is None or score > previous_best_score - SCORE_TOLERANCE
        )
        would_be_stale_count = previous_stale_count + 1
        force_promote = would_be_stale_count >= STALENESS_LIMIT

        if gate["passed"] or is_improvement or force_promote:
            shutil.copy2(controller_result["checkpoint_path"], stable_ctrl_checkpoint)
            shutil.copy2(training_result["best_checkpoint_path"], stable_wm_checkpoint)
            genuine_improvement = (
                previous_best_score is None or score > previous_best_score
            )
            if force_promote and not (gate["passed"] or is_improvement):
                # Forced escape: drop the high-water mark to the current score so
                # future candidates are no longer measured against a lucky peak.
                new_best_score = score
            elif genuine_improvement:
                new_best_score = score
            else:
                # Tolerance promotion: refresh the policy/data but keep the
                # existing high-water mark so it does not drift down on noise.
                new_best_score = previous_best_score
            stable_metrics = {
                "best_score": new_best_score,
                "score_name": "mean_episode_return",
                "objective_version": controller_result.get("objective_version", "unknown"),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "source_run_dir": str(validation_run_dir),
                "validation_summary": validation_summary,
                "acceptance_gate": gate,
                "bootstrap": False,
                "stale_count": 0,
            }
            _atomic_write_text(stable_metrics_path, json.dumps(stable_metrics, indent=2))
            stable_promoted = True
            prev_score_str = f"{previous_best_score:.2f}" if previous_best_score is not None else "N/A"
            if gate["passed"]:
                stable_reason = "Candidate passed the absolute acceptance gate and was promoted."
            elif genuine_improvement:
                stable_reason = f"Candidate progressively promoted because it improved the stable score ({prev_score_str} -> {score:.2f})."
            elif is_improvement:
                stable_reason = f"Candidate promoted within tolerance (score {score:.2f} vs best {prev_score_str}, tol {SCORE_TOLERANCE:.1f})."
            else:
                stable_reason = f"Candidate force-promoted after {would_be_stale_count} stale cycles to escape a frozen optimum (score {score:.2f}, best was {prev_score_str})."
        else:
            # Rejected: keep the stable checkpoint but persist the incremented
            # stale counter so the staleness escape can eventually fire.
            stable_metrics = dict(stable_metrics)
            stable_metrics["stale_count"] = would_be_stale_count
            stable_metrics["updated_at"] = datetime.now(timezone.utc).isoformat()
            _atomic_write_text(stable_metrics_path, json.dumps(stable_metrics, indent=2))
            stable_reason = (
                f"Candidate rejected (stale {would_be_stale_count}/{STALENESS_LIMIT}): "
                f"score {score:.2f} below best {previous_best_score:.2f} - tol {SCORE_TOLERANCE:.1f}."
            )

    stable_decision = {
        "promoted": stable_promoted,
        "reason": stable_reason,
        "score": score,
        "score_name": "mean_episode_return",
        "previous_best_score": previous_best_score,
        "acceptance_gate": gate,
        "stable_controller_checkpoint": str(stable_ctrl_checkpoint),
        "stable_world_model_checkpoint": str(stable_wm_checkpoint),
        "stable_metrics_path": str(stable_metrics_path),
    }

    report = {
        "run_dir": str(run_dir),
        "day": day_result,
        "buffer_summary": summary,
        "training": training_result,
        "controller": {**controller_result, "health": health_signal},
        "validation": {
            **validation_result,
            "run_dir": str(validation_run_dir),
            "report_path": str(validation_report_path),
            "summary": validation_summary,
            "transfer": transfer_digest,
            "stable_decision": stable_decision,
        },
    }
    report_path = config.paths.reports_dir / "phase2_cycle_report.json"
    write_prediction_report(report_path, report)
    return {"report_path": str(report_path), **report}
