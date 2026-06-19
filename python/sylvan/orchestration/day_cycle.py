"""Day cycle orchestration.

Phase 1 starts with a bootstrap collector that can generate deterministic sample
episodes. This keeps the Python side executable before the full Godot bridge is
connected, while preserving the exact transition contract expected by the real
collector.
"""

from __future__ import annotations

import math
import random
import shutil
import subprocess
from datetime import datetime, timezone
from os import environ
from pathlib import Path

from ..buffer.schema import Observation, Transition, TransitionInfo
from ..buffer.writer import EpisodeWriter
from ..config import SylvanConfig
from ..constants import LOCOMOTION_METRIC_KEYS, REPLAY_BUFFER_DIR


def prepare_day_run(config: SylvanConfig, run_name: str | None = None) -> Path:
    config.ensure_directories()
    resolved_run_name = run_name or datetime.now(timezone.utc).strftime("day_%Y%m%d_%H%M%S")
    run_dir = REPLAY_BUFFER_DIR / resolved_run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _resolve_godot_executable(config: SylvanConfig) -> str:
    if config.godot.executable:
        return config.godot.executable
        
    # Check local tools folder first
    local_godot = Path("tools/godot/godot")
    if local_godot.exists() and local_godot.is_file():
        return str(local_godot.absolute())
        
    for candidate in ("godot-4", "godot4", "godot"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise FileNotFoundError(
        "Godot executable not found. Set SylvanConfig.godot.executable or GODOT_BIN."
    )


def _build_godot_cmd_env(
    config: SylvanConfig,
    run_dir: Path,
    *,
    num_episodes: int,
    seed: int,
    policy_json: Path | None = None,
    policy_server_host: str | None = None,
    policy_server_port: int | None = None,
    exploration_noise_initial: float = 0.0,
    exploration_noise_final: float = 0.0,
    collector_mode: str = "babbling",
    perturbation_strength: float = 0.0,
) -> tuple[list[str], dict]:
    """Build the (command, env) for ONE Godot collection process. Shared by the
    blocking single-instance path (run_godot_day) and the parallel workers
    (spawn_godot_worker) so the Vulkan-disabling env never drifts between them."""
    godot_bin = environ.get("GODOT_BIN", "") or _resolve_godot_executable(config)
    env = environ.copy()
    env.update(
        {
            "SYLVAN_COLLECT": "1",
            "SYLVAN_RUN_DIR": str(run_dir),
            "SYLVAN_NUM_EPISODES": str(num_episodes),
            "SYLVAN_MAX_EPISODE_STEPS": str(config.env.max_episode_steps),
            "SYLVAN_SEED": str(seed),
            "SYLVAN_COLLECTOR_MODE": collector_mode,
            "SYLVAN_POLICY_EXPLORATION_STD_INITIAL": str(exploration_noise_initial),
            "SYLVAN_POLICY_EXPLORATION_STD_FINAL": str(exploration_noise_final),
            # External-push disturbance (Godot's existing perturbation curriculum
            # reads this env var). NOT part of the action vector / schema — an
            # unobserved disturbance used to create off-balance states (WM data
            # diversity) and to stress-test the planner (J1b A/B). 0.0 = disabled.
            "SYLVAN_PERTURBATION_STRENGTH": str(perturbation_strength),
        }
    )
    if policy_json is not None:
        env["SYLVAN_POLICY_JSON"] = str(policy_json)
    if policy_server_host is not None:
        env["SYLVAN_POLICY_HOST"] = policy_server_host
    if policy_server_port is not None:
        env["SYLVAN_POLICY_PORT"] = str(policy_server_port)
    command = [
        godot_bin,
        "--headless",
        "--rendering-driver",
        "dummy",
        "--audio-driver",
        "Dummy",
        "--fixed-fps",
        "2000",
        "--path",
        str(config.godot.project_dir),
        "--",
        "--speedup=24",
    ]
    # Godot 4 initialise Vulkan même en --headless, ce qui entre en conflit
    # avec ROCm/HSA de PyTorch → HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION.
    # On force Godot à utiliser le rendu software/dummy (pas de GPU du tout)
    # pour laisser le GPU exclusivement à PyTorch pendant l'entraînement.
    env.update({
        # Désactive complètement l'accès GPU au niveau ROCm/HIP/CUDA
        "ROCR_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "CUDA_VISIBLE_DEVICES": "",
        # Désactive l'accès GPU Vulkan/AMD pour Godot
        "DISABLE_LAYER_AMD_SWITCHABLE_GRAPHICS_1": "1",
        "AMD_VULKAN_ICD": "NONE",
        "VK_ICD_FILENAMES": "/dev/null",  # Force le loader Vulkan à ignorer les ICD systèmes
        "VK_DRIVER_FILES": "/dev/null",   # Alternative moderne pour empêcher tout chargement de driver Vulkan
        "VK_ADD_LAYER_PATH": "/dev/null",
        "VK_LAYER_PATH": "/dev/null",
        "VK_LOADER_DRIVERS_DISABLE": "*",  # Désactive tous les drivers Vulkan
        "LIBGL_ALWAYS_SOFTWARE": "1",    # Force OpenGL software si Godot fallback OpenGL
        "GALLIUM_DRIVER": "softpipe",    # Driver software Mesa
    })
    return command, env


def run_godot_day(
    config: SylvanConfig,
    run_dir: Path,
    *,
    policy_json: Path | None = None,
    policy_server_host: str | None = None,
    policy_server_port: int | None = None,
    exploration_noise_initial: float = 0.0,
    exploration_noise_final: float = 0.0,
    collector_mode: str = "babbling",
    perturbation_strength: float = 0.0,
) -> dict[str, object]:
    command, env = _build_godot_cmd_env(
        config,
        run_dir,
        num_episodes=config.day.num_episodes,
        seed=config.env.seed,
        policy_json=policy_json,
        policy_server_host=policy_server_host,
        policy_server_port=policy_server_port,
        exploration_noise_initial=exploration_noise_initial,
        exploration_noise_final=exploration_noise_final,
        collector_mode=collector_mode,
        perturbation_strength=perturbation_strength,
    )
    print(f"[Python] Launching Godot: {' '.join(command)}")
    completed = subprocess.run(command, env=env, check=False)

    if completed.returncode != 0:
        raise RuntimeError("Godot collection failed (see output above).")
    return {
        "run_dir": str(run_dir),
        "collector": "godot",
        "collector_mode": collector_mode,
        "stdout": "Streamed to console",
    }


def spawn_godot_worker(
    config: SylvanConfig,
    run_dir: Path,
    *,
    num_episodes: int,
    seed: int,
    policy_server_host: str,
    policy_server_port: int,
    exploration_noise_initial: float = 0.0,
    exploration_noise_final: float = 0.0,
    collector_mode: str = "policy_server",
    perturbation_strength: float = 0.0,
    log_path: Path | None = None,
) -> subprocess.Popen:
    """Launch ONE Godot collection process NON-BLOCKING (for parallel collection).
    Each worker gets its own run_dir (run_dir/wK) and its own policy-server port, so
    N workers run concurrently on the idle CPU cores. Returns the Popen — the caller
    waits on it. stdout/stderr go to log_path (per-worker) so failures are inspectable
    without flooding the console (a single worker prints per-step lines)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    command, env = _build_godot_cmd_env(
        config,
        run_dir,
        num_episodes=num_episodes,
        seed=seed,
        policy_server_host=policy_server_host,
        policy_server_port=policy_server_port,
        exploration_noise_initial=exploration_noise_initial,
        exploration_noise_final=exploration_noise_final,
        collector_mode=collector_mode,
        perturbation_strength=perturbation_strength,
    )
    out = open(log_path, "wb") if log_path is not None else subprocess.DEVNULL
    return subprocess.Popen(command, env=env, stdout=out, stderr=subprocess.STDOUT)


def collect_bootstrap_day(config: SylvanConfig, run_dir: Path) -> dict[str, object]:
    rng = random.Random(config.env.seed)
    writer = EpisodeWriter(run_dir)
    writer.write_run_metadata(
        {
            "collector": "bootstrap_python_collector",
            "num_episodes": config.day.num_episodes,
            "seed": config.env.seed,
        }
    )

    proprio_dim = config.env.proprio_dim
    action_dim = config.env.action_dim
    max_steps = min(config.env.max_episode_steps, 64)

    for episode_index in range(config.day.num_episodes):
        episode_id = f"episode_{episode_index:04d}"
        transitions: list[Transition] = []
        phase = rng.random() * math.pi
        energy = 100.0
        health = 100.0
        for step_id in range(max_steps):
            action = [
                math.sin(phase + step_id * 0.05 + action_index * 0.1)
                for action_index in range(action_dim)
            ]
            proprio = [
                math.sin(phase + step_id * 0.05 + proprio_index * 0.03)
                for proprio_index in range(proprio_dim)
            ]
            effort = sum(abs(value) for value in action) / max(1, action_dim)
            next_proprio = [
                proprio[index] + 0.05 * action[index % action_dim]
                for index in range(proprio_dim)
            ]
            energy = max(0.0, energy - 0.08 - effort * 0.02)
            done = step_id == max_steps - 1 or energy <= 0.0 or health <= 0.0
            reward = 0.02 - effort * 0.01
            metrics = {
                "uprightness": max(0.0, 1.0 - abs(proprio[0])),
                "forward_velocity": max(0.0, sum(action[:2]) * 0.25),
                "torso_tilt": proprio[0],
                "height": 1.0 - min(0.8, abs(proprio[1]) * 0.5),
                "ground_contact": 1.0,
                "effort": effort,
            }
            obs = Observation(
                proprio=proprio,
                vision=[],
                energy=energy,
                health=health,
                metrics=metrics,
            )
            next_obs = Observation(
                proprio=next_proprio,
                vision=[],
                energy=max(0.0, energy - 0.02),
                health=health,
                metrics={
                    key: float(metrics[key]) for key in LOCOMOTION_METRIC_KEYS
                },
            )
            transition = Transition(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                truncated=step_id == max_steps - 1 and not done,
                info=TransitionInfo(
                    episode_id=episode_id,
                    step_id=step_id,
                    seed=config.env.seed,
                    scene_version="bootstrap_day_v0",
                    agent_version="bootstrap_agent_v0",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
            )
            transition.validate(
                proprio_dim=config.env.proprio_dim, action_dim=config.env.action_dim
            )
            transitions.append(transition)
            if done:
                break
        writer.write_episode(episode_id, transitions)

    return {"run_dir": str(run_dir), "collector": "bootstrap_python_collector"}
