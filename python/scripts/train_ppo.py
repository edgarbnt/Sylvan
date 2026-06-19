"""J0 — model-free PPO controller trained on REAL Godot rollouts (grounded).

Each iteration: snapshot the current policy (behavior) → collect real episodes in
Godot serving that stochastic policy (Godot exploration noise = 0) → recompute
old_log_prob/GAE from the snapshot on the real rewards → K-epoch PPO update → save.

The headline metric is the REAL mean episode return (must go UP). The world model
is NOT used here (BLUEPRINT.md §8: prove balance on the real body first).

Usage (from the python/ directory):
    python3 -m scripts.train_ppo --iterations 2 --run-prefix ppo_smoke   # smoke
    python3 -m scripts.train_ppo --iterations 150 --run-prefix ppo_j0    # real run
    python3 -m scripts.diagnose_run --prefix ppo_j0 --trend              # watch return ↑ fall% ↓
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from sylvan.config import SylvanConfig
from sylvan.constants import CHECKPOINTS_DIR
from sylvan.buffer.reader import iter_episodes
from sylvan.control.ppo.policy import GaussianActorCritic
from sylvan.control.ppo.rollout import build_rollout_batch
from sylvan.control.ppo.stochastic_server import serve_stochastic_policy
from sylvan.control.ppo.symmetry import self_check as sym_self_check
from sylvan.control.ppo.update import PPOConfig, ppo_update
from sylvan.orchestration.day_cycle import prepare_day_run, run_godot_day, spawn_godot_worker
from sylvan.training.checkpointing import load_checkpoint, save_checkpoint


def _prune_old_cycles(run_dir: Path, run_prefix: str, keep_it: int) -> None:
    """Disk hygiene: PPO is on-policy, so once a cycle's batch is consumed its episodes are
    dead weight. Delete every PRIOR cycle dir for this prefix (keep only the current one) so
    the replay buffer stays FLAT (~one cycle) instead of growing unbounded — it silently
    filled a 136G disk at ~100MB/iter before this guard existed."""
    base = run_dir.parent
    for d in base.glob(f"{run_prefix}_cycle_*"):
        try:
            idx = int(d.name.rsplit("_", 1)[1])
        except (ValueError, IndexError):
            continue
        if idx < keep_it and d.is_dir():
            shutil.rmtree(d, ignore_errors=True)


def _quick_digest(run_dir: Path) -> dict[str, float]:
    """Minimal real-rollout digest (self-contained; the headline metric). Also
    reports mean forward velocity (the J_walk locomotion headline)."""
    episodes = [ep for ep in iter_episodes(Path(run_dir)) if ep]
    if not episodes:
        return {"mean_return": 0.0, "mean_length": 0.0, "fall_rate": 0.0,
                "mean_fwd_vel": 0.0, "num_episodes": 0.0}
    returns = [sum(t.reward for t in ep) for ep in episodes]
    lengths = [len(ep) for ep in episodes]
    falls = sum(1 for ep in episodes if ep[-1].done)
    fwd_vals = [float(t.obs.metrics.get("forward_velocity", 0.0)) for ep in episodes for t in ep]
    n = len(episodes)
    return {
        "mean_return": sum(returns) / n,
        "mean_length": sum(lengths) / n,
        "fall_rate": falls / n,
        "mean_fwd_vel": (sum(fwd_vals) / len(fwd_vals)) if fwd_vals else 0.0,
        "num_episodes": float(n),
    }


def _checkpoint_score(digest: dict[str, float], metric: str) -> float:
    """Score used to pick policy_best.pt. LESSON (J_walk): best-by-return saved a
    checkpoint that falls 50% while the stable (fall ~20%) windows were lost. So the
    default weights the score by STABILITY (1 - fall_rate), never by raw return alone.
      - stability    : mean_length * (1 - fall_rate)        — re-balance (J0-bis)
      - stable_fwd   : max(0, fwd_vel) * (1 - fall_rate) * mean_length — re-walk
      - stable_return: mean_return * (1 - fall_rate)
      - return       : legacy mean_return (NOT recommended)
    """
    fall = digest["fall_rate"]
    length = digest["mean_length"]
    stab = 1.0 - fall
    if metric == "return":
        return digest["mean_return"]
    if metric == "stable_return":
        return digest["mean_return"] * stab
    if metric == "stable_fwd":
        return max(0.0, digest["mean_fwd_vel"]) * stab * length
    # default: pure stability (survive long AND don't fall)
    return length * stab


class ParallelCollectorPool:
    """Parallel Godot collection via N PERSISTENT policy-server SUBPROCESSES.

    Why subprocesses, not threads: N server threads in the trainer share ONE Python
    interpreter, so the GIL serialises their per-step request handling → only ~1.8x even
    with 8 workers (the machine stays half-idle). Each server here is its OWN process
    (own GIL) → true parallelism. torch is imported ONCE per server at pool start (NOT per
    iteration). Each iteration: write the new behavior checkpoint, SIGHUP every server to
    hot-reload it (while idle, between collections), then launch the Godot workers — each
    worker writes to run_dir/wK and iter_episodes' rglob reads them all.
    """

    def __init__(self, config, init_policy, *, num_workers: int, host: str = "127.0.0.1") -> None:
        self.config = config
        self.host = host
        self.n = max(1, min(num_workers, int(config.day.num_episodes)))
        self.dir = Path(tempfile.mkdtemp(prefix="sylvan_pool_"))
        self.ckpt = self.dir / "behavior.pt"
        self._save(init_policy)
        self.servers: list[dict] = []
        for k in range(self.n):
            port_file = self.dir / f"{k}.port"
            ack_file = self.dir / f"{k}.ack"
            log_file = self.dir / f"{k}.log"
            proc = subprocess.Popen(
                [
                    sys.executable, "-m", "scripts.serve_ppo_collect",
                    "--checkpoint", str(self.ckpt), "--host", host, "--port", "0",
                    "--seed", str(1000 + k),
                    "--port-file", str(port_file), "--ack-file", str(ack_file),
                ],
                stdout=open(log_file, "wb"), stderr=subprocess.STDOUT,
            )
            self.servers.append({"proc": proc, "port_file": port_file, "ack_file": ack_file, "port": None})
        for s in self.servers:
            s["port"] = int(self._wait_file(s["port_file"], timeout=120.0))

    def _save(self, policy) -> None:
        tmp = self.ckpt.with_suffix(".tmp")
        torch.save({"model_state_dict": policy.state_dict()}, tmp)
        tmp.rename(self.ckpt)  # atomic publish so a SIGHUP'd server never reads a partial file

    @staticmethod
    def _wait_file(path: Path, *, timeout: float) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.exists():
                return path.read_text().strip()
            time.sleep(0.05)
        raise RuntimeError(f"timeout waiting for {path}")

    def collect(self, run_dir: Path, behavior, *, seed: int) -> None:
        # 1. Publish the new behavior, then SIGHUP every (idle) server to hot-reload it.
        for s in self.servers:
            s["ack_file"].unlink(missing_ok=True)
        self._save(behavior)
        for s in self.servers:
            if s["proc"].poll() is not None:
                raise RuntimeError(f"collection server died (see {self.dir}/*.log)")
            s["proc"].send_signal(signal.SIGHUP)
        for s in self.servers:
            self._wait_file(s["ack_file"], timeout=30.0)
        # 2. Split episodes across workers, launch their Godots concurrently, wait all.
        total = int(self.config.day.num_episodes)
        chunks = [total // self.n + (1 if k < total % self.n else 0) for k in range(self.n)]
        procs = []
        for k, eps in enumerate(chunks):
            if eps <= 0:
                continue
            wdir = run_dir / f"w{k}"
            p = spawn_godot_worker(
                self.config, wdir,
                num_episodes=eps, seed=seed + 1000 * (k + 1),
                policy_server_host=self.host, policy_server_port=self.servers[k]["port"],
                collector_mode="policy_server", log_path=wdir / "godot.log",
            )
            procs.append((k, p))
        failed = [k for k, p in procs if p.wait() != 0]
        if failed:
            raise RuntimeError(f"Godot worker(s) {failed} failed — see {run_dir}/w<k>/godot.log")

    def close(self) -> None:
        for s in self.servers:
            try:
                s["proc"].send_signal(signal.SIGTERM)
            except Exception:
                pass
        for s in self.servers:
            try:
                s["proc"].wait(timeout=5)
            except Exception:
                s["proc"].kill()
        shutil.rmtree(self.dir, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="J0 model-free PPO on real Godot rollouts.")
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--run-prefix", default="ppo_j0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num-episodes", type=int, default=None, help="Override config.day.num_episodes.")
    ap.add_argument("--max-steps", type=int, default=None, help="Override config.env.max_episode_steps.")
    ap.add_argument("--ppo-epochs", type=int, default=10)
    ap.add_argument("--minibatch-size", type=int, default=256)
    ap.add_argument(
        "--entropy-coef",
        type=float,
        default=0.003,
        help="Exploration bonus. Lowered from 0.01 so std anneals and the final policy "
        "settles into a calm balance instead of constant high-noise thrashing.",
    )
    ap.add_argument(
        "--sym-coef",
        type=float,
        default=0.0,
        help="Mirror-symmetry (equivariance) regularizer weight. >0 enforces "
        "pi(mirror(obs))==mirror(pi(obs)) → kills the learned left-right gait drift on the "
        "symmetric quad body. Try ~1.0. The radar/command channel mirrors with it.",
    )
    ap.add_argument(
        "--sym-v-coef",
        type=float,
        default=-1.0,
        help="Value-invariance weight: enforce V(mirror(obs))==V(obs). <0 mirrors --sym-coef "
        "(on whenever symmetry is on); >=0 sets it explicitly. Closes the actor-only gap "
        "(obs 233) where an asymmetric critic re-injects drift through the advantages.",
    )
    ap.add_argument("--sym-coef-start", type=float, default=None,
                    help="Symmetry curriculum: if set, ramp sym_coef from this start value up to "
                         "--sym-coef over --sym-coef-cycles iters. Ramping in gently (0→1.0) keeps the "
                         "gait alive — a hard sym_coef=1.0 from iter 0 collapsed it (sym1). sym_v_coef "
                         "follows sym_coef when --sym-v-coef<0.")
    ap.add_argument("--sym-coef-cycles", type=int, default=35,
                    help="Iterations over which to ramp sym_coef start->end.")
    ap.add_argument("--mirror-augment", action="store_true",
                    help="HARD symmetry: augment each PPO batch with the left-right mirror of every "
                         "transition (mirror_obs/mirror_action, same advantage/return — the locomotion "
                         "reward is chirality-invariant). Makes symmetry a property of the DATA, killing "
                         "the asymmetric gait attractor (dragging 'kickstand' leg) that the soft "
                         "equivariance penalty alone leaves intact. Compose with --sym-coef.")
    ap.add_argument("--ckpt-name", default="ppo_j0",
                    help="Checkpoint subdir under data/checkpoints (use a fresh name, e.g. ppo_walk, "
                         "to avoid overwriting the J0 policy_best.pt).")
    ap.add_argument("--init-from", default=None,
                    help="Warm-start the policy from a checkpoint (e.g. the J0 stander) so it builds "
                         "the new skill on a stable base instead of relearning balance from scratch.")
    ap.add_argument("--init-log-std", type=float, default=-0.5,
                    help="When warm-starting, reset log_std to this so exploration restarts (J0's "
                         "annealed std is too low to explore the new behaviour).")
    ap.add_argument("--best-metric", default="stability",
                    choices=["stability", "stable_fwd", "stable_return", "return"],
                    help="Score for policy_best.pt. Default 'stability' (mean_length*(1-fall_rate)) "
                         "for re-balance; use 'stable_fwd' for re-walk. NEVER 'return' alone "
                         "(saves a 50%%-fall checkpoint — the J_walk lesson).")
    ap.add_argument("--action-repeat", type=int, default=1,
                    help="Frame-skip: hold each action for N physics steps (Godot reads "
                         "SYLVAN_ACTION_REPEAT). 2 = 30Hz control, ~halves collection round-trips.")
    ap.add_argument("--num-workers", type=int, default=1,
                    help="Parallel Godot collection workers (1 = legacy single instance). The "
                         "collector is the bottleneck and uses 1 core; N workers split the "
                         "episodes across the idle cores → ~N-fold faster. Try 6-8 on a 16-core box.")
    # Locomotion velocity curriculum (sets SYLVAN_TARGET_VELOCITY per iteration so the
    # reward's target speed ramps; a low start gives a strong gradient out of standstill).
    ap.add_argument("--target-vel-start", type=float, default=None,
                    help="Curriculum start target speed (m/s). If set, ramps to --target-vel-end.")
    ap.add_argument("--target-vel-end", type=float, default=0.6)
    ap.add_argument("--target-vel-cycles", type=int, default=60,
                    help="Iterations over which to ramp the target speed start->end.")
    # Perturbation curriculum (sets SYLVAN_PERTURBATION_STRENGTH per iteration so random
    # horizontal pushes ramp in). Start at 0 to let the policy first re-adapt to a body
    # change (e.g. smaller feet), then ramp to light recoverable shoves → robust active balance.
    ap.add_argument("--perturb-start", type=float, default=None,
                    help="Curriculum start push impulse (N·s). If set, ramps to --perturb-end.")
    ap.add_argument("--perturb-end", type=float, default=8.0)
    ap.add_argument("--perturb-cycles", type=int, default=60,
                    help="Iterations over which to ramp the push strength start->end.")
    # Phase A command curriculum (sets SYLVAN_CMD_WMAX per iteration so the |omega| command range
    # widens 0->end). Root fix for the frozen turn: uniform large-range omega sampling makes RL fail
    # (research_omnidirectional_locomotion.md); start near zero and expand. Godot relaunches per iter
    # and samples within this range (main.gd, SYLVAN_CMD_CURRIC=1). Needs SYLVAN_CMD_CURRIC=1 exported.
    ap.add_argument("--cmd-wmax-start", type=float, default=None,
                    help="Curriculum start |omega| command range (rad/s). If set, ramps to --cmd-wmax-end.")
    ap.add_argument("--cmd-wmax-end", type=float, default=0.6)
    ap.add_argument("--cmd-wmax-cycles", type=int, default=50,
                    help="Iterations over which to ramp the omega command range start->end.")
    # Body-MORPH curriculum (2026-06-16 speed redesign): morph sprawled→runner geometry over N iters so a
    # WARM-STARTED stable policy adapts continuously (from-scratch on the runner body fell 80%; an annealed
    # posture change is the research-prescribed alternative). Sets SYLVAN_SPRAWL_SPLAY/LEG_UP/LEG_LOW per iter.
    ap.add_argument("--body-morph-cycles", type=int, default=0,
                    help="If >0, morph splay 0.45→0.25, legs 0.20/0.15→0.27/0.21 over this many iters.")
    args = ap.parse_args()

    config = SylvanConfig()
    if args.num_episodes is not None:
        config.day.num_episodes = args.num_episodes
    if args.max_steps is not None:
        config.env.max_episode_steps = args.max_steps

    os.environ["SYLVAN_ACTION_REPEAT"] = str(args.action_repeat)  # propagates to Godot workers
    torch.manual_seed(args.seed)
    obs_dim = config.env.policy_input_dim  # proprio ++ vision (food radar)
    action_dim = config.env.action_dim
    policy = GaussianActorCritic(
        obs_dim=obs_dim, hidden_dim=config.controller.hidden_dim, action_dim=action_dim
    )
    if args.init_from:
        load_checkpoint(Path(args.init_from), policy)
        with torch.no_grad():
            policy.log_std.fill_(args.init_log_std)  # restart exploration from the stable base
        print(f"[PPO] warm-started from {args.init_from} | log_std reset to {args.init_log_std}")
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    ppo_cfg = PPOConfig(
        epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        entropy_coef=args.entropy_coef,
        sym_coef=args.sym_coef,
        sym_v_coef=args.sym_v_coef,
    )
    if ppo_cfg.sym_coef > 0.0:
        # Guard the hand-built mirror maps (involution + empirical turnL<->turnR) BEFORE
        # training — they are tied to the 94-dim proprio layout and would silently corrupt
        # the regularizer if the layout drifted (e.g. the recent bipedal->quad pivot).
        sym_self_check()
        v_eff = ppo_cfg.sym_v_coef if ppo_cfg.sym_v_coef >= 0.0 else ppo_cfg.sym_coef
        print(f"[PPO] symmetry ON | sym_coef={ppo_cfg.sym_coef} sym_v_coef(eff)={v_eff} | "
              "mirror self_check passed", flush=True)
    ckpt_dir = CHECKPOINTS_DIR / args.ckpt_name
    best_score = float("-inf")  # track the PEAK policy by the stability-weighted score

    print(
        f"[PPO] J0 start | obs={obs_dim} act={action_dim} iters={args.iterations} "
        f"episodes/iter={config.day.num_episodes} max_steps={config.env.max_episode_steps}"
    )

    pool = None
    if args.num_workers > 1:
        pool = ParallelCollectorPool(
            config, policy, num_workers=args.num_workers, host=config.godot.policy_host
        )
        import atexit
        atexit.register(pool.close)  # terminate the server subprocesses on exit
        print(f"[PPO] parallel pool: {pool.n} persistent server processes (own GIL)", flush=True)

    for it in range(args.iterations):
        # 0. Velocity curriculum: ramp the reward's target speed (Godot reads the env).
        if args.target_vel_start is not None:
            frac = min(1.0, it / max(1, args.target_vel_cycles))
            target_v = args.target_vel_start + frac * (args.target_vel_end - args.target_vel_start)
            os.environ["SYLVAN_TARGET_VELOCITY"] = f"{target_v:.4f}"
        # 0b. Perturbation curriculum: ramp random push strength (Godot reads the env).
        if args.perturb_start is not None:
            pfrac = min(1.0, it / max(1, args.perturb_cycles))
            perturb = args.perturb_start + pfrac * (args.perturb_end - args.perturb_start)
            os.environ["SYLVAN_PERTURBATION_STRENGTH"] = f"{perturb:.4f}"
        # 0b2. Command curriculum (Phase A): ramp the |omega| command range so turning is introduced
        # gradually (Godot relaunches per iter and re-reads SYLVAN_CMD_WMAX; main.gd samples within it).
        if args.cmd_wmax_start is not None:
            wfrac = min(1.0, it / max(1, args.cmd_wmax_cycles))
            cmd_wmax = args.cmd_wmax_start + wfrac * (args.cmd_wmax_end - args.cmd_wmax_start)
            os.environ["SYLVAN_CMD_WMAX"] = f"{cmd_wmax:.4f}"
        # 0c. Body-morph curriculum: anneal sprawled→runner geometry so a warm-started policy follows it.
        if args.body_morph_cycles > 0:
            bfrac = min(1.0, it / max(1, args.body_morph_cycles))
            os.environ["SYLVAN_SPRAWL_SPLAY"] = f"{0.45 + bfrac * (0.25 - 0.45):.4f}"
            os.environ["SYLVAN_LEG_UP"] = f"{0.20 + bfrac * (0.27 - 0.20):.4f}"
            os.environ["SYLVAN_LEG_LOW"] = f"{0.15 + bfrac * (0.21 - 0.15):.4f}"
        # 0c. Symmetry curriculum: ramp sym_coef start->end so the equivariance pressure comes in
        # gently and never collapses the gait (the sym1 failure). sym_v_coef mirrors it when <0.
        if args.sym_coef_start is not None:
            sfrac = min(1.0, it / max(1, args.sym_coef_cycles))
            ppo_cfg.sym_coef = args.sym_coef_start + sfrac * (args.sym_coef - args.sym_coef_start)

        # 1. Frozen behavior snapshot (the policy that collects this iteration).
        behavior = copy.deepcopy(policy).eval()
        for p in behavior.parameters():
            p.requires_grad_(False)

        # 2. Collect real episodes in Godot, serving the stochastic behavior policy.
        run_dir = prepare_day_run(config, run_name=f"{args.run_prefix}_cycle_{it:04d}")
        _prune_old_cycles(run_dir, args.run_prefix, keep_it=it)
        if pool is not None:
            # Parallel collection across N persistent-server Godot workers.
            pool.collect(run_dir, behavior, seed=args.seed + it * 1000)
        else:
            with serve_stochastic_policy(
                behavior, host=config.godot.policy_host, port=config.godot.policy_port, seed=args.seed + it
            ) as srv:
                run_godot_day(
                    config,
                    run_dir,
                    policy_server_host=srv["host"],
                    policy_server_port=srv["port"],
                    exploration_noise_initial=0.0,
                    exploration_noise_final=0.0,
                    collector_mode="policy_server",
                )

        # 3. Real rollouts -> PPO batch (old_log_prob/GAE recomputed from behavior).
        batch, roll_stats = build_rollout_batch(run_dir, behavior, gamma=args.gamma, lam=args.lam,
                                                mirror_augment=args.mirror_augment)
        if batch is None:
            print(f"[PPO] iter {it}: no transitions collected — skipping update.")
            continue

        # 4. PPO update on the live policy.
        upd = ppo_update(policy, optimizer, batch, ppo_cfg)

        # 5. Save + 6. log the headline real metric (must go UP).
        digest = _quick_digest(run_dir)
        score = _checkpoint_score(digest, args.best_metric)
        metrics = {
            "mean_return": digest["mean_return"],
            "fall_rate": digest["fall_rate"],
            "mean_length": digest["mean_length"],
            "mean_fwd_vel": digest["mean_fwd_vel"],
            "best_metric": args.best_metric,
            "score": score,
        }
        save_checkpoint(
            destination=ckpt_dir / "policy_latest.pt",
            model=policy,
            optimizer=optimizer,
            epoch=it,
            metrics=metrics,
        )
        # Keep the PEAK policy by the STABILITY-weighted score (not raw return):
        # the J_walk best-by-return checkpoint fell 50% while stable windows were lost.
        is_best = score > best_score
        if is_best:
            best_score = score
            save_checkpoint(
                destination=ckpt_dir / "policy_best.pt",
                model=policy,
                optimizer=optimizer,
                epoch=it,
                metrics=metrics,
            )
        print(
            "[PPO] iter %d | return=%.2f len=%.0f fall=%.0f%% fwd_vel=%.3f score=%.1f tgt=%s | kl=%.4f clip=%.2f std=%.3f vloss=%.3f sym=%.4f symv=%.4f (%d transitions)%s"
            % (
                it,
                digest["mean_return"],
                digest["mean_length"],
                digest["fall_rate"] * 100.0,
                digest["mean_fwd_vel"],
                score,
                os.environ.get("SYLVAN_TARGET_VELOCITY", "-"),
                upd["approx_kl"],
                upd["clip_frac"],
                upd["mean_std"],
                upd["value_loss"],
                upd.get("sym_loss", 0.0),
                upd.get("sym_loss_v", 0.0),
                int(roll_stats["num_transitions"]),
                "  <- best" if is_best else "",
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
