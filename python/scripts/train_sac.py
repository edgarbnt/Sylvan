"""Off-policy SAC on real Godot rollouts — the sample-efficiency lever for agile turning.

Motivation: the exhaustive turn-agility investigation proved the ~14 deg/s yaw ceiling is a
SAMPLE-EFFICIENCY wall (8 CPU envs, not Isaac's 4096), not the body or the reward — a
PPO spin-specialist with a linear high-yaw reward still capped at ~15 deg/s. PPO is
on-policy (one update per transition); SAC reuses every transition many times from a
persistent replay buffer, extracting far more learning per env-step.

CRITICAL: SAC from scratch faces the SAME exploration wall as PPO (8 envs can't DISCOVER
walking). So we WARM-START the actor from the working CPG-residual walker (--init-from),
exactly as BC-bootstrap unblocked the learned walker. The CPG gives walk-by-construction;
SAC's residual then has only to refine the agile turning PPO's residual couldn't reach.

Each iteration: collect fresh episodes serving the CURRENT stochastic actor → push their
(s,a,r,s',done) into the replay buffer → run many SAC gradient steps over uniform
minibatches → publish the new actor so the collection servers hot-reload it.

Usage (from python/, with the SAME SYLVAN_* env as the spin1 turn task):
    python3 -m scripts.train_sac --iterations 120 --num-workers 8 \
        --init-from ../data/checkpoints/ppo_cpg_residual7/policy_best.pt \
        --ckpt-name sac_spin1 --run-prefix sac_spin1
"""

from __future__ import annotations

import argparse
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
from sylvan.control.sac.models import SacActor, TwinQ
from sylvan.control.sac.replay import ReplayBuffer
from sylvan.control.sac.update import SacLearner
from sylvan.orchestration.day_cycle import prepare_day_run, run_godot_day, spawn_godot_worker
from sylvan.training.checkpointing import save_checkpoint

# reuse the PPO digest (identical real-rollout headline metrics)
from scripts.train_ppo import _quick_digest, _checkpoint_score, _prune_old_cycles


class SacCollectorPool:
    """Parallel Godot collection via N persistent serve_sac_collect subprocesses.

    Same lifecycle as the PPO ParallelCollectorPool, but serves the SAC actor: each
    iteration publish the live actor, SIGHUP every (idle) server to hot-reload it, launch
    the Godot workers, wait. The served action is SAMPLED (SAC entropy = exploration).
    """

    def __init__(self, config, init_actor, *, num_workers: int, host: str = "127.0.0.1") -> None:
        self.config = config
        self.host = host
        self.n = max(1, min(num_workers, int(config.day.num_episodes)))
        self.dir = Path(tempfile.mkdtemp(prefix="sylvan_sacpool_"))
        self.ckpt = self.dir / "behavior.pt"
        self._save(init_actor)
        self.servers: list[dict] = []
        for k in range(self.n):
            port_file = self.dir / f"{k}.port"
            ack_file = self.dir / f"{k}.ack"
            log_file = self.dir / f"{k}.log"
            proc = subprocess.Popen(
                [
                    sys.executable, "-m", "scripts.serve_sac_collect",
                    "--checkpoint", str(self.ckpt), "--host", host, "--port", "0",
                    "--seed", str(2000 + k),
                    "--port-file", str(port_file), "--ack-file", str(ack_file),
                ],
                stdout=open(log_file, "wb"), stderr=subprocess.STDOUT,
            )
            self.servers.append({"proc": proc, "port_file": port_file, "ack_file": ack_file, "port": None})
        for s in self.servers:
            s["port"] = int(self._wait_file(s["port_file"], timeout=120.0))

    def _save(self, actor) -> None:
        tmp = self.ckpt.with_suffix(".tmp")
        torch.save({"model_state_dict": actor.state_dict()}, tmp)
        tmp.rename(self.ckpt)  # atomic publish

    @staticmethod
    def _wait_file(path: Path, *, timeout: float) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.exists():
                return path.read_text().strip()
            time.sleep(0.05)
        raise RuntimeError(f"timeout waiting for {path}")

    def collect(self, run_dir: Path, actor, *, seed: int) -> None:
        for s in self.servers:
            s["ack_file"].unlink(missing_ok=True)
        self._save(actor)
        for s in self.servers:
            if s["proc"].poll() is not None:
                raise RuntimeError(f"SAC collection server died (see {self.dir}/*.log)")
            s["proc"].send_signal(signal.SIGHUP)
        for s in self.servers:
            self._wait_file(s["ack_file"], timeout=30.0)
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


def _warm_start_actor(actor: SacActor, init_from: Path, log_std_init: float) -> None:
    """Map a GaussianActorCritic (residual/BC walker) into the SAC actor: the backbone
    (fc1, fc2) and the mean head (its fc3) transfer 1:1; log_std_head starts fresh small so
    SAC begins with modest exploration around the known-good gait instead of relearning it."""
    ckpt = torch.load(init_from, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    with torch.no_grad():
        actor.fc1.weight.copy_(sd["actor.fc1.weight"]); actor.fc1.bias.copy_(sd["actor.fc1.bias"])
        actor.fc2.weight.copy_(sd["actor.fc2.weight"]); actor.fc2.bias.copy_(sd["actor.fc2.bias"])
        actor.mean_head.weight.copy_(sd["actor.fc3.weight"]); actor.mean_head.bias.copy_(sd["actor.fc3.bias"])
        actor.log_std_head.weight.zero_()
        actor.log_std_head.bias.fill_(log_std_init)
    print(f"[SAC] warm-started actor from {init_from} (backbone+mean) | log_std init={log_std_init}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Off-policy SAC on real Godot rollouts.")
    ap.add_argument("--iterations", type=int, default=120)
    ap.add_argument("--run-prefix", default="sac")
    ap.add_argument("--ckpt-name", default="sac")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--alpha-init", type=float, default=0.2)
    ap.add_argument("--alpha-min", type=float, default=0.05,
                    help="Floor on the temperature so entropy regularization can't vanish "
                         "(run1 collapsed when auto-alpha drove it to ~0 → 100%% falls).")
    ap.add_argument("--fixed-alpha", type=float, default=None,
                    help="Disable auto-temperature; use this constant alpha. Needed because the "
                         "warm-started saturating actor inflates the tanh entropy estimate, so "
                         "auto-alpha makes the entropy bonus dwarf the task (run3 jitter-standstill). "
                         "~0.01 keeps entropy a minor regularizer.")
    ap.add_argument("--num-episodes", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--action-repeat", type=int, default=1)
    ap.add_argument("--capacity", type=int, default=300_000, help="Replay buffer capacity (transitions).")
    ap.add_argument("--grad-steps", type=int, default=400, help="SAC updates per iteration.")
    ap.add_argument("--critic-warmup-iters", type=int, default=4,
                    help="Initial iterations that train ONLY the critic (actor frozen at the "
                         "warm-started gait). Lets Q^pi become meaningful before actor updates, "
                         "so the from-scratch critic doesn't destroy the good warm-started policy.")
    ap.add_argument("--critic-warmup-mult", type=int, default=6,
                    help="During critic warmup, do grad_steps*this critic-only updates per iter.")
    ap.add_argument("--bc-coef", type=float, default=2.0,
                    help="Initial BC-anchor weight: keeps the actor near the warm-started gait "
                         "(in-distribution, where Q is accurate) so it improves turning without "
                         "the critic-extrapolation collapse. 0 disables.")
    ap.add_argument("--bc-decay-iters", type=int, default=80,
                    help="Iterations (after warmup) over which the BC anchor decays linearly to 0, "
                         "freeing the policy to deviate from the gait and turn harder.")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=2000, help="Min buffer size before updating.")
    ap.add_argument("--init-from", default=None, help="Warm-start actor from a GaussianActorCritic walker.")
    ap.add_argument("--init-log-std", type=float, default=-1.0)
    ap.add_argument("--best-metric", default="stability",
                    choices=["stability", "stable_fwd", "stable_return", "return"])
    args = ap.parse_args()

    config = SylvanConfig()
    if args.num_episodes is not None:
        config.day.num_episodes = args.num_episodes
    if args.max_steps is not None:
        config.env.max_episode_steps = args.max_steps

    os.environ["SYLVAN_ACTION_REPEAT"] = str(args.action_repeat)
    torch.manual_seed(args.seed)
    obs_dim = config.env.policy_input_dim
    action_dim = config.env.action_dim
    h = config.controller.hidden_dim

    actor = SacActor(obs_dim=obs_dim, hidden_dim=h, action_dim=action_dim)
    critic = TwinQ(obs_dim=obs_dim, hidden_dim=h, action_dim=action_dim)
    if args.init_from:
        _warm_start_actor(actor, Path(args.init_from), args.init_log_std)
    learner = SacLearner(
        actor=actor, critic=critic, action_dim=action_dim,
        gamma=args.gamma, tau=args.tau, lr=args.lr, alpha_init=args.alpha_init,
        alpha_min=args.alpha_min, fixed_alpha=args.fixed_alpha,
    )
    if args.init_from and args.bc_coef > 0.0:
        learner.set_bc_reference(actor)  # snapshot the warm-started gait as the BC anchor
    buffer = ReplayBuffer(capacity=args.capacity, obs_dim=obs_dim, action_dim=action_dim, gamma=args.gamma)
    gen = torch.Generator().manual_seed(args.seed)

    ckpt_dir = CHECKPOINTS_DIR / args.ckpt_name
    best_score = float("-inf")
    print(f"[SAC] start | obs={obs_dim} act={action_dim} iters={args.iterations} "
          f"episodes/iter={config.day.num_episodes} grad_steps={args.grad_steps} cap={args.capacity}", flush=True)

    pool = SacCollectorPool(config, actor, num_workers=args.num_workers, host=config.godot.policy_host)
    import atexit
    atexit.register(pool.close)
    print(f"[SAC] parallel pool: {pool.n} persistent SAC server processes", flush=True)

    for it in range(args.iterations):
        # 1. Collect with the CURRENT stochastic actor (SAC entropy = exploration).
        run_dir = prepare_day_run(config, run_name=f"{args.run_prefix}_cycle_{it:04d}")
        _prune_old_cycles(run_dir, args.run_prefix, keep_it=it)
        pool.collect(run_dir, learner.actor, seed=args.seed + it * 1000)

        # 2. Ingest into the persistent replay buffer (off-policy: kept across iterations).
        added = buffer.ingest(run_dir)

        # 3. SAC gradient steps over uniform minibatches (the sample reuse).
        # During critic-warmup iters, train ONLY the critic (more steps) with the actor frozen
        # at the warm-started gait — Q^pi must be meaningful before actor improvement starts.
        ustats = {}
        in_warmup = it < args.critic_warmup_iters
        # BC anchor decays linearly to 0 over bc_decay_iters AFTER the critic warmup.
        post = max(0, it - args.critic_warmup_iters)
        bc_coef = args.bc_coef * max(0.0, 1.0 - post / max(1, args.bc_decay_iters))
        if len(buffer) >= max(args.batch_size, 1000 if in_warmup else args.warmup):
            n_steps = args.grad_steps * (args.critic_warmup_mult if in_warmup else 1)
            for _ in range(n_steps):
                batch = buffer.sample(args.batch_size, generator=gen)
                if in_warmup:
                    ustats = learner.mc_warmup(batch)  # supervised MC value regression
                else:
                    ustats = learner.update(batch, bc_coef=bc_coef)

        # 4. Publish + log the real headline metric.
        digest = _quick_digest(run_dir)
        score = _checkpoint_score(digest, args.best_metric)
        metrics = {
            "mean_return": digest["mean_return"], "fall_rate": digest["fall_rate"],
            "mean_length": digest["mean_length"], "mean_fwd_vel": digest["mean_fwd_vel"],
            "best_metric": args.best_metric, "score": score,
        }
        save_checkpoint(destination=ckpt_dir / "policy_latest.pt", model=learner.actor,
                        optimizer=learner.actor_opt, epoch=it, metrics=metrics)
        is_best = score > best_score
        if is_best:
            best_score = score
            save_checkpoint(destination=ckpt_dir / "policy_best.pt", model=learner.actor,
                            optimizer=learner.actor_opt, epoch=it, metrics=metrics)
        print(
            "[SAC] iter %d | return=%.2f len=%.0f fall=%.0f%% fwd_vel=%.3f score=%.1f | "
            "buf=%d(+%d) closs=%.3f aloss=%.3f Q=%.2f bc=%.2f(w%.2f)%s"
            % (
                it, digest["mean_return"], digest["mean_length"], digest["fall_rate"] * 100.0,
                digest["mean_fwd_vel"], score, len(buffer), added,
                ustats.get("critic_loss", 0.0), ustats.get("actor_loss", 0.0),
                ustats.get("q_mean", 0.0), ustats.get("bc_loss", 0.0), bc_coef,
                ("  [critic-warmup]" if in_warmup else "") + ("  <- best" if is_best else ""),
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
