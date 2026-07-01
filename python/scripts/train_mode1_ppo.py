"""Mode-1 Phase 2 : ENTRAÎNEUR PPO de la politique drive-symétrique (récompense survie-pure).

Boucle on-policy « option A » (UN Godot + UN serveur de collecte par itération) :
  1. snapshot BEHAVIOR = deepcopy(policy) gelé, sauvé au format BC ({"model","meta"}) → temp ;
  2. lance `scripts.serve_mode1_collect --policy <temp>` (échantillonne la gaussienne, logge le buffer) ;
  3. lance Godot dans le RÉGIME PROPRE (env forké de gate1_mode1_bc.sh) → collecte N épisodes ;
  4. tue serveur+Godot et VÉRIFIE 0 orphelin (teardown en finally/atexit) ;
  5. `build_rollout_batch_mode1` → RolloutBatch ; `ppo_update` (symétrie OFF, lr 1e-4) ;
  6. sauve policy.pt (dernier) + policy_best.pt quand la SURVIE (mean_episode_steps) s'améliore.

Exploration : la stochasticité vient du `log_std` de la politique (échantillonné côté serveur) + du
bonus d'entropie de ppo_update. Godot n'ajoute AUCUN bruit (STD_INITIAL/FINAL=0, comme gate1). Il
existe `SYLVAN_INIT_ENERGY/THIRST` mais ils FIXENT (ne randomisent pas) le niveau de départ → non
utilisés ici pour rester byte-identique au régime gate1 (aucun confond) ; `--init-log-std` relance
l'exploration au warm-start.

⚠️ SCOPE : ce script est PROUVÉ par un smoke de PLOMBERIE (1 itération, 2 épisodes courts). Un vrai run
long (« Gate-2 ») est une étape coûteuse déclenchée délibérément par l'owner (CLAUDE.md §1).

Usage (smoke de plomberie — PAS un entraînement) :
    PYTHONPATH=python env_pytorch_3.12/bin/python -m scripts.train_mode1_ppo \\
        --iterations 1 --episodes-per-iter 2 --episode-cap 200 --lr 1e-4
"""

from __future__ import annotations

import argparse
import atexit
import copy
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch

from sylvan.control.mode1.policy import DriveSymmetricPolicy
from sylvan.control.mode1.rollout_mode1 import build_rollout_batch_mode1
from sylvan.control.ppo.update import PPOConfig, ppo_update

ROOT = Path(__file__).resolve().parents[2]           # racine du repo (…/SylvanV1)
PROPRIO_DIM = 132


# --------------------------------------------------------------------------- #
# Sauvegarde / chargement au format BC (celui que serve_mode1* consomme)
# --------------------------------------------------------------------------- #
def _save_bc_format(policy: DriveSymmetricPolicy, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": policy.state_dict(), "meta": {"proprio_dim": policy.proprio_dim}}, path)


# --------------------------------------------------------------------------- #
# Discipline de kill (CLAUDE.md) : pkill NE suffit pas seul → tuer PUIS vérifier.
# --------------------------------------------------------------------------- #
def _pkill_all() -> None:
    for pat in ("serve_mode1_collect", "godot --path godot"):
        subprocess.run(["pkill", "-9", "-f", pat], check=False)


def _verify_no_orphans() -> int:
    """Renvoie le nb de process 'godot' vivants (doit être 0 après teardown)."""
    out = subprocess.run(["pgrep", "-xc", "godot"], capture_output=True, text=True)
    try:
        return int(out.stdout.strip() or "0")
    except ValueError:
        return -1


def _wait_port(host: str, port: int, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.5)
    return False


def _godot_env(base_seed: int, episodes: int, cap: int, port: int, run_dir: Path) -> dict:
    """Env Godot forké EXACTEMENT de gate1_mode1_bc.sh (régime propre, éco de vie 0.05, food/water=5,
    rétine, policy_server) — SEULS varient épisodes/cap/seed/port/run_dir. Aucun confond."""
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": "python",
        # --- régime propre hexapode (CLAUDE.md) ---
        "SYLVAN_CPG": "1", "SYLVAN_RESIDUAL_GAIN": "0.4", "SYLVAN_TURN_FADE": "0",
        "SYLVAN_FOOT_FRICTION": "7", "SYLVAN_CPG_SPEEDCAD": "0.6", "SYLVAN_CPG_PERIOD": "0.5",
        # --- perception rétine + planner CPG ---
        "SYLVAN_CPG_PLANNER": "1", "SYLVAN_RETINA_PLANNER": "1", "SYLVAN_WM_USE_RETINA": "1",
        "SYLVAN_EAT_RADIUS": "1.0", "SYLVAN_DRINK_RADIUS": "1.0",
        # --- économie de vie multi-pulsions ---
        "SYLVAN_FOOD_COUNT": "5", "SYLVAN_WATER_COUNT": "5",
        "SYLVAN_ENERGY_DRAIN": "0.05", "SYLVAN_THIRST_DRAIN": "0.05",
        # --- collecte ---
        "SYLVAN_COLLECT": "1", "SYLVAN_NUM_EPISODES": str(episodes),
        "SYLVAN_MAX_EPISODE_STEPS": str(cap), "SYLVAN_SEED": str(base_seed),
        "SYLVAN_COLLECTOR_MODE": "policy_server",
        "SYLVAN_POLICY_HOST": "127.0.0.1", "SYLVAN_POLICY_PORT": str(port),
        # bruit Godot OFF : l'exploration vient du log_std de la politique (échantillonné côté serveur)
        "SYLVAN_POLICY_EXPLORATION_STD_INITIAL": "0", "SYLVAN_POLICY_EXPLORATION_STD_FINAL": "0",
        "SYLVAN_REFLEX_STRENGTH": "0", "SYLVAN_ASSIST_RATIO": "0",
        "SYLVAN_RUN_DIR": str(run_dir),
    })
    return env


def run_iteration(
    it: int, policy: DriveSymmetricPolicy, optimizer, args, out_dir: Path,
) -> dict | None:
    """Une itération on-policy complète. Renvoie les stats du batch (ou None si buffer vide)."""
    iter_dir = out_dir / f"iter_{it:03d}"
    buffer_dir = iter_dir / "buffer"
    godot_run_dir = iter_dir / "godot_run"
    buffer_dir.mkdir(parents=True, exist_ok=True)
    godot_run_dir.mkdir(parents=True, exist_ok=True)
    behavior_ckpt = iter_dir / "behavior.pt"
    port = args.base_port + it

    # 1) snapshot behavior gelé, sauvé au format BC lisible par serve_mode1_collect
    behavior = copy.deepcopy(policy).eval()
    for p in behavior.parameters():
        p.requires_grad_(False)
    _save_bc_format(behavior, behavior_ckpt)

    srv_proc = None
    godot_proc = None
    try:
        # 2) serveur de collecte : commande python SEULE (pas de préambule kill → sinon exit1)
        srv_log = open(iter_dir / "server.log", "w")
        srv_proc = subprocess.Popen(
            [sys.executable, "-m", "scripts.serve_mode1_collect",
             "--residual", str(args.residual), "--policy", str(behavior_ckpt),
             "--out", str(buffer_dir), "--seed", str(it),
             "--host", "127.0.0.1", "--port", str(port),
             "--replan-every", str(args.replan_every)],
            cwd=str(ROOT), env={**os.environ, "PYTHONPATH": "python"},
            stdout=srv_log, stderr=subprocess.STDOUT,
        )
        if not _wait_port("127.0.0.1", port, timeout=60.0):
            raise RuntimeError(f"serve_mode1_collect n'écoute pas sur :{port} (voir {iter_dir/'server.log'})")

        # 3) Godot headless dans le régime propre
        godot_bin = os.environ.get("GODOT_BIN", str(ROOT / "tools/godot/godot"))
        godot_log = open(iter_dir / "godot.log", "w")
        godot_proc = subprocess.Popen(
            [godot_bin, "--path", "godot", "--headless"],
            cwd=str(ROOT),
            env=_godot_env(args.seed + it, args.episodes_per_iter, args.episode_cap, port, godot_run_dir),
            stdout=godot_log, stderr=subprocess.STDOUT,
        )
        godot_proc.wait(timeout=args.godot_timeout)
    finally:
        # 4) teardown + vérif orphelins (même en cas de crash)
        for proc in (godot_proc, srv_proc):
            if proc and proc.poll() is None:
                proc.send_signal(signal.SIGKILL)
        _pkill_all()
        time.sleep(1.0)
        n_orphan = _verify_no_orphans()
        print(f"[train-mode1] it={it} teardown: pgrep -xc godot = {n_orphan}", flush=True)

    # 5) buffer → batch
    batch, stats = build_rollout_batch_mode1(buffer_dir, behavior, gamma=args.gamma, lam=args.lam)
    if batch is None:
        print(f"[train-mode1] it={it} buffer VIDE ({stats}) → skip update", flush=True)
        return None

    # 6) PPO update (symétrie OFF, lr 1e-4)
    cfg = PPOConfig(
        clip=args.clip, value_coef=args.value_coef, entropy_coef=args.entropy_coef,
        epochs=args.epochs, minibatch_size=args.minibatch, max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl, sym_coef=0.0, sym_v_coef=0.0,
    )
    up = ppo_update(policy, optimizer, batch, cfg)
    stats.update(up)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Sylvan Mode-1 Phase-2 PPO trainer (survival-pure).")
    ap.add_argument("--policy", default="data/checkpoints/mode1_bc/policy.pt", help="warm-start BC")
    ap.add_argument("--residual", default="data/checkpoints/hexapod_v2/policy_best.pt")
    ap.add_argument("--out", default="data/checkpoints/mode1_ppo")
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--episodes-per-iter", type=int, default=8)
    ap.add_argument("--episode-cap", type=int, default=3000, help="SYLVAN_MAX_EPISODE_STEPS")
    ap.add_argument("--replan-every", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4, help="OBLIGATOIRE 1e-4 (3e-4 = instable)")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--init-log-std", type=float, default=-0.5, help="relance l'exploration au warm-start")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--base-port", type=int, default=6060)
    ap.add_argument("--godot-timeout", type=float, default=1800.0)
    # PPOConfig
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--target-kl", type=float, default=0.03)
    args = ap.parse_args()

    if abs(args.lr - 1e-4) > 1e-9:
        print(f"[train-mode1] ATTENTION lr={args.lr} != 1e-4 (défaut recommandé, cf CLAUDE.md)", flush=True)

    torch.manual_seed(args.seed)
    out_dir = ROOT / args.out if not os.path.isabs(args.out) else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # teardown de sécurité au cas où le process meurt (atexit)
    atexit.register(_pkill_all)

    # --- Politique : warm-start acteur BC (strict=False → critique frais), reset log_std ------------
    policy = DriveSymmetricPolicy(proprio_dim=PROPRIO_DIM)
    bc_path = ROOT / args.policy if not os.path.isabs(args.policy) else Path(args.policy)
    ck = torch.load(bc_path, map_location="cpu", weights_only=False)
    missing, unexpected = policy.load_state_dict(ck["model"], strict=False)
    with torch.no_grad():
        policy.log_std.fill_(float(args.init_log_std))
    print(f"[train-mode1] warm-start {bc_path.name} : missing={list(missing)} unexpected={list(unexpected)} "
          f"| log_std reset→{args.init_log_std} (std={policy.mean_std():.3f})", flush=True)

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    best_survival = float("-inf")
    for it in range(args.iterations):
        stats = run_iteration(it, policy, optimizer, args, out_dir)
        if stats is None:
            continue
        _save_bc_format(policy, out_dir / "policy.pt")  # dernier (format BC → rechargeable par serve_mode1)
        surv = stats.get("mean_episode_steps", 0.0)
        tag = ""
        if surv > best_survival:
            best_survival = surv
            _save_bc_format(policy, out_dir / "policy_best.pt")
            tag = " ★best"
        print(
            f"[train-mode1] it={it} n_ep={stats['n_episodes']:.0f} mean_steps={surv:.1f} "
            f"mean_reward={stats['mean_reward']:.3f} approx_kl={stats['approx_kl']:.4f} "
            f"value_loss={stats['value_loss']:.3f} mean_std={stats['mean_std']:.3f}{tag}",
            flush=True,
        )

    print("[train-mode1] DONE", flush=True)


if __name__ == "__main__":
    main()
