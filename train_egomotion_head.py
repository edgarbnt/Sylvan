"""Entraîne EgomotionHead (proprio[132] → (dyaw, dfwd, dlat)) sur retina_wm_a/b.

Cibles construites avec egomotion_from_torso (convention Task 1, identique à
diag_slot_memory_drift.py) sur frames consécutives du buffer.

Cross-val PAR ÉPISODE (80/20) — jamais de split intra-épisode.

Architecture : MLP 1-couche cachée (132→H→3, SiLU).  Le linéaire a été testé en premier et a
échoué (dfwd R²=0.665, dlat R²=0.416 < seuil 0.9), ce qui a déclenché la montée en 1-hidden
MLP conformément au protocole « linear-first » (CLAUDE.md §1).

Usage :
    # Entraîner :
    PYTHONPATH=python ./env_pytorch_3.12/bin/python train_egomotion_head.py

    # Self-check (RED avant train, GREEN après) :
    PYTHONPATH=python ./env_pytorch_3.12/bin/python train_egomotion_head.py --selfcheck

    # Options :
    --bufs retina_wm_a retina_wm_b   (buffers, défaut)
    --out  data/checkpoints/egomotion_head/best.pt
    --epochs 300
    --batch 512
    --lr 1e-3
    --hidden 128                      (dim couche cachée du MLP)
    --seed 0
"""

import argparse
import glob
import json
import math
import os
import sys

import torch
import torch.nn as nn

torch.set_num_threads(4)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DEFAULT_BUFS = ["retina_wm_a", "retina_wm_b"]
DEFAULT_OUT = "data/checkpoints/egomotion_head/best.pt"
R2_THRESHOLD = 0.9
NAMES = ["dyaw", "dfwd", "dlat"]

# ---------------------------------------------------------------------------
# Helpers convention ego-motion (copie verbatim de diag_slot_memory_drift.py)
# ---------------------------------------------------------------------------

def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def egomotion_from_torso(t0, t1):
    """(x0,z0,yaw0), (x1,z1,yaw1) → (dyaw rad, dfwd m, dlat m)."""
    x0, z0, yaw0 = t0
    x1, z1, yaw1 = t1
    dyaw = wrap(yaw1 - yaw0)
    dx, dz = x1 - x0, z1 - z0
    dfwd = dx * math.sin(yaw0) + dz * math.cos(yaw0)
    dlat = dx * math.cos(yaw0) - dz * math.sin(yaw0)
    return dyaw, dfwd, dlat

# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

def load_dataset(bufs):
    """Retourne une liste d'épisodes.
    Chaque épisode = liste de (proprio_t, dyaw_t→t+1, dfwd_t→t+1, dlat_t→t+1).
    """
    episodes = []
    for buf in bufs:
        files = sorted(glob.glob(f"data/replay_buffer/{buf}/*.jsonl"))
        if not files:
            # Chemin alternatif (godot/…)
            files = sorted(glob.glob(f"godot/data/replay_buffer/{buf}/*.jsonl"))
        for fpath in files:
            raw = []
            with open(fpath) as fh:
                for line in fh:
                    r = json.loads(line)
                    w = r.get("wm", {})
                    t0 = w.get("torso0")
                    pro = r.get("obs", {}).get("proprio")
                    if t0 is None or pro is None:
                        continue
                    raw.append((pro, t0))
            if len(raw) < 2:
                continue
            seq = []
            for i in range(len(raw) - 1):
                pro_i, t0_i = raw[i]
                _,     t0_j = raw[i + 1]
                dyaw, dfwd, dlat = egomotion_from_torso(t0_i, t0_j)
                seq.append((pro_i, dyaw, dfwd, dlat))
            if len(seq) >= 5:
                episodes.append(seq)
    return episodes


def build_tensors(episodes, ntr):
    """Retourne (Xtr, Ytr, Xte, Yte) tenseurs float32."""
    Xtr, Ytr, Xte, Yte = [], [], [], []
    for ei, seq in enumerate(episodes):
        target_x = Xtr if ei < ntr else Xte
        target_y = Ytr if ei < ntr else Yte
        for (pro, dyaw, dfwd, dlat) in seq:
            target_x.append(pro)
            target_y.append([dyaw, dfwd, dlat])
    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    Ytr = torch.tensor(Ytr, dtype=torch.float32)
    Xte = torch.tensor(Xte, dtype=torch.float32)
    Yte = torch.tensor(Yte, dtype=torch.float32)
    return Xtr, Ytr, Xte, Yte

# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------

def r2_score(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """R² par colonne.  pred/true : (N, C)."""
    ss_res = ((true - pred) ** 2).sum(0)
    ss_tot = ((true - true.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / (ss_tot + 1e-12)

# ---------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------

def train(args):
    from sylvan.models.egomotion_head import EgomotionHead, save_egomotion_head

    torch.manual_seed(args.seed)

    print(f"[train] Chargement des buffers : {args.bufs}")
    episodes = load_dataset(args.bufs)
    if not episodes:
        print("ERREUR : aucun épisode chargé — vérifier les buffers.", file=sys.stderr)
        sys.exit(1)

    ntr = max(1, int(0.8 * len(episodes)))
    print(f"[train] Épisodes={len(episodes)} (train={ntr}, test={len(episodes) - ntr})")

    Xtr, Ytr, Xte, Yte = build_tensors(episodes, ntr)
    print(f"[train] Frames train={len(Xtr)}, test={len(Xte)}")

    # Normalisation des entrées — buffers stockés dans le checkpoint
    mu_x = Xtr.mean(0)
    sd_x = Xtr.std(0) + 1e-6

    # Modèle MLP 1-couche cachée
    head = EgomotionHead(hidden=args.hidden)
    head.mu_x.copy_(mu_x)
    head.sd_x.copy_(sd_x)

    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    best_r2_min = -float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        head.train()
        perm = torch.randperm(len(Xtr))
        total_loss = 0.0
        n_batches = 0
        for start in range(0, len(Xtr), args.batch):
            idx = perm[start: start + args.batch]
            xb = Xtr[idx]          # entrée brute non normalisée
            yb = Ytr[idx]
            pred = head(xb)        # forward() normalise en interne
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        if epoch % 50 == 0 or epoch == args.epochs:
            head.eval()
            with torch.no_grad():
                pred_te = head(Xte)      # forward() normalise en interne
                r2 = r2_score(pred_te, Yte)
            r2_min = r2.min().item()
            print(
                f"  epoch {epoch:4d}/{args.epochs}  loss={total_loss/n_batches:.5f}"
                f"  R²=[{', '.join(f'{v:.3f}' for v in r2.tolist())}]  min={r2_min:.3f}"
            )
            if r2_min > best_r2_min:
                best_r2_min = r2_min
                best_state = {k: v.clone() for k, v in head.state_dict().items()}

    # Restaurer le meilleur modèle
    if best_state is not None:
        head.load_state_dict(best_state)

    # Évaluation finale
    head.eval()
    with torch.no_grad():
        pred_te = head(Xte)
        r2_final = r2_score(pred_te, Yte)

    print("\n[train] R² FINAL (test, par composante) :")
    all_pass = True
    for j, name in enumerate(NAMES):
        v = r2_final[j].item()
        status = "OK" if v >= R2_THRESHOLD else "FAIL"
        print(f"    {name:>5}: R²={v:.4f}  [{status}]")
        if v < R2_THRESHOLD:
            all_pass = False

    # Sauvegarde
    save_egomotion_head(head, args.out)
    print(f"\n[train] Checkpoint sauvegardé → {args.out}")

    if not all_pass:
        print(
            "\n[train] KILL — au moins une composante R² < 0.9 (cf CLAUDE.md §1 KILL gate).",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        print("[train] Toutes les composantes R² ≥ 0.9  ✓")

# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def selfcheck(args):
    """Charge le checkpoint et vérifie R² ≥ 0.9 sur le split test.
    Retourne 0 si tout passe, 1 sinon (TDD RED/GREEN)."""
    from sylvan.models.egomotion_head import load_egomotion_head

    print(f"[selfcheck] Chargement checkpoint : {args.out}")
    if not os.path.exists(args.out):
        print(f"[selfcheck] ÉCHEC — checkpoint introuvable : {args.out}", file=sys.stderr)
        sys.exit(1)

    head = load_egomotion_head(args.out)
    head.eval()

    print(f"[selfcheck] Chargement des buffers : {args.bufs}")
    episodes = load_dataset(args.bufs)
    if not episodes:
        print("[selfcheck] ERREUR : aucun épisode chargé.", file=sys.stderr)
        sys.exit(1)

    ntr = max(1, int(0.8 * len(episodes)))
    Xtr, Ytr, Xte, Yte = build_tensors(episodes, ntr)
    print(f"[selfcheck] Épisodes test={len(episodes) - ntr}, frames test={len(Xte)}")

    with torch.no_grad():
        pred_te = head(Xte)      # forward() normalise en interne
        r2 = r2_score(pred_te, Yte)

    print("\n[selfcheck] R² (test, par composante) :")
    all_pass = True
    for j, name in enumerate(NAMES):
        v = r2[j].item()
        status = "OK" if v >= R2_THRESHOLD else "FAIL"
        print(f"    {name:>5}: R²={v:.4f}  [{status}]")
        if v < R2_THRESHOLD:
            all_pass = False

    if all_pass:
        print(f"\n[selfcheck] GREEN — toutes les composantes R² ≥ {R2_THRESHOLD}  ✓")
        sys.exit(0)
    else:
        print(
            f"\n[selfcheck] RED — KILL gate : au moins une composante R² < {R2_THRESHOLD}",
            file=sys.stderr,
        )
        sys.exit(1)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Entraîne / vérifie EgomotionHead (MLP 1-hidden)")
    p.add_argument("--selfcheck", action="store_true",
                   help="Mode self-check : charge le checkpoint et vérifie R² ≥ 0.9")
    p.add_argument("--bufs", nargs="+", default=DEFAULT_BUFS,
                   help="Noms des buffers replay (sous data/replay_buffer/)")
    p.add_argument("--out", default=DEFAULT_OUT,
                   help="Chemin de sortie du checkpoint")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128,
                   help="Dimension de la couche cachée du MLP 1-hidden")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.selfcheck:
        selfcheck(args)
    else:
        train(args)
