"""Entraîne l'affinity_head de l'encodeur d'attention sur des labels ORACLE.

POURQUOI. L'encodeur d'attention (wm_foret_attn_hue) classifie les rayons à 99,7 % dans son
latent, mais le slot lit cos>threshold sur la rétine brute → 2,18 m d'erreur. L'affinity_head
(64→32→1 par ressource, branchée dans l'étape 1-3) projette les tokens encodeur [B,36,64]
vers une affinité par rayon [B,36,K] que le slot utilise comme filtre.

Ce script :
  1. Charge le corpus gate_foret_cl et label chaque rayon (food=1, pas-food=0) via food_rel0
  2. Charge wm_foret_attn_hue, override with_slot=True → nouveau modules random init
  3. Gèle TOUT le WM SAUF affinity_attn + affinity_head + token + slot_encoder.score
  4. Entraîne BCE(affinity_head(attention(tok)), labels) avec poids de classe
  5. Mesure l'erreur de localisation slot et compare au cosinus

CLI :
    PYTHONPATH=python SYLVAN_RETINA_FOV_DEG=120 env_pytorch_3.12/bin/python \
        scripts/train_slot_affinity.py \
        --corpus data/replay_buffer/gate_foret_cl \
        --wm data/checkpoints/wm_foret_attn_hue/wm_best.pt \
        --out data/checkpoints/slot_affinity/slot_aff_best.pt \
        --epochs 50
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import warnings

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from sylvan.models.command_wm import CommandWorldModel  # noqa: E402

NRAY = 36
RANGE_M = 10.0
DEPTH_OFFSET = 0.35
_HALF_FOV = float(os.environ.get("SYLVAN_RETINA_FOV_DEG", "360")) / 2.0
FOV_DEG = float(os.environ.get("SYLVAN_RETINA_FOV_DEG", "360"))


def ray_bearing(k: int) -> float:
    return (k if k <= NRAY // 2 else k - NRAY) * FOV_DEG / NRAY


def load_data(corpus: str):
    obs_list, rets, truths = [], [], []
    for f in sorted(glob.glob(os.path.join(corpus, "*.jsonl"))):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            v = r.get("wm", {}).get("food_rel0")
            if not v or len(v) < 3 or v[2] <= 0.5:
                continue
            bearing = math.degrees(math.atan2(v[0], v[1]))
            if abs(bearing) > _HALF_FOV:
                continue
            o = r["obs"]
            obs_list.append(o["proprio"] + o["retina"] + [o["energy"] / 100.0])
            rets.append(o["retina"])
            truths.append([v[0], v[1]])
    return (torch.tensor(obs_list, dtype=torch.float32),
            torch.tensor(rets, dtype=torch.float32),
            torch.tensor(truths, dtype=torch.float32))


def label_rays(retina: torch.Tensor, fx: float, fz: float) -> torch.Tensor:
    labels = torch.zeros(NRAY, dtype=torch.float32)
    bearing = math.degrees(math.atan2(fx, fz))
    true_dist = math.hypot(fx, fz)
    for k in range(NRAY):
        rb = ray_bearing(k)
        ad = abs(rb - bearing)
        if ad > 180:
            ad = 360 - ad
        ang_tol = max(3.5, math.degrees(math.atan2(0.25, true_dist)))
        if ad > ang_tol:
            continue
        d = float(retina[k * 4]) * RANGE_M + DEPTH_OFFSET
        if abs(d - true_dist) < 3.0:
            labels[k] = 1.0
    return labels


def train(
    wm: CommandWorldModel,
    obs: torch.Tensor,
    all_labels: torch.Tensor,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    valid_frac: float = 0.2,
    seed: int = 42,
) -> tuple[dict, dict]:
    """Entraîne affinity_attn + affinity_head + token + slot.score sur labels oracle."""

    torch.manual_seed(seed)
    n = obs.shape[0]
    perm = torch.randperm(n)
    split = int(n * (1 - valid_frac))
    train_idx, valid_idx = perm[:split], perm[split:]

    # Figer tout, puis dégeler les modules à entraîner
    wm.eval()
    train_modules = [
        wm.encoder.affinity_attn,
        wm.encoder.affinity_head,
        wm.encoder.token,
        wm.slot_encoder.score,
    ]
    trainable = set()
    for mod in train_modules:
        for p in mod.parameters():
            p.requires_grad = True
            trainable.add(p)
    n_trainable = sum(p.numel() for p in trainable)
    print(f"    Paramètres entraînables : {n_trainable}")

    opt = torch.optim.Adam([
        {"params": wm.encoder.token.parameters(), "lr": lr * 0.1},
        {"params": wm.encoder.affinity_attn.parameters(), "lr": lr * 0.1},
        {"params": wm.encoder.affinity_head.parameters(), "lr": lr},
        {"params": wm.slot_encoder.score.parameters(), "lr": lr * 0.1},
    ])

    n_pos = float(all_labels[train_idx].sum().item())
    n_neg = float(all_labels[train_idx].numel()) - n_pos
    pos_weight = n_neg / max(n_pos, 1.0)
    print(f"    Rayons food : {100 * n_pos / all_labels[train_idx].numel():.1f}%, "
          f"pos_weight={pos_weight:.0f}")

    best_f1, best_state = 0.0, None
    history = {"train_loss": [], "valid_f1": [], "valid_recall": [],
               "valid_prec": [], "valid_acc": [], "best_epoch": 0}

    for epoch in range(epochs):
        # ── Train ──
        wm.train()
        # Re-geler les parties non entraînées
        for name, p in wm.named_parameters():
            if p not in trainable:
                p.requires_grad = False
        total_loss = 0.0
        bs = 256
        idx = torch.randperm(len(train_idx))
        for start in range(0, len(train_idx), bs):
            bi = idx[start:start + bs]
            ob = obs[train_idx[bi]]
            lb = all_labels[train_idx[bi]]
            opt.zero_grad()
            # Utiliser get_affinity() qui passe par self-attention + MLP
            aff_raw = wm.encoder.get_affinity(ob)              # [B, 36, n_res]
            loss = torch.tensor(0.0)
            for k in range(aff_raw.shape[-1]):
                pred = aff_raw[..., k]                          # [B, 36]
                loss = loss + F.binary_cross_entropy(
                    pred, lb,
                    weight=torch.where(lb > 0.5,
                                       torch.tensor(pos_weight),
                                       torch.tensor(1.0)),
                )
            loss.backward()
            opt.step()
            total_loss += float(loss) * ob.shape[0]
        history["train_loss"].append(total_loss / len(train_idx))

        # ── Valid ──
        wm.eval()
        with torch.no_grad():
            ob_v = obs[valid_idx]
            lb_v = all_labels[valid_idx]
            aff_v = wm.encoder.get_affinity(ob_v)              # [B, 36, n_res]
            pred_bin = (aff_v[..., 0] > 0.5).float()
            tp = (pred_bin * lb_v).sum().item()
            fp = (pred_bin * (1 - lb_v)).sum().item()
            fn = ((1 - pred_bin) * lb_v).sum().item()
            tn = ((1 - pred_bin) * (1 - lb_v)).sum().item()

            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            acc = (tp + tn) / max(tp + tn + fp + fn, 1)

        history["valid_f1"].append(f1)
        history["valid_recall"].append(rec)
        history["valid_prec"].append(prec)
        history["valid_acc"].append(acc)

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in wm.state_dict().items()
                          if any(k.startswith(p) for p in
                                 ["encoder.affinity_attn", "encoder.affinity_head",
                                  "encoder.token", "slot_encoder.score"])}
            history["best_epoch"] = epoch

    wm.load_state_dict(best_state, strict=False)
    wm.eval()
    return history, best_state


@torch.no_grad()
def measure_slot(wm: CommandWorldModel, obs: torch.Tensor, retina: torch.Tensor,
                 truths: torch.Tensor, use_affinity: bool = True) -> dict:
    if use_affinity and hasattr(wm.encoder, "get_affinity"):
        aff_raw = wm.encoder.get_affinity(obs)
        aff = aff_raw.permute(0, 2, 1)
        pos = wm.slot_encoder.positions(retina, affinity=aff)[:, 0, :]
    else:
        pos = wm.slot_encoder.positions(retina)[:, 0, :]

    err = (pos - truths).norm(dim=1)
    live = pos.abs().sum(dim=1) > 1e-6
    err_live = err[live]
    return {
        "n": retina.shape[0],
        "n_live": int(live.sum().item()),
        "err_median": float(err_live.median()) if err_live.numel() > 0 else float("nan"),
        "err_mean": float(err_live.mean()) if err_live.numel() > 0 else float("nan"),
        "pct_lt_05": float((err_live < 0.5).float().mean()) if err_live.numel() > 0 else 0.0,
        "pct_lt_10": float((err_live < 1.0).float().mean()) if err_live.numel() > 0 else 0.0,
        "pct_dead": float((~live).float().mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="data/replay_buffer/gate_foret_cl")
    ap.add_argument("--wm", default="data/checkpoints/wm_foret_attn_hue/wm_best.pt")
    ap.add_argument("--out", default="data/checkpoints/slot_affinity/slot_aff_best.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    print(f"=== ENTRAÎNEMENT AFFINITY_HEAD | {a.corpus} ===")
    print(f"    FOV={FOV_DEG:.0f}° epochs={a.epochs} lr={a.lr}")

    # 1. Données
    obs, retina, truths = load_data(a.corpus)
    print(f"\n1. Données : {obs.shape[0]} ticks, obs_dim={obs.shape[1]}")

    all_labels = torch.zeros(obs.shape[0], NRAY, dtype=torch.float32)
    n_labeled = 0
    for i in range(obs.shape[0]):
        labs = label_rays(retina[i], float(truths[i, 0]), float(truths[i, 1]))
        all_labels[i] = labs
        if labs.sum() > 0:
            n_labeled += 1
    print(f"    Ticks avec ≥1 rayon food : {n_labeled}/{obs.shape[0]} "
          f"({100 * n_labeled / obs.shape[0]:.0f}%)")
    print(f"    Rayons food : {int(all_labels.sum().item())}/{all_labels.numel()} "
          f"({100 * all_labels.sum().item() / all_labels.numel():.1f}%)")

    # 2. WM + slot override
    payload = torch.load(a.wm, map_location="cpu", weights_only=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wm = CommandWorldModel.from_checkpoint(payload, with_slot=True, slot_resources=1)
    print(f"\n2. WM chargé avec slot override")

    # 3. Mesure AVANT
    print(f"\n3. Mesure AVANT entraînement :")
    r_cos = measure_slot(wm, obs, retina, truths, use_affinity=False)
    print(f"    COSINUS  : méd={r_cos['err_median']:.2f}m  "
          f"<0.5m={100 * r_cos['pct_lt_05']:.1f}%")
    r_enc = measure_slot(wm, obs, retina, truths, use_affinity=True)
    print(f"    ENCODEUR : méd={r_enc['err_median']:.2f}m  "
          f"<0.5m={100 * r_enc['pct_lt_05']:.1f}%  (random init)")

    # 4. Entraîner
    print(f"\n4. Entraînement ({a.epochs} époques) :")
    history, _ = train(wm, obs, all_labels, epochs=a.epochs, lr=a.lr, seed=a.seed)

    be = history["best_epoch"]
    print(f"    Meilleure époque {be}: F1={history['valid_f1'][be]:.3f}  "
          f"rappel={history['valid_recall'][be]:.3f}  "
          f"préc={history['valid_prec'][be]:.3f}  acc={history['valid_acc'][be]:.3f}")

    # 5. Mesure APRÈS
    print(f"\n5. Mesure APRÈS entraînement :")
    r_enc2 = measure_slot(wm, obs, retina, truths, use_affinity=True)
    print(f"    ENCODEUR : méd={r_enc2['err_median']:.2f}m  "
          f"<0.5m={100 * r_enc2['pct_lt_05']:.1f}%  "
          f"<1.0m={100 * r_enc2['pct_lt_10']:.1f}%")

    # 6. Sauvegarde
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    payload["meta"]["with_slot"] = True
    payload["meta"]["slot_resources"] = 1
    payload["meta"]["slot_affinity_trained"] = True
    payload["model"] = wm.state_dict()
    payload["affinity_train_history"] = history
    torch.save(payload, a.out)
    print(f"\n6. Checkpoint : {a.out}  ({os.path.getsize(a.out) / 1024 / 1024:.1f} MB)")

    # 7. Verdict
    print(f"\n{'─' * 60}")
    print(f"    RÉSUMÉ :")
    print(f"    COSINUS (avant) :      méd={r_cos['err_median']:.2f}m  "
          f"<0.5m={100 * r_cos['pct_lt_05']:.1f}%")
    print(f"    ENCODEUR (avant) :     méd={r_enc['err_median']:.2f}m  "
          f"<0.5m={100 * r_enc['pct_lt_05']:.1f}%  (random init)")
    print(f"    ENCODEUR (entraîné) :  méd={r_enc2['err_median']:.2f}m  "
          f"<0.5m={100 * r_enc2['pct_lt_05']:.1f}%")

    gain = r_enc['err_median'] - r_enc2['err_median']
    if r_enc2['err_median'] < 0.80:
        print(f"\n    ✅ G0 PASS : affinité encodeur {r_enc2['err_median']:.2f}m < 0.80m "
              f"(gain {gain:.2f}m)")
        return 0
    else:
        print(f"\n    ❌ G0 ÉCHEC : {r_enc2['err_median']:.2f}m ≥ 0.80m (gain {gain:.2f}m)")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
