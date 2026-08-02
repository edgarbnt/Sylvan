"""L'affinité APPRISE (MLP 4→32→1 par rayon) bat-elle le cos>threshold codé-main ?

POURQUOI. Le soft-argmax localise à 0,24 m quand le filtre est parfait, mais à 1,56 m avec
le cos>threshold. L'Option A (2026-07-30) remplace le filtre cosinus par un petit MLP qui
classifie chaque rayon. Ce diag mesure le gain SUR LE CORPUS, avec des labels ORACLE (issus
de food_rel0) — l'entraînement sur CONSÉQUENCES viendra après.

MÉTHODE :
  1. Sur le corpus gate_foret_cl, pour chaque tick où food_rel0 est visible :
     - Le(s) rayon(s) pointant vers la bouffe → label 1 (food)
     - Les autres rayons → label 0 (not-food)
  2. On split train/test, on entraîne le MLP d'affinité avec poids de classe (food=4.4% des rayons!)
  3. On mesure l'erreur de localisation slot avec affinité APPRISE vs COSINUS vs ORACLE

CRITÈRES PRÉ-ENREGISTRÉS :
  G0 GRATUIT : affinité apprise < 0,80 m d'erreur médiane (barre = 2× l'oracle 0,24 m)
  → le MLP sait classifier les rayons → l'architecture est VALIDE
  → entraînement sur conséquences justifié

CLI :
    PYTHONPATH=python SYLVAN_RETINA_FOV_DEG=120 env_pytorch_3.12/bin/python \
        diagnostics/diag_slot_learned_affinity.py \
        --corpus data/replay_buffer/gate_foret_cl
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from sylvan.models.slot_head import SelfSupervisedSlotHead, NRAY  # noqa: E402

RANGE_M = 10.0
DEPTH_OFFSET = 0.35
_HALF_FOV = float(os.environ.get("SYLVAN_RETINA_FOV_DEG", "360")) / 2.0
FOV = float(os.environ.get("SYLVAN_RETINA_FOV_DEG", "360"))

# ─── Chargement / filtrage ────────────────────────────────────────────────


def ray_bearing(k: int) -> float:
    return (k if k <= NRAY // 2 else k - NRAY) * FOV / NRAY


def load_data(corpus: str):
    """Charge (retina, food_rel0) pour les ticks où la nourriture est visible et dans le FOV."""
    retinas, truths = [], []
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
            retinas.append(r["obs"]["retina"])
            truths.append([v[0], v[1]])
    if not retinas:
        raise SystemExit(f"aucun tick avec food visible dans {corpus}")
    return (torch.tensor(retinas, dtype=torch.float32),
            torch.tensor(truths, dtype=torch.float32))


def label_rays(retina: torch.Tensor, fx: float, fz: float) -> torch.Tensor:
    """Label chaque rayon [0..35] : 1.0 si le rayon touche la bouffe, 0.0 sinon."""
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


# ─── Entraînement de l'affinité ───────────────────────────────────────────


@torch.enable_grad()
def train_affinity(
    retinas: torch.Tensor,
    all_labels: torch.Tensor,  # [N, NRAY] = label oracle par rayon
    n_resources: int = 1,
    epochs: int = 20,
    lr: float = 1e-3,
    valid_frac: float = 0.2,
) -> tuple[SelfSupervisedSlotHead, dict]:
    """Entraîne l'affinity_net du slot SUR LES LABELS ORACLE."""
    n = retinas.shape[0]
    perm = torch.randperm(n)
    split = int(n * (1 - valid_frac))
    train_idx, valid_idx = perm[:split], perm[split:]
    train_r, train_l = retinas[train_idx], all_labels[train_idx]
    valid_r, valid_l = retinas[valid_idx], all_labels[valid_idx]

    # POIDS DE CLASSE : ~4.4% des rayons sont FOOD → sans poids, le MLP apprend à toujours dire NON
    n_pos = float(train_l.sum().item())
    n_neg = float(train_l.numel()) - n_pos
    pos_weight = n_neg / max(n_pos, 1.0)
    print(f"    Rayons food={100*n_pos/train_l.numel():.1f}%, pos_weight={pos_weight:.0f}")

    os.environ["SYLVAN_SLOT_LEARNED_AFF"] = "1"
    head = SelfSupervisedSlotHead(n_resources=n_resources, learned_affinity=True)
    opt = torch.optim.Adam(head.affinity_net.parameters(), lr=lr)

    best_f1, best_state = 0.0, None
    metrics = {"train_loss": [], "valid_f1": [], "valid_recall": [], "valid_prec": [],
               "best_epoch": 0}

    for epoch in range(epochs):
        head.train()
        total_loss = 0.0
        bs = 256
        for i in range(0, train_r.shape[0], bs):
            rb = train_r[i:i + bs]
            lb = train_l[i:i + bs]
            opt.zero_grad()
            r4 = rb.reshape(-1, NRAY, 4)
            preds = [net(r4).squeeze(-1) for net in head.affinity_net]
            loss = torch.tensor(0.0)
            for k in range(n_resources):
                loss = loss + F.binary_cross_entropy(
                    preds[k], lb,
                    weight=torch.where(lb > 0.5, torch.tensor(pos_weight), torch.tensor(1.0)),
                )
            loss.backward()
            opt.step()
            total_loss += float(loss) * rb.shape[0]
        train_loss = total_loss / train_r.shape[0]
        metrics["train_loss"].append(train_loss)

        # Validation — F1/RAPPEL/PRÉCISION sur la classe FOOD (pas l'accuracy globale trompeuse)
        head.eval()
        with torch.no_grad():
            r4v = valid_r.reshape(-1, NRAY, 4)
            preds_v = [net(r4v).squeeze(-1) for net in head.affinity_net]
            pred_bin = (preds_v[0] > 0.5).float()
            tp = (pred_bin * valid_l).sum().item()
            fp = (pred_bin * (1 - valid_l)).sum().item()
            fn = ((1 - pred_bin) * valid_l).sum().item()
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        metrics["valid_f1"].append(f1)
        metrics["valid_recall"].append(rec)
        metrics["valid_prec"].append(prec)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            metrics["best_epoch"] = epoch

    head.load_state_dict(best_state)
    head.eval()
    return head, metrics


# ─── Mesure de localisation ───────────────────────────────────────────────


@torch.no_grad()
def measure(head: SelfSupervisedSlotHead, retinas: torch.Tensor,
            truths: torch.Tensor) -> dict:
    """Erreur de localisation du slot."""
    pos = head.positions(retinas)
    err = (pos[:, 0, :] - truths).norm(dim=1)
    live = pos[:, 0, :].abs().sum(dim=1) > 1e-6
    err_live = err[live]
    return {
        "n": retinas.shape[0],
        "n_live": int(live.sum().item()),
        "err_median": float(err_live.median()) if err_live.numel() > 0 else float("nan"),
        "err_mean": float(err_live.mean()) if err_live.numel() > 0 else float("nan"),
        "err_p90": float(err_live.quantile(0.9)) if err_live.numel() > 0 else float("nan"),
        "pct_lt_05": float((err_live < 0.5).float().mean()) if err_live.numel() > 0 else 0.0,
        "pct_lt_10": float((err_live < 1.0).float().mean()) if err_live.numel() > 0 else 0.0,
        "pct_dead": float((~live).float().mean()),
    }


# ─── Main ──────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="data/replay_buffer/gate_foret_cl")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)

    print(f"=== AFFINITÉ APPRISE vs COSINUS | {a.corpus} ===")
    print(f"    FOV={FOV:.0f}° epochs={a.epochs} seed={a.seed}")

    # 1. Charger + labelliser
    retinas, truths = load_data(a.corpus)
    print(f"\n1. Données : {retinas.shape[0]} ticks, {truths.shape[0]} positions nourriture")

    all_labels = torch.zeros(retinas.shape[0], NRAY, dtype=torch.float32)
    n_labeled = 0
    for i in range(retinas.shape[0]):
        labs = label_rays(retinas[i], float(truths[i, 0]), float(truths[i, 1]))
        all_labels[i] = labs
        if labs.sum() > 0:
            n_labeled += 1
    n_food_rays = int(all_labels.sum().item())
    n_all_rays = int(all_labels.numel())
    print(f"    Ticks avec ≥1 rayon food : {n_labeled}/{retinas.shape[0]} "
          f"({100 * n_labeled / retinas.shape[0]:.0f}%)")
    print(f"    Rayons food : {n_food_rays}/{n_all_rays} "
          f"({100 * n_food_rays / n_all_rays:.1f}%)")

    # 2. Baseline COSINUS
    print(f"\n2. Baseline COSINUS :")
    slot_cos = SelfSupervisedSlotHead(n_resources=1)
    r_cos = measure(slot_cos, retinas, truths)
    print(f"    Erreur  : méd={r_cos['err_median']:.2f}m  moy={r_cos['err_mean']:.2f}m  "
          f"p90={r_cos['err_p90']:.2f}m")
    print(f"    <0.5m   : {100*r_cos['pct_lt_05']:.1f}%")
    print(f"    slots morts : {100*r_cos['pct_dead']:.1f}%")

    # 3. Oracle (filtre parfait)
    print(f"\n3. ORACLE (filtre parfait, plafond) :")
    @torch.no_grad()
    def oracle_positions(retina_batch: torch.Tensor, label_batch: torch.Tensor) -> torch.Tensor:
        B = retina_batch.shape[0]
        r4 = retina_batch.reshape(B, NRAY, 4)
        dist = r4[..., 0] * RANGE_M + DEPTH_OFFSET
        positions = []
        for b in range(B):
            labs = label_batch[b]
            if labs.sum() == 0:
                positions.append(torch.tensor([0.0, 0.0]))
                continue
            w = labs / labs.sum()
            px = (w * dist[b] * slot_cos.sin).sum()
            pz = (w * dist[b] * slot_cos.cos).sum()
            positions.append(torch.stack([px, pz]))
        return torch.stack(positions)

    pos_oracle = oracle_positions(retinas, all_labels)
    err_oracle = (pos_oracle - truths).norm(dim=1)
    live_o = pos_oracle.abs().sum(dim=1) > 1e-6
    err_o = err_oracle[live_o]
    print(f"    Erreur  : méd={err_o.median():.2f}m  moy={err_o.mean():.2f}m  "
          f"p90={err_o.quantile(0.9):.2f}m")
    print(f"    <0.5m   : {100*(err_o < 0.5).float().mean():.1f}%")
    print(f"    slots morts : {100*(~live_o).float().mean():.1f}%")

    # 4. Affinité apprise
    print(f"\n4. AFFINITÉ APPRISE (MLP 4→32→1, {a.epochs} époques) :")
    head_learned, train_metrics = train_affinity(
        retinas, all_labels, n_resources=1, epochs=a.epochs,
    )
    be = train_metrics["best_epoch"]
    print(f"    Meilleure époque {be}: F1={train_metrics['valid_f1'][be]:.3f}  "
          f"rappel={train_metrics['valid_recall'][be]:.3f}  "
          f"précision={train_metrics['valid_prec'][be]:.3f}")

    r_learned = measure(head_learned, retinas, truths)
    print(f"    Erreur  : méd={r_learned['err_median']:.2f}m  moy={r_learned['err_mean']:.2f}m  "
          f"p90={r_learned['err_p90']:.2f}m")
    print(f"    <0.5m   : {100*r_learned['pct_lt_05']:.1f}%")
    print(f"    slots morts : {100*r_learned['pct_dead']:.1f}%")

    # 5. Verdict
    print(f"\n{'─' * 60}")
    print(f"    RÉSUMÉ :")
    print(f"    COSINUS (codé-main) :       méd={r_cos['err_median']:.2f}m  "
          f"<0.5m={100*r_cos['pct_lt_05']:.1f}%")
    print(f"    AFFINITÉ APPRISE (oracle) : méd={r_learned['err_median']:.2f}m  "
          f"<0.5m={100*r_learned['pct_lt_05']:.1f}%")
    print(f"    ORACLE (filtre parfait) :    méd={err_o.median():.2f}m  "
          f"<0.5m={100*(err_o < 0.5).float().mean():.1f}%")

    if r_learned['err_median'] < 0.80:
        print(f"\n    ✅ G0 PASS : affinité apprise {r_learned['err_median']:.2f}m < 0.80m")
        print(f"       → L'architecture MLP sait classifier les rayons.")
        print(f"       → Entraînement sur CONSÉQUENCES justifié.")
        return 0
    else:
        print(f"\n    ❌ G0 ÉCHEC : affinité apprise {r_learned['err_median']:.2f}m ≥ 0.80m")
        print(f"       → Le MLP ne bat pas la barre. Investiguer l'architecture.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
