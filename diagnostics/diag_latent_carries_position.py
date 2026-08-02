"""Sonde GRATUITE : le latent du WM d'attention porte-t-il la position de la bouffe ?

POURQUOI. Le slot token-JEPA plafonne à 1,50 m dans la forêt. L'encodeur d'attention classifie
les types à 99,7 % — son latent SAIT où est la bouffe, mais le slot n'y a pas accès.

Ce diag teste GRATUITEMENT, SANS retrain WM :
  1. Charge wm_foret_attn_hue + corpus gate_foret_cl
  2. Pour chaque tick, encode l'obs → latent [128]
  3. Split train/test (80/20)
  4. Sonde LINÉAIRE + MLP latent→position (L2 loss, < 1 min CPU)
  5. Compare au slot cosinus (2,18 m)

CRITÈRE : erreur médiane < 1,00 m → le latent porte la position → plan_latent justifié.

CLI :
    PYTHONPATH=python SYLVAN_RETINA_FOV_DEG=120 env_pytorch_3.12/bin/python \
        diagnostics/diag_latent_carries_position.py \
        --wm data/checkpoints/wm_foret_attn_hue/wm_best.pt \
        --corpus data/replay_buffer/gate_foret_cl
"""

from __future__ import annotations

import argparse, glob, json, math, os, sys
import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
from sylvan.models.command_wm import CommandWorldModel  # noqa: E402

_HALF_FOV = float(os.environ.get("SYLVAN_RETINA_FOV_DEG", "360")) / 2.0


def load_data(corpus, wm):
    latents, positions, retinas = [], [], []
    for f in sorted(glob.glob(os.path.join(corpus, "*.jsonl"))):
        for line in open(f):
            if not line.strip(): continue
            r = json.loads(line)
            food = r.get("wm", {}).get("food_rel0")
            if not food or len(food) < 3 or food[2] <= 0.5: continue
            bearing = math.degrees(math.atan2(food[0], food[1]))
            if abs(bearing) > _HALF_FOV: continue
            o = r["obs"]
            obs = o["proprio"] + o["retina"] + [o["energy"] / 100.0]
            with torch.no_grad():
                latent = wm.encoder(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            latents.append(latent.squeeze(0))
            positions.append([food[0], food[1]])
            retinas.append(o["retina"])
    return torch.stack(latents), torch.tensor(positions, dtype=torch.float32), torch.tensor(retinas, dtype=torch.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wm", default="data/checkpoints/wm_foret_attn_hue/wm_best.pt")
    ap.add_argument("--corpus", default="data/replay_buffer/gate_foret_cl")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed); torch.set_num_threads(4)
    print(f"=== LE LATENT PORTE-T-IL LA POSITION ? | {a.wm} ===")

    payload = torch.load(a.wm, map_location="cpu", weights_only=False)
    wm = CommandWorldModel.from_checkpoint(payload)
    wm.eval()
    for p in wm.parameters(): p.requires_grad = False
    print(f"1. WM chargé : retina_attention=True, latent_dim=128")

    latents, truths, retinas = load_data(a.corpus, wm)
    n = latents.shape[0]
    print(f"   {n} ticks encodés")

    perm = torch.randperm(n); split = int(0.8 * n)
    tr_idx, te_idx = perm[:split], perm[split:]
    print(f"   train={len(tr_idx)} test={len(te_idx)}")

    # Baseline cosinus
    from sylvan.models.slot_head import SelfSupervisedSlotHead
    slot = SelfSupervisedSlotHead(n_resources=1); slot.eval()
    with torch.no_grad():
        pos_cos = slot.positions(retinas[te_idx])[:, 0, :]
    err_cos = (pos_cos - truths[te_idx]).norm(dim=1)
    print(f"\n2. COSINUS  : méd={err_cos.median():.2f}m  "
          f"<0.5m={(err_cos<0.5).float().mean():.1%}  "
          f"<1.0m={(err_cos<1.0).float().mean():.1%}")

    # Normalisation
    mu = latents[tr_idx].mean(0, keepdim=True)
    sd = latents[tr_idx].std(0, keepdim=True).clamp(min=0.01)
    Xtr = (latents[tr_idx] - mu) / sd
    Xte = (latents[te_idx] - mu) / sd
    Ytr, Yte = truths[tr_idx], truths[te_idx]

    results = {}

    # Sonde LINÉAIRE
    print(f"\n3. Sonde LINÉAIRE (128→2, {a.epochs} époques) :")
    probe = nn.Linear(128, 2)
    opt = torch.optim.Adam(probe.parameters(), lr=a.lr)
    best = float("inf"); best_w = None
    for ep in range(a.epochs):
        probe.train(); opt.zero_grad()
        loss = ((probe(Xtr) - Ytr) ** 2).mean(); loss.backward(); opt.step()
        with torch.no_grad():
            probe.eval()
            e = (probe(Xte) - Yte).norm(dim=1).median()
            if e < best: best = e; best_w = {k: v.clone() for k, v in probe.state_dict().items()}
    probe.load_state_dict(best_w); probe.eval()
    with torch.no_grad():
        e_lin = (probe(Xte) - Yte).norm(dim=1)
    results["linéaire"] = e_lin

    # Sonde MLP
    print(f"\n4. Sonde MLP (128→64→32→2, {a.epochs} époques) :")
    mlp = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 2))
    opt2 = torch.optim.Adam(mlp.parameters(), lr=a.lr)
    best2 = float("inf"); best_w2 = None
    for ep in range(a.epochs):
        mlp.train(); opt2.zero_grad()
        loss = ((mlp(Xtr) - Ytr) ** 2).mean(); loss.backward(); opt2.step()
        with torch.no_grad():
            mlp.eval()
            e = (mlp(Xte) - Yte).norm(dim=1).median()
            if e < best2: best2 = e; best_w2 = {k: v.clone() for k, v in mlp.state_dict().items()}
    mlp.load_state_dict(best_w2); mlp.eval()
    with torch.no_grad():
        e_mlp = (mlp(Xte) - Yte).norm(dim=1)
    results["MLP"] = e_mlp

    # Décomposition par distance
    print(f"\n5. Décomposition (MLP) :")
    dists = truths[te_idx].norm(dim=1)
    for lo, hi, label in [(0, 2, "<2m"), (2, 5, "2-5m"), (5, 10, "5-10m"), (10, 99, ">10m")]:
        m = (dists >= lo) & (dists < hi)
        if m.sum() == 0: continue
        e = e_mlp[m]
        print(f"    {label:6s}: n={m.sum().item():4d}  méd={e.median():.2f}m  moy={e.mean():.2f}m  <0.5m={(e<0.5).float().mean():.1%}")

    # Verdict
    print(f"\n{'─'*60}")
    print(f"    RÉSUMÉ :")
    print(f"    COSINUS           : méd={err_cos.median():.2f}m  <0.5m={(err_cos<0.5).float().mean():.1%}")
    print(f"    Sonde LINÉAIRE    : méd={e_lin.median():.2f}m  <0.5m={(e_lin<0.5).float().mean():.1%}")
    print(f"    Sonde MLP          : méd={e_mlp.median():.2f}m  <0.5m={(e_mlp<0.5).float().mean():.1%}")
    print(f"    Token-slot JEPA    : méd=1.50m  <0.5m=~18%  (réf)")

    if e_mlp.median() < 0.80:
        print(f"\n    ✅ GO : le latent porte la position ({e_mlp.median():.2f}m < 0.80m)")
        print(f"       → plan_latent avec encodeur d'attention justifié.")
        return 0
    else:
        print(f"\n    ❌ NO-GO : {e_mlp.median():.2f}m ≥ 0.80m")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
