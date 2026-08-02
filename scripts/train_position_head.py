"""Entraîne la position_head (128→64→32→2) sur L2(latent, food_rel0), WM GELÉ.

🚨 CE SCRIPT PRODUIT UN CHECKPOINT CONTAMINÉ — NE JAMAIS PROMOUVOIR SA SORTIE.
La cible `food_rel0` est « la position de la chose rouge/rose » : la tête encode donc la règle
d'apparence qu'on cherche précisément à faire disparaître (§3 CLAUDE.md). Changer la couleur de
la nourriture la casse. Le planner REFUSE désormais ce chemin (`command_planner.plan_wm_slot`
n'accepte que `with_slot`).

Il est conservé parce que sa MESURE est l'acquis le plus solide du chantier : servi une nuit en
bootstrap, il a fait passer la survie de 380 à 1833 ticks et les repas de 1,3 à 10,5 par vie —
même monde, même corps, même planner, SEULE la lecture de position changeait. C'est ce contrôle
qui prouve que le goulot est la perception et non le monde, le corps ou le coût.


La sonde MLP prouve que le latent porte la position à 0,55 m. Ce script entraîne
la tête dans le WM pour que rollout_open_loop la serve directement. Gratuit — zéro
retrain WM, juste la tête de lecture (~1 min CPU).

CLI:
    PYTHONPATH=python SYLVAN_RETINA_FOV_DEG=120 env_pytorch_3.12/bin/python \
        scripts/train_position_head.py \
        --wm data/checkpoints/wm_foret_attn_hue/wm_best.pt \
        --corpus data/replay_buffer/gate_foret_cl \
        --out data/checkpoints/wm_foret_attn_hue_pos/wm_best.pt \
        --epochs 200
"""

from __future__ import annotations
import argparse, glob, json, math, os, sys, warnings
import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
from sylvan.models.command_wm import CommandWorldModel

_HALF_FOV = float(os.environ.get("SYLVAN_RETINA_FOV_DEG", "360")) / 2.0


def load_data(corpus, wm):
    latents, truths = [], []
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
            truths.append([food[0], food[1]])
    return torch.stack(latents), torch.tensor(truths, dtype=torch.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wm", default="data/checkpoints/wm_foret_attn_hue/wm_best.pt")
    ap.add_argument("--corpus", default="data/replay_buffer/gate_foret_cl")
    ap.add_argument("--out", default="data/checkpoints/wm_foret_attn_hue_pos/wm_best.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed); torch.set_num_threads(4)
    print(f"=== ENTRAÎNEMENT POSITION_HEAD (WM gelé) | {a.wm} ===")

    # 1. Charger WM original
    payload = torch.load(a.wm, map_location="cpu", weights_only=False)
    meta = payload["meta"]
    print(f"1. WM chargé : retina_attention={meta.get('retina_attention')}")

    # 2. Recharger avec with_position_head=True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wm = CommandWorldModel.from_checkpoint(payload, with_position_head=True)
    wm.eval()
    for p in wm.parameters(): p.requires_grad = False
    for p in wm.position_head.parameters(): p.requires_grad = True
    n_params = sum(p.numel() for p in wm.position_head.parameters())
    print(f"2. position_head ajoutée : {n_params} params")

    # 3. Données
    latents, truths = load_data(a.corpus, wm)
    n = latents.shape[0]
    print(f"3. {n} ticks encodés")
    perm = torch.randperm(n); split = int(0.8 * n)
    tr_idx, te_idx = perm[:split], perm[split:]
    mu = latents[tr_idx].mean(0, keepdim=True)
    sd = latents[tr_idx].std(0, keepdim=True).clamp(min=0.01)
    Xtr = (latents[tr_idx] - mu) / sd
    Xte = (latents[te_idx] - mu) / sd
    Ytr, Yte = truths[tr_idx], truths[te_idx]

    # 4. Entraîner position_head
    opt = torch.optim.Adam(wm.position_head.parameters(), lr=a.lr)
    best_err = float("inf"); best_state = None
    for ep in range(a.epochs):
        wm.train()
        # Re-geler tout sauf position_head
        for p in wm.parameters(): p.requires_grad = False
        for p in wm.position_head.parameters(): p.requires_grad = True
        opt.zero_grad()
        loss = ((wm.position_head(Xtr) - Ytr) ** 2).mean()
        loss.backward(); opt.step()
        with torch.no_grad():
            wm.eval()
            e = (wm.position_head(Xte) - Yte).norm(dim=1).median()
            if e < best_err:
                best_err = e
                best_state = {k: v.clone() for k, v in wm.state_dict().items()
                              if k.startswith("position_head")}
    wm.load_state_dict(best_state, strict=False)
    wm.eval()

    with torch.no_grad():
        pred = wm.position_head(Xte)
        err = (pred - Yte).norm(dim=1)
    print(f"4. Entraîné {a.epochs} époques : méd={err.median():.2f}m moy={err.mean():.2f}m "
          f"<0.5m={(err<0.5).float().mean():.1%} <1.0m={(err<1.0).float().mean():.1%}")

    # 5. Sauver le checkpoint complet (WM + position_head)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    payload["meta"]["with_position_head"] = True
    payload["model"] = wm.state_dict()
    torch.save(payload, a.out)
    mb = os.path.getsize(a.out) / 1024 / 1024
    print(f"5. Checkpoint : {a.out} ({mb:.1f} MB)")

    # 6. Smoke-test rollback (just verify architecture, not accuracy)
    print(f"6. Smoke-test : position_head chargée dans le WM, prête pour rollout_open_loop.")

    if err.median() < 0.80:
        print(f"\n✅ GO : position_head {err.median():.2f}m < 0.80m — prêt pour le planner.")
        return 0
    else:
        print(f"\n❌ {err.median():.2f}m ≥ 0.80m")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
