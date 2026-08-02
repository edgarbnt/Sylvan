"""Entraîne ValueHead(latent) → « vais-je manger ? » sur CONSEQUENCES VECUES.

POURQUOI. Le plan_latent du planner a besoin d'une tête de valeur qui lit les latents RÊVÉS
et prédit « cet état du monde → je mange ». La tête existante (value_head_food) a été entraînée
sur un ancien corpus avec un encodeur dense. Celle-ci est entraînée sur le corpus FORET avec
l'encodeur d'ATTENTION (99,7% de perception type).

PROPRETÉ (§3 CLAUDE.md). La cible est `ate` — un flag binaire écrit par Godot quand l'agent
MANGE (body overlap + trigger → jauge monte). C'est une CONSÉQUENCE VÉCUE, pas un oracle :
- Zéro food_rel0 dans la perte
- Zéro couleur dans le code (le WM lit la rétine → latent → value_head)
- Si on change la couleur de la bouffe, l'agent la mange quand même (ate=1), et la nuit
  il ré-apprend

MÉTHODE. Latents RSSM teacher-forced (1-step) sous les commandes EXÉCUTÉES — même distribution
que plan_latent à t=0. Hypothèse : les latents rêvés multi-step ont une structure similaire
(validé par la value_head originale qui généralisait du teacher-forced au rêvé).

Usage :
    PYTHONPATH=python env_pytorch_3.12/bin/python scripts/train_value_foret.py \
        --wm data/checkpoints/wm_foret_attn_hue/wm_best.pt \
        --corpus data/replay_buffer/gate_foret_cl \
        --out data/checkpoints/value_head_foret \
        --K 20 --epochs 800
"""

from __future__ import annotations
import argparse, glob, json, math, os, sys, warnings
from pathlib import Path

import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from sylvan.models.command_wm import CommandWorldModel  # noqa: E402
from sylvan.models.value_head import ValueHead           # noqa: E402


def load_teacher_forced_latents(corpus: str, wm: CommandWorldModel, K: int, mode: str = "delta",
                                dream: bool = False, dream_h: int = 30,
                                start_stride: int = 4, depth_stride: int = 2):
    """Extrait les latents RSSM pour CHAQUE tick valide.

    Retourne (latents [N,128], labels [N], episode_ids [N]).
    dream=False → teacher-forced 1-step (latent[t,0])
    dream=True  → rêvé multi-step (latent[t,d] à profondeurs variées, mêmes que plan_latent)
    """
    episodes = []
    for f in sorted(glob.glob(os.path.join(corpus, "*.jsonl"))):
        rows = [json.loads(l) for l in open(f)]
        ate = [1.0 if r.get("wm", {}).get("ate", 0.0) > 0.5 else 0.0 for r in rows]
        de = [max(0.0, rows[i + 1]["obs"]["energy"] - rows[i]["obs"]["energy"])
              if i + 1 < len(rows) else 0.0 for i in range(len(rows))]
        obs_ep, cmd_ep = [], []
        for r in rows:
            ret = r.get("wm", {}).get("retina0")
            if ret and len(ret) == 144:
                o = r["obs"]
                obs_ep.append(o["proprio"] + ret + [o["energy"] / 100.0])
            else:
                obs_ep.append(None)
            cmd = r.get("wm", {}).get("cmd")
            cmd_ep.append((cmd[:2] if cmd and len(cmd) >= 2 else [0.0, 0.0]))
        episodes.append({"obs": obs_ep, "cmd": cmd_ep, "ate": ate, "de": de})

    lat_l, lab_l, eid_l = [], [], []
    DELTA_CAP = 140.0
    with torch.no_grad():
        for eid, ep in enumerate(episodes):
            T = len(ep["ate"])
            if not dream:
                valid = [(i, ep["obs"][i]) for i in range(T) if ep["obs"][i] is not None]
                for s in range(0, len(valid), 4096):
                    chunk = valid[s:s + 4096]
                    O = torch.tensor([o for _, o in chunk], dtype=torch.float32)
                    C = torch.tensor([ep["cmd"][i] for i, _ in chunk],
                                     dtype=torch.float32).reshape(-1, 1, 2)
                    lt = wm.rollout_open_loop(O, C)["predicted_latents"][:, 0, :]
                    lat_l.append(lt)
                    if mode == "delta":
                        lab_l += [min(ep["de"][i] / DELTA_CAP, 1.0) for i, _ in chunk]
                    else:
                        lab_l += [_label(ep["ate"], i, K) for i, _ in chunk]
                    eid_l += [eid] * len(chunk)
            else:
                # DREAM MULTI-STEP — mêmes latents rêvés que plan_latent consomme
                starts = [i for i in range(0, T, start_stride) if ep["obs"][i] is not None]
                if not starts: continue
                O = torch.tensor([ep["obs"][i] for i in starts], dtype=torch.float32)
                seqs = []
                for i in starts:
                    seq = [ep["cmd"][min(i + t, T - 1)] for t in range(dream_h)]
                    seqs.append(seq)
                C = torch.tensor(seqs, dtype=torch.float32)
                lats = wm.rollout_open_loop(O, C)["predicted_latents"]
                for si, i in enumerate(starts):
                    for d in range(0, dream_h, depth_stride):
                        j = i + 1 + d
                        if j > T - 1: break
                        lat_l.append(lats[si, d:d + 1, :])
                        if mode == "delta":
                            lab_l.append(min(ep["de"][j] / DELTA_CAP, 1.0))
                        else:
                            lab_l.append(_label(ep["ate"], j, K))
                        eid_l.append(eid)
    if dream:
        return torch.cat(lat_l, dim=0), torch.tensor(lab_l), torch.tensor(eid_l)
    return torch.cat(lat_l), torch.tensor(lab_l), torch.tensor(eid_l)


def _label(ate, j, K):
    """1.0 si l'agent mange dans les K prochains ticks à partir de j+1."""
    return 1.0 if sum(ate[min(j + 1, len(ate) - 1):min(j + 1 + K, len(ate))]) > 0 else 0.0


def auc(score, label):
    s, l = score.flatten(), label.flatten()
    o = torch.argsort(s)
    rk = torch.empty_like(s); rk[o] = torch.arange(1, len(s) + 1, dtype=s.dtype)
    np_, nn_ = l.sum().item(), (1 - l).sum().item()
    if np_ == 0 or nn_ == 0: return float("nan")
    return (rk[l == 1].sum().item() - np_ * (np_ + 1) / 2) / (np_ * nn_)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wm", default="data/checkpoints/wm_foret_attn_hue/wm_best.pt")
    ap.add_argument("--corpus", default="data/replay_buffer/gate_foret_cl")
    ap.add_argument("--out", default="data/checkpoints/value_head_foret")
    ap.add_argument("--K", type=int, default=20, help="fenêtre 'repas dans K pas' (ignoré en mode ΔE)")
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--mode", choices=["ate", "delta"], default="delta",
                    help="ate=binaire 'dans K pas' | delta=régression ΔE continu")
    ap.add_argument("--dream", action="store_true",
                    help="Mode RÊVÉ multi-step (mêmes latents que plan_latent consomme)")
    ap.add_argument("--dream-h", type=int, default=30,
                    help="Profondeur de rêve (--dream only)")
    a = ap.parse_args()

    torch.set_num_threads(4)
    print(f"=== VALUE HEAD FORET (conséquences vécues) | mode={a.mode}"
          f" dream={a.dream} ===")

    # 1. Charger WM (gelé)
    payload = torch.load(a.wm, map_location="cpu", weights_only=False)
    meta = payload["meta"]
    with warnings.catch_warnings(): warnings.simplefilter("ignore")
    wm = CommandWorldModel.from_checkpoint(payload)
    wm.eval()
    for p in wm.parameters(): p.requires_grad = False
    L = meta.get("latent_dim", 128)
    print(f"1. WM : {a.wm}  latent_dim={L}  retina_attention={meta.get('retina_attention')}")

    # 2. Extraire latents teacher-forced
    print(f"2. Extraction latents RSSM depuis {a.corpus}...")
    LAT, LAB, EID = load_teacher_forced_latents(a.corpus, wm, a.K, mode=a.mode,
                                                  dream=a.dream, dream_h=a.dream_h)
    n_total = LAB.shape[0]

    if a.mode == "delta":
        n_repas = int((LAB > 0.0).sum().item())
        print(f"   {n_total} latents  |  repas={n_repas}  "
              f"LAB∈[{LAB.min():.3f}, {LAB.max():.3f}]  "
              f"LAB>0={100*(LAB>0).float().mean():.1f}%")
    else:
        n_pos = int(LAB.sum().item())
        print(f"   {n_total} latents  |  positifs={n_pos} ({100*n_pos/n_total:.1f}%)")

    # 3. Split train/test par épisode
    cut = int(EID.max() * 0.7)
    tr = EID < cut; te = ~tr
    print(f"3. train={int(tr.sum())} test={int(te.sum())}")

    # 4. ValueHead
    head = ValueHead(L)
    head.mu.copy_(LAT[tr].mean(0))
    head.sd.copy_(LAT[tr].std(0).clamp(min=1e-4))
    opt = torch.optim.Adam(head.parameters(), lr=a.lr, weight_decay=1e-4)
    if a.mode == "delta":
        lossf = nn.MSELoss()
        print(f"   Régression MSE sur ΔE ∈ [0,1]")
    else:
        pw = ((1 - LAB[tr]).sum() / (LAB[tr].sum() + 1e-6)).clamp(1, 50)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        print(f"   BCE pos_weight={pw:.1f}")

    # 5. Entraîner
    print(f"4. Entraînement ({a.epochs} époques)...")
    best_metric = -float("inf") if a.mode == "delta" else 0.0
    for ep in range(a.epochs):
        head.train(); opt.zero_grad()
        if a.mode == "delta":
            pred = torch.sigmoid(head.logit(LAT[tr]))   # [0,1] pour MSE
            loss = lossf(pred, LAB[tr])
        else:
            loss = lossf(head.logit(LAT[tr]), LAB[tr])
        loss.backward(); opt.step()

        if (ep + 1) % 100 == 0:
            head.eval()
            with torch.no_grad():
                if a.mode == "delta":
                    pred_te = torch.sigmoid(head.logit(LAT[te]))
                    # Corrélation de Pearson pred vs ΔE (le vrai signal est rare mais fort)
                    pc = pred_te.squeeze(); lc = LAB[te]
                    vx, vy = pc - pc.mean(), lc - lc.mean()
                    corr = (vx * vy).sum() / ((vx**2).sum() * (vy**2).sum() + 1e-8).sqrt()
                    mse_val = float(((pred_te - LAB[te]) ** 2).mean())
                    # Bonus: AUC binaire (LAB>0 comme seuil)
                    de_ate = (LAB[te] > 0.0).float()
                    a_val = auc(torch.sigmoid(head.logit(LAT[te])), de_ate)
                    print(f"   ep{ep+1:4d} loss={loss.item():.4f}  corr={corr:.3f}  "
                          f"AUC(ΔE>0)={a_val:.3f}")
                    metric = float(corr)
                else:
                    a_val = auc(head.logit(LAT[te]), LAB[te])
                    print(f"   ep{ep+1:4d} loss={loss.item():.4f} AUC_te={a_val:.3f}")
                    metric = a_val
            if metric > best_metric:
                best_metric = metric
                out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "state_dict": head.state_dict(),
                    "latent_dim": L, "hidden": 256, "K": a.K,
                    "wm_ckpt": a.wm, "corpus": a.corpus,
                    "mode": a.mode,
                    "best_corr" if a.mode == "delta" else "auc_heldout":
                        float(corr) if a.mode == "delta" else a_val,
                    "n_train": int(tr.sum()),
                    "consequence_trained": True,
                }, out_dir / "value_best.pt")

    # 6. Verdict
    print(f"\n{'─'*60}")
    if a.mode == "delta":
        print(f"Corrélation best={best_metric:.3f}  |  repas={100*(LAB>0).float().mean():.1f}%")
        print(f"Checkpoint → {a.out}/value_best.pt")
        if best_metric > 0.10:
            print(f"✅ corr > 0.10 : le latent prédit ΔE mieux que le hasard.")
            print(f"   → Prête pour plan_latent (--value-head {a.out}/value_best.pt)")
            return 0
        else:
            print(f"❌ corr {best_metric:.3f} ≤ 0.10 : pas de signal prédictible.")
            return 1
    else:
        n_pos = int(LAB.sum().item())
        print(f"AUC held-out best={best_metric:.3f}  |  positifs={100*n_pos/n_total:.1f}%")
        print(f"Checkpoint → {a.out}/value_best.pt")
        if best_metric > 0.65:
            return 0
        else:
            print(f"❌ AUC {best_metric:.3f} ≤ 0.65")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
