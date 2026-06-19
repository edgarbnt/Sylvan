"""Entraîner la TÊTE DE VALEUR V(latent) (🅑-pur) — cible 'repas dans K pas' sur le latent GELÉ du WM eat-riche.
Sauve data/checkpoints/value_head_food/value_best.pt (state_dict + mu/sd buffers + meta). AUC held-out par épisode.
Usage: SYLVAN_WM_USE_RETINA=1 PYTHONPATH=python ./env_pytorch_3.12/bin/python train_value_head.py [wm_ckpt]
"""
import sys, json, glob, math
from pathlib import Path
import torch
from sylvan.models.command_wm import CommandWorldModel
from sylvan.models.value_head import ValueHead

WM = sys.argv[1] if len(sys.argv) > 1 else "data/checkpoints/wm_command_hex_retina_eat_v1/wm_best.pt"
K = 20
OUT = Path("data/checkpoints/value_head_food"); OUT.mkdir(parents=True, exist_ok=True)
DIRS = ["godot/data/replay_buffer/retina_eat_a", "godot/data/replay_buffer/retina_eat_b"]

pl = torch.load(WM, map_location="cpu", weights_only=False); meta = pl["meta"]
wm = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
                       predictor_arch=meta.get("predictor_arch", "shallow"))
wm.load_state_dict(pl["model"]); wm.eval()
L = meta.get("latent_dim", 128)
print(f"WM={WM} latent={L} | cible 'repas<{K}pas'")

obs_l, cmd_l, lab_l, eid_l = [], [], [], []
eid = 0
for d in DIRS:
    for f in sorted(glob.glob(f"{d}/episode_*.jsonl")):
        rows = [json.loads(l) for l in open(f)]
        ate = [1.0 if r.get("wm", {}).get("ate") else 0.0 for r in rows]
        for i, r in enumerate(rows):
            ret = r.get("wm", {}).get("retina0")
            if not ret:
                continue
            obs_l.append(r["obs"]["proprio"] + ret + [r["obs"]["energy"] / 100.0])
            cmd_l.append((r["wm"].get("cmd") or [0.0, 0.0])[:2])
            lab_l.append(1.0 if sum(ate[i + 1:i + 1 + K]) > 0 else 0.0)
            eid_l.append(eid)
        eid += 1
OBS = torch.tensor(obs_l, dtype=torch.float32); CMD = torch.tensor(cmd_l, dtype=torch.float32)
LAB = torch.tensor(lab_l); EID = torch.tensor(eid_l)
print(f"frames={len(LAB)} épisodes={eid} positifs={100*LAB.mean():.1f}%")

lats = []
with torch.no_grad():
    for s in range(0, OBS.shape[0], 4096):
        lats.append(wm.rollout_open_loop(OBS[s:s + 4096], CMD[s:s + 4096].reshape(-1, 1, 2))["predicted_latents"][:, 0, :])
LAT = torch.cat(lats)

cut = int(eid * 0.7); tr = EID < cut; te = ~tr
head = ValueHead(L)
head.mu.copy_(LAT[tr].mean(0)); head.sd.copy_(LAT[tr].std(0) + 1e-6)
opt = torch.optim.Adam(head.parameters(), lr=2e-3, weight_decay=1e-4)
pw = ((1 - LAB[tr]).sum() / (LAB[tr].sum() + 1e-6)).clamp(1, 50)
lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)


def auc(score, label):
    s, l = score.flatten(), label.flatten()
    o = torch.argsort(s); rk = torch.empty_like(s); rk[o] = torch.arange(1, len(s) + 1, dtype=s.dtype)
    np_, nn_ = l.sum().item(), (1 - l).sum().item()
    return float("nan") if np_ == 0 or nn_ == 0 else (rk[l == 1].sum().item() - np_ * (np_ + 1) / 2) / (np_ * nn_)


best = 0.0
for ep in range(800):
    head.train(); opt.zero_grad()
    loss = lossf(head.logit(LAT[tr]), LAB[tr]); loss.backward(); opt.step()
    if (ep + 1) % 100 == 0:
        head.eval()
        with torch.no_grad():
            a = auc(head.logit(LAT[te]), LAB[te])
        print(f"  ep{ep+1} loss={loss.item():.3f} AUC_te={a:.3f}")
        if a > best:
            best = a
            torch.save({"state_dict": head.state_dict(), "latent_dim": L, "hidden": 256, "K": K,
                        "wm_ckpt": WM, "auc_heldout": a}, OUT / "value_best.pt")
print(f"AUC held-out best={best:.3f} → {OUT/'value_best.pt'}")
