"""Entraîner la TÊTE D'ORIENTATION OrientHead (🅑-pur, 2026-06-21) — l'analogue LATENT du terme de cap.

Cible = (cos, sin) du BEARING ÉGOCENTRIQUE de la ressource (food_rel0 = (right, fwd) → atan2(right,fwd)), lue
depuis le latent GELÉ du WM. Comme la value : entraînée sur les latents RÊVÉS multi-pas (la distribution que le
planner voit), commandes exécutées, label = bearing de la frame RÉELLE correspondante (honnête). Sauve
data/checkpoints/orient_head_food/orient_best.pt. Métrique held-out = erreur angulaire + acc devant/derrière.
Usage: SYLVAN_WM_USE_RETINA=1 PYTHONPATH=python ./env_pytorch_3.12/bin/python train_orient_head.py [wm_ckpt]
"""
import sys, os, json, glob, math
from pathlib import Path
import torch
from sylvan.models.command_wm import CommandWorldModel
from sylvan.models.value_head import OrientHead

WM = sys.argv[1] if len(sys.argv) > 1 else "data/checkpoints/wm_rich_fidele_sym/wm_best.pt"
OUT = Path(os.environ.get("SYLVAN_ORIENT_OUT", "data/checkpoints/orient_head_food")); OUT.mkdir(parents=True, exist_ok=True)
DIRS = ["godot/data/replay_buffer/retina_eat_a", "godot/data/replay_buffer/retina_eat_b",
        "godot/data/replay_buffer/retina_forage"]
H = int(os.environ.get("SYLVAN_ORIENT_DREAM_H", "30"))
SS = int(os.environ.get("SYLVAN_ORIENT_START_STRIDE", "4"))
DS = int(os.environ.get("SYLVAN_ORIENT_DEPTH_STRIDE", "2"))
CAP = int(os.environ.get("SYLVAN_ORIENT_CAP", "80000"))

pl = torch.load(WM, map_location="cpu", weights_only=False); meta = pl["meta"]
wm = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
                       predictor_arch=meta.get("predictor_arch", "shallow"))
wm.load_state_dict(pl["model"]); wm.eval()
L = meta.get("latent_dim", 128)
print(f"WM={WM} latent={L} | cible (cos,sin) bearing égocentrique sur latents RÊVÉS")

# Charge par épisode : obs (None si pas de rétine), cmd, et bearing-label (cos,sin,valid) par frame.
episodes = []
for d in DIRS:
    for f in sorted(glob.glob(f"{d}/episode_*.jsonl")):
        rows = [json.loads(l) for l in open(f)]
        obs_ep, cmd_ep, lab_ep = [], [], []
        for r in rows:
            w = r.get("wm", {}); ret = w.get("retina0"); fr = w.get("food_rel0")
            obs_ep.append((r["obs"]["proprio"] + ret + [r["obs"]["energy"] / 100.0]) if ret else None)
            cmd_ep.append((w.get("cmd") or [0.0, 0.0])[:2])
            if fr and fr[2] >= 0.5:
                b = math.atan2(fr[0], fr[1]); lab_ep.append((math.cos(b), math.sin(b), 1.0))
            else:
                lab_ep.append((0.0, 0.0, 0.0))
        episodes.append({"obs": obs_ep, "cmd": cmd_ep, "lab": lab_ep})

lat_l, cos_l, sin_l, eid_l = [], [], [], []
with torch.no_grad():
    for eid, ep in enumerate(episodes):
        obs_ep, cmd_ep, lab = ep["obs"], ep["cmd"], ep["lab"]; T = len(lab)
        starts = [i for i in range(0, T, SS) if obs_ep[i] is not None]
        if not starts:
            continue
        O = torch.tensor([obs_ep[i] for i in starts], dtype=torch.float32)
        C = torch.tensor([[cmd_ep[min(i + t, T - 1)] for t in range(H)] for i in starts], dtype=torch.float32)
        lats = wm.rollout_open_loop(O, C)["predicted_latents"]            # [num_starts, H, L]
        for si, i in enumerate(starts):
            for dd in range(0, H, DS):
                j = i + 1 + dd
                if j > T - 1:
                    break
                c, s, v = lab[j]
                if v < 0.5:                                              # cible non visible à cette frame → pas de label
                    continue
                lat_l.append(lats[si, dd:dd + 1, :]); cos_l.append(c); sin_l.append(s); eid_l.append(eid)

LAT = torch.cat(lat_l, dim=0); COS = torch.tensor(cos_l); SIN = torch.tensor(sin_l); EID = torch.tensor(eid_l)
if LAT.shape[0] > CAP:
    g = torch.Generator().manual_seed(0); sel = torch.randperm(LAT.shape[0], generator=g)[:CAP]
    LAT, COS, SIN, EID = LAT[sel], COS[sel], SIN[sel], EID[sel]
BR = torch.atan2(SIN, COS)
print(f"latents={LAT.shape[0]} épisodes={len(episodes)} | %derrière(|brg|>90°)={100*(BR.abs()>math.pi/2).float().mean():.0f}%")

ne = int(EID.max()) + 1; cut = int(ne * 0.7); tr = EID < cut; te = ~tr
head = OrientHead(L)
head.mu.copy_(LAT[tr].mean(0)); head.sd.copy_(LAT[tr].std(0) + 1e-6)
opt = torch.optim.Adam(head.parameters(), lr=2e-3, weight_decay=1e-4)
Ytr = torch.stack([COS[tr], SIN[tr]], 1)


def metrics():
    head.eval()
    with torch.no_grad():
        p = head.cos_sin(LAT[te]); pb = torch.atan2(p[:, 1], p[:, 0])
    err = torch.atan2(torch.sin(pb - BR[te]), torch.cos(pb - BR[te])).abs() * 180 / math.pi
    ahead = ((BR[te].abs() < math.pi / 2) == (pb.abs() < math.pi / 2)).float().mean()
    return float(err.median()), float(ahead)


best = 1e9
for ep in range(800):
    head.train(); opt.zero_grad()
    loss = ((head.cos_sin(LAT[tr]) - Ytr) ** 2).mean(); loss.backward(); opt.step()
    if (ep + 1) % 100 == 0:
        em, ah = metrics()
        print(f"  ep{ep+1} loss={loss.item():.3f} err_med={em:.0f}° devant/derrière={ah*100:.0f}%")
        if em < best:
            best = em
            torch.save({"state_dict": head.state_dict(), "latent_dim": L, "hidden": 256,
                        "wm_ckpt": WM, "err_med_deg": em, "ahead_acc": ah}, OUT / "orient_best.pt")
print(f"BEST err_med={best:.0f}° → {OUT/'orient_best.pt'}")
