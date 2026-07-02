"""TEST 6 / S1-FAISABILITÉ (design WM factorisé) — la displacement-head EXISTANTE du WM transporte-t-elle un SLOT ?
(offline, 0 retrain). Gate : S1 est-il inference-only ou faut-il retrain ?

Idée S1 (persistance-first) : un SLOT = coordonnée ego de l'objet, initialisée depuis la perception (ici la vraie
food_rel0 à t0, comme le ferait retina_head), puis TRANSPORTÉE le long du rêve par l'ego-motion que le WM prédit DÉJÀ
(displacement_head → d_fwd,d_lat,d_yaw). Le slot PERSISTE (état explicite) → ne peut pas être perdu comme dans le latent.

On compare le transport du bearing (corr poolée cos(brg), par horizon + ARRIÈRE) :
  - SLOT (transporté par la displacement-head du WM)   = ce que S1 ferait à l'inférence
  - dreamed-latent (réf, sonde sur latents rêvés)      ≈ +0.30 (test3) = le WM actuel
  - STATIC slot (gèle t0)                              = contrôle persistance pure
Calibration : on fit 3 scalaires (yaw/fwd/lat scale+signe) sur TRAIN pour aligner la convention de la displacement-head
au repère food_rel0 (lève l'ambiguïté de signe, cf F2) — c'est un readout, pas un entraînement du WM.

SUCCÈS : SLOT >> +0.30 (esp. ARRIÈRE) → S1 quasi inference-only (ajouter le slot au planner, peu/pas de retrain).
PARTIEL : SLOT bat le latent mais arrière faible → retrain léger (superviser le slot le long du rollout).
KILL : SLOT ≈ latent → la displacement-head dérive trop dans le rêve → retrain displacement-aware d'abord.

Usage: WM_CKPT=data/checkpoints/wm_rich_fidele_sym/wm_best.pt BUF=retina_eat_a PYTHONPATH=python \
       ./env_pytorch_3.12/bin/python diag_test6_slot_transport.py
"""
import json, glob, math, os, statistics as st
import torch
from sylvan.models.command_wm import CommandWorldModel, DISPLACEMENT_SCALE

torch.manual_seed(0); torch.set_num_threads(4)
WM = os.environ.get("WM_CKPT", "data/checkpoints/wm_rich_fidele_sym/wm_best.pt")
BUF = os.environ.get("BUF", "retina_eat_a")
L = 40; OMG = 0.30
print(f"WM={WM} BUF={BUF}")
pl = torch.load(WM, map_location="cpu", weights_only=False); meta = pl["meta"]
wm = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
                       predictor_arch=meta.get("predictor_arch", "shallow"))
wm.load_state_dict(pl["model"]); wm.eval()
files = sorted(glob.glob(f"godot/data/replay_buffer/{BUF}/episode_*.jsonl") or
               glob.glob(f"data/replay_buffer/{BUF}/episode_*.jsonl"))[:80]


def load_eps():
    eps = []
    for f in files:
        seq = []
        for line in open(f):
            r = json.loads(line); w = r.get("wm", {})
            ret = w.get("retina0"); fr = w.get("food_rel0"); cmd = w.get("cmd")
            if not ret or not fr or not cmd:
                continue
            en = r["obs"].get("energy", 50.0)
            seq.append((r["obs"]["proprio"] + ret + [en / 100.0], list(cmd[:2]),
                        [float(fr[0]), float(fr[1])], float(fr[2])))
        if len(seq) > L + 2:
            eps.append(seq)
    return eps


eps = load_eps()
ntr = max(1, int(0.8 * len(eps)))
print(f"épisodes={len(eps)} (train={ntr}, test={len(eps)-ntr})")


@torch.no_grad()
def wm_disp(win):
    obs = torch.tensor([w[0] for w in win], dtype=torch.float32).unsqueeze(0)
    cmd = torch.tensor([w[1] for w in win], dtype=torch.float32).unsqueeze(0)
    out = wm.rollout_open_loop(obs[:, 0, :], cmd)
    return out["predicted_displacement"][0] / DISPLACEMENT_SCALE     # [T,3] = (d_fwd,d_lat,d_yaw) réels


# segments de virage, objet visible à t0 ; cache la displacement rêvée du WM
segs = []
for ei, seq in enumerate(eps):
    is_tr = ei < ntr
    for s0 in range(0, len(seq) - L, L):
        win = seq[s0:s0 + L]
        if win[0][3] < 0.5 or st.mean(abs(w[1][1]) for w in win) <= OMG:
            continue
        disp = wm_disp(win)
        P = torch.tensor([w[2] for w in win]); VIS = torch.tensor([w[3] for w in win])
        segs.append((P, VIS, disp, is_tr))
print(f"segments de virage = {len(segs)} (train={sum(s[3] for s in segs)})")


def transport(p, dyaw, dfwd, dlat):
    px, pz = p[0] - dlat, p[1] - dfwd
    ca, sa = math.cos(-dyaw), math.sin(-dyaw)
    return [ca * px - sa * pz, sa * px + ca * pz]


def rollout_slot(P, disp, kyaw, kfwd, klat):
    s = [float(P[0][0]), float(P[0][1])]; out = [s]
    for t in range(len(P) - 1):
        dfwd, dlat, dyaw = float(disp[t][0]), float(disp[t][1]), float(disp[t][2])
        s = transport(s, kyaw * dyaw, kfwd * dfwd, klat * dlat); out.append(s)
    return out


def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); d = a.norm() * b.norm()
    return (a @ b).item() / d.item() if d > 1e-6 else float("nan")


# Calibration (kyaw,kfwd,klat) sur train, VECTORISÉE : aligne la displacement-head au repère food_rel0 (1-pas, visibles)
Pt, Pn, Df, Dl, Dy = [], [], [], [], []
for P, VIS, disp, is_tr in segs:
    if not is_tr:
        continue
    for t in range(len(P) - 1):
        if VIS[t] > 0.5 and VIS[t + 1] > 0.5:
            Pt.append(P[t]); Pn.append(P[t + 1]); Df.append(disp[t][0]); Dl.append(disp[t][1]); Dy.append(disp[t][2])
Pt = torch.stack(Pt); Pn = torch.stack(Pn); Df = torch.stack(Df); Dl = torch.stack(Dl); Dy = torch.stack(Dy)
print(f"transitions calibration = {len(Pt)}")
kyaw = torch.tensor(1.0, requires_grad=True); kfwd = torch.tensor(1.0, requires_grad=True); klat = torch.tensor(1.0, requires_grad=True)
opt = torch.optim.Adam([kyaw, kfwd, klat], lr=0.05)
for _ in range(1500):
    px = Pt[:, 0] - klat * Dl; pz = Pt[:, 1] - kfwd * Df
    a = -kyaw * Dy; ca, sa = torch.cos(a), torch.sin(a)
    pxn = ca * px - sa * pz; pzn = sa * px + ca * pz
    loss = ((pxn - Pn[:, 0]) ** 2 + (pzn - Pn[:, 1]) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
KY, KF, KL = kyaw.item(), kfwd.item(), klat.item()
print(f"calibration: kyaw={KY:+.2f} kfwd={KF:+.2f} klat={KL:+.2f}")


def bearing_corr(getslot, H):
    pp, tt, ppr, ttr = [], [], [], []
    for P, VIS, disp, is_tr in segs:
        if is_tr:
            continue
        slot = getslot(P, disp)
        for t in range(min(H, len(P))):
            if VIS[t] > 0.5:
                bt = math.atan2(float(P[t][0]), float(P[t][1]))
                bs = math.atan2(slot[t][0], slot[t][1])
                pp.append(math.cos(bs)); tt.append(math.cos(bt))
                if abs(bt) > math.pi / 2:
                    ppr.append(math.cos(bs)); ttr.append(math.cos(bt))
    g = corr(torch.tensor(pp), torch.tensor(tt))
    r = corr(torch.tensor(ppr), torch.tensor(ttr)) if ppr else float("nan")
    return g, r


def slot_fn(P, disp):
    return rollout_slot(P, disp, KY, KF, KL)


def static_fn(P, disp):
    return [[float(P[0][0]), float(P[0][1])]] * len(P)


print(f"\ntransport du bearing (corr poolée cos(brg), test) :")
print(f"{'H':>4} | {'SLOT':>6} | {'STATIC':>7} || {'SLOT-arr':>9} | {'STATIC-arr':>10}")
for H in (5, 10, 20, 40):
    gs, rs = bearing_corr(slot_fn, H); g0, r0 = bearing_corr(static_fn, H)
    print(f"{H:>4} | {gs:>+6.2f} | {g0:>+7.2f} || {rs:>+9.2f} | {r0:>+10.2f}")

gS, rS = bearing_corr(slot_fn, L)
print(f"\nVERDICT (H={L}) : SLOT global={gS:+.2f} arrière={rS:+.2f}  (réf dreamed-latent ≈ +0.30 ; seuils +0.8 / +0.7 arr)")
if gS >= 0.8 and (math.isnan(rS) or rS >= 0.7):
    print(">>> S1 QUASI INFERENCE-ONLY : la displacement-head transporte le slot → ajouter le slot au planner, peu/pas de retrain.")
elif gS >= 0.5:
    print(">>> PARTIEL : slot bat le latent mais arrière imparfait → retrain léger (superviser le slot le long du rollout).")
else:
    print(">>> la displacement-head dérive dans le rêve → retrain displacement-aware d'abord.")
