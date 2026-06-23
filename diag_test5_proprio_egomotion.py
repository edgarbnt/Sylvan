"""TEST 5 / F2 (design WM factorisé) — l'EGO-MOTION est-elle prédictible depuis le PROPRIO, et alimente-t-elle un
transport de slot bien meilleur que les commandes (+0.14 arrière, F1) ? (offline, 0 retrain WM).

F1 a montré : un slot (coord ego explicite) qui PERSISTE transporte le bearing à +0.90/+0.65 (vs WM +0.30), MAIS le
transform depuis les COMMANDES (vx,ω) échoue (Δbearing +0.09/+0.14) car l'ego-motion n'est pas commande-déterminée.
F2 : l'ego-motion VRAIE (Δyaw,Δfwd,Δlat, depuis torso0→torso1) est ground-truth. On teste :
  (A) BORNE HAUTE : transport du slot avec l'ego-motion VRAIE → Δbearing (valide le mécanisme+conventions).
  (B) RÉGRESSION : proprio_t → (Δyaw,Δfwd,Δlat), cross-val par épisode.
  (C) TRANSPORT proprio-prédit : Δbearing via ego-motion prédite depuis proprio, global + ARRIÈRE, vs F1 commandes +0.14.

GO build S1 si : (A) haut (mécanisme sain) ET (C) arrière >> +0.14 (proprio alimente le transform).
SINON : le slot PERSISTE quand même (+0.65 arrière F1) mais sans transform fin → build persistance-first.

Usage: BUF=retina_eat_a PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_test5_proprio_egomotion.py
"""
import json, glob, math, os, statistics as st
import torch
from torch import nn

torch.manual_seed(0); torch.set_num_threads(4)
BUF = os.environ.get("BUF", "retina_eat_a")
files = sorted(glob.glob(f"godot/data/replay_buffer/{BUF}/episode_*.jsonl") or
               glob.glob(f"data/replay_buffer/{BUF}/episode_*.jsonl"))[:80]
print(f"BUF={BUF} fichiers={len(files)}")


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


# Par frame : proprio(132), p=food_rel0(fx,fz,vis), pose torse (x,z,yaw). L'ego-motion vraie est calculée entre
# frames CONSÉCUTIVES (torso0[i]→torso0[i+1]) — le torso0/torso1 INTRA-frame est nul dans ces buffers.
raw = []
for f in files:
    seq = []
    for line in open(f):
        r = json.loads(line); w = r.get("wm", {})
        fr = w.get("food_rel0"); t0 = w.get("torso0"); pro = r["obs"].get("proprio")
        if not fr or not t0 or not pro:
            continue
        seq.append((pro, [float(fr[0]), float(fr[1])], float(fr[2]), t0[0], t0[1], t0[2]))
    if len(seq) > 6:
        raw.append(seq)
# convertir en (proprio, p, vis, dyaw, dfwd, dlat) où l'ego-motion va de la frame i à i+1
eps = []
for seq in raw:
    out = []
    for i in range(len(seq) - 1):
        pro, p, vis, x0, z0, yaw0 = seq[i]
        _, _, _, x1, z1, yaw1 = seq[i + 1]
        dyaw = wrap(yaw1 - yaw0)
        dx, dz = x1 - x0, z1 - z0
        dfwd = dx * math.sin(yaw0) + dz * math.cos(yaw0)
        dlat = dx * math.cos(yaw0) - dz * math.sin(yaw0)
        out.append((pro, p, vis, dyaw, dfwd, dlat, seq[i + 1][1], seq[i + 1][2]))  # +p_next, vis_next
    eps.append(out)
ntr = max(1, int(0.8 * len(eps)))
print(f"épisodes={len(eps)} (train={ntr}, test={len(eps)-ntr}) ; frames={sum(len(e) for e in eps)}")


def transport(p, dyaw, dfwd, dlat):
    # p (ego-coord t) → ego-coord t+1 d'un point monde-statique : translate (−déplacement) puis rotate(−dyaw)
    px, pz = p[0] - dlat, p[1] - dfwd
    ca, sa = math.cos(-dyaw), math.sin(-dyaw)
    return [ca * px - sa * pz, sa * px + ca * pz]


def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); d = a.norm() * b.norm()
    return (a @ b).item() / d.item() if d > 1e-6 else float("nan")


# (A) BORNE HAUTE : transport avec ego-motion VRAIE → Δbearing vs réel (frames rotantes, |dyaw|>0.005)
dp, dt_, dpr, dtr = [], [], [], []
for ei, seq in enumerate(eps):
    if ei < ntr:
        continue
    for (pro, p, vis, dyaw, dfwd, dlat, p1, vis1) in seq:
        if vis < 0.5 or vis1 < 0.5 or abs(dyaw) < 0.005:
            continue
        b0 = math.atan2(p[0], p[1]); b1 = math.atan2(p1[0], p1[1])
        pt = transport(p, dyaw, dfwd, dlat); bp = math.atan2(pt[0], pt[1])
        dt_.append(wrap(b1 - b0)); dp.append(wrap(bp - b0))
        if abs(b0) > math.pi / 2:
            dtr.append(wrap(b1 - b0)); dpr.append(wrap(bp - b0))
print(f"\n(A) BORNE HAUTE ego-motion VRAIE : Δbearing corr GLOBAL={corr(torch.tensor(dp),torch.tensor(dt_)):+.2f} "
      f"ARRIÈRE={corr(torch.tensor(dpr),torch.tensor(dtr)):+.2f} (n={len(dt_)}, arr={len(dtr)})")

# (B) RÉGRESSION proprio → (dyaw, dfwd, dlat)
Xtr, Ytr, Xte, Yte = [], [], [], []
for ei, seq in enumerate(eps):
    for (pro, p, vis, dyaw, dfwd, dlat, p1, vis1) in seq:
        (Xtr if ei < ntr else Xte).append(pro)
        (Ytr if ei < ntr else Yte).append([dyaw, dfwd, dlat])
Xtr = torch.tensor(Xtr); Ytr = torch.tensor(Ytr); Xte = torch.tensor(Xte); Yte = torch.tensor(Yte)
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
muY, sdY = Ytr.mean(0), Ytr.std(0) + 1e-6
net = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 3))
opt = torch.optim.Adam(net.parameters(), 1e-3)
Xn = (Xtr - mu) / sd; Yn = (Ytr - muY) / sdY
for _ in range(3000):
    bi = torch.randint(0, len(Xn), (512,))
    loss = ((net(Xn[bi]) - Yn[bi]) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    pred = net((Xte - mu) / sd) * sdY + muY
names = ["dyaw", "dfwd", "dlat"]
print("\n(B) RÉGRESSION proprio→ego-motion (test, corr par composante) :")
for j in range(3):
    print(f"    {names[j]:>5}: corr={corr(pred[:,j], Yte[:,j]):+.2f}  (réel std={Yte[:,j].std():.4f})")

# (C) TRANSPORT avec ego-motion PRÉDITE-proprio → Δbearing vs réel (mêmes frames rotantes test).
# predY est aligné à Xte = frames test dans l'ordre (épisode puis frame), même ordre que la boucle ci-dessous.
dp2, dt2, dpr2, dtr2 = [], [], [], []
with torch.no_grad():
    predY = (net((Xte - mu) / sd) * sdY + muY).tolist()
k = 0
for ei, seq in enumerate(eps):
    if ei < ntr:
        continue
    for (pro, p, vis, dyaw, dfwd, dlat, p1, vis1) in seq:
        pdyaw, pdfwd, pdlat = predY[k]
        if vis >= 0.5 and vis1 >= 0.5 and abs(dyaw) >= 0.005:
            b0 = math.atan2(p[0], p[1]); b1 = math.atan2(p1[0], p1[1])
            pt = transport(p, pdyaw, pdfwd, pdlat); bp = math.atan2(pt[0], pt[1])
            dt2.append(wrap(b1 - b0)); dp2.append(wrap(bp - b0))
            if abs(b0) > math.pi / 2:
                dtr2.append(wrap(b1 - b0)); dpr2.append(wrap(bp - b0))
        k += 1
cG = corr(torch.tensor(dp2), torch.tensor(dt2)); cR = corr(torch.tensor(dpr2), torch.tensor(dtr2)) if dtr2 else float("nan")
print(f"\n(C) TRANSPORT proprio-prédit : Δbearing corr GLOBAL={cG:+.2f}  ARRIÈRE={cR:+.2f}  (vs F1 commandes +0.09/+0.14)")
print(f"\nVERDICT : ", end="")
if cR >= 0.4 and cR == cR:
    print("GO build S1 — le proprio alimente un transform de slot nettement meilleur que les commandes.")
elif cR >= 0.2:
    print("PARTIEL — proprio aide mais arrière encore faible ; build PERSISTANCE-first (+0.65 F1) + transform best-effort.")
else:
    print("le transform reste faible même via proprio → build PERSISTANCE-first (la persistance, pas le transform, porte le gain).")
