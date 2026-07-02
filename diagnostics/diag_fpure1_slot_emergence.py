"""F-pure-1 (plan WM object-centric pur) — un SLOT SPATIAL ÉMERGE-t-il SANS label de position ? (offline, gratuit).

Test de PURETÉ central : on entraîne un encodeur retina(144)→slot(2D) UNIQUEMENT par consistance de transport
auto-supervisée — transport(slot_t, ego-motion_t→t+k) ≈ stop-grad(slot_{t+k}) — + VICReg (anti-collapse). AUCUN label
de position dans la perte. Hypothèse : seul une COORDONNÉE ego se transforme rigidement sous l'ego-motion → la
contrainte FORCE le slot à devenir spatial tout seul ; l'encodeur doit, sans qu'on lui dise, extraire la position de
l'objet (la chose monde-fixe saillante de la rétine).

Ego-motion VRAIE = pose torse entre frames (torso0[i]→torso0[i+k], cf F2). Écart k pour un transport substantiel.
SONDE (éval only, held-out par épisode) : corr(bearing décodé du slot, vrai bearing food_rel0). Contrôles : slot
SUPERVISÉ (borne haute) + slot non-entraîné (plancher).

SUCCÈS = slot auto-supervisé décode le bearing à corr ≥ +0.65 (≈ slot codé-main) EN RESTANT label-free.
KILL = ≈ plancher → l'auto-supervision seule ne fait pas émerger le spatial → anchor de position léger (à flagger).

Usage: BUF=retina_eat_a GAP=8 PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_fpure1_slot_emergence.py
"""
import json, glob, math, os, statistics as st
import torch
from torch import nn
from sylvan.models.command_wm import vicreg_terms

torch.manual_seed(0); torch.set_num_threads(4)
BUF = os.environ.get("BUF", "retina_eat_a")
GAP = int(os.environ.get("GAP", "8"))
files = sorted(glob.glob(f"godot/data/replay_buffer/{BUF}/episode_*.jsonl") or
               glob.glob(f"data/replay_buffer/{BUF}/episode_*.jsonl"))[:80]
print(f"BUF={BUF} GAP={GAP} fichiers={len(files)}")


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


# episodes: list of frames (retina[144], fx, fz, vis, x, z, yaw)
eps = []
for f in files:
    seq = []
    for line in open(f):
        r = json.loads(line); w = r.get("wm", {})
        ret = w.get("retina0"); fr = w.get("food_rel0"); t0 = w.get("torso0")
        if not ret or not fr or not t0 or len(ret) != 144:
            continue
        seq.append((ret, float(fr[0]), float(fr[1]), float(fr[2]), t0[0], t0[1], t0[2]))
    if len(seq) > GAP + 2:
        eps.append(seq)
ntr = max(1, int(0.8 * len(eps)))
print(f"épisodes={len(eps)} (train={ntr}, test={len(eps)-ntr})")

# pairs (i, i+GAP) both visible : retina_a, retina_b, net ego-motion a→b, bearings, is_train
RA, RB, DY, DF, DL, BA, BB, TR = [], [], [], [], [], [], [], []
for ei, seq in enumerate(eps):
    is_tr = ei < ntr
    for i in range(len(seq) - GAP):
        a = seq[i]; b = seq[i + GAP]
        if a[3] < 0.5 or b[3] < 0.5:
            continue
        x0, z0, y0 = a[4], a[5], a[6]; x1, z1, y1 = b[4], b[5], b[6]
        dyaw = wrap(y1 - y0); dx, dz = x1 - x0, z1 - z0
        dfwd = dx * math.sin(y0) + dz * math.cos(y0); dlat = dx * math.cos(y0) - dz * math.sin(y0)
        RA.append(a[0]); RB.append(b[0]); DY.append(dyaw); DF.append(dfwd); DL.append(dlat)
        BA.append(math.atan2(a[1], a[2])); BB.append(math.atan2(b[1], b[2])); TR.append(is_tr)
RA = torch.tensor(RA); RB = torch.tensor(RB)
DY = torch.tensor(DY); DF = torch.tensor(DF); DL = torch.tensor(DL)
BA = torch.tensor(BA); BB = torch.tensor(BB); TR = torch.tensor(TR)
tr = TR.bool(); te = ~tr
print(f"paires = {len(RA)} (train={int(tr.sum())}, test={int(te.sum())}) | |Δyaw| moy={DY.abs().mean():.3f} |Δfwd| moy={DF.abs().mean():.3f}")


def transport(p, dyaw, dfwd, dlat):
    # p:[N,2]=(fx,fz). translate -(dlat,dfwd) puis Rot(-dyaw)
    px = p[:, 0] - dlat; pz = p[:, 1] - dfwd
    ca, sa = torch.cos(-dyaw), torch.sin(-dyaw)
    return torch.stack([ca * px - sa * pz, sa * px + ca * pz], dim=1)


def make_enc():
    return nn.Sequential(nn.Linear(144, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))


def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); d = a.norm() * b.norm()
    return (a @ b).item() / d.item() if d > 1e-6 else float("nan")


def bearing_corr(enc):
    with torch.no_grad():
        sb = enc(RB[te]); dec = torch.atan2(sb[:, 0], sb[:, 1])
        return corr(torch.cos(dec), torch.cos(BB[te]))


# ---- 1) SLOT AUTO-SUPERVISÉ (consistance de transport + VICReg, ZÉRO label) ----
enc = make_enc(); opt = torch.optim.Adam(enc.parameters(), 1e-3)
rai, rbi = RA[tr], RB[tr]; dyi, dfi, dli = DY[tr], DF[tr], DL[tr]
N = len(rai)
for it in range(4000):
    bi = torch.randint(0, N, (256,))
    sa = enc(rai[bi]); sb = enc(rbi[bi])
    pred = transport(sa, dyi[bi], dfi[bi], dli[bi])
    cons = ((pred - sb.detach()) ** 2).sum(1).mean()          # JEPA : stop-grad sur la cible
    vv, vc = vicreg_terms(torch.cat([sa, sb], 0), gamma=1.0)   # anti-collapse (sinon slot→const)
    loss = cons + 1.0 * vv + 1.0 * vc
    opt.zero_grad(); loss.backward(); opt.step()
ss = bearing_corr(enc)

# ---- 2) CONTRÔLE borne haute : slot SUPERVISÉ sur la vraie position (ce que le slot DEVRAIT atteindre) ----
enc_sup = make_enc(); opt2 = torch.optim.Adam(enc_sup.parameters(), 1e-3)
# cible = vraie position ego (fx,fz) reconstruite depuis bearing+? on n'a que bearing → superviser le cos/sin du bearing
tgtA = torch.stack([torch.sin(BA[tr]), torch.cos(BA[tr])], 1)
for it in range(3000):
    bi = torch.randint(0, int(tr.sum()), (256,))
    sa = enc_sup(RA[tr][bi]); sa = sa / (sa.norm(dim=1, keepdim=True) + 1e-6)
    loss = ((sa - tgtA[bi]) ** 2).sum(1).mean()
    opt2.zero_grad(); loss.backward(); opt2.step()
sup = bearing_corr(enc_sup)

# ---- 3) plancher : encodeur non entraîné ----
floor = bearing_corr(make_enc())

print(f"\nbearing corr (held-out, sonde) :")
print(f"  plancher (enc non entraîné)            = {floor:+.2f}")
print(f"  SLOT AUTO-SUPERVISÉ (transport+VICReg) = {ss:+.2f}   <-- LE test de pureté (label-free)")
print(f"  borne haute (slot supervisé position)  = {sup:+.2f}")
print(f"\nVERDICT (seuil +0.65) :")
if ss >= 0.65:
    print(">>> SUCCÈS : un slot SPATIAL émerge de l'auto-supervision SEULE → pureté validée, on gate la Phase 1.")
elif ss >= 0.4:
    print(">>> PARTIEL : le spatial émerge mais imparfait → muscler (gap, archi, features sin/cos) ou anchor léger (à flagger).")
else:
    print(">>> KILL : l'auto-supervision seule ne fait pas émerger le spatial → repenser (anchor de position ? à flagger).")
