"""F-pure-2 (plan WM object-centric pur) — K slots se lient-ils à des OBJETS DISTINCTS ? (offline, gratuit, label-free).
Précondition de la mémoire multi-ressource.

K=2 slots à ATTENTION COMPÉTITIVE (softmax sur les SLOTS → chaque rayon-objet est assigné à UN slot, façon Slot
Attention) ; chaque slot = soft-argmax géométrique de ses rayons → coordonnée ego. Entraînés UNIQUEMENT par
transport-consistance par slot + VICReg + petite répulsion (zéro label). Vérité-terrain des objets = clustering des
rayons ROUGES (food) de la rétine (pas un label externe). Métrique : les 2 slots suivent-ils 2 objets DIFFÉRENTS ?

SUCCÈS = chaque slot corrèle fort (|.|≥0.5) avec un objet DISTINCT (slot1↔nearest, slot2↔2e), et slot1≠slot2.

Usage: BUF=retina_eat_a GAP=8 PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_fpure2_multislot.py
"""
import json, glob, math, os
import torch
from torch import nn
from sylvan.models.command_wm import vicreg_terms

torch.manual_seed(0); torch.set_num_threads(4)
BUF = os.environ.get("BUF", "retina_eat_a"); GAP = int(os.environ.get("GAP", "8")); K = 2
files = sorted(glob.glob(f"godot/data/replay_buffer/{BUF}/episode_*.jsonl") or
               glob.glob(f"data/replay_buffer/{BUF}/episode_*.jsonl"))[:60]
NRAY = 36; RANGE = 10.0
TH = torch.tensor([k * 2 * math.pi / NRAY for k in range(NRAY)]); SIN, COS = torch.sin(TH), torch.cos(TH)
print(f"BUF={BUF} GAP={GAP} K={K} fichiers={len(files)}")


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def obj_bearings(ret):
    """clusters de rayons rouges-hit → liste (bearing, depth) triée par depth (proche d'abord). Label-free (rétine)."""
    fr = []
    for k in range(NRAY):
        d, R, G, B = ret[4 * k:4 * k + 4]
        fr.append(d < 0.999 and R > 0.5 and R > G + 0.1 and R > B + 0.1)
    objs = []; k = 0
    used = [False] * NRAY
    for start in range(NRAY):
        if fr[start] and not fr[(start - 1) % NRAY]:
            ks = []; j = start
            while fr[j % NRAY] and not used[j % NRAY] and len(ks) < NRAY:
                used[j % NRAY] = True; ks.append(j % NRAY); j += 1
            if ks:
                # bearing = moyenne circulaire des angles ; depth = min
                sx = sum(math.sin(TH[i]) for i in ks); cx = sum(math.cos(TH[i]) for i in ks)
                objs.append((math.atan2(sx, cx), min(ret[4 * i] for i in ks)))
    if all(fr):
        objs = [(0.0, min(ret[4 * i] for i in range(NRAY)))]
    objs.sort(key=lambda o: o[1])
    return objs


eps = []
for f in files:
    seq = []
    for line in open(f):
        w = json.loads(line).get("wm", {}); ret = w.get("retina0"); t0 = w.get("torso0")
        if not ret or not t0 or len(ret) != 144:
            continue
        seq.append((ret, t0[0], t0[1], t0[2]))
    if len(seq) > GAP + 2:
        eps.append(seq)
ntr = max(1, int(0.8 * len(eps)))
RA, RB, DY, DF, DL, B1, B2, TR = [], [], [], [], [], [], [], []
for ei, seq in enumerate(eps):
    is_tr = ei < ntr
    for i in range(len(seq) - GAP):
        a = seq[i]; b = seq[i + GAP]
        ob_b = obj_bearings(b[0])
        if len(ob_b) < 2:
            continue
        x0, z0, y0 = a[1], a[2], a[3]; x1, z1, y1 = b[1], b[2], b[3]
        RA.append(a[0]); RB.append(b[0]); DY.append(wrap(y1 - y0))
        dx, dz = x1 - x0, z1 - z0
        DF.append(dx * math.sin(y0) + dz * math.cos(y0)); DL.append(dx * math.cos(y0) - dz * math.sin(y0))
        B1.append(ob_b[0][0]); B2.append(ob_b[1][0]); TR.append(is_tr)   # nearest, 2nd-nearest bearings (vérité rétine)
RA = torch.tensor(RA); RB = torch.tensor(RB); DY = torch.tensor(DY); DF = torch.tensor(DF); DL = torch.tensor(DL)
B1 = torch.tensor(B1); B2 = torch.tensor(B2); TR = torch.tensor(TR); tr = TR.bool(); te = ~tr
print(f"épisodes={len(eps)} paires(≥2 objets)={len(RA)} (train={int(tr.sum())} test={int(te.sum())})")


class MultiSlot(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.key = nn.Sequential(nn.Linear(6, 32), nn.SiLU(), nn.Linear(32, 32))
        self.q = nn.Parameter(torch.randn(K, 32) * 0.5)

    def forward(self, ret):
        N = ret.shape[0]; r = ret.reshape(N, NRAY, 4)
        feat = torch.cat([r, SIN.expand(N, NRAY).unsqueeze(-1), COS.expand(N, NRAY).unsqueeze(-1)], dim=-1)
        key = self.key(feat)                                   # [N,36,32]
        score = torch.einsum("kc,nrc->nkr", self.q, key)       # [N,K,36]
        attn = torch.softmax(score, dim=1)                     # COMPÉTITION sur les slots
        d, R, G, B = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
        mask = (torch.sigmoid(20 * (R - G - 0.1)) * torch.sigmoid(20 * (0.999 - d)))  # soft food-hit
        w = attn * mask.unsqueeze(1)                            # [N,K,36]
        wn = w / (w.sum(-1, keepdim=True) + 1e-6)
        dist = (d * RANGE).unsqueeze(1)                        # [N,1,36]
        px = (wn * (dist * SIN)).sum(-1); pz = (wn * (dist * COS)).sum(-1)  # [N,K]
        return torch.stack([px, pz], dim=-1)                   # [N,K,2]


def transport(p, dyaw, dfwd, dlat):     # p [N,K,2]
    px = p[..., 0] - dlat.unsqueeze(1); pz = p[..., 1] - dfwd.unsqueeze(1)
    ca, sa = torch.cos(-dyaw).unsqueeze(1), torch.sin(-dyaw).unsqueeze(1)
    return torch.stack([ca * px - sa * pz, sa * px + ca * pz], dim=-1)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); d = a.norm() * b.norm()
    return (a @ b).item() / d.item() if d > 1e-6 else float("nan")


def slot_bearings(enc, R):
    with torch.no_grad():
        s = enc(R); return torch.atan2(s[..., 0], s[..., 1])    # [N,K]


enc = MultiSlot(K); opt = torch.optim.Adam(enc.parameters(), 2e-3)
rai, rbi, dyi, dfi, dli = RA[tr], RB[tr], DY[tr], DF[tr], DL[tr]; N = len(rai)
for it in range(7000):
    bi = torch.randint(0, N, (256,))
    sa = enc(rai[bi]); sb = enc(rbi[bi])
    pred = transport(sa, dyi[bi], dfi[bi], dli[bi])
    cons = ((pred - sb.detach()) ** 2).sum(-1).mean()
    vv, vc = vicreg_terms(sa.reshape(-1, 2), gamma=1.0)
    rep = torch.exp(-((sa[:, 0] - sa[:, 1]) ** 2).sum(-1)).mean()   # répulsion : slots distincts
    loss = cons + 1.0 * vv + 1.0 * vc + 0.3 * rep
    opt.zero_grad(); loss.backward(); opt.step()

# métrique : assigner slot↔objet par corrélation sur le test, vérifier 2 objets DISTINCTS
sb = slot_bearings(enc, RB[te]); c = torch.cos(sb)               # [Nte,K]
gt1, gt2 = torch.cos(B1[te]), torch.cos(B2[te])
# matrice de corr slot×objet
M = [[abs(corr(c[:, k], g)) for g in (gt1, gt2)] for k in range(K)]
print(f"\n|corr| slot×objet (lignes=slots, col=[nearest, 2e]) :")
for k in range(K):
    print(f"  slot{k}: nearest={M[k][0]:+.2f}  2e={M[k][1]:+.2f}")
# meilleure assignation (slot0→a, slot1→b) vs croisée
import itertools
best = max(((sum(M[k][p[k]] for k in range(K)), p) for p in itertools.permutations(range(2))), key=lambda x: x[0])
score, perm = best
inter = abs(corr(c[:, 0], c[:, 1]))    # les 2 slots sont-ils le MÊME objet ?
print(f"\nmeilleure assignation: slot0→obj{perm[0]} slot1→obj{perm[1]} | somme|corr|={score:.2f} | corr(slot0,slot1)={inter:.2f}")
ok = (M[0][perm[0]] >= 0.5 and M[1][perm[1]] >= 0.5 and inter < 0.85)
print(">>> " + ("SUCCÈS : 2 slots se lient à 2 objets DISTINCTS (label-free) → multi-slot viable, précondition mémoire OK."
               if ok else
               "PARTIEL/KILL : séparation imparfaite → la compétition/association demande + de soin (Slot Attention itératif)."))
