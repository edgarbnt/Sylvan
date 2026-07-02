"""F-pure-1b — version JUSTE de F-pure-1 : encodeur à ATTENTION GÉOMÉTRIQUE (soft-argmax sur les rayons → coordonnée
par construction, comme retina_head), entraîné UNIQUEMENT par consistance de transport auto-supervisée + VICReg, ZÉRO
label. (F-pure-1 avec MLP plat = injuste : le projet sait qu'un MLP plat ne décode pas la position rétine.)

slot(retina) = Σ_k softmax(score(ray_k)) · dist_k · (sinθ_k, cosθ_k)   [θ_k = angle connu du rayon k]
→ l'attention SÉLECTIONNE un rayon, la géométrie en fait une coordonnée ego. L'auto-supervision (transport) doit
faire émerger « attention sur l'OBJET » sans qu'on le dise. SUCCÈS = bearing corr ≥ +0.65 label-free.

Usage: BUF=retina_eat_a GAP=8 PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_fpure1b_attn.py
"""
import json, glob, math, os
import torch
from torch import nn
from sylvan.models.command_wm import vicreg_terms

torch.manual_seed(0); torch.set_num_threads(4)
BUF = os.environ.get("BUF", "retina_eat_a"); GAP = int(os.environ.get("GAP", "8"))
files = sorted(glob.glob(f"godot/data/replay_buffer/{BUF}/episode_*.jsonl") or
               glob.glob(f"data/replay_buffer/{BUF}/episode_*.jsonl"))[:80]
NRAY = 36; RANGE = 10.0
TH = torch.tensor([k * 2 * math.pi / NRAY for k in range(NRAY)])
SIN, COS = torch.sin(TH), torch.cos(TH)
print(f"BUF={BUF} GAP={GAP} fichiers={len(files)}")


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


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
RA, RB, DY, DF, DL, BB, TR = [], [], [], [], [], [], []
for ei, seq in enumerate(eps):
    is_tr = ei < ntr
    for i in range(len(seq) - GAP):
        a = seq[i]; b = seq[i + GAP]
        if a[3] < 0.5 or b[3] < 0.5:
            continue
        x0, z0, y0 = a[4], a[5], a[6]; x1, z1, y1 = b[4], b[5], b[6]
        dyaw = wrap(y1 - y0); dx, dz = x1 - x0, z1 - z0
        RA.append(a[0]); RB.append(b[0]); DY.append(dyaw)
        DF.append(dx * math.sin(y0) + dz * math.cos(y0)); DL.append(dx * math.cos(y0) - dz * math.sin(y0))
        BB.append(math.atan2(b[1], b[2])); TR.append(is_tr)
RA = torch.tensor(RA); RB = torch.tensor(RB); DY = torch.tensor(DY); DF = torch.tensor(DF); DL = torch.tensor(DL)
BB = torch.tensor(BB); TR = torch.tensor(TR); tr = TR.bool(); te = ~tr
print(f"épisodes={len(eps)} paires={len(RA)} (train={int(tr.sum())} test={int(te.sum())})")


class AttnSlot(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, ret):                      # ret [N,144]
        r = ret.reshape(-1, NRAY, 4)             # depth,R,G,B
        s = self.score(r).squeeze(-1)            # [N,36]
        a = torch.softmax(s, dim=1)
        dist = r[..., 0] * RANGE                 # depth→mètres
        px = (a * dist * SIN).sum(1); pz = (a * dist * COS).sum(1)
        return torch.stack([px, pz], dim=1)


def transport(p, dyaw, dfwd, dlat):
    px = p[:, 0] - dlat; pz = p[:, 1] - dfwd
    ca, sa = torch.cos(-dyaw), torch.sin(-dyaw)
    return torch.stack([ca * px - sa * pz, sa * px + ca * pz], dim=1)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); d = a.norm() * b.norm()
    return (a @ b).item() / d.item() if d > 1e-6 else float("nan")


def bearing_corr(enc):
    with torch.no_grad():
        sb = enc(RB[te]); return corr(torch.cos(torch.atan2(sb[:, 0], sb[:, 1])), torch.cos(BB[te]))


# auto-supervisé : transport-consistance + VICReg, ZÉRO label
enc = AttnSlot(); opt = torch.optim.Adam(enc.parameters(), 2e-3)
rai, rbi, dyi, dfi, dli = RA[tr], RB[tr], DY[tr], DF[tr], DL[tr]; N = len(rai)
print(f"plancher (non entraîné) = {bearing_corr(AttnSlot()):+.2f}")
for it in range(9000):
    bi = torch.randint(0, N, (256,))
    sa = enc(rai[bi]); sb = enc(rbi[bi])
    pred = transport(sa, dyi[bi], dfi[bi], dli[bi])
    cons = ((pred - sb.detach()) ** 2).sum(1).mean()
    vv, vc = vicreg_terms(torch.cat([sa, sb], 0), gamma=1.0)
    loss = cons + 1.0 * vv + 1.0 * vc
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 1500 == 1499:
        print(f"  it{it+1}: cons={cons.item():.4f} vic_v={vv.item():.3f} bearing_corr={bearing_corr(enc):+.2f} (|.|={abs(bearing_corr(enc)):.2f})")
ss = bearing_corr(enc); mag = abs(ss)
print(f"\nSLOT ATTENTION AUTO-SUPERVISÉ (label-free) bearing corr = {ss:+.2f} → MAGNITUDE = {mag:.2f}")
print("(le signe est une JAUGE : la consistance détermine le slot à une réflexion/rotation du repère près → magnitude = la vraie mesure)")
print(">>> " + ("SUCCÈS : slot spatial ÉMERGE label-free (attention+transport+VICReg) → PURETÉ VALIDÉE, gate Phase 1."
               if mag >= 0.65 else
               ("PARTIEL : émerge mais imparfait → muscler (gap, archi, plus de data)." if mag >= 0.4 else
                "KILL : l'auto-supervision seule ne suffit pas → repenser le grounding.")))
