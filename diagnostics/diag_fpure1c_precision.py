"""Phase 1 PRÉ-CHECK (gratuit, décisif) — le slot AUTO-SUPERVISÉ atteint-il la précision de retina_head (supervisé-oracle) ?
Si oui → feu vert pour l'intégrer (pureté SANS régression). Si loin → muscler avant d'intégrer.

Entraîne le slot attention auto-supervisé (transport-consistance Rot(+Δyaw) CORRIGÉ + VICReg, ZÉRO label), puis mesure
sur held-out : MAE bearing (deg) + MAE position (m), COMPARÉ à retina_head (supervisé) sur les MÊMES frames.

Usage: BUF=retina_eat_a GAP=8 PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_fpure1c_precision.py
"""
import json, glob, math, os
import torch
from torch import nn
from sylvan.models.command_wm import vicreg_terms
from sylvan.models.perception_head import RetinaPerceptionHead, RETINA_DIM

torch.manual_seed(0); torch.set_num_threads(4)
BUF = os.environ.get("BUF", "retina_eat_a"); GAP = int(os.environ.get("GAP", "8"))
files = sorted(glob.glob(f"godot/data/replay_buffer/{BUF}/episode_*.jsonl") or
               glob.glob(f"data/replay_buffer/{BUF}/episode_*.jsonl"))[:60]
NRAY = 36; RANGE = 10.0; OFF = 0.35
TH = torch.tensor([k * 2 * math.pi / NRAY for k in range(NRAY)]); SIN, COS = torch.sin(TH), torch.cos(TH)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


eps = []
for f in files:
    seq = []
    for line in open(f):
        w = json.loads(line).get("wm", {}); ret = w.get("retina0"); fr = w.get("food_rel0"); t0 = w.get("torso0")
        if not ret or not fr or not t0 or len(ret) != 144:
            continue
        seq.append((ret, float(fr[0]), float(fr[1]), float(fr[2]), t0[0], t0[1], t0[2]))
    if len(seq) > GAP + 2:
        eps.append(seq)
ntr = max(1, int(0.8 * len(eps)))
RA, RB, DY, DF, DL, FXB, FZB, TR = [], [], [], [], [], [], [], []
for ei, seq in enumerate(eps):
    is_tr = ei < ntr
    for i in range(len(seq) - GAP):
        a = seq[i]; b = seq[i + GAP]
        if a[3] < 0.5 or b[3] < 0.5:
            continue
        x0, z0, y0 = a[4], a[5], a[6]; x1, z1, y1 = b[4], b[5], b[6]
        RA.append(a[0]); RB.append(b[0]); DY.append(wrap(y1 - y0))
        dx, dz = x1 - x0, z1 - z0
        DF.append(dx * math.sin(y0) + dz * math.cos(y0)); DL.append(dx * math.cos(y0) - dz * math.sin(y0))
        FXB.append(b[1]); FZB.append(b[2]); TR.append(is_tr)
RA = torch.tensor(RA); RB = torch.tensor(RB); DY = torch.tensor(DY); DF = torch.tensor(DF); DL = torch.tensor(DL)
FXB = torch.tensor(FXB); FZB = torch.tensor(FZB); TR = torch.tensor(TR); tr = TR.bool(); te = ~tr
print(f"BUF={BUF} épisodes={len(eps)} paires={len(RA)} (train={int(tr.sum())} test={int(te.sum())})")


class AttnSlot(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, ret):
        r = ret.reshape(-1, NRAY, 4); a = torch.softmax(self.score(r).squeeze(-1), dim=1)
        dist = r[..., 0] * RANGE + OFF
        return torch.stack([(a * dist * SIN).sum(1), (a * dist * COS).sum(1)], dim=1)


def transport(p, dyaw, dfwd, dlat):     # CORRIGÉ : Rot(+dyaw)
    px = p[:, 0] - dlat; pz = p[:, 1] - dfwd
    ca, sa = torch.cos(dyaw), torch.sin(dyaw)
    return torch.stack([ca * px - sa * pz, sa * px + ca * pz], dim=1)


enc = AttnSlot(); opt = torch.optim.Adam(enc.parameters(), 2e-3)
rai, rbi, dyi, dfi, dli = RA[tr], RB[tr], DY[tr], DF[tr], DL[tr]; N = len(rai)
for it in range(9000):
    bi = torch.randint(0, N, (256,))
    sa = enc(rai[bi]); sb = enc(rbi[bi])
    cons = ((transport(sa, dyi[bi], dfi[bi], dli[bi]) - sb.detach()) ** 2).sum(1).mean()
    vv, vc = vicreg_terms(torch.cat([sa, sb], 0), gamma=1.0)
    (cons + vv + vc).backward(); opt.step(); opt.zero_grad()


def metrics(pred, fx, fz, name):
    bt = torch.atan2(fx, fz); bp = torch.atan2(pred[:, 0], pred[:, 1])
    bmae = torch.atan2(torch.sin(bp - bt), torch.cos(bp - bt)).abs().mean()
    pmae = ((pred[:, 0] - fx) ** 2 + (pred[:, 1] - fz) ** 2).sqrt().mean()
    print(f"  {name:>26}: bearing MAE = {math.degrees(bmae):.1f}°   position MAE = {pmae:.2f} m")
    return math.degrees(bmae), float(pmae)


with torch.no_grad():
    sp = enc(RB[te])
print("\nPRÉCISION sur held-out (test) :")
b_ss, p_ss = metrics(sp, FXB[te], FZB[te], "SLOT auto-supervisé (PUR)")

# retina_head (supervisé-oracle) sur les MÊMES frames
hck = torch.load("data/checkpoints/retina_head/head_best.pt", map_location="cpu", weights_only=False)
rh = RetinaPerceptionHead(n_resources=int(hck.get("n_resources", 1))); rh.load_state_dict(hck["state_dict"]); rh.eval()
with torch.no_grad():
    pos = torch.stack([torch.tensor(rh.locate(RB[te][i])[0]) for i in range(min(2000, int(te.sum())))])
idx = list(range(min(2000, int(te.sum()))))
b_rh, p_rh = metrics(pos, FXB[te][idx], FZB[te][idx], "retina_head (supervisé)")

print(f"\nVERDICT (be sûr que c'est ROBUSTE avant d'intégrer) :")
print(f"  bearing : pur {b_ss:.1f}° vs supervisé {b_rh:.1f}°   |  position : pur {p_ss:.2f} m vs supervisé {p_rh:.2f} m")
ok = (b_ss <= b_rh * 1.5 + 5) and (p_ss <= p_rh + 0.4)
print(">>> " + ("FEU VERT : le slot PUR atteint ~la précision du supervisé → intégrer sans régresser, puis re-gate closed-loop."
               if ok else
               "PAS ENCORE : le slot pur est nettement moins précis → muscler l'auto-supervision (capacité/data/iters) AVANT d'intégrer (ne pas régresser)."))
