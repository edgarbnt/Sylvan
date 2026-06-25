"""STEP 2 (internaliser le slot, Incrément 1) — entraîne le CANAL-SLOT du WM, WM GELÉ (risque nul pour le WM vivant).

On internalise l'échafaudage : le slot devient un composant du WM (`out["slot"]`), encodé par attention (slot_head,
warm-start) et transporté par la displacement-head que le WM prédit. On entraîne UNIQUEMENT le SlotChannel
(slot_encoder + slot_calib) par CONSISTANCE DE TRANSPORT label-free le long du rêve — la dynamique du WM (encoder/
rssm/displacement/obs/done) est GELÉE → wm_objcentric_s1 a une dynamique identique au WM vivant + le canal-slot.

Gate interne (= Step 2 du plan) : bearing corr held-out (transport du slot t0 le long du rêve vs food_rel0, EVAL-ONLY)
global ≥ +0.55, arrière reporté. eff_rank/displacement/pos triviialement inchangés (WM gelé).

Usage: WM_CKPT=data/checkpoints/wm_rich_fidele_sym_jepa/wm_best.pt SLOT=data/checkpoints/slot_head/slot_best.pt \
       PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.train_slot_channel
"""
import glob
import json
import math
import os
import statistics as st

import torch

from sylvan.models.command_wm import CommandWorldModel, DISPLACEMENT_SCALE, vicreg_terms
from sylvan.models.slot_head import load_slot_head

torch.manual_seed(0)
torch.set_num_threads(4)
WM = os.environ.get("WM_CKPT", "data/checkpoints/wm_rich_fidele_sym_jepa/wm_best.pt")
SLOT = os.environ.get("SLOT", "data/checkpoints/slot_head/slot_best.pt")
OUT = os.environ.get("OUT", "data/checkpoints/wm_objcentric_s1")
BUFS = os.environ.get("BUFS", "retina_eat_a retina_eat_b").split()
L = int(os.environ.get("L", "40"))
ITERS = int(os.environ.get("ITERS", "4000"))
LR = float(os.environ.get("LR", "1e-3"))
BATCH = int(os.environ.get("BATCH", "64"))
print(f"WM={WM}\nSLOT={SLOT}  OUT={OUT}\nBUFS={BUFS} L={L} ITERS={ITERS} LR={LR}")

pl = torch.load(WM, map_location="cpu", weights_only=False)
meta = pl["meta"]
PRO = meta["proprio_dim"]
model = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=PRO,
                          predictor_arch=meta.get("predictor_arch", "shallow"), with_slot=True)
miss, unexp = model.load_state_dict(pl["model"], strict=False)
print(f"WM chargé (slot manquant attendu : {[m for m in miss if m.startswith('slot')][:2]}...)")
# warm-start de l'encodeur slot depuis le slot_head vivant (label-free déjà appris)
sh = load_slot_head(SLOT)
model.slot_encoder.load_state_dict(sh.state_dict())
print("slot_encoder warm-start depuis slot_head ✓")
# GEL : seul slot_encoder s'entraîne. slot_calib = buffer (convention géométrique FIXE (1,−1,−1), pas appris :
# l'apprendre sur food_rel0 faisait orbiter le slot → 0 engagement, cf command_wm.py).
for n, p in model.named_parameters():
    p.requires_grad = n.startswith("slot_encoder")
trainables = [n for n, p in model.named_parameters() if p.requires_grad]
print(f"params entraînés = {trainables[:1]}... ; slot_calib={model.slot_calib.tolist()} (fixe) ; WM gelé")
model.eval()  # WM en eval (pas de dropout/scheduled-sampling), le canal apprend quand même

files = []
for b in BUFS:
    files += sorted(glob.glob(f"godot/data/replay_buffer/{b}/episode_*.jsonl") or
                    glob.glob(f"data/replay_buffer/{b}/episode_*.jsonl"))[:80]


def load_eps():
    eps = []
    for f in files:
        seq = []
        for line in open(f):
            r = json.loads(line)
            w = r.get("wm", {})
            ret = w.get("retina0")
            fr = w.get("food_rel0")
            cmd = w.get("cmd")
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
def make_windows():
    """Précompute par fenêtre : obs_seq [L,obs_dim], disp_real [L,3] (rêvée, WM gelé), P [L,2], VIS [L], is_tr."""
    wins = []
    for ei, seq in enumerate(eps):
        is_tr = ei < ntr
        for s0 in range(0, len(seq) - L, L):
            win = seq[s0:s0 + L]
            if win[0][3] < 0.5:
                continue
            obs = torch.tensor([w[0] for w in win], dtype=torch.float32)
            cmd = torch.tensor([w[1] for w in win], dtype=torch.float32)
            disp = model.rollout_open_loop(obs[0:1], cmd.unsqueeze(0))["predicted_displacement"][0] / DISPLACEMENT_SCALE
            P = torch.tensor([w[2] for w in win])
            VIS = torch.tensor([w[3] for w in win])
            wins.append((obs, disp, P, VIS, is_tr))
    return wins


wins = make_windows()
tr = [w for w in wins if w[4]]
te = [w for w in wins if not w[4]]
print(f"fenêtres = {len(wins)} (train={len(tr)}, test={len(te)})")
OBS_TR = torch.stack([w[0] for w in tr])      # [Ntr, L, obs_dim]
DISP_TR = torch.stack([w[1] for w in tr])     # [Ntr, L, 3]
VIS_TR = torch.stack([w[3] for w in tr])      # [Ntr, L]


def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); d = a.norm() * b.norm()
    return (a @ b).item() / d.item() if d > 1e-6 else float("nan")


@torch.no_grad()
def gate(H=L):
    """bearing corr (cos) sur held-out : transport du slot t0 le long du rêve vs food_rel0 (EVAL-ONLY)."""
    pp, tt, ppr, ttr = [], [], [], []
    for obs, disp, P, VIS, _ in te:
        slot = model.encode_slot(obs[0])                          # [2] depuis la rétine t0
        s = slot
        traj = [s]
        for t in range(len(P) - 1):
            s = model.transport_slot(s, disp[t]); traj.append(s)
        for t in range(min(H, len(P))):
            if VIS[t] > 0.5:
                bt = math.atan2(float(P[t][0]), float(P[t][1]))
                bs = math.atan2(float(traj[t][0]), float(traj[t][1]))
                pp.append(math.cos(bs)); tt.append(math.cos(bt))
                if abs(bt) > math.pi / 2:
                    ppr.append(math.cos(bs)); ttr.append(math.cos(bt))
    g = corr(torch.tensor(pp), torch.tensor(tt))
    r = corr(torch.tensor(ppr), torch.tensor(ttr)) if ppr else float("nan")
    return g, r


opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
g0, r0 = gate()
print(f"\navant entraînement (warm-start slot_head) : global={g0:+.2f} arrière={r0:+.2f}")
N = OBS_TR.shape[0]
for it in range(1, ITERS + 1):
    bi = torch.randint(0, N, (min(BATCH, N),))
    obs = OBS_TR[bi]                                              # [B, L, obs_dim]
    disp = DISP_TR[bi]                                            # [B, L, 3]
    vis = VIS_TR[bi]                                              # [B, L]
    slot_enc = model.encode_slot(obs)                            # [B, L, 2] (grad → slot_encoder)
    trans = model.transport_slot(slot_enc[:, :-1], disp[:, :-1])  # [B, L-1, 2] transport 1-pas
    m = (vis[:, :-1] > 0.5) & (vis[:, 1:] > 0.5)                  # paires visibles
    cons = (((trans - slot_enc[:, 1:].detach()) ** 2).sum(-1) * m).sum() / (m.sum() + 1e-6)
    vv, vc = vicreg_terms(slot_enc.reshape(-1, 2), gamma=1.0)
    loss = cons + 1.0 * vv + 1.0 * vc
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 500 == 0:
        g, r = gate()
        print(f"it {it:>5} | cons={cons.item():.4f} vic=({vv.item():.3f},{vc.item():.3f}) "
              f"| GATE global={g:+.2f} arrière={r:+.2f} | calib={model.slot_calib.data.tolist()}")

gF, rF = gate()
print(f"\nGATE FINAL : global={gF:+.2f} arrière={rF:+.2f}  (seuil interne +0.55 ; réf slot codé-main +0.89/+0.47)")
os.makedirs(OUT, exist_ok=True)
out_meta = {**meta, "with_slot": True, "slot_resources": 1}
torch.save({"model": model.state_dict(), "meta": out_meta, "gate": {"global": gF, "rear": rF}},
           os.path.join(OUT, "wm_best.pt"))
print(f"sauvé → {OUT}/wm_best.pt")
if gF >= 0.55:
    print(">>> SUCCÈS : canal-slot internalisé ≥ +0.55 → Step 3 (câbler le planner sur out['slot'] + re-gate closed-loop).")
else:
    print(">>> SOUS LE GATE : ne pas promouvoir ; diagnostiquer (calib/encodeur) avant Step 3.")
