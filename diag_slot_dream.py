"""STEP 1 (internaliser le slot, Incrément 1) — GATE GRATUIT, 0 retrain du WM.

Question : le slot APPRIS (slot_head, label-free) survit-il au transport par la displacement-head du WM
le long d'un rêve open-loop ? = la version PURE de diag_test6 (qui initialisait le slot depuis l'ORACLE
food_rel0). Si oui, la perception apprise + le transport-WM tiennent hors-ligne → on gate le retrain Step 2.

Pipeline (identique à diag_test6, SEUL l'init du slot change) :
  - WM gelé + slot_head gelé, sur held-out retina_eat_a/b (segments tournants, objet visible à t0).
  - init slot = slot_head.positions(retina0)[food]   (APPRIS)   vs   food_rel0   (ORACLE, réf haute ≈ +0.65)
  - transport le long du rêve par predicted_displacement (calibration 3-scalaires fitée sur food_rel0).
  - bearing corr (cos-bearing vs food_rel0, EVAL-ONLY) global + bucket ARRIÈRE (|brg|>90°), horizons {5,10,20,40}.

Réfs : dreamed-latent ≈ +0.30 (test3) ; oracle-slot ≈ +0.65 (test6).
GATE (plan §1) : SLOT-appris global ≥ +0.55 ET arrière ≥ +0.45 → SUCCÈS, on gate Step 2.
                 ≈ +0.30 → KILL : la displacement prédite dégrade le slot dans le rêve → retrain displacement-aware d'abord.

Usage: WM_CKPT=data/checkpoints/wm_rich_fidele_sym_jepa/wm_best.pt SLOT=data/checkpoints/slot_head/slot_best.pt \
       BUF=retina_eat_a PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_slot_dream.py
"""
import json, glob, math, os, statistics as st
import torch
from sylvan.models.command_wm import CommandWorldModel, DISPLACEMENT_SCALE
from sylvan.models.slot_head import load_slot_head

torch.manual_seed(0); torch.set_num_threads(4)
WM = os.environ.get("WM_CKPT", "data/checkpoints/wm_rich_fidele_sym_jepa/wm_best.pt")
SLOT_CKPT = os.environ.get("SLOT", "data/checkpoints/slot_head/slot_best.pt")
BUF = os.environ.get("BUF", "retina_eat_a")
L = 40; OMG = 0.30
print(f"WM={WM}\nSLOT={SLOT_CKPT}  BUF={BUF}")
pl = torch.load(WM, map_location="cpu", weights_only=False); meta = pl["meta"]
wm = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
                       predictor_arch=meta.get("predictor_arch", "shallow"))
wm.load_state_dict(pl["model"]); wm.eval()
slot_head = load_slot_head(SLOT_CKPT); slot_head.eval()
PRO = meta["proprio_dim"]
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
                        [float(fr[0]), float(fr[1])], float(fr[2]), list(ret)))
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


@torch.no_grad()
def learned_slot0(ret0):
    """slot APPRIS (food) à t0 depuis la rétine, repère ego (x_right, z_fwd) = même convention que food_rel0."""
    pos = slot_head.positions(torch.tensor(ret0, dtype=torch.float32))   # [n_res,2]
    return [float(pos[0, 0]), float(pos[0, 1])]


# segments de virage, objet visible à t0 ; cache la displacement rêvée du WM + la rétine t0
segs = []
for ei, seq in enumerate(eps):
    is_tr = ei < ntr
    for s0 in range(0, len(seq) - L, L):
        win = seq[s0:s0 + L]
        if win[0][3] < 0.5 or st.mean(abs(w[1][1]) for w in win) <= OMG:
            continue
        disp = wm_disp(win)
        P = torch.tensor([w[2] for w in win]); VIS = torch.tensor([w[3] for w in win])
        ret0 = win[0][4]
        segs.append((P, VIS, disp, is_tr, ret0))
print(f"segments de virage = {len(segs)} (train={sum(s[3] for s in segs)})")


def transport(p, dyaw, dfwd, dlat):
    px, pz = p[0] - dlat, p[1] - dfwd
    ca, sa = math.cos(-dyaw), math.sin(-dyaw)
    return [ca * px - sa * pz, sa * px + ca * pz]


def rollout_from(s0, disp, kyaw, kfwd, klat, n):
    """transport un slot initial s0 le long de n pas par la displacement calibrée."""
    s = [float(s0[0]), float(s0[1])]; out = [s]
    for t in range(n - 1):
        dfwd, dlat, dyaw = float(disp[t][0]), float(disp[t][1]), float(disp[t][2])
        s = transport(s, kyaw * dyaw, kfwd * dfwd, klat * dlat); out.append(s)
    return out


def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); d = a.norm() * b.norm()
    return (a @ b).item() / d.item() if d > 1e-6 else float("nan")


# Calibration (kyaw,kfwd,klat) sur train (1-pas, visibles) — readout, PAS un entraînement du WM.
Pt, Pn, Df, Dl, Dy = [], [], [], [], []
for P, VIS, disp, is_tr, ret0 in segs:
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

# qualité du slot appris à t0 (vs food_rel0) — contexte
e0 = [math.hypot(learned_slot0(r)[0] - float(P[0][0]), learned_slot0(r)[1] - float(P[0][1]))
      for P, VIS, disp, is_tr, r in segs if not is_tr]
print(f"erreur slot APPRIS à t0 (m) : médiane={st.median(e0):.2f}  (test={len(e0)} segments)")


def bearing_corr(getslot, H):
    pp, tt, ppr, ttr = [], [], [], []
    for P, VIS, disp, is_tr, ret0 in segs:
        if is_tr:
            continue
        slot = getslot(P, disp, ret0)
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


def learned_fn(P, disp, ret0):                       # SLOT APPRIS (le gate)
    return rollout_from(learned_slot0(ret0), disp, KY, KF, KL, len(P))


def oracle_fn(P, disp, ret0):                        # SLOT ORACLE (réf haute ≈ +0.65, = test6)
    return rollout_from([float(P[0][0]), float(P[0][1])], disp, KY, KF, KL, len(P))


def static_fn(P, disp, ret0):                        # contrôle : slot appris GELÉ à t0 (persistance nulle)
    return [learned_slot0(ret0)] * len(P)


print(f"\ntransport du bearing (corr poolée cos(brg), test) — l'ARRIÈRE est le discriminant (cible derrière) :")
print(f"{'H':>4} | {'APPRIS':>7} | {'ORACLE':>7} | {'STATIC':>7} || {'APPRIS-arr':>11} | {'ORACLE-arr':>11} | {'STATIC-arr':>11}")
for H in (5, 10, 20, 40):
    gl, rl = bearing_corr(learned_fn, H); go, ro = bearing_corr(oracle_fn, H); g0, r0 = bearing_corr(static_fn, H)
    print(f"{H:>4} | {gl:>+7.2f} | {go:>+7.2f} | {g0:>+7.2f} || {rl:>+11.2f} | {ro:>+11.2f} | {r0:>+11.2f}")

gL, rL = bearing_corr(learned_fn, L); gO, rO = bearing_corr(oracle_fn, L)
print(f"\nVERDICT (H={L}) : SLOT-APPRIS global={gL:+.2f} arrière={rL:+.2f}  "
      f"(oracle {gO:+.2f}/{rO:+.2f} ; réf latent ≈ +0.30 ; GATE +0.55 / +0.45 arr)")
if gL >= 0.55 and (math.isnan(rL) or rL >= 0.45):
    print(">>> SUCCÈS : le slot APPRIS survit au transport-WM dans le rêve → on gate le retrain Step 2 (internaliser le canal-slot).")
elif gL >= 0.40:
    print(">>> PARTIEL : bat le latent mais sous le gate → retrain léger (superviser le slot le long du rollout) à considérer.")
else:
    print(">>> KILL : ≈ latent → la displacement prédite dégrade le slot appris dans le rêve → retrain displacement-aware d'abord.")
