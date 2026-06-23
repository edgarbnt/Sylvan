"""TEST 3 (clé de voûte 3c) — ÉQUIVARIANCE : un opérateur STRUCTURÉ (générateur de rotation) transporte-t-il le
bearing à travers le rollout open-loop mieux que le prédicteur MLP du WM ? (offline, 0 retrain du WM).

Hypothèse 3c : le rollout déterministe MLP SMEAR la sous-composante fine du bearing (3a′ : rêve plateau +0.15).
Si ω agit comme une ROTATION CONNUE sur le latent, alors un opérateur structuré z_{t+1} ≈ z_t + ω·G·z_t + vx·D·z_t + b
(G = générateur de rotation appris une fois, linéaire) devrait, déroulé OPEN-LOOP, transporter le bearing.

  - structuré ≥ +0.35  → l'équivariance EST le bon biais → baker un rollout structuré dans le WM (gaté). FEU VERT.
  - structuré ≈ +0.15  → même un opérateur géométrique idéal ne récupère pas le bearing en open-loop → l'info n'est
                         pas dans un sous-espace qui tourne linéairement → autre levier (encodeur object-centric…). ROUGE.
  - ablation sans G (vx·D seul) : isole la part portée par le terme de ROTATION.

Usage: WM_CKPT=... BUF=retina_eat_a PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_test3_equivariance.py
"""
import json, glob, math, os, statistics as st
import torch
from torch import nn
from sylvan.models.command_wm import CommandWorldModel

torch.manual_seed(0); torch.set_num_threads(4)
WM = os.environ.get("WM_CKPT", "data/checkpoints/wm_rich_fidele_sym/wm_best.pt")
BUF = os.environ.get("BUF", "retina_eat_a")
L = 40; OMG = 0.30
print(f"WM = {WM} | BUF = {BUF}")
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
                        math.atan2(fr[0], fr[1]), float(fr[2])))
        if len(seq) > L + 2:
            eps.append(seq)
    return eps


eps = load_eps()
ntr = max(1, int(0.8 * len(eps)))
print(f"épisodes={len(eps)} (train={ntr}, test={len(eps)-ntr})")


@torch.no_grad()
def latents_and_cmds(win):
    obs = torch.tensor([w[0] for w in win], dtype=torch.float32).unsqueeze(0)
    cmd = torch.tensor([w[1] for w in win], dtype=torch.float32).unsqueeze(0)
    tf = wm.forward(obs, cmd)["latents"][0]                                  # (L, d) teacher-forced
    ol = wm.rollout_open_loop(obs[:, 0, :], cmd)["predicted_latents"][0]     # (L, d) WM MLP open-loop
    return tf, ol, cmd[0]                                                    # cmd[0] = (L,2) [vx, omega]


# Build segments (teacher-forced latents = the "truth" we fit the structured operator to transport).
segs = []  # (tf, ol_wm, cmd, ahead_true, vis, is_rot, is_train, crosses)
for ei, seq in enumerate(eps):
    is_train = ei < ntr
    for s0 in range(0, len(seq) - L, L):
        win = seq[s0:s0 + L]
        if win[0][3] < 0.5:
            continue
        rot = st.mean(abs(w[1][1]) for w in win) > OMG
        tf, ol, cmd = latents_and_cmds(win)
        aT = torch.tensor([math.cos(w[2]) for w in win])
        vis = torch.tensor([w[3] for w in win])
        crosses = any((abs(win[i][2]) > math.pi / 2) != (abs(win[0][2]) > math.pi / 2) for i in range(len(win)))
        segs.append((tf, ol, cmd, aT, vis, rot, is_train, crosses))
d = segs[0][0].shape[1]
print(f"segments={len(segs)} (virage={sum(s[5] for s in segs)}) | latent dim d={d}")


# --- Fit structured operator on TRAIN transitions: Δz = omega·(G z) + vx·(D z) + b  (ridge least squares) ---
def fit_operator(use_G=True):
    Phi, Y = [], []
    for s in segs:
        if not s[6]:
            continue
        tf, cmd = s[0], s[2]
        for t in range(L - 1):
            z = tf[t]; vx, om = float(cmd[t][0]), float(cmd[t][1])
            feat = torch.cat([(om * z) if use_G else torch.zeros_like(z), vx * z, torch.ones(1)])
            Phi.append(feat); Y.append(tf[t + 1] - tf[t])
    Phi = torch.stack(Phi); Y = torch.stack(Y)
    lam = 1e-2 * Phi.shape[0]
    A = Phi.T @ Phi + lam * torch.eye(Phi.shape[1])
    W = torch.linalg.solve(A, Phi.T @ Y)        # (2d+1, d)
    return W, use_G


def rollout_structured(W, use_G, z0, cmd):
    zs = [z0]; z = z0
    for t in range(L - 1):
        vx, om = float(cmd[t][0]), float(cmd[t][1])
        feat = torch.cat([(om * z) if use_G else torch.zeros_like(z), vx * z, torch.ones(1)])
        z = z + feat @ W
        zs.append(z)
    return torch.stack(zs)


# --- Bearing probe trained on a given latent source (TRAIN segs), evaluated on TEST rotation segs ---
def train_probe(getlat):
    X, Y = [], []
    for s in segs:
        if not s[6]:
            continue
        lat = getlat(s)
        for t in range(L):
            if s[4][t] > 0.5:
                X.append(lat[t]); Y.append([float(s[3][t]), float((1 - s[3][t] ** 2).clamp(min=0) ** 0.5)])
    X = torch.stack(X).detach(); Y = torch.tensor(Y)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    p = nn.Sequential(nn.Linear(d, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))
    opt = torch.optim.Adam(p.parameters(), 1e-3); Xn = (X - mu) / sd
    for _ in range(1500):
        bi = torch.randint(0, len(X), (256,))
        q = p(Xn[bi]); q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        loss = ((q - Y[bi]) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return p, mu, sd


def decode_cos(probe, mu, sd, lat):
    q = probe((lat - mu) / sd); return q[:, 0] / (q.norm(dim=-1) + 1e-6)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); dn = a.norm() * b.norm()
    return (a @ b).item() / dn.item() if dn > 1e-6 else float("nan")


def m(xs):
    xs = [x for x in xs if not math.isnan(x)]; return st.mean(xs) if xs else float("nan")


# Fit operators + attach structured open-loop rollouts to each segment (cache on the tuple via dict).
W_full, _ = fit_operator(use_G=True)
W_noG, _ = fit_operator(use_G=False)
struct_full = {}; struct_noG = {}
for i, s in enumerate(segs):
    struct_full[i] = rollout_structured(W_full, True, s[0][0], s[2]).detach()
    struct_noG[i] = rollout_structured(W_noG, False, s[0][0], s[2]).detach()

# --- WHITENED variant: removes the variance confound (bearing is a LOW-variance subcomponent that a full-latent
# least-squares ignores). PCA-whiten on TRAIN TF latents (top-K comps), fit + roll the structured operator in
# whitened space, un-whiten before probing. Fair test of "is bearing in a subspace that rotates linearly with ω?".
Ztr = torch.cat([s[0] for s in segs if s[6]], 0)
mu_w = Ztr.mean(0)
U, S, _ = torch.linalg.svd(Ztr - mu_w, full_matrices=False)
K = int((S > S.max() * 1e-3).sum().item()); K = min(K, 40)
comps = (_ if False else torch.linalg.svd(Ztr - mu_w, full_matrices=False)[2])[:K]   # (K, d) principal axes
evals = (S[:K] ** 2 / (len(Ztr) - 1)).clamp(min=1e-8)
Wproj = comps / evals.sqrt().unsqueeze(1)     # whiten:  w = (z-mu) @ Wproj.T
Winv = comps * evals.sqrt().unsqueeze(1)       # un-whiten: z = mu + w @ Winv
print(f"whitening: K={K} composantes (eff)")


def to_w(z):
    return (z - mu_w) @ Wproj.T


def from_w(w):
    return mu_w + w @ Winv


def fit_operator_w():
    Phi, Y = [], []
    for s in segs:
        if not s[6]:
            continue
        wlat = to_w(s[0]); cmd = s[2]
        for t in range(L - 1):
            w = wlat[t]; vx, om = float(cmd[t][0]), float(cmd[t][1])
            Phi.append(torch.cat([om * w, vx * w, torch.ones(1)])); Y.append(wlat[t + 1] - wlat[t])
    Phi = torch.stack(Phi); Y = torch.stack(Y)
    lam = 1e-2 * Phi.shape[0]
    return torch.linalg.solve(Phi.T @ Phi + lam * torch.eye(Phi.shape[1]), Phi.T @ Y)


def rollout_w(Ww, z0, cmd):
    w = to_w(z0); ws = [w]
    for t in range(L - 1):
        vx, om = float(cmd[t][0]), float(cmd[t][1])
        w = w + torch.cat([om * w, vx * w, torch.ones(1)]) @ Ww
        ws.append(w)
    return from_w(torch.stack(ws))


Ww = fit_operator_w()
struct_wh = {}
for i, s in enumerate(segs):
    struct_wh[i] = rollout_w(Ww, s[0][0], s[2]).detach()

# Probes: WM-OL (TEST1 ref) and STRUCTURED-OL trained each on their own latent source.
probe_wm, mwm, swm = train_probe(lambda s: s[1])
idx = {id(s): i for i, s in enumerate(segs)}
probe_st, mst, sst = train_probe(lambda s: struct_full[idx[id(s)]])
probe_ng, mng, sng = train_probe(lambda s: struct_noG[idx[id(s)]])
probe_wh, mwh, swh = train_probe(lambda s: struct_wh[idx[id(s)]])
print("sondes entraînées (WM-OL, STRUCT-OL, STRUCT-noG, STRUCT-whitened)")

R = {"WM_OL": [], "STRUCT": [], "STRUCT_noG": [], "STRUCT_WH": []}
for i, s in enumerate(segs):
    if s[6] or not s[5]:    # test only, rotation only
        continue
    aT = s[3]
    R["WM_OL"].append(corr(decode_cos(probe_wm, mwm, swm, s[1]), aT))
    R["STRUCT"].append(corr(decode_cos(probe_st, mst, sst, struct_full[i]), aT))
    R["STRUCT_noG"].append(corr(decode_cos(probe_ng, mng, sng, struct_noG[i]), aT))
    R["STRUCT_WH"].append(corr(decode_cos(probe_wh, mwh, swh, struct_wh[i]), aT))

# Per-horizon pooled corr (distinguishes wrong-function-class from compounding-drift).
print("\ntransport du bearing par HORIZON (corr poolée, frames t<H, segments virage test) :")
print(f"{'H':>4} | {'WM-OL':>7} | {'STRUCT-WH':>9}")
for H in (5, 10, 20, 40):
    pw_p, pw_t, ps_p, ps_t = [], [], [], []
    for i, s in enumerate(segs):
        if s[6] or not s[5]:
            continue
        cw = decode_cos(probe_wm, mwm, swm, s[1]); cs = decode_cos(probe_wh, mwh, swh, struct_wh[i])
        for t in range(min(H, L)):
            if s[4][t] > 0.5:
                pw_p.append(cw[t]); pw_t.append(s[3][t]); ps_p.append(cs[t]); ps_t.append(s[3][t])
    cwm = corr(torch.stack(pw_p), torch.stack(pw_t)); cst = corr(torch.stack(ps_p), torch.stack(ps_t))
    print(f"{H:>4} | {cwm:>+7.2f} | {cst:>+9.2f}")

n = len([x for x in R["WM_OL"] if not math.isnan(x)])
print(f"\nsegments de virage (test) = {n}")
print(f"  WM-MLP open-loop (réf TEST1)      = {m(R['WM_OL']):+.2f}")
print(f"  STRUCTURÉ ω·G + vx·D (équivariant) = {m(R['STRUCT']):+.2f}")
print(f"  ablation sans G (vx·D seul)        = {m(R['STRUCT_noG']):+.2f}")
print(f"  STRUCTURÉ BLANCHI (variance égale) = {m(R['STRUCT_WH']):+.2f}  <- enlève le confound basse-variance")
g = max(m(R["STRUCT"]), m(R["STRUCT_WH"]))
print(f"\nVERDICT (seuil +0.35) :")
if g >= 0.35:
    print(">>> FEU VERT 3c : un opérateur structuré transporte le bearing → baker l'équivariance (rollout structuré).")
elif g >= m(R["WM_OL"]) + 0.10:
    print(">>> PARTIEL : la structure aide nettement mais < +0.35 → équivariance prometteuse, à muscler (subspace dédié).")
else:
    print(">>> ROUGE : même un opérateur géométrique idéal ne transporte pas → l'info n'est pas dans un sous-espace"
          " qui tourne linéairement → levier ailleurs (encodeur object-centric / latent stochastique).")
