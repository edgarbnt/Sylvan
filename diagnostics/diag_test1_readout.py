"""TEST 1 (carte clé de voûte) — readout vs retrain. Le bearing est-il DANS le latent rêvé (mal lu) ou JETÉ ?
(offline, 0 entraînement du WM). wm_rich_fidele_sym.

On compare, sur les MÊMES segments de virage held-out (split par ÉPISODE, sans fuite) :
  REPR  = sonde-TF sur latents TEACHER-FORCED        (représentation, attendu ~+0.5)
  RÊVE  = sonde-TF sur latents OPEN-LOOP             (ce qu'on mesurait, ~+0.15)
  TEST1 = sonde-OL (entraînée SUR latents open-loop) sur latents open-loop
Verdict : TEST1 ≈ +0.5 → l'info EST dans le rêve, mal exposée → FIX CHEAP (readout, pas de retrain).
          TEST1 ≈ +0.15 → l'info est JETÉE par la dynamique → RETRAIN (perte aux-rollout). Seuil +0.35.

Usage: PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_test1_readout.py
"""
import json, glob, math, os, statistics as st
import torch
from torch import nn
from sylvan.models.command_wm import CommandWorldModel

torch.manual_seed(0); torch.set_num_threads(4)
WM = os.environ.get("WM_CKPT", "data/checkpoints/wm_rich_fidele_sym/wm_best.pt")
print(f"WM = {WM}")
L = 40; OMG = 0.30
pl = torch.load(WM, map_location="cpu", weights_only=False); meta = pl["meta"]
wm = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
                       predictor_arch=meta.get("predictor_arch", "shallow"))
wm.load_state_dict(pl["model"]); wm.eval()
BUF = os.environ.get("BUF", "retina_forage")
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
def lats(win):
    obs = torch.tensor([w[0] for w in win], dtype=torch.float32).unsqueeze(0)
    cmd = torch.tensor([w[1] for w in win], dtype=torch.float32).unsqueeze(0)
    tf = wm.forward(obs, cmd)["latents"][0]
    ol = wm.rollout_open_loop(obs[:, 0, :], cmd)["predicted_latents"][0]
    return tf, ol


# build segments with cached latents + flags
segs = []  # (tf, ol, ahead_true[T], vis[T], is_rot, is_train, crosses)
for ei, seq in enumerate(eps):
    is_train = ei < ntr
    for s0 in range(0, len(seq) - L, L):
        win = seq[s0:s0 + L]
        rot = st.mean(abs(w[1][1]) for w in win) > OMG
        if win[0][3] < 0.5:
            continue
        tf, ol = lats(win)
        aT = torch.tensor([math.cos(w[2]) for w in win])
        vis = torch.tensor([w[3] for w in win])
        crosses = any((abs(win[i][2]) > math.pi / 2) != (abs(win[0][2]) > math.pi / 2) for i in range(len(win)))
        segs.append((tf, ol, aT, vis, rot, is_train, crosses))
print(f"segments={len(segs)} (virage={sum(s[4] for s in segs)})")


def train_probe(getlat):
    X, Y = [], []
    for s in segs:
        if not s[5]:
            continue
        lat = getlat(s)
        for t in range(L):
            if s[3][t] > 0.5:
                X.append(lat[t]); Y.append([float(s[2][t]), float((1 - s[2][t] ** 2).clamp(min=0) ** 0.5)])
    X = torch.stack(X); Y = torch.tensor(Y)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    p = nn.Sequential(nn.Linear(X.shape[1], 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))
    opt = torch.optim.Adam(p.parameters(), 1e-3); Xn = (X - mu) / sd
    for _ in range(1500):
        bi = torch.randint(0, len(X), (256,))
        q = p(Xn[bi]); q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        loss = ((q - Y[bi]) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return p, mu, sd


# Y = (cos, |sin|) — on ne suit que "ahead" (cos) ; |sin| juste pour normaliser la sortie. Corr mesurée sur cos.
def decode_cos(probe, mu, sd, lat):
    q = probe((lat - mu) / sd); return q[:, 0] / (q.norm(dim=-1) + 1e-6)


probe_tf, mtf, stf = train_probe(lambda s: s[0])
probe_ol, mol, sol = train_probe(lambda s: s[1])
print("sondes entraînées (TF sur latents TF, OL sur latents OL)")


def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); d = a.norm() * b.norm()
    return (a @ b).item() / d.item() if d > 1e-6 else float("nan")


R = {"front": {"REPR": [], "REVE": [], "TEST1": []}, "cross": {"REPR": [], "REVE": [], "TEST1": []}}
for s in segs:
    if s[5] or not s[4]:   # test only, rotation only
        continue
    aT = s[2]
    k = "cross" if s[6] else "front"
    R[k]["REPR"].append(corr(decode_cos(probe_tf, mtf, stf, s[0]), aT))
    R[k]["REVE"].append(corr(decode_cos(probe_tf, mtf, stf, s[1]), aT))
    R[k]["TEST1"].append(corr(decode_cos(probe_ol, mol, sol, s[1]), aT))


def m(xs):
    xs = [x for x in xs if not math.isnan(x)]; return st.mean(xs) if xs else float("nan")


print(f"\n{'cas':>6} | {'n':>3} | REPR (TF/TF) | RÊVE (TF/OL) | TEST1 (OL/OL)")
allt1 = []
for k in ("front", "cross"):
    n = len(R[k]['REPR'])
    if n:
        allt1 += [x for x in R[k]['TEST1'] if not math.isnan(x)]
        print(f"{k:>6} | {n:>3} |    {m(R[k]['REPR']):+.2f}     |    {m(R[k]['REVE']):+.2f}     |    {m(R[k]['TEST1']):+.2f}")
t1 = st.mean(allt1) if allt1 else float("nan")
print(f"\nTEST1 global = {t1:+.2f}  (seuil +0.35)")
print(">>> " + ("READOUT : l'info EST dans le rêve, mal exposée → fix CHEAP (pas de retrain)."
               if t1 >= 0.35 else
               "RETRAIN : l'info est JETÉE par la dynamique → perte aux-rollout (GATE 3)."))
