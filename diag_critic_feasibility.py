"""GATE GRATUIT — un CRITIQUE/valeur appris sur le latent est-il FAISABLE ? (avant de l'entraîner)

🅑 par coût-énergie est mort. Question décisive : l'échec vient-il du READOUT (énergie-MSE molle) ou de la
REPRÉSENTATION ? On teste si le latent encode la POSITION de la bouffe (régression linéaire latent→(dx,dz)),
en distinguant DEUX régimes :
  (1) TEACHER-FORCED (latent au pas 0, depuis l'obs réelle incl. rétine) — le latent encode-t-il la bouffe ?
      C'est la condition NÉCESSAIRE pour tout critique. (rétine décodable à 0.08 m → attendu OUI.)
  (2) RÊVÉ à t croissant (open-loop) — le dream PRÉSERVE-t-il cette position quand l'agent imagine bouger ?
      Sonde entraînée sur (1), appliquée aux latents rêvés, erreur vs t = COURBE DE DÉGRADATION du rêve.
Cela localise le bottleneck SANS l'artefact "0% atteint" : (1) bas → représentation à revoir ; (1) haut mais
(2) s'effondre vite → le rêve perd la bouffe → un critique latent ne marche qu'à HORIZON COURT (+ replan) ou
il faut re-feeder la rétine ; (1) ET (2) tiennent → critique pleinement faisable.
Usage: SYLVAN_WM_USE_RETINA=1 PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_critic_feasibility.py [wm_ckpt]
"""
import sys, json, glob, math
import torch
from sylvan.models.command_wm import CommandWorldModel, DISPLACEMENT_SCALE
from sylvan.control.planning.command_planner import CommandPlanner, CommandPlanConfig

WM = sys.argv[1] if len(sys.argv) > 1 else "data/checkpoints/wm_command_hex_retina_eat_v1/wm_best.pt"
H = 120
TPROBE = [0, 10, 30, 60, 100]   # pas où on évalue le dream-decay (0 = teacher-forced)
N_TF = 4000        # frames pour la sonde teacher-forced (>> 128 dims, sinon surajustement → R² faux)
N_DREAM = 300      # frames roulés pour la courbe de dégradation du rêve
POS_SCALE = 10.0

pl = torch.load(WM, map_location="cpu", weights_only=False); meta = pl["meta"]
wm = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
                       predictor_arch=meta.get("predictor_arch", "shallow"))
wm.load_state_dict(pl["model"]); wm.eval()
print(f"WM={WM} obs_dim={meta['obs_dim']} | teacher-forced ({N_TF}) + dream-decay ({N_DREAM})")

files = sorted(glob.glob("data/replay_buffer/retina_forage/episode_*.jsonl") or
               glob.glob("godot/data/replay_buffer/retina_forage/episode_*.jsonl"))


def gen_frames():
    for f in files:
        for line in open(f):
            r = json.loads(line); w = r.get("wm", {})
            ret = w.get("retina0"); fr = w.get("food_rel0"); cmd = w.get("cmd")
            if not ret or not fr or not cmd or fr[2] < 0.5: continue
            d = math.hypot(fr[0], fr[1])
            if d < 1.0 or d > 4.0: continue
            yield r["obs"]["proprio"], ret, r["obs"]["energy"] / 100.0, fr[0], fr[1], cmd[:2]


# ── (1) TEACHER-FORCED : latent au pas 0 sur BEAUCOUP de frames (batché, H=1) → sonde bien posée ──
obs_tf, tgt_tf = [], []
for proprio, ret, e0, fx, fz, cmd in gen_frames():
    obs_tf.append(proprio + ret + [e0]); tgt_tf.append([fx / POS_SCALE, fz / POS_SCALE])
    if len(obs_tf) >= N_TF: break
obs_tf = torch.tensor(obs_tf, dtype=torch.float32); tgt_tf = torch.tensor(tgt_tf)
# une commande nulle, H=1 : latent[:,0] = latent teacher-forced (1 pas depuis l'obs réelle)
with torch.no_grad():
    lat0 = wm.rollout_open_loop(obs_tf, torch.zeros(obs_tf.shape[0], 1, 2))["predicted_latents"][:, 0, :]
n = lat0.shape[0]; cut = int(n * 0.7)
LAT = {0: lat0}; TGT = {0: tgt_tf}
print(f"teacher-forced frames={n} (train={cut}, test={n-cut})")

# ── (2) DREAM-DECAY : rouler la commande réelle, sonde t0 appliquée aux latents rêvés à t ──
for t in TPROBE[1:]:
    LAT[t] = []; TGT[t] = []
nd = 0; skip = 0
for proprio, ret, e0, fx, fz, cmd in gen_frames():
    if skip < N_TF:   # disjoint des frames teacher-forced (pas de fuite)
        skip += 1; continue
    seq = torch.tensor([cmd] * H, dtype=torch.float32).reshape(1, H, 2)
    with torch.no_grad():
        out = wm.rollout_open_loop(torch.tensor(proprio + ret + [e0], dtype=torch.float32).reshape(1, -1), seq)
    disp = (out["predicted_displacement"] / DISPLACEMENT_SCALE)[0]; lat = out["predicted_latents"][0]
    x = z = yaw = 0.0; traj = []
    for t in range(H):
        d_fwd, d_lat, d_yaw = disp[t].tolist(); s, c = math.sin(yaw), math.cos(yaw)
        x += d_fwd * s + d_lat * c; z += d_fwd * c - d_lat * s; yaw += d_yaw
        traj.append((x, z, yaw))
    for t in TPROBE[1:]:
        xt, zt, yt = traj[t]; s, c = math.sin(yt), math.cos(yt)
        dx = (fx - xt) * c - (fz - zt) * s; dz = (fx - xt) * s + (fz - zt) * c
        LAT[t].append(lat[t]); TGT[t].append([dx / POS_SCALE, dz / POS_SCALE])
    nd += 1
    if nd >= N_DREAM: break
for t in TPROBE[1:]:
    LAT[t] = torch.stack(LAT[t]); TGT[t] = torch.tensor(TGT[t])
print(f"dream-decay frames={nd}")

tr0 = torch.arange(n) < cut


# Probe = petit MLP (NON-linéaire) : la position bouffe est une fonction non-linéaire du latent (cf
# retina_head = attention/soft-argmax). Un probe linéaire sous-estimerait la décodabilité.
class MLP(torch.nn.Module):
    def __init__(self, d, out=2):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.SiLU(),
                                        torch.nn.Linear(256, 256), torch.nn.SiLU(), torch.nn.Linear(256, out))
    def forward(self, x): return self.net(x)


def train_probe(X, Y, epochs=800, lr=2e-3, wd=1e-4):
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-6
    net = MLP(X.shape[1], Y.shape[1]); opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    Xn = (X - mu) / sd
    for _ in range(epochs):
        opt.zero_grad(); loss = ((net(Xn) - Y) ** 2).mean(); loss.backward(); opt.step()
    net.eval()
    return net, mu, sd


train_probe2 = train_probe   # alias (sortie de dim quelconque)


def r2_and_err(probe, X, Y):
    net, mu, sd = probe
    with torch.no_grad():
        pred = net((X - mu) / sd)
    ss_res = ((Y - pred) ** 2).sum(0); ss_tot = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-9
    r2 = (1 - ss_res / ss_tot).mean().item()
    err_m = ((pred - Y) * POS_SCALE).pow(2).sum(1).sqrt().mean().item()   # erreur position (m)
    return r2, err_m


def r2_only(probe, X, Y):
    net, mu, sd = probe
    with torch.no_grad():
        pred = net((X - mu) / sd)
    ss_res = ((Y - pred) ** 2).sum(0); ss_tot = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-9
    return (1 - ss_res / ss_tot).mean().item()


# sonde entraînée sur le TEACHER-FORCED (t=0), train split
W = train_probe(LAT[0][tr0], TGT[0][tr0])
print(f"\n=== Sonde linéaire latent→(dx,dz) bouffe, entraînée TEACHER-FORCED (t=0) ===")
print(f"{'t (pas rêvés)':>14} | {'R² held-out':>11} | {'err pos (m)':>11}")
res = {}
for t in TPROBE:
    Xte, Yte = (LAT[0][~tr0], TGT[0][~tr0]) if t == 0 else (LAT[t], TGT[t])
    r2, err = r2_and_err(W, Xte, Yte)
    res[t] = (r2, err)
    tag = "  ← teacher-forced (condition nécessaire)" if t == 0 else ""
    print(f"{t:>14} | {r2:>11.3f} | {err:>11.3f}{tag}")

# ── CONTRÔLES (le probe & l'info marchent-ils ?) ──
print(f"\n=== CONTRÔLES ===")
# (a) latent → vitesse torse (proprio[1:4]) : le latent encode-t-il QUELQUE CHOSE de décodable ?
prop_tgt = obs_tf[:, 1:4]
pa = train_probe2(LAT[0][tr0], prop_tgt[tr0]); r2a = r2_only(pa, LAT[0][~tr0], prop_tgt[~tr0])
# (b) RÉTINE brute (144) → bouffe : l'info est-elle dans l'INPUT ? (réf retina_head ~0.08 m)
ret_in = obs_tf[:, 132:276]
pb = train_probe(ret_in[tr0], TGT[0][tr0]); r2b, errb = r2_and_err(pb, ret_in[~tr0], TGT[0][~tr0])
# (c) sortie ENCODEUR (pré-RSSM) → bouffe : l'encodeur garde-t-il la bouffe avant le RSSM ?
with torch.no_grad():
    enc = wm.encoder(obs_tf)
pc = train_probe(enc[tr0], TGT[0][tr0]); r2c, errc = r2_and_err(pc, enc[~tr0], TGT[0][~tr0])
print(f"  (a) latent → vitesse torse      : R²={r2a:.3f}   (contrôle pipeline : doit être bon)")
print(f"  (b) RÉTINE brute → bouffe        : R²={r2b:.3f} err={errb:.2f}m  (l'info EST dans l'input ?)")
print(f"  (c) sortie ENCODEUR → bouffe     : R²={r2c:.3f} err={errc:.2f}m  (avant le RSSM)")

r2_0 = res[0][0]; r2_60 = res[60][0]
print(f"\n=== VERDICT critique ===")
if r2_0 < 0.4:
    print(f"🔴 REPRÉSENTATION : même teacher-forced le latent n'encode PAS bien la bouffe (R²={r2_0:.2f}) →")
    print(f"   bottleneck = représentation/WM, un critique n'aide pas. Revoir l'intégration rétine→latent.")
elif r2_60 >= 0.4:
    print(f"🟢 CRITIQUE FAISABLE : latent encode la bouffe (t0 R²={r2_0:.2f}) ET le rêve la préserve jusqu'à t60")
    print(f"   (R²={r2_60:.2f}) → le souci de 🅑 était le READOUT (énergie-MSE). Construire le critique.")
else:
    print(f"🟡 REPRÉSENTATION OK MAIS RÊVE COURT : t0 R²={r2_0:.2f} bon, mais s'effondre (t60 R²={r2_60:.2f}) →")
    print(f"   le latent rêvé perd la bouffe en open-loop. Critique viable seulement à HORIZON COURT + replan,")
    print(f"   ou re-feeder la rétine dans le rollout. = piste claire, pas un mur.")
