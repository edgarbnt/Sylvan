"""CARTE FALSIFIABLE — qu'est-ce qui manque au WM pour IMAGINER la rotation / la perception sous auto-mouvement ?
(offline, 0 entraînement du WM). wm_rich_fidele_sym.

Sur de vrais segments de VIRAGE du replay (|ω| élevé), on suit le bearing de la bouffe (cos = "ahead") :
  TRUTH   = bearing réel (food_rel0 par frame).
  REPR    = bearing décodé des latents TEACHER-FORCED (vraie obs chaque pas) → l'ENCODEUR/représentation suit-il ?
  RÊVE    = bearing décodé des latents OPEN-LOOP (rêve depuis la frame 0, vraies commandes) → la DYNAMIQUE suit-elle ?
Décodeur de bearing = sonde fraîche entraînée sur latents TF (même espace pour les deux).
Split : segments où la bouffe RESTE devant vs CROISE derrière (le cas dur de l'acquisition-par-virage).
+ Sonde UNCERTAINTY : aux "reveals" (bouffe 0→1 visible), le WM déterministe peut-il l'anticiper (value 1 pas avant) ?

Lecture : REPR mauvais sous rotation → manque ENCODEUR (représentation rotation-aware). REPR bon mais RÊVE mauvais
→ manque DYNAMIQUE (transport/fidélité du rollout en rotation). Reveal raté → manque UNCERTAINTY (latent stochastique).

Usage: PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_wm_rotation.py
"""
import json, glob, math
import torch
from torch import nn
from sylvan.models.command_wm import CommandWorldModel
from sylvan.models.value_head import load_value_head

torch.manual_seed(0); torch.set_num_threads(2)
WM = "data/checkpoints/wm_rich_fidele_sym/wm_best.pt"
VH = "data/checkpoints/value_head_food_dream/value_best.pt"
L = 40            # longueur de segment
OMG = 0.30        # |ω| moyen min pour "virage"
pl = torch.load(WM, map_location="cpu", weights_only=False); meta = pl["meta"]
wm = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
                       predictor_arch=meta.get("predictor_arch", "shallow"))
wm.load_state_dict(pl["model"]); wm.eval()
val = load_value_head(VH)
files = sorted(glob.glob("godot/data/replay_buffer/retina_forage/episode_*.jsonl") or
               glob.glob("data/replay_buffer/retina_forage/episode_*.jsonl"))[:25]


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
                        math.atan2(fr[0], fr[1]), float(fr[2]), math.hypot(fr[0], fr[1])))
        if len(seq) > L + 2:
            eps.append(seq)
    return eps


eps = load_eps()
print(f"épisodes={len(eps)} (segments L={L}, virage |ω|>{OMG})")


@torch.no_grad()
def tf_latents(seq):
    obs = torch.tensor([s[0] for s in seq], dtype=torch.float32).unsqueeze(0)
    cmd = torch.tensor([s[1] for s in seq], dtype=torch.float32).unsqueeze(0)
    return wm.forward(obs, cmd)["latents"][0]            # [T,L] teacher-forced


@torch.no_grad()
def ol_latents(seq):
    obs0 = torch.tensor(seq[0][0], dtype=torch.float32).reshape(1, -1)
    cmd = torch.tensor([s[1] for s in seq], dtype=torch.float32).unsqueeze(0)
    return wm.rollout_open_loop(obs0, cmd)["predicted_latents"][0]   # [T,L] rêve


# --- sonde de bearing entraînée sur latents TF (frames visibles) ---
LAT, TGT = [], []
for seq in eps[:20]:
    lat = tf_latents(seq)
    for t, s in enumerate(seq):
        if s[3] > 0.5:
            LAT.append(lat[t]); TGT.append([math.cos(s[2]), math.sin(s[2])])
LAT = torch.stack(LAT); TGT = torch.tensor(TGT)
mu, sd = LAT.mean(0), LAT.std(0) + 1e-6
probe = nn.Sequential(nn.Linear(LAT.shape[1], 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))
opt = torch.optim.Adam(probe.parameters(), 1e-3)
LATn = (LAT - mu) / sd
for _ in range(1500):
    bi = torch.randint(0, len(LAT), (256,))
    p = probe(LATn[bi]); p = p / (p.norm(dim=-1, keepdim=True) + 1e-6)
    loss = ((p - TGT[bi]) ** 2).sum(1).mean()
    opt.zero_grad(); loss.backward(); opt.step()
print(f"sonde bearing entraînée (loss={loss.item():.3f}, n={len(LAT)})")


@torch.no_grad()
def decode_ahead(lat):  # lat [T,L] -> cos(bearing) [T]
    p = probe((lat - mu) / sd); return (p[:, 0] / (p.norm(dim=-1) + 1e-6))


def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); d = a.norm() * b.norm()
    return (a @ b).item() / d.item() if d > 1e-6 else float("nan")


import statistics as st
res = {"front": {"repr": [], "reve": []}, "cross": {"repr": [], "reve": []}}
nseg = 0
for seq in eps:
    for s0 in range(0, len(seq) - L, L):
        win = seq[s0:s0 + L]
        if st.mean(abs(w[1][1]) for w in win) < OMG:  # pas un virage
            continue
        if win[0][3] < 0.5:                            # bouffe visible au départ
            continue
        aT = torch.tensor([math.cos(w[2]) for w in win])     # ahead vrai
        aR = decode_ahead(tf_latents(win))                   # ahead représentation (TF)
        aD = decode_ahead(ol_latents(win))                   # ahead rêve (open-loop)
        crosses = any((win[i][2] > math.pi / 2) != (win[0][2] > math.pi / 2) or
                      (abs(win[i][2]) > math.pi / 2) != (abs(win[0][2]) > math.pi / 2) for i in range(len(win)))
        key = "cross" if crosses else "front"
        res[key]["repr"].append(corr(aR, aT)); res[key]["reve"].append(corr(aD, aT))
        nseg += 1
print(f"segments de virage analysés = {nseg}\n")
print(f"{'cas':>8} | {'n':>3} | corr(REPR/TF , vrai) | corr(RÊVE/OL , vrai)")
for k in ("front", "cross"):
    rr = [x for x in res[k]["repr"] if not math.isnan(x)]
    dd = [x for x in res[k]["reve"] if not math.isnan(x)]
    if rr:
        print(f"{k:>8} | {len(rr):>3} |        {st.mean(rr):+.2f}         |        {st.mean(dd):+.2f}")
print("  front = bouffe reste devant ; cross = la bouffe CROISE devant↔derrière pendant le virage (cas dur)")

# --- Sonde UNCERTAINTY : reveal (bouffe 0->1 visible). Le WM (déterministe) l'anticipe-t-il 1 pas avant ? ---
@torch.no_grad()
def value_next(seq, t):  # value prédite du latent open-loop au pas t+1 depuis obs t (1 pas de rêve)
    obs0 = torch.tensor(seq[t][0], dtype=torch.float32).reshape(1, -1)
    cmd = torch.tensor([seq[t][1]], dtype=torch.float32).unsqueeze(0)
    lat = wm.rollout_open_loop(obs0, cmd)["predicted_latents"][0, 0]
    return float(torch.sigmoid(val.logit(lat)))

reveal_v, normal_v, nrev = [], [], 0
for seq in eps:
    for t in range(len(seq) - 1):
        appears = seq[t][3] < 0.5 and seq[t + 1][3] > 0.5 and seq[t + 1][4] < 2.5  # bouffe proche apparaît
        if appears:
            reveal_v.append(value_next(seq, t)); nrev += 1
        elif seq[t][3] > 0.5 and seq[t + 1][4] < 2.5:
            normal_v.append(value_next(seq, t))
        if nrev >= 60:
            break
    if nrev >= 60:
        break
if reveal_v:
    print(f"\nUNCERTAINTY/reveal (n={len(reveal_v)}): value prédite 1 pas AVANT que la bouffe (proche) apparaisse "
          f"= {st.mean(reveal_v):.3f}  vs déjà-visible-proche = {st.mean(normal_v) if normal_v else float('nan'):.3f}")
    print("  (value reveal ≈ 0 et << déjà-visible ⇒ le WM déterministe N'ANTICIPE PAS le reveal → manque uncertainty)")
