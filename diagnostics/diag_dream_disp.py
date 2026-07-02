"""Mesure GRATUITE de la FIDÉLITÉ DYNAMIQUE OPEN-LOOP d'un WM (le cœur du mur 🅑 du 2026-06-19) :
combien le RÊVE (rollout open-loop, commande droite) transporte-t-il le corps, et le latent reste-t-il
cohérent ? + eff_rank du latent (le rêve doit locomoter SANS re-collapser la représentation).

Sert d'INSTRUMENT D'ABLATION mse-vs-cosine : le WM-rétine (cosine+VICReg) rêve ~0.19 m/120 pas (latent
saute cos→0.77 dès t=1) ; le live v2 (mse) rêve ~0.75 m (latent cohérent cos 0.94@119). Hypothèse testée :
le latent-loss MSE restaure le déplacement open-loop sans tuer eff_rank.

Usage: SYLVAN_WM_USE_RETINA=1 PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_dream_disp.py <wm_ckpt> [withretina=1]
"""
import sys, json, glob, math, torch
from sylvan.models.command_wm import CommandWorldModel, DISPLACEMENT_SCALE

CK = sys.argv[1]
WITHRET = (sys.argv[2] if len(sys.argv) > 2 else "1") == "1"
H = 120

pl = torch.load(CK, map_location="cpu", weights_only=False); m = pl["meta"]
wm = CommandWorldModel(obs_dim=m["obs_dim"], proprio_dim=m["proprio_dim"],
                       predictor_arch=m.get("predictor_arch", "shallow"))
wm.load_state_dict(pl["model"]); wm.eval()

files = (sorted(glob.glob("godot/data/replay_buffer/retina_forage/episode_*.jsonl")) if WITHRET
         else sorted(glob.glob("data/replay_buffer/wm_hex_v2_a/episode_*.jsonl")))


def build_obs(r, w):
    if WITHRET:
        return torch.tensor(r["obs"]["proprio"] + w["retina0"] + [r["obs"]["energy"] / 100.0])
    obs = torch.zeros(m["obs_dim"]); obs[:132] = torch.tensor(r["obs"]["proprio"])
    obs[132:132 + 12] = torch.tensor(w["radar0"]); obs[-1] = r["obs"]["energy"] / 100.0
    return obs


# ── 1. DÉPLACEMENT DU RÊVE (commande droite) + cohérence latente, moyenné sur N frames ──
obss = []
for f in files:
    for line in open(f):
        r = json.loads(line); w = r.get("wm", {})
        if WITHRET and not w.get("retina0"):
            continue
        obss.append(build_obs(r, w))
        if len(obss) >= 64:
            break
    if len(obss) >= 64:
        break
O = torch.stack(obss)
seqs = torch.zeros(O.shape[0], H, 2); seqs[..., 0] = 0.65
with torch.no_grad():
    out = wm.rollout_open_loop(O, seqs)
disp = (out["predicted_displacement"] / DISPLACEMENT_SCALE)
paths = disp[..., :2].norm(dim=-1).sum(dim=1)            # [N] longueur de chemin / 120 pas
lat = out["predicted_latents"]                           # [N,H,D]
cos1 = torch.cosine_similarity(lat[:, 1], lat[:, 0], dim=-1).mean()
cos_end = torch.cosine_similarity(lat[:, -1], lat[:, 0], dim=-1).mean()
print(f"WM={CK}  obs_dim={m['obs_dim']} retina={WITHRET}")
print(f"  RÊVE droit 120 pas : path médian = {float(paths.median()):.3f} m  (moy {float(paths.mean()):.3f})")
print(f"  cohérence latente   : cos(lat_1,lat_0)={float(cos1):+.3f}  cos(lat_119,lat_0)={float(cos_end):+.3f}")

# ── 2. eff_rank du latent (participation ratio) sur les mêmes frames (latent au pas 0 du rollout) ──
z = lat[:, 0]
z = z - z.mean(0, keepdim=True)
cov = (z.T @ z) / max(1, z.shape[0] - 1)
ev = torch.linalg.eigvalsh(cov).clamp(min=0)
eff_rank = float((ev.sum() ** 2) / (ev.pow(2).sum() + 1e-12))
print(f"  eff_rank (participation ratio) = {eff_rank:.1f} / {z.shape[1]}")

# ── 3. one-step teacher-forced (sanity : la tête déplacement doit être bonne quoi qu'il arrive) ──
seqs1 = torch.zeros(O.shape[0], 1, 2); seqs1[..., 0] = 0.65
with torch.no_grad():
    d1 = (wm.rollout_open_loop(O, seqs1)["predicted_displacement"] / DISPLACEMENT_SCALE)[:, 0, :2].norm(dim=-1)
print(f"  1-pas teacher-forced disp = {float(d1.mean())*1000:.1f} mm (réel ~5.0 mm)")

# ── 4. FIDÉLITÉ DU RÊVE ↔ RÉEL (LA métrique du chantier archi) : sur une vraie trajectoire consécutive,
#       le latent RÊVÉ (open-loop) suit-il le latent RÉEL (teacher-forced) ? cos haut = imagination fidèle. ──
import json as _json
rows = []
for f in files:
    rows = [_json.loads(l) for l in open(f)][:60]
    if len(rows) >= 50:
        break
if WITHRET:
    seq = torch.tensor([[r["obs"]["proprio"] + r["wm"]["retina0"] + [r["obs"]["energy"] / 100.0]] for r in rows])
else:
    seq = torch.zeros(len(rows), 1, m["obs_dim"])
    for i, r in enumerate(rows):
        seq[i, 0, :132] = torch.tensor(r["obs"]["proprio"]); seq[i, 0, 132:132 + 12] = torch.tensor(r["wm"]["radar0"])
        seq[i, 0, -1] = r["obs"]["energy"] / 100.0
seq = seq.transpose(0, 1)                                                   # [1,T,obs]
cmd = torch.tensor([[r["wm"]["cmd"][:2]] for r in rows]).float().transpose(0, 1)
with torch.no_grad():
    tf = wm.forward(seq, cmd)["latents"][0]                                 # latent RÉEL (teacher-forced)
    ol = wm.rollout_open_loop(seq[:, 0], cmd)["predicted_latents"][0]       # latent RÊVÉ (open-loop)
fid = {t: float(torch.cosine_similarity(ol[t:t + 1], tf[t:t + 1])[0]) for t in (1, 10, 40) if t < tf.shape[0]}
print("  fidélité rêve↔réel cos : " + "  ".join(f"t={t}:{c:+.3f}" for t, c in fid.items())
      + "   (haut = imagination fidèle ; chantier vise ≥~0.85@40)")
