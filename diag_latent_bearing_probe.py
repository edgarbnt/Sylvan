"""Où meurt le bearing ARRIÈRE ? rétine(87%) -> encodeur -> latent RSSM(OrientHead 35%). Sonde fraîche à chaque étage.

La rétine brute porte le bearing derrière (87% f/b, sonde fraîche). OrientHead sur le latent ne le lit qu'à 35%.
On entraîne une sonde FRAÎCHE (même cible bearing) sur deux représentations internes du WM :
  (a) encoder(obs0)                  = l'obs encodée (goulot 128-d que voit la GRU)
  (b) predicted_latents[:,0]         = le latent RSSM t0 (l'espace que LIT OrientHead)
Si (b) reste ~35% -> l'info est jetée par l'encodeur/GRU -> lire le bearing depuis la RÉTINE (perception-pure).
Si (b) remonte ~80% -> l'info EST dans le latent, OrientHead est juste un mauvais readout -> sonde latente
   (le fix dead-reckon revit en latent-PUR).

Usage: PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_latent_bearing_probe.py
"""
import json, glob, math
import torch
from torch import nn
from sylvan.models.command_wm import CommandWorldModel

torch.manual_seed(0)
WM = "data/checkpoints/wm_rich_fidele_sym/wm_best.pt"
pl = torch.load(WM, map_location="cpu", weights_only=False); meta = pl["meta"]
wm = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
                       predictor_arch=meta.get("predictor_arch", "shallow"))
wm.load_state_dict(pl["model"]); wm.eval()

files = sorted(glob.glob("data/replay_buffer/retina_forage/episode_*.jsonl") or
               glob.glob("godot/data/replay_buffer/retina_forage/episode_*.jsonl"))
OBS, TH, FZ = [], [], []
for f in files:
    for line in open(f):
        r = json.loads(line); w = r.get("wm", {})
        ret = w.get("retina0"); fr = w.get("food_rel0"); pr = r["obs"]["proprio"]
        if not ret or not fr or fr[2] < 0.5:
            continue
        OBS.append(pr + ret + [0.5]); TH.append(math.atan2(fr[0], fr[1])); FZ.append(fr[1])
        if len(OBS) >= 18000:
            break
    if len(OBS) >= 18000:
        break
OBS = torch.tensor(OBS, dtype=torch.float32); th = torch.tensor(TH); FZ = torch.tensor(FZ)
print(f"frames={len(OBS)} | obs_dim={OBS.shape[1]} (attendu {meta['obs_dim']})")
with torch.no_grad():
    enc = wm.encoder(OBS)
    cmd = torch.tensor([0.65, 0.0]).reshape(1, 1, 2).expand(len(OBS), 1, 2).contiguous()
    rssm0 = wm.rollout_open_loop(OBS, cmd)["predicted_latents"][:, 0]
Y = torch.stack([th.cos(), th.sin()], 1)


def run(name, Z):
    ntr = int(0.8 * len(Z))
    mu, sd = Z[:ntr].mean(0), Z[:ntr].std(0) + 1e-6; Zn = (Z - mu) / sd
    Ztr, Ytr = Zn[:ntr], Y[:ntr]; Zte = Zn[ntr:]; THte, FZte = th[ntr:], FZ[ntr:]
    probe = nn.Sequential(nn.Linear(Z.shape[1], 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))
    opt = torch.optim.Adam(probe.parameters(), 1e-3)
    for _ in range(1500):
        bi = torch.randint(0, ntr, (512,))
        p = probe(Ztr[bi]); pn = p / (p.norm(dim=-1, keepdim=True) + 1e-6)
        loss = ((pn - Ytr[bi]) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p = probe(Zte); pn = p / (p.norm(dim=-1, keepdim=True) + 1e-6)
    pred = torch.atan2(pn[:, 1], pn[:, 0])
    print(f"\n{name} (dim {Z.shape[1]}):")
    for lab, mask in (("DERRIÈRE", FZte < 0), ("DEVANT", FZte > 0)):
        if mask.sum() == 0:
            continue
        pt, tt = pred[mask], THte[mask]
        fb = ((pt.cos() > 0) == (tt.cos() > 0)).float().mean().item()
        side = ((pt.sin() > 0) == (tt.sin() > 0)).float().mean().item()
        d = ((pt - tt + math.pi) % (2 * math.pi) - math.pi).abs()
        print(f"  {lab}: n={int(mask.sum()):5d}  devant/derrière OK={fb:.0%}  côté OK={side:.0%}  |Δθ| méd={math.degrees(d.median().item()):3.0f}°")


run("encoder(obs0)", enc)
run("latent RSSM t0 = predicted_latents[:,0] (ce que lit OrientHead)", rssm0)
print("\nRappel : rétine brute DERRIÈRE = 87%/72%/34° ; OrientHead latent DERRIÈRE = 35%/13%/120°.")
