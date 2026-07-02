"""L'info 'cible derrière' est-elle dans la RÉTINE BRUTE elle-même ? (probe direct, indépendant du WM)

On entraîne une petite sonde retina0(144) -> (cos,sin) du bearing de la bouffe la plus proche (food_rel0),
puis on évalue séparément cible DEVANT vs DERRIÈRE. Tranche deux fix très différents :
 - si la rétine brute LIT le derrière (mais le latent non, 35%) -> l'ENCODEUR jette l'info -> retrain encodeur.
 - si la rétine brute NE lit PAS le derrière non plus -> l'INPUT rétine n'a pas l'arrière (FOV avant) -> SCAN.
Indépendant de la lecture-code du FOV (preuve par les données, pas par le code).

Usage: PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_retina_fov_probe.py
"""
import json, glob, math
import torch
from torch import nn

torch.manual_seed(0)
files = sorted(glob.glob("data/replay_buffer/retina_forage/episode_*.jsonl") or
               glob.glob("godot/data/replay_buffer/retina_forage/episode_*.jsonl"))
assert files
X, TH, FZ = [], [], []
for f in files:
    for line in open(f):
        r = json.loads(line); w = r.get("wm", {})
        ret = w.get("retina0"); fr = w.get("food_rel0")
        if not ret or not fr or fr[2] < 0.5:
            continue
        X.append(ret); TH.append(math.atan2(fr[0], fr[1])); FZ.append(fr[1])
        if len(X) >= 30000:
            break
    if len(X) >= 30000:
        break
X = torch.tensor(X, dtype=torch.float32); th = torch.tensor(TH); FZ = torch.tensor(FZ)
n = len(X); ntr = int(0.8 * n)
print(f"frames={n} (dim rétine={X.shape[1]}) | train={ntr} test={n-ntr}")
mu, sd = X[:ntr].mean(0), X[:ntr].std(0) + 1e-6
Xn = (X - mu) / sd
Y = torch.stack([th.cos(), th.sin()], 1)
Xtr, Ytr = Xn[:ntr], Y[:ntr]
Xte, THte, FZte = Xn[ntr:], th[ntr:], FZ[ntr:]

probe = nn.Sequential(nn.Linear(X.shape[1], 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))
opt = torch.optim.Adam(probe.parameters(), 1e-3)
for step in range(1500):
    bi = torch.randint(0, ntr, (512,))
    p = probe(Xtr[bi]); pn = p / (p.norm(dim=-1, keepdim=True) + 1e-6)
    loss = ((pn - Ytr[bi]) ** 2).sum(1).mean()
    opt.zero_grad(); loss.backward(); opt.step()
print(f"loss train final (0=parfait, 2=hasard) = {loss.item():.3f}")

with torch.no_grad():
    p = probe(Xte); pn = p / (p.norm(dim=-1, keepdim=True) + 1e-6)
pred = torch.atan2(pn[:, 1], pn[:, 0])


def stats(mask):
    pt, tt = pred[mask], THte[mask]
    fb = ((pt.cos() > 0) == (tt.cos() > 0)).float().mean().item()
    side = ((pt.sin() > 0) == (tt.sin() > 0)).float().mean().item()
    d = ((pt - tt + math.pi) % (2 * math.pi) - math.pi).abs()
    return int(mask.sum()), fb, side, math.degrees(d.median().item())


print("\n=== lisibilité du bearing DEPUIS LA RÉTINE BRUTE (sonde fraîche) ===")
for name, mask in (("DERRIÈRE", FZte < 0), ("DEVANT", FZte > 0)):
    if mask.sum() == 0:
        print(f"{name}: aucune frame"); continue
    nm, fb, side, dd = stats(mask)
    print(f"{name}: n={nm:5d}  devant/derrière OK={fb:.0%}  côté OK={side:.0%}  |Δθ| méd={dd:3.0f}°")
print(f"\npart cible-derrière dans le test = {(FZte < 0).float().mean():.0%}")
print("Lecture: si DERRIÈRE reste ~50%/~50%/~90° => l'arrière n'est PAS dans la rétine (FOV avant) -> SCAN.")
print("         si DERRIÈRE est bon (>80%) => l'info Y EST, c'est l'encodeur qui la jette -> retrain encodeur.")
