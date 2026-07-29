"""LE LATENT PORTE-T-IL LA POSITION, ou seulement le TYPE ? (offline, gratuit)

POURQUOI. A1 a prouvé que l'encodeur du WM porte le TYPE de la ressource (99,9 %). Mais le SLOT,
qui fournit la POSITION au planner, ne lit pas le WM : il lit la rétine avec ses propres requêtes
cosinus. Deux organes séparés — le WM sait QUOI, le slot calcule OÙ — et le succès d'A1 n'a donc
jamais atteint le ciblage. La question de l'owner : le slot pourrait-il lire le latent ?

Ça ne se déduit pas d'A1. Un type est une catégorie parmi quatre ; une position est deux réels.
Un latent peut très bien coder « il y a du rouge » sans coder « à 4,2 m, 30° à gauche ».

CE QU'ON MESURE. Une tête de lecture (linéaire, puis MLP) entraînée à prédire la position vraie de
la ressource depuis le latent, sur un held-out. On la compare à ce que fait le slot actuel et au
témoin nul.

CRITÈRES PRÉ-ENREGISTRÉS :
  T1 ... erreur médiane < 1,93 m (le slot actuel sur la même vérité) ⇒ le latent fait MIEUX,
         la piste est réelle.
  T2 ... erreur médiane <= 1,0 m (barre historique du projet) ⇒ la piste suffit telle quelle.
  KILL . erreur >= témoin nul ⇒ le latent ne porte PAS la position ; inutile d'y brancher le slot,
         et il faudra chercher le ciblage ailleurs.

CLI : PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_latent_carries_position.py
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from sylvan.models.command_wm import CommandWorldModel  # noqa: E402

HALF_FOV = float(os.environ.get("SYLVAN_RETINA_FOV_DEG", "360")) / 2.0
SLOT_NOW = 1.93     # m — le slot actuel, mesuré par diag_slot_localise sur la MÊME vérité
BAR = 1.0           # m — barre historique du projet


def load(corpus: str, wm_path: str, key: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(latents encodeur, positions vraies) sur les ticks où la cible est visible ET dans le champ."""
    payload = torch.load(wm_path, map_location="cpu", weights_only=False)
    wm = CommandWorldModel.from_checkpoint(payload)
    wm.load_state_dict(payload["model"])
    wm.eval()
    P = payload["meta"]["proprio_dim"]

    obs, tgt = [], []
    for f in sorted(glob.glob(os.path.join(corpus, "*.jsonl"))):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            v = r.get("wm", {}).get(key)
            if not v or len(v) < 3 or v[2] <= 0.5:
                continue
            # Même restriction que diag_slot_localise : la vérité de Godot est à 360°, le monde sert
            # un cône. On ne juge que là où la question a un sens.
            if abs(math.degrees(math.atan2(v[0], v[1]))) > HALF_FOV:
                continue
            o = r["obs"]
            obs.append(o["proprio"][:P] + o["retina"] + [o["energy"] / 100.0])
            tgt.append([v[0], v[1]])
    x = torch.tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        z = wm.encoder(x)
    return z, torch.tensor(tgt, dtype=torch.float32)


def probe(z: torch.Tensor, y: torch.Tensor, hidden: int, steps: int = 3000) -> float:
    """Erreur médiane held-out d'une tête entraînée à lire la position dans le latent."""
    torch.manual_seed(0)
    n_tr = int(0.7 * len(z))
    net = (nn.Linear(z.shape[1], 2) if hidden == 0 else
           nn.Sequential(nn.Linear(z.shape[1], hidden), nn.SiLU(), nn.Linear(hidden, 2)))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(steps):
        i = torch.randperm(n_tr)[:256]
        opt.zero_grad()
        nn.functional.mse_loss(net(z[:n_tr][i]), y[:n_tr][i]).backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        err = (net(z[n_tr:]) - y[n_tr:]).norm(dim=1)
    return float(err.median())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wm", default="data/checkpoints/wm_foret_v2/wm_best.pt")
    ap.add_argument("--corpus", default="data/replay_buffer/gate_foret_cl")
    a = ap.parse_args()

    print(f"=== LE LATENT PORTE-T-IL LA POSITION ? | {a.wm} ===")
    ok = True
    for name, key in (("NOURRITURE", "food_rel0"), ("EAU", "water_rel0")):
        z, y = load(a.corpus, a.wm, key)
        null = float(y.norm(dim=1).median())
        lin, mlp = probe(z, y, 0), probe(z, y, 256)
        best = min(lin, mlp)
        print(f"\n  {name}  ({len(z)} ticks)")
        print(f"    témoin NUL (prédire 0,0)  {null:.2f} m")
        print(f"    slot ACTUEL (rétine)      {SLOT_NOW:.2f} m")
        print(f"    latent → linéaire         {lin:.2f} m")
        print(f"    latent → MLP              {mlp:.2f} m")
        if best >= null:
            print("    🛑 KILL : le latent ne porte PAS la position — brancher le slot dessus"
                  " n'apporterait rien.")
            ok = False
        elif best <= BAR:
            print(f"    ✅ le latent porte la position ({best:.2f} m ≤ {BAR:.1f} m) — la piste SUFFIT")
        elif best < SLOT_NOW:
            print(f"    ⚠️  mieux que le slot actuel ({best:.2f} < {SLOT_NOW:.2f} m) mais au-dessus"
                  f" de la barre {BAR:.1f} m — piste réelle, pas suffisante seule")
        else:
            print(f"    ❌ pas mieux que le slot actuel ({best:.2f} ≥ {SLOT_NOW:.2f} m)")
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
