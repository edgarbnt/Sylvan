"""Sanity offline (0 Godot) : (1) le serveur importe sans erreur, (2) plan_latent expose 'engage' et il est
BAS pour une cible derrière / HAUT pour une cible devant (le déclencheur CHERCHER est correctement câblé)."""
import json, glob, math
import torch
import scripts.serve_planner_command  # check syntaxe/import du serveur modifié
from sylvan.models.command_wm import CommandWorldModel
from sylvan.models.value_head import load_value_head
from sylvan.control.planning.command_planner import CommandPlanner, CommandPlanConfig

print("import serve_planner_command : OK")
WM = "data/checkpoints/wm_rich_fidele_sym/wm_best.pt"
VH = "data/checkpoints/value_head_food_dream/value_best.pt"
pl = torch.load(WM, map_location="cpu", weights_only=False); meta = pl["meta"]
wm = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
                       predictor_arch=meta.get("predictor_arch", "shallow"))
wm.load_state_dict(pl["model"]); wm.eval()
val = load_value_head(VH)
planner = CommandPlanner(wm, CommandPlanConfig(horizon=120))
files = sorted(glob.glob("data/replay_buffer/retina_forage/episode_*.jsonl") or
               glob.glob("godot/data/replay_buffer/retina_forage/episode_*.jsonl"))


def grab(behind):
    for f in files:
        for line in open(f):
            r = json.loads(line); w = r.get("wm", {})
            ret = w.get("retina0"); fr = w.get("food_rel0")
            if not ret or not fr or fr[2] < 0.5:
                continue
            fx, fz = fr[0], fr[1]
            if behind and fz >= 0:
                continue
            if (not behind) and (fz <= 0 or math.hypot(fx, fz) > 1.5):
                continue
            return torch.tensor(r["obs"]["proprio"] + ret + [0.4], dtype=torch.float32), (fx, fz)
    return None, None


import os
os.environ["SYLVAN_VALUE_AGG"] = "mean"
for behind in (True, False):
    obs, food = grab(behind)
    if obs is None:
        print(f"{'derrière' if behind else 'devant'}: aucune frame"); continue
    res = planner.plan_latent(obs, val, energy=0.4)
    lab = "derrière" if behind else "devant  "
    print(f"{lab}: engage={res['engage']:.3f}  cmd=({res['command'][0]:.2f},{res['command'][1]:+.2f})  "
          f"food=({food[0]:+.2f},{food[1]:+.2f})  → {'CHERCHE (engage<0.5)' if res['engage'] < 0.5 else 'APPROCHE'}")
