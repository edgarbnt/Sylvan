"""DIAG GRATUIT (Step 3 régression) — le signal d'ENGAGEMENT (min_dist + commande choisie) du planner WM-slot
diffère-t-il du codé-main sur des frames ARRIÈRE ? (offline, 0 godot). La perception est OK (diag_slot_encode_cmp) →
on regarde le transport/coût. Suspect : slot_calib kfwd=0.31 (slot s'approche ~3× trop lentement dans le rêve →
min_dist plat → pas de discrimination des candidats → pas d'engagement).

Pour des frames où food est DERRIÈRE (|bearing|>120°), on appelle :
  - planner WM-slot   : CommandPlanner(wm_objcentric_s1, with_slot) → branche plan_wm_slot (out["slot"])
  - planner codé-main : CommandPlanner(wm_rich_fidele_sym_jepa) + food_override=slot_head.locate (branche validée 13/16)
et on compare (best min_dist imaginé, |omega| choisi, signe omega vers la cible). Un bon engagement = min_dist bas
(le rêve montre qu'on PEUT atteindre) + omega fort dans le bon sens.

Usage: PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_plan_rear_cmp.py
"""
import glob
import json
import math
import os
import statistics as st

import torch

from sylvan.control.planning.command_planner import CommandPlanner, CommandPlanConfig
from sylvan.models.command_wm import CommandWorldModel
from sylvan.models.slot_head import load_slot_head

torch.manual_seed(0)
torch.set_num_threads(4)
WM_SLOT = "data/checkpoints/wm_objcentric_s1/wm_best.pt"
WM_BASE = "data/checkpoints/wm_rich_fidele_sym_jepa/wm_best.pt"
SLOT = "data/checkpoints/slot_head/slot_best.pt"
BUFS = os.environ.get("BUF", "retina_eat_a retina_eat_b").split()


def load_wm(path, with_slot):
    pl = torch.load(path, map_location="cpu", weights_only=False)
    m = pl["meta"]
    wm = CommandWorldModel(obs_dim=m["obs_dim"], proprio_dim=m["proprio_dim"],
                           predictor_arch=m.get("predictor_arch", "shallow"),
                           with_slot=with_slot, slot_resources=m.get("slot_resources", 1))
    wm.load_state_dict(pl["model"]); wm.eval()
    return wm, m


cfg = CommandPlanConfig(horizon=160)
cfg.heading_weight = 0.0   # config vivante (hw=0)
wm_slot, meta = load_wm(WM_SLOT, True)
if os.environ.get("CALIB"):
    wm_slot.slot_calib.data = torch.tensor([float(x) for x in os.environ["CALIB"].split()])
    print(f"CALIB FORCÉE = {wm_slot.slot_calib.data.tolist()}")
wm_base, _ = load_wm(WM_BASE, False)
pl_slot = CommandPlanner(wm_slot, cfg)
pl_base = CommandPlanner(wm_base, cfg)
sh = load_slot_head(SLOT)
PRO = meta["proprio_dim"]

files = []
for b in BUFS:
    files += sorted(glob.glob(f"godot/data/replay_buffer/{b}/episode_*.jsonl") or
                    glob.glob(f"data/replay_buffer/{b}/episode_*.jsonl"))[:40]

rear = []
for f in files:
    for line in open(f):
        r = json.loads(line); w = r.get("wm", {})
        ret = w.get("retina0"); fr = w.get("food_rel0")
        if not ret or not fr or float(fr[2]) < 0.5:
            continue
        bt = math.degrees(math.atan2(float(fr[0]), float(fr[1])))
        dd = math.hypot(float(fr[0]), float(fr[1]))
        if abs(bt) > 120 and dd > float(os.environ.get("DMIN", "2.0")):   # food DERRIÈRE et LOIN
            obs = list(r["obs"]["proprio"]) + list(ret) + [r["obs"].get("energy", 50.0) / 100.0]
            rear.append((torch.tensor(obs, dtype=torch.float32), ret, [float(fr[0]), float(fr[1])], bt))
    if len(rear) >= 120:
        break
dists = [math.hypot(x[2][0], x[2][1]) for x in rear]
print(f"frames ARRIÈRE+LOIN (|brg|>120°, d>{os.environ.get('DMIN','2.0')}m) = {len(rear)} ; "
      f"dist méd={st.median(dists) if dists else float('nan'):.2f}m")


def turn_ok(om, bt):
    """omega choisi va-t-il dans le sens qui ramène la cible vers l'avant ? (bt>0 = cible à droite)."""
    return (om < 0 and bt > 0) or (om > 0 and bt < 0)   # convention: om négatif tourne vers la droite


res = {"wm_slot": {"mind": [], "om": [], "turn": 0}, "base": {"mind": [], "om": [], "turn": 0}}
for obs, ret, fr, bt in rear[:120]:
    rs = pl_slot.plan(obs, [0.0] * 12)
    food_pos = sh.locate(torch.tensor(ret, dtype=torch.float32))[0]
    rb = pl_base.plan(obs, [0.0] * 12, override_pos=True, food_override=tuple(food_pos))
    for name, r in (("wm_slot", rs), ("base", rb)):
        res[name]["mind"].append(float(r.get("pred_min_dist", float("nan"))))
        om = float(r["command"][1])
        res[name]["om"].append(abs(om))
        res[name]["turn"] += int(turn_ok(om, bt))

n = len(rear[:120])
print(f"\n{'planner':>10} | {'min_dist méd':>12} | {'|omega| méd':>11} | {'tourne vers cible':>18}")
for name in ("wm_slot", "base"):
    d = res[name]
    print(f"{name:>10} | {st.median(d['mind']):>12.2f} | {st.median(d['om']):>11.2f} | {d['turn']:>3}/{n} "
          f"({100*d['turn']//n}%)")
print(f"\nréf : reason WM-slot={pl_slot.plan(rear[0][0], [0.0]*12).get('reason')} ; "
      f"calib WM-slot={wm_slot.slot_calib.data.tolist()}")
print("→ si wm_slot min_dist >> base (reste haut) ET tourne-vers-cible << base : le slot ne 's'approche' pas dans le")
print("  rêve → coût plat → pas d'engagement. Cause = transport/échelle (slot_calib), pas la perception.")
