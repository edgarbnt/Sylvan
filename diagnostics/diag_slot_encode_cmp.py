"""DIAG GRATUIT (Step 3 régression closed-loop) — le fine-tuning du slot_encoder a-t-il DÉGRADÉ la perception
BRUTE t0 (par-frame, sans transport) vs le slot_head d'origine ? + jitter tick-à-tick. (offline, 0 godot)

Hypothèse : en closed-loop le WM ré-encode le slot FRAIS à chaque replan ; si le fine-tuning transport-consistance a
abîmé l'encodeur en perception single-frame (surtout à l'arrière / bord de champ), il perçoit moins bien que le slot_head
codé-main → régression (7/16, arrière 0/4) malgré le +0.68 offline (qui mesurait le TRANSPORT, pas l'encode frais).

On compare, par-frame (objet visible), erreur de bearing + position vs food_rel0, bucket front/side/arrière :
  - slot_head (origine, slot_best.pt)
  - WM.slot_encoder (fine-tuné, wm_objcentric_s1)
+ jitter = |Δposition| moyen entre frames consécutives (instabilité tick-à-tick).

Usage: WM_CKPT=data/checkpoints/wm_objcentric_s1/wm_best.pt SLOT=data/checkpoints/slot_head/slot_best.pt \
       BUF=retina_eat_a PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_slot_encode_cmp.py
"""
import glob
import json
import math
import os
import statistics as st

import torch

from sylvan.models.command_wm import CommandWorldModel
from sylvan.models.slot_head import load_slot_head

torch.manual_seed(0)
torch.set_num_threads(4)
WM = os.environ.get("WM_CKPT", "data/checkpoints/wm_objcentric_s1/wm_best.pt")
SLOT = os.environ.get("SLOT", "data/checkpoints/slot_head/slot_best.pt")
BUFS = os.environ.get("BUF", "retina_eat_a retina_eat_b").split()
print(f"WM={WM}\nSLOT={SLOT}  BUFS={BUFS}")

pl = torch.load(WM, map_location="cpu", weights_only=False)
meta = pl["meta"]
wm = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
                       predictor_arch=meta.get("predictor_arch", "shallow"),
                       with_slot=True, slot_resources=meta.get("slot_resources", 1))
wm.load_state_dict(pl["model"]); wm.eval()
sh = load_slot_head(SLOT); sh.eval()

files = []
for b in BUFS:
    files += sorted(glob.glob(f"godot/data/replay_buffer/{b}/episode_*.jsonl") or
                    glob.glob(f"data/replay_buffer/{b}/episode_*.jsonl"))[:60]


def load():
    eps = []
    for f in files:
        seq = []
        for line in open(f):
            r = json.loads(line); w = r.get("wm", {})
            ret = w.get("retina0"); fr = w.get("food_rel0")
            if not ret or not fr:
                continue
            seq.append((ret, [float(fr[0]), float(fr[1])], float(fr[2])))
        if len(seq) > 5:
            eps.append(seq)
    return eps


eps = load()
print(f"épisodes={len(eps)} frames={sum(len(e) for e in eps)}")


@torch.no_grad()
def wm_enc(ret):
    obs = torch.zeros(meta["obs_dim"])
    obs[meta["proprio_dim"]:meta["proprio_dim"] + len(ret)] = torch.tensor(ret, dtype=torch.float32)
    return wm.encode_slot(obs).tolist()


@torch.no_grad()
def sh_enc(ret):
    return sh.locate(torch.tensor(ret, dtype=torch.float32))[0]


def bearing_err(p, fr):
    bs = math.atan2(p[0], p[1]); bt = math.atan2(fr[0], fr[1])
    return abs(math.degrees(math.atan2(math.sin(bs - bt), math.cos(bs - bt))))


# accuracy par bucket
buckets = {"front |b|<45": [], "side 45-135": [], "rear |b|>135": []}
errs = {"slot_head": {k: [] for k in buckets}, "wm_encoder": {k: [] for k in buckets}}
pos_err = {"slot_head": [], "wm_encoder": []}
jit = {"slot_head": [], "wm_encoder": []}
for seq in eps:
    prev = {"slot_head": None, "wm_encoder": None}
    for ret, fr, vis in seq:
        ph = sh_enc(ret); pw = wm_enc(ret)
        for name, p in (("slot_head", ph), ("wm_encoder", pw)):
            if prev[name] is not None:
                jit[name].append(math.hypot(p[0] - prev[name][0], p[1] - prev[name][1]))
            prev[name] = p
        if vis < 0.5:
            continue
        bt = abs(math.degrees(math.atan2(fr[0], fr[1])))
        bk = "front |b|<45" if bt < 45 else ("side 45-135" if bt <= 135 else "rear |b|>135")
        errs["slot_head"][bk].append(bearing_err(ph, fr))
        errs["wm_encoder"][bk].append(bearing_err(pw, fr))
        pos_err["slot_head"].append(math.hypot(ph[0] - fr[0], ph[1] - fr[1]))
        pos_err["wm_encoder"].append(math.hypot(pw[0] - fr[0], pw[1] - fr[1]))


def med(x):
    return st.median(x) if x else float("nan")


print("\nERREUR DE BEARING par-frame (médiane °, plus bas = mieux) :")
print(f"{'bucket':>14} | {'slot_head':>10} | {'wm_encoder':>11} | {'n':>5}")
for bk in buckets:
    print(f"{bk:>14} | {med(errs['slot_head'][bk]):>10.1f} | {med(errs['wm_encoder'][bk]):>11.1f} | "
          f"{len(errs['slot_head'][bk]):>5}")
print(f"\nerreur POSITION médiane (m) : slot_head={med(pos_err['slot_head']):.2f}  wm_encoder={med(pos_err['wm_encoder']):.2f}")
print(f"JITTER tick-à-tick |Δpos| médian (m) : slot_head={med(jit['slot_head']):.3f}  wm_encoder={med(jit['wm_encoder']):.3f}")
print("\n→ si wm_encoder >> slot_head (surtout arrière) : le fine-tuning a dégradé la perception frais → fix = "
      "ne PAS fine-tuner l'encodeur (garder slot_head gelé, n'apprendre QUE la calib).")
print("→ si wm_encoder ≈ slot_head : la perception n'est pas la cause → regarder le transport/coût/EMA en closed-loop.")
