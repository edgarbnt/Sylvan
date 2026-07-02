"""Construit le WM object-centric VIVANT (wm_objcentric_s1) = WM gelé + canal-slot, SANS entraînement.

Build canonique de l'internalisation (2026-06-25). Le canal-slot = slot_encoder (= slot_head, perception label-free
PROUVÉE) + transport GÉOMÉTRIQUE FIXE (1,-1,-1). Pas de fine-tuning : l'avoir fine-tuné par consistance-de-transport
a APPRIS une calib (0.31,-0.30,-0.93) qui faisait ORBITER le slot (bearing OK mais distance constante → 0 engagement
closed-loop) ; la calib est une géométrie (inverse exact du transport-agent codé-main), pas une quantité à fitter.
→ slot_head copié tel quel + calib fixe = engagement 15/16 > échafaudage 14/16, foraging à parité. WM IDENTIQUE
(gelé → displacement au bit, eff_rank/pos intacts). train_slot_channel.py (le fine-tune) = superseded, gardé pour le récit.

Usage: WM_CKPT=data/checkpoints/wm_rich_fidele_sym_jepa/wm_best.pt SLOT=data/checkpoints/slot_head/slot_best.pt \
       OUT=data/checkpoints/wm_objcentric_s1 PYTHONPATH=python ./env_pytorch_3.12/bin/python build_slot_channel.py
"""
import os

import torch

from sylvan.models.command_wm import CommandWorldModel
from sylvan.models.slot_head import load_slot_head

WM = os.environ.get("WM_CKPT", "data/checkpoints/wm_rich_fidele_sym_jepa/wm_best.pt")
SLOT = os.environ.get("SLOT", "data/checkpoints/slot_head/slot_best.pt")
OUT = os.environ.get("OUT", "data/checkpoints/wm_objcentric_s1")

pl = torch.load(WM, map_location="cpu", weights_only=False)
meta = pl["meta"]
# modèle avec canal-slot (slot_calib = buffer fixe (1,-1,-1) défini dans CommandWorldModel)
model = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
                          predictor_arch=meta.get("predictor_arch", "shallow"), with_slot=True)
model.load_state_dict(pl["model"], strict=False)               # WM gelé inchangé ; slot_* restent init
model.slot_encoder.load_state_dict(load_slot_head(SLOT).state_dict())   # encodeur = slot_head PROUVÉ
model.eval()

os.makedirs(OUT, exist_ok=True)
out_meta = {**meta, "with_slot": True, "slot_resources": 1,
            "slot_note": "encoder=slot_head (label-free) ; transport géométrique fixe (1,-1,-1) ; WM gelé"}
torch.save({"model": model.state_dict(), "meta": out_meta}, os.path.join(OUT, "wm_best.pt"))
print(f"slot_calib={model.slot_calib.tolist()} (fixe) ; encoder<-slot_head ; WM gelé")
print(f"sauvé → {OUT}/wm_best.pt  (out['slot'] actif ; promu forager vivant)")
