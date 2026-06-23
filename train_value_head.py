"""Entraîner la TÊTE DE VALEUR V(latent) (🅑-pur) — cible 'repas dans K pas' sur le latent GELÉ du WM eat-riche.
Sauve data/checkpoints/value_head_food/value_best.pt (state_dict + mu/sd buffers + meta). AUC held-out par épisode.
Usage: SYLVAN_WM_USE_RETINA=1 PYTHONPATH=python ./env_pytorch_3.12/bin/python train_value_head.py [wm_ckpt]
"""
import sys, json, glob, math
from pathlib import Path
import torch
from sylvan.models.command_wm import CommandWorldModel
from sylvan.models.value_head import ValueHead

import os
WM = sys.argv[1] if len(sys.argv) > 1 else "data/checkpoints/wm_command_hex_retina_eat_v1/wm_best.pt"
K = int(os.environ.get("SYLVAN_VALUE_K", "20"))          # 'repas dans K pas' ; K court = pique au CONTACT (close)
DIRS = ["godot/data/replay_buffer/retina_eat_a", "godot/data/replay_buffer/retina_eat_b"]
# ── HYPOTHÈSE TRANSFERT (2026-06-21) ────────────────────────────────────────────────────────────────
# DEFAUT (DREAM=0) : la value est entraînée sur les latents TEACHER-FORCED 1-pas (predicted_latents[:,0]),
# alors que le planner l'applique sur les latents RÊVÉS MULTI-PAS (rollout_open_loop, profondeurs 0..H).
# DREAM=1 : on entraîne sur la MÊME distribution que l'usage — latents rêvés à profondeurs VARIÉES, sous les
# commandes RÉELLEMENT exécutées (labels honnêtes), pour tester si le mismatch teacher-forced→rêvé plafonne
# la précision au close. Source de latents = SEULE différence ; cible/archi/AUC identiques.
DREAM = os.environ.get("SYLVAN_VALUE_DREAM", "0") == "1"
# Sortie par défaut DISTINCTE en mode rêve → ne JAMAIS écraser le value_head_food teacher-forced (comparaison).
_default_out = "data/checkpoints/value_head_food_dream" if DREAM else "data/checkpoints/value_head_food"
OUT = Path(os.environ.get("SYLVAN_VALUE_OUT", _default_out)); OUT.mkdir(parents=True, exist_ok=True)
H = int(os.environ.get("SYLVAN_VALUE_DREAM_H", "30"))      # profondeur de rêve échantillonnée (≈ portée K)
SS = int(os.environ.get("SYLVAN_VALUE_START_STRIDE", "4")) # 1 départ sur SS frames (limite le volume)
DS = int(os.environ.get("SYLVAN_VALUE_DEPTH_STRIDE", "2")) # 1 profondeur sur DS (couvre 0..H uniformément)
CAP = int(os.environ.get("SYLVAN_VALUE_CAP", "60000"))     # plafond #latents (sous-échantillonnage si dépassé)

pl = torch.load(WM, map_location="cpu", weights_only=False); meta = pl["meta"]
wm = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
                       predictor_arch=meta.get("predictor_arch", "shallow"))
wm.load_state_dict(pl["model"]); wm.eval()
L = meta.get("latent_dim", 128)
print(f"WM={WM} latent={L} | cible 'repas<{K}pas' | mode={'DREAM-multistep' if DREAM else 'teacher-forced'}")

# Charge en gardant la STRUCTURE PAR ÉPISODE (nécessaire pour rêver sous les commandes exécutées).
episodes = []  # liste de dicts {obs:[T,obs_dim] (None si pas de rétine), cmd:[T,2], ate:[T]}
for d in DIRS:
    for f in sorted(glob.glob(f"{d}/episode_*.jsonl")):
        rows = [json.loads(l) for l in open(f)]
        ate = [1.0 if r.get("wm", {}).get("ate") else 0.0 for r in rows]
        obs_ep, cmd_ep = [], []
        for r in rows:
            ret = r.get("wm", {}).get("retina0")
            obs_ep.append((r["obs"]["proprio"] + ret + [r["obs"]["energy"] / 100.0]) if ret else None)
            cmd_ep.append(((r["wm"].get("cmd") or [0.0, 0.0])[:2]))
        episodes.append({"obs": obs_ep, "cmd": cmd_ep, "ate": ate})


def _label(ate, j):  # 'repas dans K pas' à partir de la frame j+1 (convention identique au depth-0)
    return 1.0 if sum(ate[j + 1:j + 1 + K]) > 0 else 0.0


lat_l, lab_l, eid_l = [], [], []
with torch.no_grad():
    for eid, ep in enumerate(episodes):
        obs_ep, cmd_ep, ate = ep["obs"], ep["cmd"], ep["ate"]
        T = len(ate)
        if not DREAM:
            # TEACHER-FORCED (défaut) : latent 1-pas par frame valide — comportement d'origine, par batch.
            valid = [(i, obs_ep[i]) for i in range(T) if obs_ep[i] is not None]
            for s in range(0, len(valid), 4096):
                chunk = valid[s:s + 4096]
                O = torch.tensor([o for _, o in chunk], dtype=torch.float32)
                C = torch.tensor([cmd_ep[i] for i, _ in chunk], dtype=torch.float32).reshape(-1, 1, 2)
                lt = wm.rollout_open_loop(O, C)["predicted_latents"][:, 0, :]
                lat_l.append(lt)
                lab_l += [_label(ate, i) for i, _ in chunk]
                eid_l += [eid] * len(chunk)
            continue
        # DREAM MULTI-PAS : depuis chaque frame de départ valide, rêve H pas sous les commandes exécutées
        # (paddées par la dernière dispo), puis émet les latents à profondeurs variées avec labels honnêtes.
        starts = [i for i in range(0, T, SS) if obs_ep[i] is not None]
        if not starts:
            continue
        O = torch.tensor([obs_ep[i] for i in starts], dtype=torch.float32)
        seqs = []
        for i in starts:
            seq = [cmd_ep[min(i + t, T - 1)] for t in range(H)]
            seqs.append(seq)
        C = torch.tensor(seqs, dtype=torch.float32)                       # [num_starts, H, 2]
        lats = wm.rollout_open_loop(O, C)["predicted_latents"]            # [num_starts, H, L]
        for si, i in enumerate(starts):
            for d in range(0, H, DS):
                j = i + 1 + d                                            # frame réelle correspondant à la prof. d
                if j > T - 1:
                    break
                lat_l.append(lats[si, d:d + 1, :])
                lab_l.append(_label(ate, j))
                eid_l.append(eid)

LAT = torch.cat(lat_l) if not DREAM else torch.cat(lat_l, dim=0)
LAB = torch.tensor(lab_l); EID = torch.tensor(eid_l)
# Plafond mémoire/calcul : sous-échantillonne (déterministe) si on dépasse CAP latents.
if LAT.shape[0] > CAP:
    g = torch.Generator().manual_seed(0)
    sel = torch.randperm(LAT.shape[0], generator=g)[:CAP]
    LAT, LAB, EID = LAT[sel], LAB[sel], EID[sel]
print(f"latents={LAT.shape[0]} épisodes={len(episodes)} positifs={100*LAB.float().mean():.1f}%")

cut = int(eid * 0.7); tr = EID < cut; te = ~tr
head = ValueHead(L)
head.mu.copy_(LAT[tr].mean(0)); head.sd.copy_(LAT[tr].std(0) + 1e-6)
opt = torch.optim.Adam(head.parameters(), lr=2e-3, weight_decay=1e-4)
pw = ((1 - LAB[tr]).sum() / (LAB[tr].sum() + 1e-6)).clamp(1, 50)
lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)


def auc(score, label):
    s, l = score.flatten(), label.flatten()
    o = torch.argsort(s); rk = torch.empty_like(s); rk[o] = torch.arange(1, len(s) + 1, dtype=s.dtype)
    np_, nn_ = l.sum().item(), (1 - l).sum().item()
    return float("nan") if np_ == 0 or nn_ == 0 else (rk[l == 1].sum().item() - np_ * (np_ + 1) / 2) / (np_ * nn_)


best = 0.0
for ep in range(800):
    head.train(); opt.zero_grad()
    loss = lossf(head.logit(LAT[tr]), LAB[tr]); loss.backward(); opt.step()
    if (ep + 1) % 100 == 0:
        head.eval()
        with torch.no_grad():
            a = auc(head.logit(LAT[te]), LAB[te])
        print(f"  ep{ep+1} loss={loss.item():.3f} AUC_te={a:.3f}")
        if a > best:
            best = a
            torch.save({"state_dict": head.state_dict(), "latent_dim": L, "hidden": 256, "K": K,
                        "wm_ckpt": WM, "auc_heldout": a}, OUT / "value_best.pt")
print(f"AUC held-out best={best:.3f} → {OUT/'value_best.pt'}")
