"""Free JEPA-ification diagnostic (NO training) on the live WM (wm_command_hex_v2).

Answers: (1) is the latent COLLAPSED? (eff_rank / std) — decides whether we can shift weight to the
latent path at all. (2) is the latent prediction REAL or TRIVIAL? (vs a mean-predictor baseline +
cosine). (3) what do the reconstruction heads we'd DROP currently deliver vs the abstract readouts
the planner NEEDS (displacement, energy)? → tells us what's safe to drop and what to anchor.
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from sylvan.buffer.wm_dataset import load_wm_episode
from sylvan.models.command_wm import CommandWorldModel, representation_health, DISPLACEMENT_SCALE

CKPT = sys.argv[1] if len(sys.argv) > 1 else "data/checkpoints/wm_command_hex_v2/wm_best.pt"
W = 50  # window length
p = torch.load(CKPT, map_location="cpu", weights_only=False)
meta = p["meta"]; pd = meta["proprio_dim"]
model = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=pd, predictor_arch=meta.get("predictor_arch", "shallow"))
model.load_state_dict(p["model"]); model.eval()
eps = [e for e in (load_wm_episode(Path(x)) for x in meta["val_episodes"]) if e is not None]

OBS, CMD, NXT, DISP = [], [], [], []
for ep in eps:
    n = ep["obs"].shape[0]
    for s in range(0, n - W, W):
        OBS.append(ep["obs"][s:s+W]); CMD.append(ep["command"][s:s+W])
        NXT.append(ep["next_obs"][s:s+W]); DISP.append(ep["displacement"][s:s+W])
obs = torch.stack(OBS); cmd = torch.stack(CMD); nxt = torch.stack(NXT); disp = torch.stack(DISP)
print(f"[diag] ckpt={Path(CKPT).parent.name} | {len(eps)} val eps | {obs.shape[0]} windows x {W} | obs_dim={meta['obs_dim']} latent={128}")
print(f"[diag] trained loss_weights = {meta.get('loss_weights')}")
print(f"[diag] latent_loss={meta.get('latent_loss')} vicreg={meta.get('vicreg')}\n")

with torch.no_grad():
    out = model(obs, cmd, scheduled_sampling_prob=1.0)
    latents = out["latents"]                       # [B,T,L]
    pred_enc = out["predicted_next_encoded"]        # predicted NEXT encoding
    tgt_enc = model.encoder(nxt)                     # the JEPA target (stop-grad in training)

# (1) COLLAPSE — representation health of the RSSM latent AND of the encoder target
rh = representation_health(latents)
rh_t = representation_health(tgt_enc)
print("=== (1) COLLAPSE (latent richness) ===")
print(f"  RSSM latent : eff_rank={rh['eff_rank']:6.2f}/128  lat_std={rh['lat_std']:.4f} (min {rh['lat_std_min']:.4f})  offdiag={rh['offdiag']:.4f}")
print(f"  encoder tgt : eff_rank={rh_t['eff_rank']:6.2f}/128  lat_std={rh_t['lat_std']:.4f} (min {rh_t['lat_std_min']:.4f})")
print(f"  -> eff_rank<<128 = collapsed (few active dims). If low, latent-weight shift needs VICReg/cosine FIRST.\n")

# (2) LATENT PREDICTION real vs trivial
z = pred_enc.reshape(-1, pred_enc.shape[-1]); t = tgt_enc.reshape(-1, tgt_enc.shape[-1])
lat_mse = F.mse_loss(z, t).item()
mean_baseline = ((t - t.mean(0, keepdim=True))**2).mean().item()   # MSE of predicting the global mean
cos = F.cosine_similarity(z, t, dim=-1).mean().item()
print("=== (2) LATENT PREDICTION: real or trivial? ===")
print(f"  latent MSE              = {lat_mse:.5f}")
print(f"  mean-predictor baseline = {mean_baseline:.5f}   (predict the constant mean encoding)")
print(f"  skill ratio             = {lat_mse/mean_baseline:.3f}   (<<1 = real predictive content; ~1 = trivial)")
print(f"  cosine(pred, target)    = {cos:.3f}")
print(f"  ||pred_enc||={z.norm(dim=-1).mean():.3f}  ||target_enc||={t.norm(dim=-1).mean():.3f}  (shrink = collapse cheat)\n")

# (3) what the heads deliver: reconstruction (drop candidates) vs abstract readouts (anchors/planner)
po = out["predicted_next_obs"]
proprio_rmse = (po[..., :pd] - nxt[..., :pd]).pow(2).mean().sqrt().item()
radar_rmse = (po[..., pd:-1] - nxt[..., pd:-1]).pow(2).mean().sqrt().item()
energy_rmse100 = (po[..., -1] - nxt[..., -1]).pow(2).mean().sqrt().item() * 100
disp_rmse = (out["predicted_displacement"] - disp).pow(2).mean().sqrt().item() / DISPLACEMENT_SCALE
# variance of targets for context (RMSE relative to signal spread)
prop_sd = nxt[..., :pd].std().item(); rad_sd = nxt[..., pd:-1].std().item()
print("=== (3) HEAD QUALITY: drop-candidates (reconstruction) vs anchors (planner signal) ===")
print(f"  [DROP?] proprio RMSE = {proprio_rmse:.4f}  (signal std {prop_sd:.3f} -> NRMSE {proprio_rmse/prop_sd:.2f})")
print(f"  [DROP?] radar   RMSE = {radar_rmse:.4f}  (signal std {rad_sd:.3f} -> NRMSE {radar_rmse/rad_sd:.2f})")
print(f"  [ANCHOR] displacement RMSE = {disp_rmse*100:.4f} cm/step   (the planner's signal)")
print(f"  [ANCHOR] energy RMSE = {energy_rmse100:.3f} /100         (the cost readout)")
print("\n  -> high NRMSE reconstruction = head already weak = cheap to drop. displacement/energy are")
print("     decoded FROM the latent already -> JEPA shift keeps the planner signal IF latent stays rich.")
