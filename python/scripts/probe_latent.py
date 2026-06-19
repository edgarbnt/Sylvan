"""Phase B (JEPA-ification) diagnostic: how much input information lives in the latent?

A ridge LINEAR probe is fit from the (frozen) world-model latent to the next obs it is
supposed to encode (radar 12-d, proprio 94-d), on held-out val episodes. We report the
probe's test MSE next to the model's OWN reconstruction-head MSE on the same targets:

  - probe_mse ≈ recon_mse  → the latent keeps the info LINEARLY, even without leaning on the
    decoder → the JEPA win (we can drop the reconstruction weight without losing the info).
  - probe_mse >> recon_mse → the info only survives because the decoder forces it in; cutting
    reconstruction will lose it → collapse risk, VICReg (step 2) needed before going further.

Teacher-forced (real obs every step, scheduled_sampling_prob=1.0) so latent/targets align.

Usage: python -m scripts.probe_latent --checkpoint data/checkpoints/wm_command_v2/wm_best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sylvan.buffer.wm_dataset import (
    CommandSequenceDataset,
    collate_command_samples,
    load_wm_episode,
)
from sylvan.constants import DEFAULT_PROPRIO_DIM
from sylvan.models.command_wm import CommandWorldModel, representation_health
from torch.utils.data import DataLoader


def ridge_fit(x: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
    """Closed-form ridge with bias. x:[N,D] y:[N,K] -> W:[D+1,K]."""
    xb = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
    d = xb.shape[1]
    a = xb.t() @ xb + lam * torch.eye(d)
    return torch.linalg.solve(a, xb.t() @ y)


def ridge_mse(w: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> float:
    xb = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
    return float(((xb @ w - y) ** 2).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description="Linear-probe the WM latent for retained input info.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--lam", type=float, default=1.0, help="Ridge regularization.")
    ap.add_argument("--max-windows", type=int, default=400, help="Cap probe samples for speed.")
    args = ap.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu")
    meta = payload["meta"]
    proprio_dim = meta.get("proprio_dim", DEFAULT_PROPRIO_DIM)
    model = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=proprio_dim,
                              predictor_arch=meta.get("predictor_arch", "shallow"))
    model.load_state_dict(payload["model"])
    model.eval()

    episodes = [load_wm_episode(Path(p)) for p in meta["val_episodes"]]
    episodes = [e for e in episodes if e is not None]
    ds = CommandSequenceDataset.__new__(CommandSequenceDataset)
    ds.sequence_length = args.seq_len
    ds.episodes = []
    ds.windows = []
    for ep in episodes:
        n = ep["obs"].shape[0]
        if n < args.seq_len:
            continue
        idx = len(ds.episodes)
        ds.episodes.append(ep)
        for start in range(0, n - args.seq_len + 1, max(16, args.seq_len // 2)):
            ds.windows.append((idx, start))
    if not ds.windows:
        raise SystemExit("Pas de fenêtre val exploitable.")
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_command_samples)

    lat, tgt_radar, tgt_prop, recon_radar, recon_prop = [], [], [], [], []
    seen = 0
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch.obs, batch.command, scheduled_sampling_prob=1.0)
            L = outputs["latents"].reshape(-1, outputs["latents"].shape[-1])
            no = batch.next_obs.reshape(-1, batch.next_obs.shape[-1])
            po = outputs["predicted_next_obs"].reshape(-1, no.shape[-1])
            lat.append(L)
            tgt_radar.append(no[:, proprio_dim:-1])
            tgt_prop.append(no[:, :proprio_dim])
            recon_radar.append(po[:, proprio_dim:-1])
            recon_prop.append(po[:, :proprio_dim])
            seen += L.shape[0]
            if seen >= args.max_windows * args.seq_len:
                break

    lat = torch.cat(lat)
    tgt_radar, tgt_prop = torch.cat(tgt_radar), torch.cat(tgt_prop)
    recon_radar, recon_prop = torch.cat(recon_radar), torch.cat(recon_prop)

    n = lat.shape[0]
    split = int(n * 0.8)
    perm = torch.randperm(n)
    tr, te = perm[:split], perm[split:]

    health = representation_health(lat)
    print(f"[probe] {n} échantillons latents (dim {lat.shape[1]}) | "
          f"lat_std={health['lat_std']:.3f} eff_rank={health['eff_rank']:.1f}/{lat.shape[1]} "
          f"offdiag={health['offdiag']:.3f}")
    print(f"\n{'cible':>8} | {'probe MSE':>10} | {'recon MSE':>10} | {'ratio p/r':>10}")
    for name, tgt, recon in (("radar", tgt_radar, recon_radar), ("proprio", tgt_prop, recon_prop)):
        w = ridge_fit(lat[tr], tgt[tr], args.lam)
        p_mse = ridge_mse(w, lat[te], tgt[te])
        r_mse = float(((recon[te] - tgt[te]) ** 2).mean())
        print(f"{name:>8} | {p_mse:>10.5f} | {r_mse:>10.5f} | {p_mse / (r_mse + 1e-12):>10.2f}")
    print("\nratio p/r ≈1 → info retenue linéairement (JEPA OK) ; >>1 → dépend du décodeur (collapse si on coupe).")


if __name__ == "__main__":
    main()
