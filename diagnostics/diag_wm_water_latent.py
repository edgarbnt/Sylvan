#!/usr/bin/env python
"""FREE offline probe — does the WM's DREAMED latent carry WATER position, the way it carries FOOD?

No Godot, no training (except a tiny linear/MLP probe). Answers ONE falsifiable question:
prerequisite for the planned pivot (a learned multi-drive survival-value on the latent). If the
latent is water-BLIND, the pivot needs a WM/slot water channel (retrain) FIRST.

METHOD (transport test):
  - Load `gate1_mode1` episodes (multi-drive raw Godot JSONL, has food+water radars).
  - obs0 (277) = proprio[132] ++ retina[144] ++ energy/100 [1]  (matches wm_dataset._obs_at).
  - DREAM forward H steps under the ACTUALLY-EXECUTED commands (obs.metrics.cmd_vx/cmd_omega).
  - At dream depth d, predicted_latents[:,d] should encode the state at the REAL future frame t0+d.
  - Ground truth at t0+d: FOOD (vision_fine 36) and WATER (vision_water 36) proximity + bearing,
    decoded with the SAME geometry as command_planner.food_xz_from_radar.
  - Probe: linear ridge (and optional MLP) latent[128] -> target, train/test split, metric = R².
  - FOOD is the POSITIVE CONTROL (known-carried: ValueHead AUC 0.78, OrientHead bearing 84%).

VERDICT:
  - water R² ~ food R² (both high) at depth 0 AND at depth 20-30 -> latent IS water-aware & transports it.
  - food R² high, water R² ~ 0                                   -> latent is WATER-BLIND (needs WM retrain).
  - water R² decent at depth 0 but collapses by 20-30            -> perceived but NOT transported (WM work).

Run:
  PYTHONPATH=python env_pytorch_3.12/bin/python diag_wm_water_latent.py --selfcheck
  PYTHONPATH=python env_pytorch_3.12/bin/python diag_wm_water_latent.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from sylvan.models.command_wm import CommandWorldModel

# ---- constants mirrored from the live seams (verified) --------------------------------------
OBS_DIM = 277           # proprio(132) ++ retina(144) ++ energy(1)
PROPRIO_DIM = 132
LATENT_DIM = 128
RADAR_MAX_RANGE = 10.0  # command_planner.RADAR_MAX_RANGE
DEFAULT_WMS = [
    "data/checkpoints/wm_objcentric_s1/wm_best.pt",
    "data/checkpoints/wm_command_hex_retina_eat_v1/wm_best.pt",
]
DEFAULT_DATA = "godot/data/replay_buffer/gate1_mode1"


# ---- ground-truth extraction (mirrors command_planner.food_xz_from_radar) -------------------
def radar_to_prox_bearing(radar: list[float]) -> tuple[float, float, float] | None:
    """Egocentric radar (any sector count) -> (proximity, sin(bearing), cos(bearing)).

    proximity = max sector value = 1 - dist/range in [0,1] (higher = nearer). bearing decoded
    exactly like food_xz_from_radar: sector center = (best+0.5)*sector_size - pi; the returned
    (x_right, z_fwd) = dist*(sin,cos) -> unit bearing vector (sin,cos). Returns None if nothing
    in range (all-zero radar)."""
    vals = [float(v) for v in radar]
    n = len(vals)
    if n == 0:
        return None
    best = max(range(n), key=lambda s: vals[s])
    if vals[best] <= 0.0:
        return None
    sector_size = 2.0 * math.pi / n
    bearing = (best + 0.5) * sector_size - math.pi
    return (vals[best], math.sin(bearing), math.cos(bearing))


# ---- WM loading (infer arch from checkpoint so both families load) --------------------------
def build_wm_from_ckpt(path: str) -> tuple[CommandWorldModel, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"]
    meta = ck.get("meta", {})
    obs_dim = int(meta.get("obs_dim", OBS_DIM))
    proprio_dim = int(meta.get("proprio_dim", PROPRIO_DIM))
    with_slot = any(k.startswith("slot_encoder") for k in sd)
    predictor_arch = meta.get("predictor_arch") or ("deep" if "encoded_predictor.6.weight" in sd else "shallow")
    model = CommandWorldModel(
        obs_dim=obs_dim,
        proprio_dim=proprio_dim,
        predictor_arch=predictor_arch,
        with_slot=with_slot,
        slot_resources=int(meta.get("slot_resources", 1)),
    )
    model.load_state_dict(sd, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"obs_dim": obs_dim, "proprio_dim": proprio_dim,
                   "with_slot": with_slot, "predictor_arch": predictor_arch}


# ---- data loading -----------------------------------------------------------------------------
def load_episode(path: Path) -> dict | None:
    """One gate1_mode1 JSONL -> per-frame arrays. Returns None if malformed/too short.

    NOTE (key adaptation): gate1_mode1 rows have NO `wm` block (unlike WM training data). We build
    obs0 from `obs` (proprio/retina/energy) and the executed command from `obs.metrics`."""
    obs, cmd, food, water = [], [], [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            o = r.get("obs")
            if not isinstance(o, dict) or "proprio" not in o or "retina" not in o:
                return None
            m = o.get("metrics", {})
            obs.append(list(o["proprio"]) + list(o["retina"]) + [float(o["energy"]) / 100.0])
            cmd.append([float(m.get("cmd_vx", 0.0)), float(m.get("cmd_omega", 0.0))])
            food.append(radar_to_prox_bearing(o.get("vision_fine", [])))
            water.append(radar_to_prox_bearing(o.get("vision_water", [])))
    if len(obs) < 2 or any(len(x) != OBS_DIM for x in obs):
        return None
    return {"obs": np.asarray(obs, dtype=np.float32),
            "cmd": np.asarray(cmd, dtype=np.float32),
            "food": food, "water": water}


# ---- linear + MLP probes ----------------------------------------------------------------------
def _standardize(Xtr, Xte):
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _r2(yte, pred):
    yte = yte.reshape(len(yte), -1)
    pred = pred.reshape(len(pred), -1)
    ss_res = ((yte - pred) ** 2).sum()
    ss_tot = ((yte - yte.mean(0, keepdims=True)) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def ridge_r2(Xtr, ytr, Xte, yte, lam=1.0):
    Xtr, Xte = _standardize(Xtr, Xte)
    ytr = ytr.reshape(len(ytr), -1)
    yte = yte.reshape(len(yte), -1)
    ym = ytr.mean(0, keepdims=True)
    Ytr = ytr - ym
    d = Xtr.shape[1]
    W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(d, dtype=Xtr.dtype), Xtr.T @ Ytr)
    pred = Xte @ W + ym
    return _r2(yte, pred)


def mlp_r2(Xtr, ytr, Xte, yte, seed=0, steps=400):
    torch.manual_seed(seed)
    Xtr, Xte = _standardize(Xtr, Xte)
    ytr = ytr.reshape(len(ytr), -1)
    yte = yte.reshape(len(yte), -1)
    Xt = torch.tensor(Xtr); Yt = torch.tensor(ytr)
    net = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], 128), torch.nn.SiLU(), torch.nn.Linear(128, ytr.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(Xt), Yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = net(torch.tensor(Xte)).numpy()
    return _r2(yte, pred)


# ---- core: dream + collect latents & ground truth ---------------------------------------------
def collect(model, episodes, depths, horizon, n_frames, rng, mb=512):
    """Sample start frames, dream, and collect (latent_at_depth, GT_at_real_future_frame) pairs.

    Returns per-depth dict: {depth: {"lat": [N,128], "food": {prox/sin/cos}, "water": {...},
    plus validity masks}}. GT is taken from the REAL frame t0+d (transport test)."""
    # enumerate valid start frames (need t0+max_depth in range)
    max_d = max(depths)
    starts = []  # (ep_idx, t0)
    for ei, ep in enumerate(episodes):
        L = len(ep["obs"])
        for t0 in range(0, L - max_d - 1):
            starts.append((ei, t0))
    rng.shuffle(starts)
    if n_frames and len(starts) > n_frames:
        starts = starts[:n_frames]

    out = {d: {"lat": [], "food": [], "water": [], "fmask": [], "wmask": []} for d in depths}
    for i in range(0, len(starts), mb):
        chunk = starts[i:i + mb]
        obs0 = np.stack([episodes[ei]["obs"][t0] for ei, t0 in chunk])
        cmds = np.stack([episodes[ei]["cmd"][t0:t0 + horizon] for ei, t0 in chunk])
        obs0_t = torch.tensor(obs0)
        cmd_t = torch.tensor(cmds)
        with torch.no_grad():
            lat = model.rollout_open_loop(obs0_t, cmd_t)["predicted_latents"].numpy()  # [B,H,128]
        for d in depths:
            out[d]["lat"].append(lat[:, d])
            for ei, t0 in chunk:
                fr = t0 + d
                fg = episodes[ei]["food"][fr]
                wg = episodes[ei]["water"][fr]
                out[d]["food"].append(fg if fg is not None else (0.0, 0.0, 0.0))
                out[d]["water"].append(wg if wg is not None else (0.0, 0.0, 0.0))
                out[d]["fmask"].append(fg is not None)
                out[d]["wmask"].append(wg is not None)
    for d in depths:
        out[d]["lat"] = np.concatenate(out[d]["lat"], 0)
        out[d]["food"] = np.asarray(out[d]["food"], dtype=np.float32)
        out[d]["water"] = np.asarray(out[d]["water"], dtype=np.float32)
        out[d]["fmask"] = np.asarray(out[d]["fmask"])
        out[d]["wmask"] = np.asarray(out[d]["wmask"])
    return out, len(starts)


def probe_all(data, depths, use_mlp, seed):
    """Return {depth: {water_prox, water_brg, food_prox, food_brg}} R² dict (linear; +mlp if asked)."""
    rows = {}
    for d in depths:
        lat = data[d]["lat"]
        res = {}
        for name, radar, mask in (("food", "food", "fmask"), ("water", "water", "wmask")):
            m = data[d][mask]
            X = lat[m]
            gt = data[d][radar][m]  # [N,3] = (prox, sin, cos)
            n = len(X)
            rs = np.random.RandomState(seed)
            perm = rs.permutation(n)
            cut = int(0.7 * n)
            tr, te = perm[:cut], perm[cut:]
            prox_r2 = ridge_r2(X[tr], gt[tr, 0], X[te], gt[te, 0])
            brg_r2 = ridge_r2(X[tr], gt[tr, 1:3], X[te], gt[te, 1:3])
            res[f"{name}_prox"] = prox_r2
            res[f"{name}_brg"] = brg_r2
            res[f"{name}_n"] = n
            if use_mlp:
                res[f"{name}_prox_mlp"] = mlp_r2(X[tr], gt[tr, 0], X[te], gt[te, 0], seed)
                res[f"{name}_brg_mlp"] = mlp_r2(X[tr], gt[tr, 1:3], X[te], gt[te, 1:3], seed)
        rows[d] = res
    return rows


def print_table(name, rows, depths, use_mlp):
    print(f"\n=== {name} ===")
    hdr = f"{'depth':>5} | {'food_prox':>9} {'food_brg':>8} | {'water_prox':>10} {'water_brg':>9} | {'gap_prox':>8} {'gap_brg':>7} | {'n':>6}"
    print(hdr)
    print("-" * len(hdr))
    for d in depths:
        r = rows[d]
        gp = r["food_prox"] - r["water_prox"]
        gb = r["food_brg"] - r["water_brg"]
        print(f"{d:>5} | {r['food_prox']:>9.3f} {r['food_brg']:>8.3f} | "
              f"{r['water_prox']:>10.3f} {r['water_brg']:>9.3f} | {gp:>8.3f} {gb:>7.3f} | {r['water_n']:>6}")
    if use_mlp:
        print("  [MLP]")
        for d in depths:
            r = rows[d]
            print(f"{d:>5} | {r['food_prox_mlp']:>9.3f} {r['food_brg_mlp']:>8.3f} | "
                  f"{r['water_prox_mlp']:>10.3f} {r['water_brg_mlp']:>9.3f} |")


def verdict(rows, depths):
    d0 = depths[0]
    deep = [d for d in depths if d >= 20] or [depths[-1]]
    wp0 = rows[d0]["water_prox"]; fp0 = rows[d0]["food_prox"]
    wpD = min(rows[d]["water_prox"] for d in deep)
    fpD = min(rows[d]["food_prox"] for d in deep)
    print("\n=== VERDICT ===")
    print(f"depth0: food_prox R²={fp0:.3f}  water_prox R²={wp0:.3f}")
    print(f"deep(>=20): food_prox R²(min)={fpD:.3f}  water_prox R²(min)={wpD:.3f}")
    HI, LO = 0.30, 0.10
    if fp0 < HI:
        print("!! POSITIVE CONTROL WEAK: food R² low at depth 0 -> probe/method suspect, do NOT trust water verdict.")
        return
    if wp0 >= HI and wpD >= HI:
        print("--> LATENT IS WATER-AWARE and TRANSPORTS it. Pivot can extend value to multi-drive on the EXISTING latent (cheap value-training).")
    elif wp0 >= HI and wpD < HI:
        print("--> WATER PERCEIVED at depth 0 but NOT TRANSPORTED (collapses by depth 20-30). Pivot needs WM rotation/permanence work.")
    elif wp0 < LO:
        print("--> LATENT IS WATER-BLIND (water R² ~ 0 while food R² high). Pivot needs a WM/slot WATER channel (RETRAIN) FIRST.")
    else:
        print("--> WATER PARTIALLY present at depth 0 (weak). Likely needs WM work; treat as blind-ish for the pivot.")


# ---- selfcheck --------------------------------------------------------------------------------
def selfcheck():
    print("[selfcheck] building WMs...")
    for wm in DEFAULT_WMS:
        model, info = build_wm_from_ckpt(wm)
        assert info["obs_dim"] == OBS_DIM, f"obs_dim {info['obs_dim']} != 277 ({wm})"
        B, H = 4, 31
        obs0 = torch.randn(B, OBS_DIM)
        cmds = torch.randn(B, H, 2)
        with torch.no_grad():
            out = model.rollout_open_loop(obs0, cmds)
        assert "predicted_latents" in out, "missing predicted_latents"
        lat = out["predicted_latents"]
        assert lat.shape == (B, H, LATENT_DIM), f"latent shape {tuple(lat.shape)} != (4,31,128)"
        assert torch.isfinite(lat).all(), "non-finite latents"
        if info["with_slot"]:
            assert "slot" in out and out["slot"].shape == (B, H, 2), "bad slot output"
        print(f"  OK {wm}  arch={info['predictor_arch']} with_slot={info['with_slot']} latent={tuple(lat.shape)}")
    # ground-truth extraction finite
    gt = radar_to_prox_bearing([0.0, 0.0, 0.6, 0.0, 0.3, 0.0])
    assert gt is not None and all(math.isfinite(v) for v in gt), "GT extraction non-finite"
    assert radar_to_prox_bearing([0.0, 0.0, 0.0]) is None, "all-zero radar should be None"
    print(f"  OK radar_to_prox_bearing -> {tuple(round(v,3) for v in gt)}; all-zero -> None")
    # tiny synthetic probe sanity: linear-signal -> high R², random -> ~0
    rs = np.random.RandomState(0)
    X = rs.randn(600, 128).astype(np.float32)
    w = rs.randn(128).astype(np.float32)
    y_sig = (X @ w + 0.05 * rs.randn(600)).astype(np.float32)
    y_rnd = rs.randn(600).astype(np.float32)
    r_sig = ridge_r2(X[:400], y_sig[:400], X[400:], y_sig[400:])
    r_rnd = ridge_r2(X[:400], y_rnd[:400], X[400:], y_rnd[400:])
    assert r_sig > 0.9, f"synthetic signal R² too low: {r_sig}"
    assert r_rnd < 0.2, f"synthetic random R² too high: {r_rnd}"
    print(f"  OK probe sanity: signal R²={r_sig:.3f} (>0.9), random R²={r_rnd:.3f} (<0.2)")
    print("[selfcheck] PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm-ckpt", nargs="*", default=DEFAULT_WMS)
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--episodes", type=int, default=8, help="max episodes to load")
    ap.add_argument("--n-frames", type=int, default=4000, help="max start frames sampled (total)")
    ap.add_argument("--depths", type=int, nargs="*", default=[0, 10, 20, 30])
    ap.add_argument("--horizon", type=int, default=31)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mlp", action="store_true", help="also fit a 1-hidden-layer MLP probe")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.selfcheck:
        selfcheck()
        return

    args.horizon = max(args.horizon, max(args.depths) + 1)
    data_dir = Path(args.data_dir)
    ep_paths = sorted(data_dir.glob("*.jsonl"))[:args.episodes]
    print(f"[data] {data_dir}: loading {len(ep_paths)} episodes (of {len(sorted(data_dir.glob('*.jsonl')))})")
    episodes = []
    for p in ep_paths:
        ep = load_episode(p)
        if ep is None:
            print(f"  skip {p.name} (malformed)")
            continue
        episodes.append(ep)
    tot_frames = sum(len(e["obs"]) for e in episodes)
    print(f"[data] {len(episodes)} episodes, {tot_frames} frames total")

    rng = np.random.RandomState(args.seed)
    for wm in args.wm_ckpt:
        model, info = build_wm_from_ckpt(wm)
        print(f"\n[wm] {wm}  arch={info['predictor_arch']} with_slot={info['with_slot']}")
        data, n_sampled = collect(model, episodes, args.depths, args.horizon, args.n_frames, rng)
        print(f"[wm] dreamed {n_sampled} start frames x H={args.horizon}")
        rows = probe_all(data, args.depths, args.mlp, args.seed)
        print_table(wm, rows, args.depths, args.mlp)
        verdict(rows, args.depths)


if __name__ == "__main__":
    sys.exit(main())
