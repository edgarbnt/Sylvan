#!/usr/bin/env python
"""FREE offline gate G2 — can a SURVIVAL value on the WM's DREAMED latent predict multi-drive
survival AND capture ARBITRATION? (No Godot, no long training — only a tiny value head is fit.)

CONTEXT (the strategic pivot):
  G1 (`diag_wm_water_latent.py`) proved the dreamed latent carries BOTH food and water position.
  G2 asks the decisive follow-up: if we train a SURVIVAL value on that FROZEN dreamed latent,
    (a) does it PREDICT multi-drive survival (return-to-go), and is it sensitive to BOTH drives?
    (b) does it capture ARBITRATION — in a state where ONE drive is low, does the value rate
        "head to the depleted drive's resource" ABOVE "head to the other (satisfied) resource"?
  If yes  -> the pivot (learned value on the raw latent -> planner cost) is viable & CHEAP.
  If no   -> fallback = a 2-resource slot (WM retrain).

METHOD:
  - Load `gate1_mode1` episodes (multi-drive raw Godot JSONL; per-frame obs.energy + obs.thirst,
    food radar = vision_fine, water radar = vision_water). One JSONL = one episode. Episodes end
    in DEATH (min(energy,thirst)->0, row.done) or TRUNCATION (row.truncated, drives still >0).
  - obs0 (277) = proprio[132] ++ retina[144] ++ energy/100 [1]   (mirrors wm_dataset._obs_at).
  - DREAM H steps under ACTUALLY-EXECUTED commands (obs.metrics.cmd_vx/cmd_omega); emit latents at
    depths 0..H, each labeled by the survival-return-to-go G at the REAL future frame t0+depth.
  - Train ValueHead(128) (frozen WM) with MSE on G in [0,1]; split by EPISODE (no leakage).

LABELS (flagged assumptions):
  - G_t (survival-return-to-go): discounted count of steps survived from t until episode end,
    normalized to [0,1]. Closed form (geometric sum) = 1 - gamma^(remaining), remaining = L - t,
    gamma=0.99 (effective horizon ~100). Low near death, ~1 when safe. NOTE: for the 1 TRUNCATED
    episode the last ~few-hundred frames get an artificially-low G (episode ended safe, not dead) —
    minor bias, flagged. Death definition = row.done (min(energy,thirst)->0).
  - surv100 (binary, interpretable) = min(energy,thirst) > 0 across the next up-to-100 frames. For
    death episodes near the end the drive reaches 0 -> surv100=0 naturally; truncated episode stays
    >0 -> surv100=1. No special-casing needed.

VERDICT:
  - G2a R²>0.3 AND sensitive to BOTH drives AND G2b >65% -> value arbitrates on the raw latent ->
    pivot viable (cheap), proceed to planner integration (G3).
  - value ignores thirst OR G2b ~50%                     -> raw-latent value doesn't arbitrate
    (lossiness bites) -> fallback = 2-resource slot (retrain).

Run:
  PYTHONPATH=python env_pytorch_3.12/bin/python diag_mode1_survival_value.py --selfcheck
  PYTHONPATH=python env_pytorch_3.12/bin/python diag_mode1_survival_value.py
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
from sylvan.models.value_head import ValueHead

# ---- constants mirrored from the live seams (verified) --------------------------------------
OBS_DIM = 277           # proprio(132) ++ retina(144) ++ energy(1)
PROPRIO_DIM = 132
LATENT_DIM = 128
RADAR_MAX_RANGE = 10.0  # command_planner.RADAR_MAX_RANGE
GAMMA = 0.99            # survival discount; effective horizon 1/(1-gamma) = 100
DEFAULT_WMS = [
    "data/checkpoints/wm_objcentric_s1/wm_best.pt",
    "data/checkpoints/wm_command_hex_retina_eat_v1/wm_best.pt",
]
DEFAULT_DATA = "godot/data/replay_buffer/gate1_mode1"


# ---- ground-truth radar decode (mirrors command_planner.food_xz_from_radar) -----------------
def radar_to_prox_bearing(radar) -> tuple[float, float, float] | None:
    """Egocentric radar -> (proximity, sin(bearing), cos(bearing)); None if nothing in range.

    proximity = max sector value = 1 - dist/range in [0,1] (higher = nearer). bearing decoded
    exactly like food_xz_from_radar: sector center = (best+0.5)*sector_size - pi."""
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


def radar_bearing_angle(radar) -> float | None:
    """Egocentric bearing angle (rad, 0 = straight ahead, +right) or None if nothing in range."""
    pb = radar_to_prox_bearing(radar)
    if pb is None:
        return None
    return math.atan2(pb[1], pb[2])


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
    """One gate1_mode1 JSONL -> per-frame arrays (obs[277], executed cmd, energy, thirst, food/water
    radar decodes, death-cause). Returns None if malformed/too short."""
    obs, cmd, energy, thirst, food, water = [], [], [], [], [], []
    done, trunc = False, False
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            o = r.get("obs")
            if not isinstance(o, dict) or "proprio" not in o or "retina" not in o:
                return None
            m = o.get("metrics", {})
            obs.append(list(o["proprio"]) + list(o["retina"]) + [float(o["energy"]) / 100.0])
            cmd.append([float(m.get("cmd_vx", 0.0)), float(m.get("cmd_omega", 0.0))])
            energy.append(float(o["energy"]))
            thirst.append(float(o["thirst"]))
            food.append(radar_to_prox_bearing(o.get("vision_fine", [])))
            water.append(radar_to_prox_bearing(o.get("vision_water", [])))
            done = done or bool(r.get("done"))
            trunc = trunc or bool(r.get("truncated"))
    if len(obs) < 32 or any(len(x) != OBS_DIM for x in obs):
        return None
    e, t = energy[-1], thirst[-1]
    cause = "trunc" if (trunc and not done) else ("energy" if e <= t else "thirst")
    return {"obs": np.asarray(obs, dtype=np.float32),
            "cmd": np.asarray(cmd, dtype=np.float32),
            "energy": np.asarray(energy, dtype=np.float32),
            "thirst": np.asarray(thirst, dtype=np.float32),
            "food": food, "water": water,
            "done": done, "trunc": trunc, "cause": cause}


def survival_labels(ep: dict) -> tuple[np.ndarray, np.ndarray]:
    """G_t (survival-return-to-go, normalized [0,1]) and surv100 (binary) per frame.

    G_t = 1 - gamma^(remaining), remaining = L - t (steps from t to EPISODE END). Low near death,
    ~1 when safe. surv100_t = 1 iff the agent is still alive 100 steps from t.

    IMPORTANT (label operationalization, flagged): the logged drives decay ~linearly and the sim
    TERMINATES at drive~=0 (last logged min is ~0.01, never exactly 0) -> a `drive>0` test is
    always true and useless. Death is therefore keyed off the EPISODE-END flag: for a DEATH episode
    (done & not truncated) the agent dies at the last frame, so surv100=0 for frames within 100 of
    the end; a TRUNCATED episode never died -> surv100=1 throughout (and its G under-counts true
    survival near the end = the flagged truncation bias, 1/12 episodes)."""
    L = len(ep["obs"])
    t = np.arange(L)
    remaining = (L - t).astype(np.float64)
    G = (1.0 - GAMMA ** remaining).astype(np.float32)
    is_death = bool(ep.get("done")) and not bool(ep.get("trunc"))
    surv100 = np.ones(L, dtype=np.float32)
    if is_death:
        remaining_to_end = (L - 1) - t             # frames from t until the (death) last frame
        surv100[remaining_to_end <= 100] = 0.0
    return G, surv100


# ---- collect dreamed latents + survival labels ------------------------------------------------
def collect(model, episodes, H, depth_stride, start_stride, cap, rng):
    """Dream H steps under executed commands from strided start frames; emit (latent@depth, G, surv100,
    energy, thirst, eid) with labels taken at the REAL future frame t0+depth."""
    depths = list(range(0, H, depth_stride))
    lat_l, G_l, s100_l, e_l, th_l, eid_l = [], [], [], [], [], []
    for eid, ep in enumerate(episodes):
        L = len(ep["obs"])
        G, s100 = survival_labels(ep)
        starts = [i for i in range(0, L - 1, start_stride)]
        if not starts:
            continue
        O = torch.tensor(np.stack([ep["obs"][i] for i in starts]))
        seqs = np.stack([ep["cmd"][[min(i + t, L - 1) for t in range(H)]] for i in starts])
        C = torch.tensor(seqs)
        with torch.no_grad():
            lat = model.rollout_open_loop(O, C)["predicted_latents"].numpy()  # [S,H,128]
        for si, i in enumerate(starts):
            for d in depths:
                fr = i + d
                if fr > L - 1:
                    break
                lat_l.append(lat[si, d])
                G_l.append(G[fr]); s100_l.append(s100[fr])
                e_l.append(ep["energy"][fr]); th_l.append(ep["thirst"][fr])
                eid_l.append(eid)
    LAT = np.asarray(lat_l, dtype=np.float32)
    G = np.asarray(G_l, dtype=np.float32); S = np.asarray(s100_l, dtype=np.float32)
    E = np.asarray(e_l, dtype=np.float32); TH = np.asarray(th_l, dtype=np.float32)
    EID = np.asarray(eid_l, dtype=np.int64)
    n = len(LAT)
    if cap and n > cap:
        sel = rng.permutation(n)[:cap]
        LAT, G, S, E, TH, EID = LAT[sel], G[sel], S[sel], E[sel], TH[sel], EID[sel]
    return {"lat": LAT, "G": G, "s100": S, "energy": E, "thirst": TH, "eid": EID}


# ---- stratified train/test split BY EPISODE (no leakage) --------------------------------------
def split_episodes(episodes, seed):
    """Stratify by death-cause so both drives appear in train AND test (thirst deaths are rare)."""
    rng = np.random.RandomState(seed)
    by_cause: dict[str, list[int]] = {}
    for i, ep in enumerate(episodes):
        by_cause.setdefault(ep["cause"], []).append(i)
    test = set()
    for cause, ids in by_cause.items():
        ids = list(ids); rng.shuffle(ids)
        k = max(1, int(round(0.3 * len(ids)))) if len(ids) > 1 else 0
        test.update(ids[:k])
    train = [i for i in range(len(episodes)) if i not in test]
    return train, sorted(test)


# ---- value head training (MSE on G in [0,1]) --------------------------------------------------
def train_value(data, train_eids, test_eids, seed, steps=800):
    torch.manual_seed(seed)
    tr = np.isin(data["eid"], list(train_eids))
    te = np.isin(data["eid"], list(test_eids))
    LAT = torch.tensor(data["lat"]); G = torch.tensor(data["G"])
    head = ValueHead(LATENT_DIM)
    head.mu.copy_(LAT[tr].mean(0)); head.sd.copy_(LAT[tr].std(0) + 1e-6)
    opt = torch.optim.Adam(head.parameters(), lr=2e-3, weight_decay=1e-4)
    Xtr, Ytr = LAT[tr], G[tr]
    for _ in range(steps):
        head.train(); opt.zero_grad()
        loss = torch.nn.functional.mse_loss(head.value(Xtr), Ytr)
        loss.backward(); opt.step()
    head.eval()
    with torch.no_grad():
        pred_te = head.value(LAT[te]).numpy()
    return head, te, pred_te


def r2(y, p):
    y = y.reshape(-1); p = p.reshape(-1)
    ss_res = ((y - p) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def auc(score, label):
    s = np.asarray(score).reshape(-1); l = np.asarray(label).reshape(-1)
    o = np.argsort(s); rk = np.empty_like(s); rk[o] = np.arange(1, len(s) + 1)
    npos, nneg = l.sum(), (1 - l).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((rk[l == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


# ---- G2b arbitration -------------------------------------------------------------------------
def arbitration_test(model, head, episodes, k_pursuit, T, vx, low, high, seed, max_frames):
    """On frames with EXACTLY ONE drive low (and both resources visible), compare value of dreaming
    toward the depleted resource vs toward the other. Return fraction depleted>other + coverage."""
    rng = np.random.RandomState(seed)
    cands = []  # (eid, t, depleted_bearing, other_bearing)
    for eid, ep in enumerate(episodes):
        L = len(ep["obs"])
        for t in range(L - 1):
            e, th = ep["energy"][t], ep["thirst"][t]
            fb = radar_bearing_angle_from(ep["food"][t])
            wb = radar_bearing_angle_from(ep["water"][t])
            if fb is None or wb is None:
                continue
            if e < low and th > high:          # energy depleted -> food is the depleted resource
                cands.append((eid, t, fb, wb))
            elif th < low and e > high:        # thirst depleted -> water is the depleted resource
                cands.append((eid, t, wb, fb))
    rng.shuffle(cands)
    if max_frames and len(cands) > max_frames:
        cands = cands[:max_frames]
    if not cands:
        return {"n": 0, "frac": float("nan")}
    obs0 = torch.tensor(np.stack([episodes[eid]["obs"][t] for eid, t, _, _ in cands]))

    def dream(bearings):
        om = np.clip(k_pursuit * np.asarray(bearings), -0.6, 0.6)
        cmds = np.zeros((len(cands), T, 2), dtype=np.float32)
        cmds[:, :, 0] = vx
        cmds[:, :, 1] = om[:, None]
        with torch.no_grad():
            lat = model.rollout_open_loop(obs0, torch.tensor(cmds))["predicted_latents"]
            return lat, head.value(lat).mean(dim=1).numpy(), om   # latents[B,T,128], mean-value, omega

    lat_dep, v_dep, om_dep = dream([c[2] for c in cands])
    lat_oth, v_oth, om_oth = dream([c[3] for c in cands])
    frac = float((v_dep > v_oth).mean())
    is_elow = np.array([episodes[c[0]]["energy"][c[1]] < low for c in cands])
    n_e = int(is_elow.sum())
    frac_elow = float((v_dep[is_elow] > v_oth[is_elow]).mean()) if n_e else float("nan")
    frac_tlow = float((v_dep[~is_elow] > v_oth[~is_elow]).mean()) if (~is_elow).any() else float("nan")
    # --- INTERPRETABILITY probes (rule out §2 self-deception): are the two dreams even different,
    #     and does the value have any range?  If the dreams barely diverge -> the WM open-loop dream
    #     is direction-BLIND (known limitation) and G2b is inconclusive, NOT "value can't arbitrate".
    lat_div = float((lat_dep - lat_oth).pow(2).sum(-1).sqrt().mean())   # mean per-step L2 latent gap
    lat_scale = float(lat_dep.pow(2).sum(-1).sqrt().mean())             # typical latent norm (ref)
    val_gap = float(np.abs(v_dep - v_oth).mean())                       # mean |Δvalue| between plans
    val_std = float(np.concatenate([v_dep, v_oth]).std())              # value spread across frames
    om_gap = float(np.abs(om_dep - om_oth).mean())                      # mean |Δomega| commanded
    return {"n": len(cands), "frac": frac, "n_energy_low": n_e, "n_thirst_low": len(cands) - n_e,
            "frac_elow": frac_elow, "frac_tlow": frac_tlow,
            "mean_v_dep": float(v_dep.mean()), "mean_v_oth": float(v_oth.mean()),
            "lat_div": lat_div, "lat_scale": lat_scale, "val_gap": val_gap,
            "val_std": val_std, "om_gap": om_gap}


def radar_bearing_angle_from(pb) -> float | None:
    if pb is None:
        return None
    return math.atan2(pb[1], pb[2])


# ---- drive-sensitivity ------------------------------------------------------------------------
def drive_sensitivity(head, data, te):
    LAT = torch.tensor(data["lat"][te])
    with torch.no_grad():
        v = head.value(LAT).numpy()
    e = data["energy"][te]; th = data["thirst"][te]
    def corr(a, b):
        if a.std() < 1e-9 or b.std() < 1e-9:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])
    corr_e = corr(v, e); corr_th = corr(v, th)
    def delta(drive):
        lo = v[drive < 30]; hi = v[drive > 70]
        if len(lo) < 5 or len(hi) < 5:
            return float("nan"), len(lo), len(hi)
        return float(lo.mean() - hi.mean()), len(lo), len(hi)
    d_e, nlo_e, nhi_e = delta(e)
    d_th, nlo_th, nhi_th = delta(th)
    return {"corr_energy": corr_e, "corr_thirst": corr_th,
            "dval_lowE": d_e, "n_lowE": nlo_e, "n_hiE": nhi_e,
            "dval_lowTh": d_th, "n_lowTh": nlo_th, "n_hiTh": nhi_th}


# ---- selfcheck --------------------------------------------------------------------------------
def selfcheck():
    print("[selfcheck] building WMs + rollout dims...")
    for wm in DEFAULT_WMS:
        model, info = build_wm_from_ckpt(wm)
        assert info["obs_dim"] == OBS_DIM, f"obs_dim {info['obs_dim']} != 277 ({wm})"
        B, H = 4, 31
        with torch.no_grad():
            out = model.rollout_open_loop(torch.randn(B, OBS_DIM), torch.randn(B, H, 2))
        lat = out["predicted_latents"]
        assert lat.shape == (B, H, LATENT_DIM), f"latent shape {tuple(lat.shape)} != (4,31,128)"
        assert torch.isfinite(lat).all(), "non-finite latents"
        print(f"  OK {wm} arch={info['predictor_arch']} slot={info['with_slot']} latent={tuple(lat.shape)}")
    # G monotone on a synthetic episode (decreasing toward death)
    fake = {"obs": np.zeros((400, OBS_DIM), np.float32),
            "energy": np.linspace(100, 0, 400).astype(np.float32),
            "thirst": np.full(400, 80.0, np.float32), "done": True, "trunc": False}
    G, s100 = survival_labels(fake)
    assert np.all(np.diff(G) <= 1e-6), "G not monotone-decreasing toward death"
    assert 0.0 <= G.min() and G.max() <= 1.0, "G out of [0,1]"
    assert s100[-1] == 0.0 and s100[0] == 1.0, f"surv100 endpoints wrong: {s100[0]},{s100[-1]}"
    assert 0.0 < s100.mean() < 1.0, f"surv100 has no pos/neg mix: {s100.mean()}"
    print(f"  OK survival labels: G[0]={G[0]:.3f} G[-1]={G[-1]:.4f} monotone; surv100 0/1 endpoints")
    # radar decode finite
    gt = radar_to_prox_bearing([0.0, 0.0, 0.6, 0.0, 0.3, 0.0])
    assert gt is not None and all(math.isfinite(v) for v in gt), "radar decode non-finite"
    assert radar_to_prox_bearing([0.0, 0.0, 0.0]) is None, "all-zero radar should be None"
    ang = radar_bearing_angle([0.0, 0.0, 0.9, 0.0])
    assert ang is not None and math.isfinite(ang), "bearing angle non-finite"
    print(f"  OK radar decode -> prox/sin/cos {tuple(round(v,3) for v in gt)}; bearing {ang:.3f} rad")
    # value head trains one step (loss finite, decreases)
    torch.manual_seed(0)
    X = torch.randn(400, LATENT_DIM); w = torch.randn(LATENT_DIM)
    y = torch.sigmoid(X @ w / 4)
    head = ValueHead(LATENT_DIM); head.mu.copy_(X.mean(0)); head.sd.copy_(X.std(0) + 1e-6)
    opt = torch.optim.Adam(head.parameters(), lr=2e-3)
    l0 = torch.nn.functional.mse_loss(head.value(X), y).item()
    for _ in range(50):
        opt.zero_grad(); loss = torch.nn.functional.mse_loss(head.value(X), y); loss.backward(); opt.step()
    assert loss.item() < l0, f"value head did not train: {l0:.4f}->{loss.item():.4f}"
    assert math.isfinite(loss.item()), "value loss non-finite"
    print(f"  OK value head trains: MSE {l0:.4f} -> {loss.item():.4f}")
    # metric sanity
    assert abs(r2(y.numpy(), y.numpy()) - 1.0) < 1e-6, "R² of perfect fit != 1"
    assert auc(np.array([0.1, 0.9, 0.4, 0.8]), np.array([0, 1, 0, 1])) == 1.0, "AUC sanity failed"
    print("  OK r2/auc metric sanity")
    print("[selfcheck] PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm-ckpt", nargs="*", default=DEFAULT_WMS)
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--episodes", type=int, default=12, help="max episodes to load")
    ap.add_argument("--horizon", type=int, default=30, help="dream depth H (G2a)")
    ap.add_argument("--depth-stride", type=int, default=6)
    ap.add_argument("--start-stride", type=int, default=8)
    ap.add_argument("--cap", type=int, default=25000, help="max #latents (subsample)")
    ap.add_argument("--seed", type=int, default=0)
    # G2b
    ap.add_argument("--arb-low", type=float, default=40.0)
    ap.add_argument("--arb-high", type=float, default=60.0)
    ap.add_argument("--arb-k", type=float, default=1.0, help="pursuit gain omega=clamp(k*bearing,±0.6)")
    ap.add_argument("--arb-T", type=int, default=20)
    ap.add_argument("--arb-vx", type=float, default=0.65)
    ap.add_argument("--arb-max-frames", type=int, default=2000)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.selfcheck:
        selfcheck(); return

    data_dir = Path(args.data_dir)
    ep_paths = sorted(data_dir.glob("*.jsonl"))[:args.episodes]
    print(f"[data] {data_dir}: loading {len(ep_paths)} episodes")
    episodes = []
    for p in ep_paths:
        ep = load_episode(p)
        if ep is None:
            print(f"  skip {p.name} (malformed/short)"); continue
        episodes.append(ep)
    tot = sum(len(e["obs"]) for e in episodes)
    causes = {}
    for e in episodes:
        causes[e["cause"]] = causes.get(e["cause"], 0) + 1
    print(f"[data] {len(episodes)} episodes, {tot} frames, death-cause {causes}")
    train_eids, test_eids = split_episodes(episodes, args.seed)
    print(f"[split] train episodes {train_eids} | test episodes {test_eids}")
    print(f"[note] thirst deaths are RARE ({causes.get('thirst',0)}/{len(episodes)}) -> thirst-side "
          f"sensitivity/arbitration is under-powered; flag in verdict.")

    rng = np.random.RandomState(args.seed)
    for wm in args.wm_ckpt:
        model, info = build_wm_from_ckpt(wm)
        print(f"\n================ WM: {wm}  arch={info['predictor_arch']} slot={info['with_slot']} ================")
        data = collect(model, episodes, args.horizon, args.depth_stride, args.start_stride, args.cap, rng)
        print(f"[collect] {len(data['lat'])} dreamed latents (H={args.horizon}, depth_stride="
              f"{args.depth_stride}, start_stride={args.start_stride}); G mean={data['G'].mean():.3f} "
              f"surv100 pos={100*data['s100'].mean():.1f}%")
        head, te, pred_te = train_value(data, train_eids, test_eids, args.seed)
        # ---- G2a ----
        Gte = data["G"][te]; Ste = data["s100"][te]
        r2_val = r2(Gte, pred_te)
        auc_val = auc(pred_te, Ste)
        ds = drive_sensitivity(head, data, te)
        print("\n--- G2a: survival value on dreamed latent (held-out by episode) ---")
        print(f"  R²(value vs G)        = {r2_val:+.3f}   (>0.3 = non-trivial)")
        print(f"  AUC(value vs surv100) = {auc_val:.3f}   (held-out surv100 pos={100*Ste.mean():.1f}%)")
        print("  drive-sensitivity:")
        print(f"    corr(value, energy) = {ds['corr_energy']:+.3f}   corr(value, thirst) = {ds['corr_thirst']:+.3f}")
        print(f"    Δvalue low-E vs hi-E  = {ds['dval_lowE']:+.3f}  (n {ds['n_lowE']}/{ds['n_hiE']}; expect <0)")
        print(f"    Δvalue low-Th vs hi-Th= {ds['dval_lowTh']:+.3f}  (n {ds['n_lowTh']}/{ds['n_hiTh']}; expect <0)")
        both_drives = (ds['dval_lowE'] < -0.01) and (ds['dval_lowTh'] < -0.01) \
            and math.isfinite(ds['corr_thirst']) and ds['corr_thirst'] > 0.02
        # NOTE: G saturates (~95% of frames are far from death, G~1) so R² is fragile; AUC(surv100),
        # a danger-detection metric, is the ROBUST readout. "predictive" = non-trivial by EITHER.
        predictive = (r2_val > 0.3) or (math.isfinite(auc_val) and auc_val > 0.70)
        # ---- G2b (only if G2a predictive) ----
        arb = None
        if predictive:
            arb = arbitration_test(model, head, episodes, args.arb_k, args.arb_T, args.arb_vx,
                                   args.arb_low, args.arb_high, args.seed, args.arb_max_frames)
            print("\n--- G2b: arbitration (one drive low, both resources visible) ---")
            if arb["n"] == 0:
                print("  NO qualifying frames (need one drive <low, other >high, both radars visible).")
            else:
                print(f"  frames={arb['n']} (energy-low {arb['n_energy_low']}, thirst-low {arb['n_thirst_low']})")
                print(f"  mean value: toward-depleted={arb['mean_v_dep']:.4f}  toward-other={arb['mean_v_oth']:.4f}")
                print(f"  frac[ value(toward-depleted) > value(toward-other) ] = {arb['frac']:.3f}   (>0.65 = arbitrates)")
                print(f"    by drive: energy-low frac={arb['frac_elow']:.3f} | thirst-low frac={arb['frac_tlow']:.3f} "
                      f"(asymmetry exposes energy-bias)")
                print(f"  [probe] Δomega commanded={arb['om_gap']:.3f} | dream latent-divergence={arb['lat_div']:.3f} "
                      f"(latent-norm ref {arb['lat_scale']:.3f}, ratio {arb['lat_div']/(arb['lat_scale']+1e-6):.3f})")
                print(f"  [probe] |Δvalue| between plans={arb['val_gap']:.4f} | value-spread(std)={arb['val_std']:.4f}")
                if arb['lat_div'] / (arb['lat_scale'] + 1e-6) < 0.05:
                    print("  [probe] !! dreams barely diverge -> WM open-loop dream is direction-BLIND -> G2b INCONCLUSIVE "
                          "(bottleneck = WM dream, not the value readout).")
        else:
            print("\n--- G2b: SKIPPED (G2a not predictive: R²<=0.3 AND AUC<=0.70) ---")
        # ---- verdict ----
        print("\n=== VERDICT ===")
        print(f"  G2a R²={r2_val:+.3f} (R²>0.3? {r2_val>0.3}) | AUC(surv100)="
              f"{auc_val:.3f} (>0.70? {math.isfinite(auc_val) and auc_val>0.70}) "
              f"-> predictive={predictive}; both-drive-sensitive={both_drives}")
        if arb and arb["n"] > 0:
            print(f"  G2b frac={arb['frac']:.3f} ({'arbitrates' if arb['frac']>0.65 else 'NO arbitration'})")
        if predictive and both_drives and arb and arb["n"] > 0 and arb["frac"] > 0.65:
            print("  --> PIVOT VIABLE on the raw latent (cheap). Value predicts multi-drive survival AND")
            print("      arbitrates. Proceed to planner integration (G3: learned value -> planner cost).")
        elif not predictive:
            print("  --> value NOT predictive on the raw latent -> FALLBACK = 2-resource slot (retrain).")
        elif not both_drives:
            print("  --> value tracks one drive / IGNORES thirst -> can't arbitrate for water ->")
            print("      FALLBACK = 2-resource slot (retrain). (Caveat: thirst deaths rare in data.)")
        elif not arb or arb["n"] == 0:
            print("  --> arbitration UNTESTABLE (no qualifying frames) -> inconclusive; need targeted data.")
        else:
            print("  --> value predicts survival but does NOT arbitrate (frac~0.5; lossiness bites) ->")
            print("      FALLBACK = 2-resource slot (retrain).")


if __name__ == "__main__":
    sys.exit(main())
