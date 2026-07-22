"""Does the frozen WM still work under a REAL cone? — FREE (0 run, 0 Godot, 0 training).

THE DEBT THIS PAYS. On 2026-07-22 the retina was changed from 360 degrees to a 120-degree cone by
REDISTRIBUTING the 36 rays (perception.gd), and memory then produced +2.17 meals. But the WM was
trained on 360-degree retinas: under a cone it sees a distribution it has never seen (rays packed
forward, a different emptiness pattern). Nothing was measured. The lives-level result stands either
way — it was measured in lives — but the ARCHITECTURAL claim does not: is the WM still doing its
job, or is the geometric slot carrying the whole thing on its own?

WHAT IS MEASURED, and why it needs no extra ground truth:

  1. SLOT FIDELITY. The retina IS the ground truth of perception. If some ray sees food-red at
     bearing theta and distance d, the slot's soft-argmax must land near (d*sin theta, d*cos theta).
     We compare the slot's own output to that geometric reference, computed with the angle table
     that matches the FOV actually served. A slot that transfers keeps the same error under a cone.
     This is the measure that matters most: the planner localises food through the slot, not
     through the latent.

  2. LATENT DISPLACEMENT. The slot could be fine while the encoder is out of distribution. We
     therefore also check that the retina still lands where the encoder expects it, by comparing
     the raw retina statistics the encoder was trained on (occupancy, depth profile) between the
     two regimes. This does not prove the latent is healthy — it bounds how far it has moved.

WHAT THIS DIAG DOES NOT SAY. It does not judge the entity, and it cannot prove the WM's dynamics
are intact — only that the perception path it actually uses is. A large slot error under the cone
would mean the +2.17 came from something other than a working perception; a small one means the
geometric transfer argued for in slot_head (attention scores [depth,R,G,B], angles are a buffer)
holds up in practice.

Usage:
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_wm_cone_fidelity.py [--selfcheck]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st

import torch

RB = "/home/edgarbrunet/Documents/PERSO/SylvanV1/data/replay_buffer"
WM_CKPT = "data/checkpoints/wm_objcentric_kin/wm_best.pt"
NRAY = 36
RANGE_M = 10.0
DEPTH_OFFSET = 0.35
RED = (1.0, 0.0, 0.0)
THR = 0.55


def ray_angles(fov_deg: float) -> list[float]:
    """Bearing of each ray, EXACTLY as perception.gd lays them out: ray 0 forward, index growing
    rightwards, wrapping to negative. At 360 this reproduces the historical k*TAU/N."""
    fov = math.radians(fov_deg)
    return [(k if k <= NRAY // 2 else k - NRAY) * fov / NRAY for k in range(NRAY)]


def load_slot_encoder(fov_deg: float):
    """Load the frozen WM's slot encoder and set its angle table to the FOV actually served.

    The sin/cos buffers are PERSISTENT: load_state_dict restores the 360-degree table, so a cone
    retina would otherwise be decoded with the wrong angles and return wrong coordinates silently.
    serve_planner_command does the same fix-up; we replicate it so the diag measures what runs.
    """
    from sylvan.models.command_wm import CommandWorldModel

    payload = torch.load(WM_CKPT, map_location="cpu", weights_only=False)
    meta = payload["meta"]
    # construction IDENTIQUE a serve_planner_command : sinon le slot_encoder n existe pas
    wm = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
                           predictor_arch=meta.get("predictor_arch", "shallow"),
                           with_slot=meta.get("with_slot", False),
                           slot_resources=meta.get("slot_resources", 1))
    wm.load_state_dict(payload["model"])
    wm.eval()
    enc = wm.slot_encoder
    if abs(fov_deg - 360.0) > 1e-6:
        th = torch.tensor(ray_angles(fov_deg), dtype=torch.float32)
        with torch.no_grad():
            enc.sin.copy_(torch.sin(th))
            enc.cos.copy_(torch.cos(th))
    return enc


def geometric_food(retina: list[float], angles: list[float]) -> tuple[float, float] | None:
    """Saliency-weighted centroid of the food-red rays — the reference the slot should reproduce."""
    wx = wz = wsum = 0.0
    for k in range(NRAY):
        depth, r, g, b = retina[k * 4: k * 4 + 4]
        if depth >= 0.999:
            continue
        n = math.sqrt(r * r + g * g + b * b)
        if n < 1e-6 or r / n < THR:
            continue
        sat = max(r, g, b) - min(r, g, b)
        d = depth * RANGE_M + DEPTH_OFFSET
        wx += sat * d * math.sin(angles[k])
        wz += sat * d * math.cos(angles[k])
        wsum += sat
    return (wx / wsum, wz / wsum) if wsum > 0 else None


def read_retinas(tag: str, nmax: int) -> list[list[float]]:
    out: list[list[float]] = []
    for f in sorted(glob.glob(f"{RB}/{tag}/*.jsonl")):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line).get("wm", {}).get("retina0")
            if r:
                out.append(r)
            if len(out) >= nmax:
                return out
    return out


def assess(tag: str, fov_deg: float, nmax: int = 4000) -> dict | None:
    retinas = read_retinas(tag, nmax)
    if not retinas:
        return None
    angles = ray_angles(fov_deg)
    enc = load_slot_encoder(fov_deg)

    errs: list[float] = []
    occ: list[float] = []
    seen = 0
    batch, refs = [], []
    for ret in retinas:
        occ.append(sum(1 for k in range(NRAY) if ret[k * 4] < 0.999) / NRAY)
        ref = geometric_food(ret, angles)
        if ref is None:
            continue
        seen += 1
        batch.append(ret)
        refs.append(ref)
    if not batch:
        return dict(n=len(retinas), seen=0, occ=st.mean(occ), err=None)

    with torch.no_grad():
        pos = enc.positions(torch.tensor(batch, dtype=torch.float32))
    pos = pos[:, 0, :] if pos.dim() == 3 else pos          # food slot
    for i, (rx, rz) in enumerate(refs):
        errs.append(math.hypot(float(pos[i, 0]) - rx, float(pos[i, 1]) - rz))
    return dict(n=len(retinas), seen=seen, occ=st.mean(occ),
                err_med=st.median(errs), err_p90=sorted(errs)[int(0.9 * len(errs))],
                ref_dist=st.median([math.hypot(*r) for r in refs]))


def selfcheck() -> int:
    a360, a120 = ray_angles(360.0), ray_angles(120.0)
    assert abs(max(a360) - math.pi) < 1e-6, "360 must span the full circle"
    assert abs(math.degrees(max(a120)) - 60.0) < 1e-6, math.degrees(max(a120))  # 120 deg field = +-60
    print(f"  [ok] ray table: 360 spans +-180 deg, cone 120 spans +-{math.degrees(max(a120)):.0f} deg")

    # a single red ray straight ahead must place the reference straight ahead at its distance
    ret = [1.0, 0.0, 0.0, 0.0] * NRAY
    ret[0:4] = [0.5, 0.9, 0.1, 0.1]
    ref = geometric_food(ret, a120)
    assert ref is not None and abs(ref[0]) < 1e-6 and abs(ref[1] - (0.5 * RANGE_M + DEPTH_OFFSET)) < 1e-6, ref
    print(f"  [ok] a single forward red ray -> reference {ref[1]:.2f} m straight ahead")

    # the same ray at index 9 must sit at the cone edge bearing, not at 90 deg
    ret2 = [1.0, 0.0, 0.0, 0.0] * NRAY
    ret2[9 * 4: 9 * 4 + 4] = [0.5, 0.9, 0.1, 0.1]
    r2 = geometric_food(ret2, a120)
    brg = math.degrees(math.atan2(r2[0], r2[1]))
    assert abs(brg - 30.0) < 1e-6, brg
    print(f"  [ok] ray 9 under the cone sits at {brg:.0f} deg (would be 90 deg with the 360 table)")
    print("SELFCHECK PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--n", type=int, default=4000)
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()

    cases = [("arbgrad_graded_s1", 360.0, "360 deg (regime d entrainement du WM)"),
             ("bosq_hw2.0_tr0.015_f120_t6.0_mon_d1_p4_r2500_b8_s1", 120.0, "CONE 120 deg")]
    print(f"  {'regime':<38s} {'ticks':>6s} {'bouffe vue':>11s} {'occupation':>11s} "
          f"{'err med':>9s} {'err p90':>9s} {'dist med':>9s}")
    res = {}
    for tag, fov, label in cases:
        r = assess(tag, fov, a.n)
        res[label] = r
        if r is None:
            print(f"  {label:<38s}  corpus absent")
            continue
        if r.get("err_med") is None:
            print(f"  {label:<38s} {r['n']:>6d} {'0':>11s} {100*r['occ']:>10.0f}%  (jamais de rouge)")
            continue
        print(f"  {label:<38s} {r['n']:>6d} {100*r['seen']/r['n']:>10.0f}% {100*r['occ']:>10.0f}% "
              f"{r['err_med']:>8.3f}m {r['err_p90']:>8.3f}m {r['ref_dist']:>8.2f}m")

    vals = [r for r in res.values() if r and r.get("err_med") is not None]
    if len(vals) == 2:
        a_, b_ = vals
        print(f"\n  ecart d erreur du slot cone - 360 : {b_['err_med']-a_['err_med']:+.3f} m")
        print("  LECTURE : le slot decode par ATTENTION GEOMETRIQUE sur des angles CONNUS, et son")
        print("  score d attention lit [depth,R,G,B] SANS l angle. S il transfere, l erreur doit")
        print("  rester du meme ordre — c est ce que l argument de slot_head predit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
