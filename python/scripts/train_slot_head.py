"""Entraîne le SLOT de perception AUTO-SUPERVISÉ (slot_head.py) — ZÉRO label de position.

Signal = consistance de transport sous l'ego-motion (équivariance) : transport(slot_t, ego-motion_{t→t+gap}) ≈
stop-grad(slot_{t+gap}), + VICReg (anti-collapse). L'ego-motion vraie vient de torso0 (frames consécutives). La
position de l'objet ÉMERGE (l'attention apprend à pointer l'objet) sans qu'on la donne. Le label food_rel0 ne sert
QU'À L'ÉVAL (MAE), jamais à l'entraînement (honnêteté §2, comme le débranchement oracle de retina_head).

Usage (depuis la racine, PYTHONPATH=python) :
  python -m scripts.train_slot_head --runs retina_eat_a retina_forage retina_wm_a \
      --out data/checkpoints/slot_head --gap 8 --iters 12000
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import torch
from torch import nn

from sylvan.models.command_wm import vicreg_terms
from sylvan.models.slot_head import RANGE, SelfSupervisedSlotHead


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _buf_files(name: str):
    return sorted(glob.glob(f"godot/data/replay_buffer/{name}/episode_*.jsonl") or
                  glob.glob(f"data/replay_buffer/{name}/episode_*.jsonl"))


def load_pairs(runs, gap):
    eps = []
    for run in runs:
        for f in _buf_files(run):
            seq = []
            for line in open(f):
                w = json.loads(line).get("wm", {})
                ret = w.get("retina0"); fr = w.get("food_rel0"); t0 = w.get("torso0")
                if not ret or not fr or not t0 or len(ret) != 144:
                    continue
                seq.append((ret, float(fr[0]), float(fr[1]), float(fr[2]), t0[0], t0[1], t0[2]))
            if len(seq) > gap + 2:
                eps.append(seq)
    ntr = max(1, int(0.8 * len(eps)))
    cols = {k: [] for k in ("ra", "rb", "dy", "df", "dl", "fx", "fz", "tr")}
    for ei, seq in enumerate(eps):
        is_tr = ei < ntr
        for i in range(len(seq) - gap):
            a = seq[i]; b = seq[i + gap]
            if a[3] < 0.5 or b[3] < 0.5:
                continue
            x0, z0, y0 = a[4], a[5], a[6]; x1, z1, y1 = b[4], b[5], b[6]
            dx, dz = x1 - x0, z1 - z0
            cols["ra"].append(a[0]); cols["rb"].append(b[0]); cols["dy"].append(wrap(y1 - y0))
            cols["df"].append(dx * math.sin(y0) + dz * math.cos(y0))
            cols["dl"].append(dx * math.cos(y0) - dz * math.sin(y0))
            cols["fx"].append(b[1]); cols["fz"].append(b[2]); cols["tr"].append(is_tr)
    t = {k: torch.tensor(v) for k, v in cols.items()}
    return t, len(eps)


def transport(p, dyaw, dfwd, dlat):    # Rot(+dyaw) — convention validée (diag_fpure3 : dead-reckon +0.95)
    px = p[:, 0] - dlat; pz = p[:, 1] - dfwd
    ca, sa = torch.cos(dyaw), torch.sin(dyaw)
    return torch.stack([ca * px - sa * pz, sa * px + ca * pz], dim=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["retina_eat_a"])
    ap.add_argument("--out", default="data/checkpoints/slot_head")
    ap.add_argument("--gap", type=int, default=8)
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); torch.set_num_threads(4)

    t, nep = load_pairs(args.runs, args.gap)
    tr = t["tr"].bool(); te = ~tr
    print(f"[slot] runs={args.runs} épisodes={nep} paires={len(t['ra'])} (train={int(tr.sum())} test={int(te.sum())})")
    head = SelfSupervisedSlotHead(n_resources=1)
    opt = torch.optim.Adam(head.parameters(), args.lr)
    ra, rb, dy, df, dl = (t[k][tr] for k in ("ra", "rb", "dy", "df", "dl"))
    N = len(ra)

    def slot(ret):
        return head.positions(ret)[:, 0, :]    # [N,2] (n_res=1)

    for it in range(args.iters):
        bi = torch.randint(0, N, (256,))
        sa = slot(ra[bi]); sb = slot(rb[bi])
        cons = ((transport(sa, dy[bi], df[bi], dl[bi]) - sb.detach()) ** 2).sum(1).mean()
        vv, vc = vicreg_terms(torch.cat([sa, sb], 0), gamma=1.0)
        (cons + vv + vc).backward(); opt.step(); opt.zero_grad()

    # ÉVAL (label food_rel0 utilisé ICI SEULEMENT, jamais à l'entraînement)
    with torch.no_grad():
        sp = slot(t["rb"][te]); fx, fz = t["fx"][te], t["fz"][te]
        bt = torch.atan2(fx, fz); bp = torch.atan2(sp[:, 0], sp[:, 1])
        bmae = math.degrees(torch.atan2(torch.sin(bp - bt), torch.cos(bp - bt)).abs().mean())
        pmae = float(((sp[:, 0] - fx) ** 2 + (sp[:, 1] - fz) ** 2).sqrt().mean())
    print(f"[slot] held-out (label-free training) : bearing MAE {bmae:.1f}° | position MAE {pmae:.2f} m")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ck = {"state_dict": head.state_dict(), "n_resources": 1, "heldout_mae_m": pmae,
          "heldout_bearing_deg": bmae, "runs": args.runs, "gap": args.gap, "self_supervised": True}
    torch.save(ck, out / "slot_best.pt")
    print(f"[slot] sauvé → {out / 'slot_best.pt'}")


if __name__ == "__main__":
    main()
