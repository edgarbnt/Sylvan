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
                wr = w.get("water_rel0") or [0.0, 0.0, 0.0]          # absent sur les runs mono → present=0
                if not ret or not fr or not t0 or len(ret) != 144:
                    continue
                seq.append((ret, float(fr[0]), float(fr[1]), float(fr[2]), t0[0], t0[1], t0[2],
                            float(wr[0]), float(wr[1]), float(wr[2])))
            if len(seq) > gap + 2:
                eps.append(seq)
    ntr = max(1, int(0.8 * len(eps)))
    cols = {k: [] for k in ("ra", "rb", "dy", "df", "dl", "fx", "fz", "tr", "wx", "wz", "wp")}
    for ei, seq in enumerate(eps):
        is_tr = ei < ntr
        for i in range(len(seq) - gap):
            a = seq[i]; b = seq[i + gap]
            if (a[3] < 0.5 and a[9] < 0.5) or (b[3] < 0.5 and b[9] < 0.5):
                continue                                            # au moins UNE ressource visible aux 2 bouts
            x0, z0, y0 = a[4], a[5], a[6]; x1, z1, y1 = b[4], b[5], b[6]
            dx, dz = x1 - x0, z1 - z0
            cols["ra"].append(a[0]); cols["rb"].append(b[0]); cols["dy"].append(wrap(y1 - y0))
            cols["df"].append(dx * math.sin(y0) + dz * math.cos(y0))
            cols["dl"].append(dx * math.cos(y0) - dz * math.sin(y0))
            cols["fx"].append(b[1]); cols["fz"].append(b[2]); cols["tr"].append(is_tr)
            cols["wx"].append(b[7]); cols["wz"].append(b[8]); cols["wp"].append(b[9])
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
    ap.add_argument("--n-resources", type=int, default=1)
    args = ap.parse_args()
    torch.manual_seed(args.seed); torch.set_num_threads(4)

    t, nep = load_pairs(args.runs, args.gap)
    tr = t["tr"].bool(); te = ~tr
    print(f"[slot] runs={args.runs} épisodes={nep} paires={len(t['ra'])} (train={int(tr.sum())} test={int(te.sum())})")
    K = args.n_resources
    head = SelfSupervisedSlotHead(n_resources=K)
    opt = torch.optim.Adam(head.parameters(), args.lr)
    ra, rb, dy, df, dl = (t[k][tr] for k in ("ra", "rb", "dy", "df", "dl"))
    N = len(ra)

    def slots(ret):
        return head.positions(ret)                                  # [N,K,2]

    for it in range(args.iters):
        bi = torch.randint(0, N, (256,))
        sa = slots(ra[bi]); sb = slots(rb[bi])
        cons = sum(((transport(sa[:, k], dy[bi], df[bi], dl[bi]) - sb[:, k].detach()) ** 2).sum(1).mean()
                   for k in range(K)) / K
        vv, vc = vicreg_terms(torch.cat([sa.reshape(-1, 2), sb.reshape(-1, 2)], 0), gamma=1.0)
        loss = cons + vv + vc
        if K > 1:                                                    # répulsion inter-slots (diag_fpure2)
            rep = sum(torch.exp(-((sa[:, i] - sa[:, j]) ** 2).sum(1)).mean()
                      for i in range(K) for j in range(i + 1, K))
            loss = loss + 0.3 * rep
        loss.backward(); opt.step(); opt.zero_grad()

    # ÉVAL (labels food_rel0/water_rel0 utilisés ICI SEULEMENT, jamais à l'entraînement).
    # Assignation slot→ressource LABEL-FREE via la masse d'attention couleur (rouge=bouffe, bleu=eau).
    with torch.no_grad():
        cm = head.color_masses(t["rb"][te]).mean(0)                  # [K,2] (rouge, bleu)
        food_idx = int(cm[:, 0].argmax()); water_idx = int(cm[:, 1].argmax()) if K > 1 else food_idx
        if K > 1 and food_idx == water_idx:
            water_idx = 1 - food_idx                                 # dégénéré → forcer distinct + le signaler
            print(f"[slot] ⚠️ assignation couleur dégénérée (masses {cm.tolist()}) — slots non spécialisés ?")
        sp = slots(t["rb"][te])
        def mae(idx, X, Z, present=None):
            m = present.bool() if present is not None else torch.ones(len(X), dtype=torch.bool)
            if int(m.sum()) == 0:
                return float("nan"), float("nan")
            s2 = sp[m, idx]; x, z = X[m], Z[m]
            bt = torch.atan2(x, z); bp = torch.atan2(s2[:, 0], s2[:, 1])
            b = math.degrees(torch.atan2(torch.sin(bp - bt), torch.cos(bp - bt)).abs().mean())
            return b, float(((s2[:, 0] - x) ** 2 + (s2[:, 1] - z) ** 2).sqrt().mean())
        bmae, pmae = mae(food_idx, t["fx"][te], t["fz"][te])
        wb, wp_mae = mae(water_idx, t["wx"][te], t["wz"][te], t["wp"][te]) if K > 1 else (float("nan"),) * 2
    print(f"[slot] held-out : BOUFFE(slot{food_idx}) bearing {bmae:.1f}° / pos {pmae:.2f} m"
          + (f" | EAU(slot{water_idx}) bearing {wb:.1f}° / pos {wp_mae:.2f} m" if K > 1 else ""))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ck = {"state_dict": head.state_dict(), "n_resources": K, "heldout_mae_m": pmae,
          "heldout_bearing_deg": bmae, "heldout_water_mae_m": (wp_mae if K > 1 else None),
          "food_idx": food_idx, "water_idx": (water_idx if K > 1 else None),
          "runs": args.runs, "gap": args.gap, "self_supervised": True}
    torch.save(ck, out / "slot_best.pt")
    print(f"[slot] sauvé → {out / 'slot_best.pt'}")


if __name__ == "__main__":
    main()
