"""G2 of the target-arbitration critic (docs/design_critique_arbitrage.md, form pinned D1 2026-07-20).

Trains THE single head of the pinned voie-A form: P(obtain | s, target), BCE on `got` over the
pursued target of every MULTI replan (both resources visible) from the 12 instrumented runs
(10 G0 runs + arb3/arb4 with flagged epsilon-target counterfactuals). Everything else is frozen
(pain/death heads, saliency lens) or measured (benefit, kappa, drain, restore, 0.02 m/step).

Deployment form (pinned):
  S(t) = dist(t) - 0.02 * max(0, P(obtain|s,t)*benefit(t) - kappa*pain(t)*100 - Pdeath(t)*kappa*100)
  choice = argmin S(t) with pro-incumbent commitment delta = 75 steps * 0.02 = 1.5 m.

Pre-registered offline gates (written in the doc BEFORE this train):
  G-rank  : AUC(P, got) > 0.70, CV-4 by LIFE (epsilon rows included)
  G-res   : simulated-choice accuracy vs bucket-best >= designed + 10 pts (held rows)
  G-consist: replayed flip rate <= 1.2x designed
  G-mono  : P*benefit strictly decreasing with target-drive satiety AND remise increasing
            with urgency (populated buckets)

Run (repo root):
  PYTHONPATH=python env_pytorch_3.12/bin/python -m scripts.train_arb_critic [--selfcheck]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import sys

import torch
from torch import nn

sys.path.insert(0, "diagnostics")
from diag_arbitrage_g0 import (  # single source of truth for the lived-stream conventions
    CONSUME_JUMP, DEATH_THR, FRESH_H, START_DRIVE, START_TOL,
)

from scripts.train_danger_saliency import DangerSaliency, saliency_points
from scripts.train_sprint_critic import SPRINT_IN_DIM, SprintCritic, sprint_inputs
from scripts.train_waypoint_pain import PainCritic
from sylvan.control.waypoint_layer import candidate_features

SEED = 0
CAP_STEPS = 600                    # pinned pursuit-window cap
DELTA_COMMIT_M = 75 * 0.02         # slot-jitter anchor (command_planner COMMITTMENT), reused as-is
M_PER_STEP = 0.02                  # calibrated body
PAIN_CKPT = "data/checkpoints/waypoint_pain_decont/pain_best.pt"
DEATH_CKPT = "data/checkpoints/sprint_critic/death_best.pt"
SAL_CKPT = "data/checkpoints/danger_saliency/saliency_best.pt"
OUT_DIR = "data/checkpoints/arb_critic"

G0_RUNS = [
    "data/replay_buffer/critic_kin_g24as1", "data/replay_buffer/critic_kin_g24as2",
    "data/replay_buffer/critic_kin_g24bs1", "data/replay_buffer/critic_kin_g24bs2",
    "data/replay_buffer/critic_kin_spx3", "data/replay_buffer/critic_kin_spx4",
    "data/replay_buffer/critic_kin_judge1", "data/replay_buffer/critic_kin_judge2",
    "data/replay_buffer/critic_kin_pure1", "data/replay_buffer/critic_kin_pure2",
]
ARB_RUNS = ["data/replay_buffer/critic_kin_arb3", "data/replay_buffer/critic_kin_arb4"]


# ---------------------------------------------------------------- corpus


def _iter_ticks(run: str):
    for ep in sorted(glob.glob(os.path.join(run, "ep_*.jsonl*"))):
        f = gzip.open(ep, "rt") if ep.endswith(".gz") else open(ep)
        with f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def load_decisions(runs: list[str]) -> tuple[list[dict], dict]:
    """One row per MULTI replan (sf/sw + both positions, no waypoint override), labeled `got`.

    Life split = the measured reset signature (e=70,t=70,h=100 reached by a jump), same
    convention as diag_arbitrage_g0 (cross-checked 24/24 vs godot.log there).
    """
    rows: list[dict] = []
    stats = {"lives": 0, "ticks": 0, "multi": 0, "wp_skipped": 0, "no_retina": 0,
             "drain_deltas": [], "restore_jumps": []}
    life_uid = -1

    def process_life(ticks: list[dict]) -> None:
        # consumption events + per-tick drives
        cons: list[tuple[int, str]] = []
        for i in range(1, len(ticks)):
            e0, t0 = ticks[i - 1]["obs"]["energy"], ticks[i - 1]["obs"]["thirst"]
            e1, t1 = ticks[i]["obs"]["energy"], ticks[i]["obs"]["thirst"]
            if e1 - e0 > CONSUME_JUMP:
                cons.append((i, "food"))
                stats["restore_jumps"].append(e1 - e0)
            elif e0 > e1:
                stats["drain_deltas"].append(e0 - e1)
            if t1 - t0 > CONSUME_JUMP:
                cons.append((i, "water"))
                stats["restore_jumps"].append(t1 - t0)
        # multi decisions (positions reliable: skip waypoint-override records)
        decs = []
        for i, r in enumerate(ticks):
            p = r.get("plan")
            if p is None or "sf" not in p or p.get("first") not in ("food", "water"):
                continue
            if "wp" in p:
                stats["wp_skipped"] += 1
                continue
            if "food" not in p or "water" not in p:
                continue
            retina = r["wm"].get("retina0")
            if not retina:
                stats["no_retina"] += 1
                continue
            decs.append((i, p, retina))
        stats["multi"] += len(decs)
        n = len(ticks)
        for k, (i, p, retina) in enumerate(decs):
            chosen = p["first"]
            switch_i = n
            for j in range(k + 1, len(decs)):
                if decs[j][1]["first"] != chosen:
                    switch_i = decs[j][0]
                    break
            end = min(switch_i, n, i + CAP_STEPS)
            got = any(i < ci <= end and ct == chosen for ci, ct in cons)
            o = ticks[i]["obs"]
            rows.append({
                "life": life_uid, "i": i, "left": n - i,
                "e": float(o["energy"]), "t": float(o["thirst"]),
                "h": float(o.get("health", 100.0)),
                "food": tuple(p["food"]), "water": tuple(p["water"]),
                "sf": float(p["sf"]), "sw": float(p["sw"]),
                "chosen": chosen, "explore": bool(p.get("explore_target")),
                "retina": retina, "got": bool(got),
            })

    for run in runs:
        cur: list[dict] = []
        prev_e = prev_t = None
        prev_h = 100.0
        for rec in _iter_ticks(run):
            o = rec["obs"]
            e, t, h = float(o["energy"]), float(o["thirst"]), float(o.get("health", 100.0))
            at_start = (abs(e - START_DRIVE) < START_TOL and abs(t - START_DRIVE) < START_TOL
                        and h >= FRESH_H)
            jumped = prev_e is not None and (abs(e - prev_e) > 1.0 or abs(t - prev_t) > 1.0
                                             or h - prev_h > 1.0)
            if cur and at_start and jumped:
                life_uid += 1
                stats["lives"] += 1
                process_life(cur)
                cur = []
            cur.append(rec)
            stats["ticks"] += 1
            prev_e, prev_t, prev_h = e, t, h
        if cur:
            life_uid += 1
            stats["lives"] += 1
            process_life(cur)
    return rows, stats


# ---------------------------------------------------------------- featurization (parity)


def _load_state(path: str) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    return ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck


def featurize(rows: list[dict]) -> dict:
    """X_food/X_water [N,14] via the EXACT deployment contract; frozen pain/death readouts."""
    sal = DangerSaliency()
    sal.load_state_dict(_load_state(SAL_CKPT))
    sal.eval()
    pain_net = PainCritic()
    pain_net.load_state_dict(_load_state(PAIN_CKPT))
    pain_net.eval()
    death_net = SprintCritic()
    death_net.load_state_dict(_load_state(DEATH_CKPT))
    death_net.eval()

    feats = {"food": [], "water": []}
    with torch.no_grad():
        for r in rows:
            greens = saliency_points(sal, r["retina"])
            for tgt in ("food", "water"):
                pos = r[tgt]
                feats[tgt].append(candidate_features(pos, pos, greens))
    out: dict = {}
    with torch.no_grad():
        for tgt in ("food", "water"):
            f = torch.tensor(feats[tgt], dtype=torch.float32)
            pain = pain_net.pain(f)
            x = torch.stack([
                sprint_inputs([feats[tgt][i]], (rows[i]["e"], rows[i]["t"], rows[i]["h"]),
                              [float(pain[i])])[0]
                for i in range(len(rows))
            ])
            out[f"X_{tgt}"] = x
            out[f"pain_{tgt}"] = pain
            out[f"pdeath_{tgt}"] = death_net.p(x)
    return out


# ---------------------------------------------------------------- form (pinned) + gates


def _auc(score: torch.Tensor, label: torch.Tensor) -> float:
    pos, neg = score[label], score[~label]
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos.unsqueeze(1) > neg.unsqueeze(0)).float().mean()
                 + 0.5 * (pos.unsqueeze(1) == neg.unsqueeze(0)).float().mean())


def benefit_steps(drive: torch.Tensor, restore: float, drain: float) -> torch.Tensor:
    return torch.minimum(torch.full_like(drive, restore), 100.0 - drive) / drain


def remise_m(p_obtain: torch.Tensor, drive: torch.Tensor, pain: torch.Tensor,
             pdeath: torch.Tensor, kappa: float, restore: float, drain: float) -> torch.Tensor:
    ben = benefit_steps(drive, restore, drain)
    return M_PER_STEP * torch.clamp(p_obtain * ben - kappa * pain * 100.0
                                    - pdeath * kappa * 100.0, min=0.0)


def scores_m(idxs: list[int], rows: list[dict], fx: dict, model: SprintCritic, kappa: float,
             restore: float, drain: float) -> tuple[torch.Tensor, torch.Tensor]:
    """S(food), S(water) in meters (lower = better) for the given row indices, with the given
    net — G-res passes each fold's HELD indices with that fold's net (no train/eval leak)."""
    ii = torch.tensor(idxs)
    e = torch.tensor([rows[i]["e"] for i in idxs])
    t = torch.tensor([rows[i]["t"] for i in idxs])
    df = torch.tensor([math.hypot(*rows[i]["food"]) for i in idxs])
    dw = torch.tensor([math.hypot(*rows[i]["water"]) for i in idxs])
    with torch.no_grad():
        pf = model.p(fx["X_food"][ii])
        pw = model.p(fx["X_water"][ii])
    sf = df - remise_m(pf, e, fx["pain_food"][ii], fx["pdeath_food"][ii], kappa, restore, drain)
    sw = dw - remise_m(pw, t, fx["pain_water"][ii], fx["pdeath_water"][ii], kappa, restore, drain)
    return sf, sw


def bucket_of(r: dict) -> tuple[int, int]:
    gap_drive = r["e"] - r["t"]
    gap_dist = math.hypot(*r["food"]) - math.hypot(*r["water"])
    b1 = 0 if gap_drive < -20 else (1 if gap_drive <= 20 else 2)
    b2 = 0 if gap_dist < -1.0 else (1 if gap_dist <= 1.0 else 2)
    return b1, b2


def bucket_best(rows: list[dict], min_n: int = 8) -> dict:
    """Empirically better target per bucket = higher paid-rate among pursued rows."""
    agg: dict = {}
    for r in rows:
        key = bucket_of(r)
        cell = agg.setdefault(key, {"food": [0, 0], "water": [0, 0]})
        cell[r["chosen"]][0] += int(r["got"])
        cell[r["chosen"]][1] += 1
    best = {}
    for key, cell in agg.items():
        if cell["food"][1] >= min_n and cell["water"][1] >= min_n:
            rf = cell["food"][0] / cell["food"][1]
            rw = cell["water"][0] / cell["water"][1]
            if abs(rf - rw) > 1e-9:
                best[key] = "food" if rf > rw else "water"
    return best


def replay_flips(order: list[int], choose) -> int:
    """Flip count over a life's decision sequence under a sequential chooser fn(idx, incumbent)."""
    inc = None
    flips = 0
    for i in order:
        c = choose(i, inc)
        if inc is not None and c != inc:
            flips += 1
        inc = c
    return flips


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=G0_RUNS + ARB_RUNS)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(SEED)

    runs = [r for r in args.runs if glob.glob(os.path.join(r, "ep_*.jsonl*"))]
    rows, stats = load_decisions(runs)
    n_eps = sum(1 for r in rows if r["explore"])
    got_rate = sum(r["got"] for r in rows) / max(len(rows), 1)
    drain = sorted(stats["drain_deltas"])[len(stats["drain_deltas"]) // 2]
    restore = sorted(stats["restore_jumps"])[len(stats["restore_jumps"]) // 2]
    kappa = sorted(r["left"] for r in rows)[len(rows) // 2] / 100.0
    print(f"[corpus] {len(runs)} runs, {stats['lives']} vies, {stats['ticks']} ticks, "
          f"{len(rows)} décisions multi ({n_eps} ε), got={got_rate:.2f}, "
          f"wp-override écartés={stats['wp_skipped']}, sans-rétine={stats['no_retina']}")
    print(f"[mesures] drain={drain:.4f}/pas restore={restore:.2f} κ_data={kappa:.2f} "
          f"δ_commit={DELTA_COMMIT_M:.2f} m")

    if args.selfcheck:
        assert len(rows) >= 5000, "corpus trop petit"
        assert n_eps >= 500, "contrefactuels ε insuffisants"
        assert 0.05 < got_rate < 0.95, "label dégénéré"
        sub = rows[:64]
        fx = featurize(sub)
        assert fx["X_food"].shape == (64, SPRINT_IN_DIM)
        folds = {r["life"] % 4 for r in sub}
        assert folds <= {0, 1, 2, 3}
        assert all(len(candidate_features(r["food"], r["food"], [])) == 10 for r in sub[:4])
        print("[selfcheck] OK")
        return

    fx = featurize(rows)
    y = torch.tensor([float(r["got"]) for r in rows])
    X_pursued = torch.stack([
        fx["X_food"][i] if rows[i]["chosen"] == "food" else fx["X_water"][i]
        for i in range(len(rows))
    ])

    # G-rank : CV-4 by life (fold nets kept for the held-fold G-res below)
    fold_of = torch.tensor([r["life"] % 4 for r in rows])
    aucs = []
    fold_nets: list[SprintCritic] = []
    for f in range(4):
        tr, te = fold_of != f, fold_of == f
        net = SprintCritic()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        lossf = nn.BCEWithLogitsLoss()
        for _ in range(args.epochs):
            opt.zero_grad()
            loss = lossf(net.q(X_pursued[tr]), y[tr])
            loss.backward()
            opt.step()
        with torch.no_grad():
            aucs.append(_auc(net.p(X_pursued[te]), y[te].bool()))
        fold_nets.append(net)
    g_rank = sum(aucs) / 4.0
    print(f"[G-rank] AUC folds = {[f'{a:.3f}' for a in aucs]} -> moyenne {g_rank:.3f} (barre 0.70)")

    # final head on ALL rows (gates below on held folds where applicable)
    model = SprintCritic()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(args.epochs):
        opt.zero_grad()
        loss = lossf(model.q(X_pursued), y)
        loss.backward()
        opt.step()

    sf_all, sw_all = scores_m(list(range(len(rows))), rows, fx, model, kappa, restore, drain)

    # G-res : held-fold accuracy vs bucket-best — bucket table from TRAIN folds only, scores from
    # the fold net that never saw those lives (leak-free).
    ok_des = ok_lrn = n_j = 0
    for f in range(4):
        te_idx = [i for i in range(len(rows)) if rows[i]["life"] % 4 == f]
        tr_rows = [rows[i] for i in range(len(rows)) if rows[i]["life"] % 4 != f]
        best = bucket_best(tr_rows)
        sf_te, sw_te = scores_m(te_idx, rows, fx, fold_nets[f], kappa, restore, drain)
        for k, i in enumerate(te_idx):
            b = best.get(bucket_of(rows[i]))
            if b is None:
                continue
            n_j += 1
            ok_des += int(("food" if rows[i]["sf"] >= rows[i]["sw"] else "water") == b)
            ok_lrn += int(("food" if sf_te[k] <= sw_te[k] else "water") == b)
    acc_des, acc_lrn = ok_des / max(n_j, 1), ok_lrn / max(n_j, 1)
    print(f"[G-res] jugeables tenus={n_j} designé={100*acc_des:.1f}% appris={100*acc_lrn:.1f}% "
          f"(barre designé+10)")

    # G-consist : sequential replay per life
    by_life: dict = {}
    for i, r in enumerate(rows):
        by_life.setdefault(r["life"], []).append(i)
    fl_des = fl_lrn = n_seq = 0
    for order in by_life.values():
        if len(order) < 2:
            continue
        n_seq += len(order) - 1
        fl_des += replay_flips(order, lambda i, inc: (
            "food" if rows[i]["sf"] >= rows[i]["sw"] else "water"))

        def choose_lrn(i: int, inc: str | None) -> str:
            f, w = float(sf_all[i]), float(sw_all[i])
            if inc == "food":
                return "food" if f <= w + DELTA_COMMIT_M else "water"
            if inc == "water":
                return "water" if w <= f + DELTA_COMMIT_M else "food"
            return "food" if f <= w else "water"
        fl_lrn += replay_flips(order, choose_lrn)
    rate_des, rate_lrn = fl_des / max(n_seq, 1), fl_lrn / max(n_seq, 1)
    print(f"[G-consist] bascules designé={100*rate_des:.1f}% appris={100*rate_lrn:.1f}% "
          f"(barre {100*1.2*rate_des:.1f}%)")

    # G-mono : pursued rows, bands of the target drive
    drive_t = torch.tensor([r["e"] if r["chosen"] == "food" else r["t"] for r in rows])
    with torch.no_grad():
        p_pur = model.p(X_pursued)
    pben = p_pur * benefit_steps(drive_t, restore, drain)
    rem_pur = torch.where(torch.tensor([r["chosen"] == "food" for r in rows]),
                          torch.tensor([math.hypot(*r["food"]) for r in rows]) - sf_all,
                          torch.tensor([math.hypot(*r["water"]) for r in rows]) - sw_all)
    bands = [(0, 30), (30, 60), (60, 100.1)]
    mb = [float(pben[(drive_t >= a) & (drive_t < b)].mean()) for a, b in bands]
    urg = 100.0 - drive_t
    q1, q2 = torch.quantile(urg, 0.33), torch.quantile(urg, 0.66)
    mu = [float(rem_pur[urg <= q1].mean()), float(rem_pur[(urg > q1) & (urg <= q2)].mean()),
          float(rem_pur[urg > q2].mean())]
    g_mono = (mb[0] > mb[1] > mb[2]) and (mu[0] < mu[1] < mu[2])
    print(f"[G-mono] P̂·bén par satiété {[f'{v:.0f}' for v in mb]} (strict ↓) ; "
          f"remise par urgence {[f'{v:.2f}' for v in mu]} (strict ↑)")

    gates = {"g_rank": g_rank > 0.70, "g_res": acc_lrn >= acc_des + 0.10,
             "g_consist": rate_lrn <= 1.2 * rate_des, "g_mono": g_mono}
    print("[gates] " + "  ".join(f"{k}={'✅' if v else '❌'}" for k, v in gates.items()))
    print(f"[proxy] loss finale {float(loss):.3f} (n'est PAS le but — les gates le sont)")

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "in_dim": SPRINT_IN_DIM, "form": "arb_p_v1",
                "kappa_data": kappa, "drain": drain, "restore": restore,
                "pain_ckpt": PAIN_CKPT, "death_ckpt": DEATH_CKPT, "sal_ckpt": SAL_CKPT,
                "delta_commit_m": DELTA_COMMIT_M, "auc_folds": aucs,
                "gates_pass": all(gates.values()), "gates": {k: bool(v) for k, v in gates.items()},
                "runs": runs}, os.path.join(OUT_DIR, "arb_best.pt"))
    print(f"[ckpt] {OUT_DIR}/arb_best.pt (gates_pass={all(gates.values())})")


if __name__ == "__main__":
    main()
