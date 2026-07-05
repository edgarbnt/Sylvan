"""Entraîne le CRITIQUE DE SURVIE drive-symétrique (chantier critique-appris, 2026-07-05).

Le « trainable critic » de LeCun pour Mode-2 : apprend du VÉCU « combien d'avenir depuis cet
état » — destiné à remplacer la queue analytique (alternance+drain codées-main) du coût survie.
Drive-SYMÉTRIQUE : un token par pulsion [niveau, dist, cos(brg), sin(brg), connu] → encodeur
partagé + pooling → une 3ᵉ pulsion = un token de plus, zéro retrain (contrat Gate-S).

Données = le vécu déjà loggé (buffers hesit_probe_* : plan.food/water crus + drives par replan).
Labels = pipeline G2/B0 : G = 1−γ^(replans restants) + surv (mort ≤10 replans). AUCUN oracle :
positions = ce que le planner CROYAIT (slots), issues = ce qui est réellement arrivé.

Gates offline pré-enregistrés (avant run, principe de travail n°1) :
  1. AUC(V, surv) held-out ≥ 0.85 (référence G2 : 0.88)
  2. NON-SATURATION (le décisif) : sur les replans où les 2 ordres analytiques étaient SATURÉS
     (sf,sw ≥ 2999, écarts 1-6 pts = la racine de l'errance-du-repu), le critique garde ≥ 50%
     de son pouvoir discriminant global (std_V_saturés / std_V_tous ≥ 0.5).
  3. Équilibre drives : ΔV négatif pour CHAQUE drive bas (contrefactuel sur tokens).
  4. Arbitrage swap (B0) : V préfère la config où la ressource de la pulsion basse est proche ;
     rappel B0 : la valeur statique seule avait ÉCHOUÉ ce test (hasard) — ici les features sont
     les tokens du critique, le verdict dira si la forme drive-symétrique fait mieux ;
     le juge FINAL reste la Phase B (critique au bout du rollout, closed-loop).

Usage : PYTHONPATH=python env_pytorch_3.12/bin/python -m scripts.train_survival_critic \
            [--glob 'data/replay_buffer/hesit_probe_*_surv'] [--out data/checkpoints/survival_critic]
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import math
from pathlib import Path

import torch
from torch import nn

GAMMA = 0.99            # par replan (10 pas Godot) — même échelle que G2/B0
H_SURV = 10             # « vivant dans 10 replans » (= 100 pas Godot, G2)
LOW, HIGH = 0.30, 0.50
TOK_DIM = 5             # [niveau, dist/10, cos, sin, connu]


def token(level: float, pos: list[float] | None) -> list[float]:
    if pos is None:
        return [level, 1.0, 0.0, 0.0, 0.0]
    d = math.hypot(pos[0], pos[1])
    return [level, min(d, 10.0) / 10.0, pos[0] / (d + 1e-6), pos[1] / (d + 1e-6), 1.0]


def load(dirs: list[str]) -> tuple[torch.Tensor, ...]:
    """→ (X [N,2,TOK], G [N], S [N], EID [N]) ; épisodes coupés aux respawns (drives ↑ ensemble)."""
    X, G, S, EID = [], [], [], []
    eid = 0
    for d in dirs:
        f = Path(d) / "ep_0000.jsonl"
        if not f.exists():
            continue
        rows = []
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = r.get("plan")
            if p is None:
                continue
            rows.append((float(r["obs"]["energy"]) / 100.0, float(r["obs"]["thirst"]) / 100.0,
                         p.get("food"), p.get("water"), p.get("sf"), p.get("sw")))
        # split épisodes par respawn
        segs, cur = [], []
        for i, row in enumerate(rows):
            if cur and (row[0] - cur[-1][0] > 0.5 or row[1] - cur[-1][1] > 0.5):
                segs.append(cur); cur = []
            cur.append(row)
        if cur:
            segs.append(cur)
        for seg in segs:
            L = len(seg)
            if L < 15:
                continue
            death = min(seg[-1][0], seg[-1][1]) < 0.03      # fin par mort (drive ~0) vs troncature
            for t, (e, th, fp, wp, sf, sw) in enumerate(seg):
                X.append([token(e, fp), token(th, wp)])
                G.append(1.0 - GAMMA ** (L - t))
                S.append(0.0 if (death and (L - 1 - t) <= H_SURV) else 1.0)
                EID.append(eid)
            eid += 1
    return (torch.tensor(X), torch.tensor(G), torch.tensor(S), torch.tensor(EID))


class SurvivalCritic(nn.Module):
    """Tokens [B, K, TOK_DIM] → V [B]. Drive-symétrique : encodeur PARTAGÉ + somme (invariant
    à l'ordre et au NOMBRE de pulsions → pulsion nouvelle = token nouveau, zéro retrain)."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(TOK_DIM, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def value(self, toks: torch.Tensor) -> torch.Tensor:
        return self.head(self.enc(toks).sum(dim=-2)).squeeze(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/replay_buffer/hesit_probe_*_surv")
    ap.add_argument("--out", default="data/checkpoints/survival_critic")
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)

    dirs = sorted(globmod.glob(args.glob))
    X, G, S, EID = load(dirs)
    n_ep = int(EID.max()) + 1
    te_mask = (EID % 4 == 3)                                # split par ÉPISODE (déterministe, ~25%)
    tr = ~te_mask
    print(f"[critic] dirs={len(dirs)} épisodes={n_ep} replans={len(X)} (train={int(tr.sum())} test={int(te_mask.sum())})")

    critic = SurvivalCritic()
    opt = torch.optim.Adam(critic.parameters(), 2e-3, weight_decay=1e-4)
    Xt, Gt = X[tr], G[tr]
    for it in range(args.iters):
        bi = torch.randint(0, len(Xt), (512,))
        loss = nn.functional.mse_loss(critic.value(Xt[bi]), Gt[bi])
        loss.backward(); opt.step(); opt.zero_grad()
    critic.eval()

    with torch.no_grad():
        v = critic.value(X[te_mask])
        s = S[te_mask]
        # 1. AUC
        o = torch.argsort(v); rk = torch.empty_like(v); rk[o] = torch.arange(1, len(v) + 1, dtype=v.dtype)
        npos, nneg = float(s.sum()), float((1 - s).sum())
        auc = float((rk[s == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else float("nan")
        # 2. NON-SATURATION : replans held-out où l'analytique était saturée
        sat = torch.tensor([(r[0] >= 2999 and r[1] >= 2999) if r[0] is not None else False
                            for r in _sat_flags(dirs)])[: len(X)]
        sat_te = sat[te_mask.nonzero().squeeze(-1)] if len(sat) == len(X) else None
        ratio = float(v[sat_te].std() / (v.std() + 1e-9)) if sat_te is not None and int(sat_te.sum()) > 30 else float("nan")
        # 3. équilibre drives (contrefactuel niveau bas/haut par token)
        dvs = []
        for k in (0, 1):
            lo, hi = X[te_mask].clone(), X[te_mask].clone()
            lo[:, k, 0] = 0.15; hi[:, k, 0] = 0.85
            dvs.append(float((critic.value(lo) - critic.value(hi)).mean()))
        # 4. arbitrage swap : une pulsion basse → sa ressource proche doit gagner
        e_lvl, t_lvl = X[te_mask][:, 0, 0], X[te_mask][:, 1, 0]
        onelow = ((e_lvl < LOW) & (t_lvl > HIGH)) | ((t_lvl < LOW) & (e_lvl > HIGH))
        both_known = (X[te_mask][:, 0, 4] > 0.5) & (X[te_mask][:, 1, 4] > 0.5)
        gap = (X[te_mask][:, 0, 1] - X[te_mask][:, 1, 1]).abs() > 0.05
        sel = onelow & both_known & gap
        Xs = X[te_mask][sel]
        sw_ = Xs.clone(); sw_[:, 0, 1:] = Xs[:, 1, 1:]; sw_[:, 1, 1:] = Xs[:, 0, 1:]   # swap positions
        v0, v1 = critic.value(Xs), critic.value(sw_)
        dep_is_e = Xs[:, 0, 0] < LOW
        dep_nearer = torch.where(dep_is_e, Xs[:, 0, 1] < Xs[:, 1, 1], Xs[:, 1, 1] < Xs[:, 0, 1])
        correct = torch.where(dep_nearer, v0 > v1, v1 > v0)
        frac = float(correct.float().mean()) if len(Xs) else float("nan")

    print(f"[critic] 1. AUC held-out = {auc:.3f} (gate ≥0.85, réf G2 0.88)")
    print(f"[critic] 2. NON-SATURATION : std_V(saturés)/std_V(tous) = {ratio:.2f} (gate ≥0.5)")
    print(f"[critic] 3. équilibre : ΔV(e bas)={dvs[0]:+.3f} ΔV(t bas)={dvs[1]:+.3f} (les 2 <0)")
    print(f"[critic] 4. arbitrage swap = {frac:.2f} sur n={int(sel.sum())} (gate ≥0.7 ; B0 valeur-plate : hasard)")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": critic.state_dict(), "auc": auc, "nonsat_ratio": ratio,
                "dv": dvs, "swap": frac, "gamma": GAMMA, "tok_dim": TOK_DIM,
                "dirs": dirs, "drive_symmetric": True}, out / "critic_best.pt")
    print(f"[critic] sauvé → {out / 'critic_best.pt'}")


def _sat_flags(dirs: list[str]):
    """Re-parcourt les buffers dans le MÊME ordre que load() → (sf, sw) par replan gardé."""
    for d in dirs:
        f = Path(d) / "ep_0000.jsonl"
        if not f.exists():
            continue
        rows = []
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = r.get("plan")
            if p is None:
                continue
            rows.append((float(r["obs"]["energy"]) / 100.0, float(r["obs"]["thirst"]) / 100.0,
                         p.get("sf"), p.get("sw")))
        segs, cur = [], []
        for row in rows:
            if cur and (row[0] - cur[-1][0] > 0.5 or row[1] - cur[-1][1] > 0.5):
                segs.append(cur); cur = []
            cur.append(row)
        if cur:
            segs.append(cur)
        for seg in segs:
            if len(seg) < 15:
                continue
            for (_, _, sf, sw) in seg:
                yield (sf, sw)


if __name__ == "__main__":
    main()
