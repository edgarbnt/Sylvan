"""CRITIQUE-DOULEUR de l'étage waypoint : Q(candidat) → dégâts@200 ticks (design amendé, gate v2).

Apprend, des morsures VÉCUES, ce que le scoreur main encode par des marges géométriques (« cette
route fait-elle mal ? »). Label = dégâts subis dans les 200 ticks suivant la décision (jointure
santé Godot). Features = candidate_features (waypoint_layer, parité train=déploiement ; le vert y
est un PERCEPT — distances brutes — sa LÉTALITÉ est ce qu'on apprend ici).

GATES PRÉ-ENREGISTRÉS (docs/design_critique_waypoint.md, v2) :
  - AUC(« ≥1 dégât dans 200 ticks ») > 0.80 en CV 4 plis PAR VIE (jamais par décision) ;
  - MONOTONIE de la douleur prédite sur les buckets de dégagure (28/5/0 sans les marges main).

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python -m scripts.train_waypoint_pain \
      --runs data/replay_buffer/critic_kin_wpx1 data/replay_buffer/critic_kin_wpx2 \
      --out data/checkpoints/waypoint_pain
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
from pathlib import Path

import torch
from torch import nn

from sylvan.control.waypoint_layer import WP_FEAT_DIM

HORIZON_TICKS = 200      # fenêtre de douleur (≈ un leg complet)
_EP = re.compile(r"Episode (\d+) \| Step (\d+) .*?Health: ([\d.]+)")


def health_series(path: Path) -> tuple[list[int], list[float], list[int]]:
    """Santé échantillonnée (tous les 10 pas) → (ticks GLOBAUX cumulés, valeurs, débuts d'épisode).

    Les frontières viennent du NUMÉRO d'épisode du log (fiable) — pas des écarts de ticks, qui
    valent exactement 10 partout, frontières comprises. Longueur d'épisode approximée au dernier
    échantillon +10 (la vraie fin peut tomber mi-décade) → dérive de jointure ≤10 ticks/épisode,
    tolérée (la sonde Δsanté@200 a donné 28/5/0 net avec la même approximation)."""
    ticks, vals, starts = [], [], []
    base, prev_ep, prev_step = 0, None, 0
    for line in open(path, errors="ignore"):
        m = _EP.search(line)
        if not m:
            continue
        ep, step, h = int(m.group(1)), int(m.group(2)), float(m.group(3))
        if prev_ep is None or ep != prev_ep:
            if prev_ep is not None:
                base += prev_step + 10
            starts.append(base)
        ticks.append(base + step)
        vals.append(h)
        prev_ep, prev_step = ep, step
    return ticks, vals, starts


def load_runs(runs: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """→ (X [N,FEAT], dmg [N] dégâts@200 (≥0), LIFE [N] id de vie, explored [N])."""
    X, dmg, life_ids, expl = [], [], [], []
    next_life = 0
    for run in runs:
        gl = Path(run) / "godot.log"
        df = Path(run) / "decisions.jsonl"
        if not gl.exists() or not df.exists():
            print(f"[pain] ⚠️ {run} incomplet (godot.log/decisions.jsonl) — ignoré")
            continue
        ticks, vals, starts = health_series(gl)

        def h_at(t: int) -> float:
            i = bisect.bisect_left(ticks, t)
            return vals[min(i, len(vals) - 1)]

        # une décision et son horizon doivent rester dans la MÊME vie, sinon le Δsanté enjambe un
        # respawn (santé restaurée à 100) → fenêtre tronquée à la frontière d'épisode suivante.
        ep_bounds = starts[1:] + [ticks[-1] + 10]
        for line in open(df):
            d = json.loads(line)
            t0 = d["tick"]
            b = bisect.bisect_right(ep_bounds, t0)
            end = ep_bounds[b] if b < len(ep_bounds) else ticks[-1] + 10
            t1 = min(t0 + HORIZON_TICKS, end - 1)       # horizon TRONQUÉ à la fin de vie (pas d'enjambement)
            if t1 <= t0 + 20:                            # décision à l'agonie : fenêtre vide, sautée
                continue
            X.append(d["feats"][d["chosen"]])
            dmg.append(max(0.0, h_at(t0) - h_at(t1)))
            life_ids.append(next_life + b)
            expl.append(bool(d["explore"]))
        next_life += len(ep_bounds) + 1
    return (torch.tensor(X, dtype=torch.float32), torch.tensor(dmg), torch.tensor(life_ids),
            torch.tensor(expl))


class PainCritic(nn.Module):
    """features candidat [B, FEAT] → dégâts prédits @200 (unités : /100, ≥0 via softplus)."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(WP_FEAT_DIM, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def pain(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.softplus(self.net(x).squeeze(-1))


def _auc(score: torch.Tensor, label: torch.Tensor) -> float:
    """AUC par comptage de paires (positifs = touché)."""
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = pos.unsqueeze(1) - neg.unsqueeze(0)
    return float(((d > 0).float() + 0.5 * (d == 0).float()).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=["data/replay_buffer/critic_kin_wpx1", "data/replay_buffer/critic_kin_wpx2"])
    ap.add_argument("--out", default="data/checkpoints/waypoint_pain")
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X, dmg, life, expl = load_runs(args.runs)
    y = (dmg / 100.0).clamp(0.0, 1.0)
    hit = dmg > 1.0
    print(f"[pain] décisions={len(X)} (explorées {int(expl.sum())}) | touchées={int(hit.sum())} "
          f"({100 * float(hit.float().mean()):.0f}%) | dégâts méd si touché="
          f"{float(dmg[hit].median()) if hit.any() else 0:.1f}")

    def fit(mask: torch.Tensor) -> PainCritic:
        torch.manual_seed(args.seed)
        c = PainCritic()
        opt = torch.optim.Adam(c.parameters(), 2e-3, weight_decay=1e-4)
        Xt, yt = X[mask], y[mask]
        for _ in range(args.iters):
            bi = torch.randint(0, len(Xt), (256,))
            nn.functional.mse_loss(c.pain(Xt[bi]), yt[bi]).backward()
            opt.step()
            opt.zero_grad()
        return c.eval()

    # GATE 1a — AUC en CV 4 plis PAR VIE (les décisions d'une même vie se ressemblent → split par
    # décision fuiterait, même règle que train_survival_critic).
    aucs = []
    for k in range(4):
        te = (life % 4 == k)
        if int(hit[te].sum()) == 0 or int((~hit[te]).sum()) == 0:
            print(f"[pain]   pli {k} : classe vide, sauté")
            continue
        c_k = fit(~te)
        with torch.no_grad():
            aucs.append(_auc(c_k.pain(X[te]), hit[te]))
        print(f"[pain]   pli {k} : AUC={aucs[-1]:.3f} (n_te={int(te.sum())}, touchés={int(hit[te].sum())})")
    auc = sum(aucs) / max(len(aucs), 1)

    # GATE 1b — MONOTONIE sans les marges main : douleur prédite par bucket de dégagure brute.
    critic = fit(torch.ones(len(X), dtype=torch.bool))
    with torch.no_grad():
        pred = critic.pain(X) * 100.0
    dg = 10.0 * torch.minimum(X[:, 7], X[:, 8])
    buckets = [("<0.5m", dg < 0.5), ("0.5-1.5m", (dg >= 0.5) & (dg <= 1.5)), (">1.5m", dg > 1.5)]
    means = []
    print("[pain] douleur PRÉDITE par dégagure (monotonie attendue, réel : 28/5/0% touchés) :")
    for name, m in buckets:
        mp = float(pred[m].mean()) if m.any() else float("nan")
        ma = float(dmg[m].mean()) if m.any() else float("nan")
        means.append(mp)
        print(f"[pain]   {name:>8} : n={int(m.sum()):>4} prédite={mp:5.1f} dégâts réels moy={ma:5.1f}")
    mono = means[0] > means[1] > means[2]

    print(f"\n[pain] === GATES v2 (pré-enregistrés) ===")
    print(f"[pain] AUC CV-4 par vie : {auc:.3f}  (gate > 0.80)  [{', '.join(f'{a:.3f}' for a in aucs)}]")
    print(f"[pain] monotonie prédite : {'OUI' if mono else 'NON'}")
    if auc > 0.80 and mono:
        print("[pain] ✅ GATES PASSÉS → brancher (SYLVAN_WP_PAIN_CRITIC) et juger par l'A/B closed-loop.")
    else:
        print("[pain] ❌ GATE ÉCHOUÉ → ne pas brancher ; commiter le négatif.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": critic.state_dict(), "feat_dim": WP_FEAT_DIM, "auc_cv": auc,
                "monotone": mono, "horizon_ticks": HORIZON_TICKS, "runs": args.runs},
               out / "pain_best.pt")
    print(f"[pain] sauvé → {out / 'pain_best.pt'}")


if __name__ == "__main__":
    main()
