"""Prédicteur d'AFFORDANCE d'obstacle (voie B, chantier CANAL OBSTACLE) — JUMEAU de la lunette
saillance-danger (scripts/train_danger_saliency.py).

Doc : docs/design_obstacle_affordance.md §G2. Apprend, du SEUL vécu moteur (label auto-supervisé
COMMANDÉ-vs-RÉEL : « j'ai commandé avance, me suis-je déplacé ? »), une tête MIL max-pool sur la
rétine brute qui dit « ce qui est devant BLOQUE le mouvement » — SANS jamais coder « cyan = mur »
(appearance-agnostic : la tête découvre la couleur qui co-occurre avec le blocage). Déployée comme un
lecteur `obstacle_points()` (drop-in du style green_points) lu par le stage waypoint comme un coût.

Architecture (identique à DangerSaliency) : s(rgb) [appearance] × g(dist) [portée apprise ρ̂], MAX-POOL
sur les 36 rayons (le blocage a UNE source la plus proche), BCE + prior de parcimonie.

Usage : PYTHONPATH=python env_pytorch_3.12/bin/python -m scripts.train_obstacle_affordance \
          --runs data/replay_buffer/obstacle_g2ds data/replay_buffer/obstacle_g2nav \
          --out data/checkpoints/obstacle_affordance
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sylvan.models.slot_head import DEPTH_OFFSET, NRAY, RANGE

N_RAY = NRAY
RETINA_RANGE_M = RANGE
SAL_THR = 0.5                 # seuil de lecture de s (PINNÉ, comme la lunette danger)
LAMBDA_S = 0.01               # prior de parcimonie « rien ne bloque sans preuve »
# Label COMMANDÉ-vs-RÉEL (mêmes constantes que diag_obstacle_g2.py) :
VX_MIN = 0.30
STEP_BLOCKED = 0.0015
STEP_FREE = 0.0030
MOVED_MIN = 0.20
TELEPORT = 0.5
CYAN_N = np.array([0.05, 0.7, 0.95]); CYAN_N = CYAN_N / np.linalg.norm(CYAN_N)


class ObstacleAffordance(nn.Module):
    """rétine[144] → P(le mouvement est bloqué devant) ; s(rgb)·g(dist), max-pool MIL."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.app = nn.Sequential(nn.Linear(3, hidden), nn.ReLU(), nn.Linear(hidden, 1))  # apparence s(rgb): 3→16→1
        self.rho = nn.Parameter(torch.tensor(1.5))       # portée de blocage apprise ρ̂ (m)
        self.tau_raw = nn.Parameter(torch.tensor(0.5))
        self.bias = nn.Parameter(torch.tensor(-3.0))

    def s(self, rgb: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.app(rgb).squeeze(-1))   # apparence SEULE, jamais la distance

    def g(self, dist_m: torch.Tensor) -> torch.Tensor:
        tau = nn.functional.softplus(self.tau_raw) + 0.05
        return torch.sigmoid((self.rho - dist_m) / tau)   # gate de portée apprise

    def rho_hat(self) -> float:
        return float(self.rho)

    def parts(self, retina: torch.Tensor):
        r = retina.view(-1, N_RAY, 4)
        d, rgb = r[..., 0], r[..., 1:]
        touch = d < 0.999
        s = self.s(rgb)
        logits = self.bias + (s * self.g(d * RETINA_RANGE_M) * touch.float()).amax(-1)   # MAX-POOL (MIL)
        return logits, s, touch

    def tick_logits(self, retina: torch.Tensor) -> torch.Tensor:
        return self.parts(retina)[0]


def obstacle_points(model: ObstacleAffordance, retina, thr: float = SAL_THR):
    """Lecteur DÉPLOYÉ (drop-in green_points) : points ego (x_right, z_fwd) des rayons où l'apparence
    est jugée BLOQUANTE (s(rgb) > thr) et touchée. Lu par le stage waypoint comme un coût d'intrusion."""
    r = torch.as_tensor(retina, dtype=torch.float32).view(N_RAY, 4)
    with torch.no_grad():
        s = model.s(r[:, 1:])
    pts = []
    for k in range(N_RAY):
        d = float(r[k, 0])
        if d >= 0.999 or float(s[k]) <= thr:
            continue
        bearing = 2.0 * math.pi * k / N_RAY
        pts.append((d * RETINA_RANGE_M * math.sin(bearing), d * RETINA_RANGE_M * math.cos(bearing)))
    return pts


def _load(run: str) -> list[dict]:
    out = []
    for fp in sorted(glob.glob(str(Path(run) / "ep_*.jsonl"))):
        for line in open(fp, errors="ignore"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def build_labels(runs: list[str]):
    """Retourne X (rétines [N,144]), y (bloqué 1 / libre 0), lives (segment id pour la CV)."""
    X, y, lives = [], [], []
    life = 0
    for run in runs:
        recs = _load(run)
        tor = [r["wm"].get("torso0") for r in recs]
        cmd = [r["wm"].get("cmd") for r in recs]
        moved = 0.0
        life += 1
        for i in range(len(recs) - 1):
            a, b = tor[i], tor[i + 1]
            if not a or not b:
                moved = 0.0
                continue
            step = math.hypot(b[0] - a[0], b[1] - a[1])
            if step > TELEPORT:              # frontière d'épisode
                moved = 0.0
                life += 1
                continue
            vx = cmd[i][0] if cmd[i] else 0.0
            if vx <= VX_MIN:
                moved += step
                continue
            blocked = step < STEP_BLOCKED and moved > MOVED_MIN
            free = step > STEP_FREE
            if blocked or free:
                ret = recs[i]["wm"].get("retina0")
                if ret and len(ret) == N_RAY * 4:
                    X.append(ret); y.append(1.0 if blocked else 0.0); lives.append(life)
            moved += step
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(lives)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores); ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels > 0.5; npos = int(pos.sum()); nneg = int((~pos).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def fit(X: torch.Tensor, y: torch.Tensor, iters: int = 2000, seed: int = 0) -> ObstacleAffordance:
    torch.manual_seed(seed)
    m = ObstacleAffordance()
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    n = len(X)
    for _ in range(iters):
        bi = torch.randint(0, n, (min(4096, n),))
        logits, s, touch = m.parts(X[bi])
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y[bi])
        if bool(touch.any()):
            loss = loss + LAMBDA_S * s[touch].mean()
        loss.backward(); opt.step(); opt.zero_grad()
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="data/checkpoints/obstacle_affordance")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--keep-neg", type=int, default=3)      # sous-échantillonne les LIBRES (négatifs)
    args = ap.parse_args()

    Xa, ya, lives = build_labels(args.runs)
    if len(Xa) == 0:
        print("[obst] ❌ aucun tick labellisé"); return
    # sous-échantillonnage déterministe des négatifs
    rng = np.random.default_rng(0)
    pos_idx = np.where(ya > 0.5)[0]; neg_idx = np.where(ya < 0.5)[0]
    keep = rng.choice(neg_idx, size=min(len(neg_idx), args.keep_neg * len(pos_idx)), replace=False)
    idx = np.concatenate([pos_idx, keep]); rng.shuffle(idx)
    X, y, L = Xa[idx], ya[idx], lives[idx]
    print(f"[obst] {len(Xa)} ticks ({int(ya.sum())} bloqués) → train {len(X)} ({int(y.sum())} bloqués), {len(set(L))} segments")

    Xt, yt = torch.tensor(X), torch.tensor(y)
    # CV par SEGMENT (pas de fuite) : AUC hors-échantillon
    uniq = sorted(set(L.tolist())); folds = 4
    aucs = []
    for f in range(folds):
        val = {u for j, u in enumerate(uniq) if j % folds == f}
        tr = np.array([i for i in range(len(X)) if L[i] not in val])
        va = np.array([i for i in range(len(X)) if L[i] in val])
        if len(va) == 0 or yt[va].sum() == 0 or (yt[va] < 0.5).sum() == 0:
            continue
        m = fit(Xt[tr], yt[tr], args.iters, seed=f)
        with torch.no_grad():
            sc = m.tick_logits(Xt[va]).numpy()
        aucs.append(_auc(sc, y[va]))
    auc_cv = float(np.nanmean(aucs)) if aucs else float("nan")

    # modèle final sur tout + sélectivité couleur (le cœur de la pureté) + ρ̂
    model = fit(Xt, yt, args.iters, seed=42)
    with torch.no_grad():
        s_cyan = float(model.s(torch.tensor([[0.05, 0.7, 0.95]])).item())
        s_red = float(model.s(torch.tensor([[1.0, 0.0, 0.0]])).item())   # bouffe (passable)
        s_blue = float(model.s(torch.tensor([[0.0, 0.0, 1.0]])).item())  # eau (passable)
        s_green = float(model.s(torch.tensor([[0.1, 0.9, 0.15]])).item())  # danger (perceptible)
    rho = model.rho_hat()

    print(f"\n[obst] AUC CV-par-segment = {auc_cv:.3f}")
    print(f"[obst] portée apprise ρ̂ = {rho:.2f} m")
    print(f"[obst] sélectivité s(couleur) : cyan(obstacle)={s_cyan:.2f}  rouge(bouffe)={s_red:.2f}  "
          f"bleu(eau)={s_blue:.2f}  vert(danger)={s_green:.2f}")

    # --- gates ---
    g_auc = np.isfinite(auc_cv) and auc_cv >= 0.90
    g_sel = s_cyan > 0.5 and max(s_red, s_blue) < 0.5     # bloque le cyan, PAS la bouffe/eau (appearance-agnostic prouvé)
    print(f"\n[obst] (G-auc) apprend bloqué (AUC≥0.90) : {'✅' if g_auc else '❌'}")
    print(f"[obst] (G-sel) SÉLECTIF : cyan bloque, bouffe/eau non : {'✅' if g_sel else '❌'} "
          f"(le prédicteur a DÉCOUVERT la couleur bloquante depuis le seul label moteur)")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "form": "obstacle_affordance_v1", "thr": SAL_THR,
                "rho_hat": rho, "auc_cv": auc_cv, "s_cyan": s_cyan, "s_red": s_red, "s_blue": s_blue,
                "runs": list(args.runs), "gates_pass": bool(g_auc and g_sel)},
               out / "obstacle_best.pt")
    print(f"\n[obst] {'✅✅ ENTRAÎNÉ + gates OK' if (g_auc and g_sel) else '⚠️ gates partiels'} → {out / 'obstacle_best.pt'}")


if __name__ == "__main__":
    main()
