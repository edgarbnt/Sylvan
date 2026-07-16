"""Requêtes de slot APPRISES du soulagement vécu — volet P6 (docs/design_purete_hjepa.md §P6).

Dissout la dernière clé-apparence : les requêtes-couleur des slots WM (« bouffe=rouge, eau=bleu,
danger=vert »). Par drive, une tête gabarit-P5 (MIL max-pool + portée + parcimonie) dont
l'apparence est LINÉAIRE en couleur NORMALISÉE :

    P(conséquence_d | rétine) = σ( b_d + max_k σ(w_d·rgbn_k + c_d) · g_d(dist_k) )

w_d·rgbn = cosinus × ‖w_d‖ = exactement la forme de l'affinité du slot (slot_head._attend) →
**q̂_d = w_d/‖w_d‖ EST la requête**, elle se branche par BUILD (build_learned_queries), WM GELÉ.

Labels VÉCUS uniquement (jamais les canaux purs — eux ne servent qu'à l'ÉVAL G-q) :
- food  : soulagement énergie au tick SUIVANT (drv[t+1]−drv[t] > +5 ; l'objet est au contact à t,
  consommé/respawné à t+1) ; water : idem soif ; danger : tick-dégâts (convention P5, même tick).

GATES OFFLINE PRÉ-ENREGISTRÉS (§P6) :
  1. G-q          : cos(q̂_d, canal pur du monde-jouet) ≥ 0.98 par drive (oracle d'éval) ET
                    affinité croisée post-seuil 0.55 = 0 (zéro fuite entre slots) ;
  2. G-slot-parité : slot head (poids du WM vivant) avec q̂ vs requêtes main sur ≥20k ticks BC :
                    masque de visibilité identique ≥99.9 % ET |Δposition| ≤ 0.05 m sur ≥99.9 %
                    des ticks visibles.

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python -m scripts.train_drive_queries [--selfcheck]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from scripts.train_danger_saliency import DMG_DROP, LAMBDA_S, LIFE_JUMP
from scripts.train_sprint_critic import DEATH_RUNS, _auc
from scripts.train_waypoint_pain import _open_text
from sylvan.models.slot_head import SelfSupervisedSlotHead
from sylvan.control.waypoint_layer import N_RAY, RETINA_RANGE_M

RELIEF = 5.0                  # remontée de drive qui signe une consommation (convention partagée)
WM_LIVING = "data/checkpoints/wm_objcentric_kin_haz/wm_best.pt"
# canaux purs du monde-jouet — ORACLE D'ÉVAL SEULEMENT (G-q), jamais un label d'entraînement.
PURE = {"food": (1.0, 0.0, 0.0), "water": (0.0, 0.0, 1.0), "danger": (0.0, 1.0, 0.0)}
DRIVES = ("food", "water", "danger")


class DriveQuery(nn.Module):
    """Tête P5 à apparence LINÉAIRE en rgb normalisé — w/‖w‖ = la requête de slot apprise.

    ⚠️ w vit dans le CÔNE POSITIF (softplus — re-train diagnostiqué, négatif n°1 §P6) : sur des
    rayons monochromes, w est libre le long de 1⃗ (jauge w+α·1⃗/c−α) → direction non-identifiée.
    La non-négativité est la parité de déploiement (l'affinité slot est un cosinus sur rgbn ≥ 0,
    requêtes = gabarits non-négatifs) et casse la jauge du bon côté (canal OFF → w_i = 0)."""

    def __init__(self) -> None:
        super().__init__()
        self.u = nn.Parameter(torch.zeros(3))        # w = softplus(u) ≥ 0
        self.c = nn.Parameter(torch.tensor(-1.0))
        self.rho = nn.Parameter(torch.tensor(1.5))
        self.tau_raw = nn.Parameter(torch.tensor(0.5))
        self.bias = nn.Parameter(torch.tensor(-3.0))

    def w(self) -> torch.Tensor:
        return nn.functional.softplus(self.u)

    def s(self, rgbn: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(rgbn @ self.w() + self.c)

    def g(self, dist_m: torch.Tensor) -> torch.Tensor:
        tau = nn.functional.softplus(self.tau_raw) + 0.05
        return torch.sigmoid((self.rho - dist_m) / tau)

    def query(self) -> torch.Tensor:
        w = self.w()
        return w / (w.norm() + 1e-8)

    def parts(self, retina: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r = retina.view(-1, N_RAY, 4)
        d, rgb = r[..., 0], r[..., 1:]
        rgbn = rgb / (rgb.norm(dim=-1, keepdim=True) + 1e-6)
        touch = d < 0.999
        s = self.s(rgbn)
        logits = self.bias + (s * self.g(d * RETINA_RANGE_M) * touch.float()).amax(-1)
        return logits, s, touch


# ------------------------------------------------------------------ corpus (flux BC par tick)

def scan_run(run: Path, life_base: int, keep_neg: int = 3,
             ) -> tuple[list[list[float]], dict[str, list[float]], list[int]]:
    """→ (retinas subsamplées, labels par drive, vies). food/water : la conséquence vit au tick
    SUIVANT (percept t → soulagement t+1) ; danger : même tick (P5)."""
    recs: list[tuple[list[float], float, float, float]] = []
    for line in _open_text(run / "ep_0000.jsonl"):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        recs.append((rec["wm"]["retina0"], float(rec["obs"]["energy"]),
                     float(rec["obs"]["thirst"]), float(rec["obs"]["health"])))
    X: list[list[float]] = []
    ys: dict[str, list[float]] = {d: [] for d in DRIVES}
    lives: list[int] = []
    life = 0
    for i, (ret, e, t, h) in enumerate(recs):
        if i + 1 >= len(recs):
            break
        e1, t1, h1 = recs[i + 1][1], recs[i + 1][2], recs[i + 1][3]
        boundary = e1 - e > LIFE_JUMP or t1 - t > LIFE_JUMP or h1 - h > LIFE_JUMP
        if boundary:
            life += 1
        y_food = float(not boundary and e1 - e > RELIEF)
        y_water = float(not boundary and t1 - t > RELIEF)
        y_dmg = float(i > 0 and recs[i - 1][3] - h > DMG_DROP
                      and not (e - recs[i - 1][1] > LIFE_JUMP or t - recs[i - 1][2] > LIFE_JUMP
                               or h - recs[i - 1][3] > LIFE_JUMP))
        if not (y_food or y_water or y_dmg) and i % keep_neg != 0:
            continue
        X.append(ret)
        ys["food"].append(y_food)
        ys["water"].append(y_water)
        ys["danger"].append(y_dmg)
        lives.append(life_base + life)
    return X, ys, lives


def fit(X: torch.Tensor, y: torch.Tensor, iters: int, seed: int) -> DriveQuery:
    torch.manual_seed(seed)
    m = DriveQuery()
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    for _ in range(iters):
        bi = torch.randint(0, len(X), (4096,))
        logits, s, touch = m.parts(X[bi])
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y[bi])
        if bool(touch.any()):
            loss = loss + LAMBDA_S * s[touch].mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    return m.eval()


# ------------------------------------------------------------------ gates

def _load_living_slot_head() -> SelfSupervisedSlotHead:
    state = torch.load(WM_LIVING, map_location="cpu", weights_only=False)["model"]
    sub = {k.removeprefix("slot_encoder."): v for k, v in state.items()
           if k.startswith("slot_encoder.")}
    head = SelfSupervisedSlotHead(n_resources=3)
    head.load_state_dict(sub)
    return head.eval()


def slot_parity(queries: torch.Tensor, X: torch.Tensor) -> tuple[float, float]:
    """→ (accord du masque de visibilité, part des ticks visibles à |Δpos| ≤ 0.05 m)."""
    hand = _load_living_slot_head()
    learned = _load_living_slot_head()
    with torch.no_grad():
        learned.color_queries.copy_(queries / queries.norm(dim=-1, keepdim=True))
        vh, vl = hand.visibility(X), learned.visibility(X)          # [N, 3]
        mask_h, mask_l = vh > 1e-6, vl > 1e-6
        mask_ok = float((mask_h == mask_l).float().mean())
        both = mask_h & mask_l
        ph, pl = hand.positions(X), learned.positions(X)            # [N, 3, 2]
        dpos = (ph - pl).norm(dim=-1)                               # [N, 3]
        pos_ok = float((dpos[both] <= 0.05).float().mean()) if bool(both.any()) else 1.0
    return mask_ok, pos_ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=DEATH_RUNS)
    ap.add_argument("--out", default="data/checkpoints/drive_queries")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parity-ticks", type=int, default=20000)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return

    Xl, ll = [], []
    ysl: dict[str, list[float]] = {d: [] for d in DRIVES}
    for k, run in enumerate(args.runs):
        X, ys, lv = scan_run(Path(run), life_base=1000 * k)
        Xl += X
        ll += lv
        for d in DRIVES:
            ysl[d] += ys[d]
        print(f"[queries] {Path(run).name}: {len(X)} ticks "
              f"(food {int(sum(ys['food']))}, water {int(sum(ys['water']))}, "
              f"dgr {int(sum(ys['danger']))})")
    X = torch.tensor(Xl, dtype=torch.float32)
    life = torch.tensor(ll)
    print(f"[queries] corpus : {len(X)} ticks | positifs "
          f"{ {d: int(sum(ysl[d])) for d in DRIVES} }")

    heads: dict[str, DriveQuery] = {}
    queries = []
    g_q = True
    for d in DRIVES:
        y = torch.tensor(ysl[d])
        aucs = []
        for k in range(4):                       # AUC CV-4 par vie (contexte, G = G-q/parité)
            te = (life % 4 == k)
            if int(y[te].sum()) == 0:
                continue
            m_k = fit(X[~te], y[~te], args.iters, args.seed)
            with torch.no_grad():
                aucs.append(_auc(m_k.parts(X[te])[0], y[te].bool()))
        m = fit(X, y, args.iters, args.seed)
        heads[d] = m
        q = m.query().detach()
        queries.append(q)
        pure = torch.tensor(PURE[d])
        cos = float(q @ (pure / pure.norm()))
        cross = [float(((q @ (torch.tensor(PURE[o]) / torch.tensor(PURE[o]).norm()))
                        - 0.55)) for o in DRIVES if o != d]
        leak = any(c > 0 for c in cross)
        ok = cos >= 0.98 and not leak
        g_q = g_q and ok
        print(f"[queries] {d:6s} : q̂=[{q[0]:+.3f} {q[1]:+.3f} {q[2]:+.3f}] "
              f"cos(canal pur)={cos:.4f} fuite croisée={'OUI' if leak else 'non'} "
              f"AUC CV-4={sum(aucs) / max(len(aucs), 1):.3f} ρ̂={float(m.rho):.2f} → "
              f"{'✅' if ok else '❌'}")

    Q = torch.stack(queries)
    step = max(1, len(X) // args.parity_ticks)
    mask_ok, pos_ok = slot_parity(Q, X[::step])
    g_par = mask_ok >= 0.999 and pos_ok >= 0.999
    print(f"\n[queries] === GATES OFFLINE (pré-enregistrés §P6) ===")
    print(f"[queries] G-q          : cos ≥ 0.98 ∧ zéro fuite post-0.55 → {'✅' if g_q else '❌'}")
    print(f"[queries] G-slot-parité : masque visibilité {100 * mask_ok:.2f}% | "
          f"Δpos ≤ 0.05 m sur {100 * pos_ok:.2f}% des visibles (≥99.9%/≥99.9%, "
          f"{len(X[::step])} ticks) → {'✅' if g_par else '❌'}")
    verdict = g_q and g_par
    print(f"[queries] {'✅ GATES PASSÉS → build + smoke licenciés' if verdict else '❌ GATE ÉCHOUÉ → diagnostiquer sur trace AVANT tout re-train (budget : 1/requête)'}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"queries": Q, "drives": list(DRIVES),
                "state_dicts": {d: heads[d].state_dict() for d in DRIVES},
                "mask_parity": mask_ok, "pos_parity": pos_ok,
                "runs": list(args.runs), "gates_pass": bool(verdict)},
               out / "queries_best.pt")
    print(f"[queries] sauvé → {out / 'queries_best.pt'}")


def selfcheck() -> None:
    torch.manual_seed(0)
    m = DriveQuery()
    with torch.no_grad():
        m.u.copy_(torch.tensor([8.0, -20.0, -20.0]))  # softplus → w ≈ [8, 0, 0] (« rouge »)
        m.c.fill_(-4.0)
    assert torch.all(m.w() >= 0.0), "w doit vivre dans le cône positif"
    assert torch.allclose(m.query(), torch.tensor([1.0, 0.0, 0.0]), atol=1e-4), m.query()
    ret = [1.0, 0.0, 0.0, 0.0] * N_RAY
    ret[0:4] = [0.05, 1.0, 0.0, 0.0]                 # rouge à 0.5 m
    ret[4:8] = [0.05, 0.0, 1.0, 0.0]                 # vert à 0.5 m
    _, s, touch = m.parts(torch.tensor([ret], dtype=torch.float32))
    assert float(s[0, 0]) > 0.9 > 0.1 > float(s[0, 1]), (float(s[0, 0]), float(s[0, 1]))
    assert int(touch.sum()) == 2
    # parité slot : requêtes IDENTIQUES aux mains ⇒ masque et positions strictement égaux
    if Path(WM_LIVING).exists():
        hand = _load_living_slot_head()
        X = torch.tensor([ret, [1.0, 0.0, 0.0, 0.0] * N_RAY], dtype=torch.float32)
        mask_ok, pos_ok = slot_parity(hand.color_queries.clone(), X)
        assert mask_ok == 1.0 and pos_ok == 1.0, (mask_ok, pos_ok)
    print("[selfcheck] OK — query=w/‖w‖, apparence linéaire sépare, parité slot à requêtes égales")


if __name__ == "__main__":
    main()
