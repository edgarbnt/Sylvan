"""Tête SAILLANCE-DANGER — perception par la CONSÉQUENCE (docs/design_purete_hjepa.md §P5).

« Dangereux » = ce qui a PRÉCÉDÉ mes dégâts. La tête apprend sur la RÉTINE BRUTE (36 rayons ×
(d,r,g,b)) à prédire la morsure VÉCUE (chute de santé par tick, BC obs.health) — AUCUN label
couleur, AUCUNE position d'oracle. Forme tranchée (§P5, une géométrie de conception) :

    P(dégâts au tick t | rétine_t) = σ( b + max_{rayons k touchants} s(rgb_k) · g(d_k) )

- s(rgb) ∈ (0,1) : saillance d'APPARENCE (MLP 3→16→1 — la couleur seule, jamais la distance) ;
- g(d)  = σ((ρ − d)/τ) : PORTÉE-MORSURE apprise ; ρ̂ (g=0.5) devient la marge MESURÉE du vécu
  qui remplace green_margin (géométrie pilier connue d'avance) au déploiement ;
- MAX-POOLING (MIL — re-train diagnostiqué, négatif n°1 §P5) : la morsure a UNE source. La
  forme SOMME donnait un crédit partiel au rouge (repas engouffrés) et séparait par le NOMBRE
  de rayons verts (proxy de proximité) → s(rouge)=0.6, ρ̂ figé à l'init ;
- prior de PARCIMONIE λ·mean(s(touchants)) : « rien n'est dangereux sans preuve vécue » —
  défaut sûr pour les apparences jamais contraintes (constante de conception, pas fittée).

Lecture déployée (parité train/déploiement — waypoint_layer importe `saliency_points` d'ici) :
rayon flaggé ⇔ d<0.999 ET s(rgb)>0.5 → mêmes points ego que green_points, lunette apprise.

GATES OFFLINE PRÉ-ENREGISTRÉS (§P5, écrits AVANT ce trainer — échec → 1 seul re-train
diagnostiqué, puis STOP négatif commité) :
  1. G-dmg  : AUC(P̂(dégâts|rétine), tick-dégâts) > 0.90, CV-4 par VIE ;
  2. G-loc  : rappel des rayons verts-règle ≥ 0.95 ET flag des touchants NON-verts ≤ 2 %
              (la règle-couleur = ORACLE D'ÉVAL seulement, licite monde-jouet) ;
  3. G-ρ    : ρ̂ ∈ [médiane, q95 + 0.3 m] des distances min au point SAILLANT aux onsets ;
  4. G-feat : lunette saillance ≡ lunette verte sur ≥ 99 % des ticks de DÉCISION
              (même cardinal ET Hausdorff ≤ 0.05 m) ⇒ dg1/dg2 reconstruits à ±0.05 m.

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python -m scripts.train_danger_saliency [--selfcheck]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

import torch
from torch import nn

from scripts.train_sprint_critic import DEATH_RUNS, _auc
from scripts.train_waypoint_pain import _open_text, _text_path
from sylvan.control.waypoint_layer import N_RAY, RETINA_RANGE_M, green_points

# Conventions de scan du flux BC (partagées avec diag_saliency_corpus — source de vérité ICI) :
DMG_DROP = 0.3    # chute de santé/tick qui signe une morsure (mesuré : −0.5/pas en zone)
LIFE_JUMP = 45.0  # remontée d'un signal vital = respawn. > 40 (restore repas/boisson mesuré),
                  # sinon chaque repas fabriquerait une fausse frontière de vie.
CLEAN_TICKS = 20  # ticks sains requis avant une morsure pour compter un ONSET
SAL_THR = 0.5     # seuil de lecture de s (PINNÉ §P5 — pas une constante à fitter)
LAMBDA_S = 0.01   # prior de parcimonie sur s (re-train diagnostiqué — conception, pas fitté)


class DangerSaliency(nn.Module):
    """Prédicteur de morsure factorisé QUOI × OÙ. Les paramètres appris : l'apparence s(rgb),
    la portée ρ/τ, le biais de base — rien d'autre (le monde n'entre que par le vécu)."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.app = nn.Sequential(nn.Linear(3, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.rho = nn.Parameter(torch.tensor(1.5))       # portée-morsure initiale (m)
        self.tau_raw = nn.Parameter(torch.tensor(0.5))   # douceur (softplus + plancher)
        self.bias = nn.Parameter(torch.tensor(-3.0))

    def s(self, rgb: torch.Tensor) -> torch.Tensor:
        """[.., 3] → saillance d'apparence (0,1) — jamais la distance en entrée."""
        return torch.sigmoid(self.app(rgb).squeeze(-1))

    def g(self, dist_m: torch.Tensor) -> torch.Tensor:
        tau = nn.functional.softplus(self.tau_raw) + 0.05
        return torch.sigmoid((self.rho - dist_m) / tau)

    def rho_hat(self) -> float:
        """Portée-morsure apprise (m) : la distance où g = 0.5 — remplace green_margin."""
        return float(self.rho)

    def parts(self, retina: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """[B, 144] → (logits [B], s [B,36], touch [B,36]) — MAX sur les rayons (MIL :
        la morsure a UNE source ; la somme comptait les rayons, négatif n°1)."""
        r = retina.view(-1, N_RAY, 4)                    # [B, 36, (d,r,g,b)]
        d, rgb = r[..., 0], r[..., 1:]
        touch = d < 0.999
        s = self.s(rgb)
        logits = self.bias + (s * self.g(d * RETINA_RANGE_M) * touch.float()).amax(-1)
        return logits, s, touch

    def tick_logits(self, retina: torch.Tensor) -> torch.Tensor:
        """[B, 144] → logit de « je prends des dégâts à ce tick »."""
        return self.parts(retina)[0]


def saliency_points(model: DangerSaliency, retina: list[float],
                    thr: float = SAL_THR) -> list[tuple[float, float]]:
    """Points-obstacles SAILLANTS perçus, en ego (x_right, z_fwd) — drop-in de green_points
    (même géométrie de lecture, lunette APPRISE). LE point de parité train/déploiement."""
    pts: list[tuple[float, float]] = []
    with torch.no_grad():
        r = torch.tensor(retina, dtype=torch.float32).view(N_RAY, 4)
        s = model.s(r[:, 1:])
        for k in range(N_RAY):
            d = float(r[k, 0])
            if d >= 0.999 or float(s[k]) <= thr:
                continue
            bearing = 2.0 * math.pi * k / N_RAY
            pts.append((d * RETINA_RANGE_M * math.sin(bearing),
                        d * RETINA_RANGE_M * math.cos(bearing)))
    return pts


# ------------------------------------------------------------------ corpus (flux BC par tick)

def scan_run(run: Path, life_base: int, keep_neg: int = 3,
             ) -> tuple[list[list[float]], list[float], list[int], list[list[float]], list[int]]:
    """→ (retinas subsamplées, labels dégât, vies, rétines d'ONSET, ticks de décision).

    Positifs : TOUS les ticks-dégâts. Négatifs : 1 sur keep_neg (déterministe par index — les
    ticks pré-morsure restent : ils ENSEIGNENT la frontière de portée à g). Frontières de vie
    par remontée vitale > LIFE_JUMP (cf diag G0)."""
    X: list[list[float]] = []
    y: list[float] = []
    lives: list[int] = []
    onsets: list[list[float]] = []
    prev_h = prev_e = prev_t = None
    life, last_dmg = 0, {}
    for i, line in enumerate(_open_text(run / "ep_0000.jsonl")):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        h, e = float(rec["obs"]["health"]), float(rec["obs"]["energy"])
        t = float(rec["obs"]["thirst"])
        if prev_h is not None and (h - prev_h > LIFE_JUMP or e - prev_e > LIFE_JUMP
                                   or t - prev_t > LIFE_JUMP):
            life += 1
            prev_h = None
        dmg = prev_h is not None and (prev_h - h) > DMG_DROP
        prev_h, prev_e, prev_t = h, e, t
        if dmg:
            prev = last_dmg.get(life)
            if prev is None or i - prev >= CLEAN_TICKS:
                onsets.append(rec["wm"]["retina0"])
            last_dmg[life] = i
        elif i % keep_neg != 0:
            continue
        X.append(rec["wm"]["retina0"])
        y.append(float(dmg))
        lives.append(life_base + life)
    dticks = []
    df = run / "decisions.jsonl"
    if _text_path(df) is not None:
        dticks = [json.loads(line)["tick"] for line in _open_text(df)]
    return X, y, lives, onsets, dticks


def decision_retinas(run: Path, dticks: list[int]) -> list[list[float]]:
    """Rétines aux ticks de décision (record BC n = tick n, la jointure vivante du corpus)."""
    want = set(dticks)
    out = {}
    for i, line in enumerate(_open_text(run / "ep_0000.jsonl")):
        if i in want:
            try:
                out[i] = json.loads(line)["wm"]["retina0"]
            except json.JSONDecodeError:
                continue
    return [out[t] for t in sorted(out)]


def _green_mask(rgb: torch.Tensor) -> torch.Tensor:
    """Règle mur-vert vectorisée (ORACLE D'ÉVAL seulement — jamais un label d'entraînement)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    sat = rgb.max(-1).values - rgb.min(-1).values
    return (g > r) & (g > b) & (sat > 0.15)


def _min_dist_to_points(pts: list[tuple[float, float]]) -> float:
    return min(math.hypot(x, z) for x, z in pts) if pts else float("nan")


# ------------------------------------------------------------------ entraînement + gates

def fit(X: torch.Tensor, y: torch.Tensor, iters: int, seed: int) -> DangerSaliency:
    torch.manual_seed(seed)
    m = DangerSaliency()
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    for _ in range(iters):
        bi = torch.randint(0, len(X), (4096,))
        logits, s, touch = m.parts(X[bi])
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y[bi])
        if bool(touch.any()):                     # parcimonie : les apparences non contraintes
            loss = loss + LAMBDA_S * s[touch].mean()   # par le vécu retombent à « pas dangereux »
        loss.backward()
        opt.step()
        opt.zero_grad()
    return m.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=DEATH_RUNS)
    ap.add_argument("--out", default="data/checkpoints/danger_saliency")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return

    Xl, yl, ll, onset_ret, dec_ret = [], [], [], [], []
    for k, run in enumerate(args.runs):
        X, y, lv, on, dticks = scan_run(Path(run), life_base=1000 * k)
        Xl += X
        yl += y
        ll += lv
        onset_ret += on
        dec_ret += decision_retinas(Path(run), dticks)
        print(f"[saliency] {Path(run).name}: {len(X)} ticks gardés ({int(sum(y))} dégâts), "
              f"{len(on)} onsets, {len(dticks)} décisions")
    X = torch.tensor(Xl, dtype=torch.float32)
    y = torch.tensor(yl)
    life = torch.tensor(ll)
    print(f"[saliency] corpus : {len(X)} ticks ({int(y.sum())} dégâts = "
          f"{100 * float(y.mean()):.1f}%), {len(onset_ret)} onsets, {len(dec_ret)} décisions")

    # GATE 1 — G-dmg : AUC tick-dégâts en CV-4 par VIE + GATE 2 — G-loc sur les ticks tenus.
    aucs, recalls, flags = [], [], []
    for k in range(4):
        te = (life % 4 == k)
        if int(y[te].sum()) == 0:
            print(f"[saliency]   pli {k} : classe vide, sauté")
            continue
        m_k = fit(X[~te], y[~te], args.iters, args.seed)
        with torch.no_grad():
            aucs.append(_auc(m_k.tick_logits(X[te]), y[te].bool()))
            r = X[te].view(-1, N_RAY, 4)
            touch = r[..., 0] < 0.999
            rgb = r[..., 1:][touch]                      # [M, 3] rayons touchants tenus
            is_green = _green_mask(rgb)
            flagged = m_k.s(rgb) > SAL_THR
        recalls.append(float(flagged[is_green].float().mean()))
        flags.append(float(flagged[~is_green].float().mean()))
        print(f"[saliency]   pli {k} : AUC={aucs[-1]:.3f} rappel_vert={recalls[-1]:.3f} "
              f"flag_non-vert={100 * flags[-1]:.2f}%")
    auc = sum(aucs) / max(len(aucs), 1)
    recall = sum(recalls) / max(len(recalls), 1)
    flag_ng = sum(flags) / max(len(flags), 1)

    # Modèle final (tout le corpus) — l'objet déployé et jugé par G-ρ / G-feat.
    model = fit(X, y, args.iters, args.seed)
    rho = model.rho_hat()

    # GATE 3 — G-ρ : la portée apprise couvre la morsure vécue sans sur-couvrir.
    od = [_min_dist_to_points(saliency_points(model, ret)) for ret in onset_ret]
    od = [d for d in od if d == d]
    blind = len(onset_ret) - len(od)
    med = st.median(od) if od else float("nan")
    q95 = st.quantiles(od, n=20)[18] if len(od) >= 20 else float("nan")
    g_rho = len(od) > 0 and med <= rho <= q95 + 0.3

    # GATE 4 — G-feat : la lunette saillance ≡ la lunette verte aux ticks de DÉCISION.
    same = 0
    for ret in dec_ret:
        gp = green_points(ret)
        sp = saliency_points(model, ret)
        if len(gp) == len(sp) and _hausdorff(gp, sp) <= 0.05:
            same += 1
    frac_same = same / max(len(dec_ret), 1)

    tau = float(nn.functional.softplus(model.tau_raw) + 0.05)
    g_dmg = auc > 0.90
    g_loc = recall >= 0.95 and flag_ng <= 0.02
    g_feat = frac_same >= 0.99
    print(f"\n[saliency] === GATES OFFLINE (pré-enregistrés §P5) ===")
    print(f"[saliency] G-dmg  : AUC CV-4 par vie = {auc:.3f} (gate > 0.90) → {'✅' if g_dmg else '❌'} "
          f"[{', '.join(f'{a:.3f}' for a in aucs)}]")
    print(f"[saliency] G-loc  : rappel vert {recall:.3f} (≥0.95) | flag non-vert "
          f"{100 * flag_ng:.2f}% (≤2%) → {'✅' if g_loc else '❌'}")
    print(f"[saliency] G-ρ    : ρ̂={rho:.2f} m (τ={tau:.2f}) ∈ [{med:.2f}, {q95 + 0.3:.2f}] ? "
          f"(onsets aveugles : {blind}) → {'✅' if g_rho else '❌'}")
    print(f"[saliency] G-feat : lunettes identiques sur {100 * frac_same:.1f}% des "
          f"{len(dec_ret)} décisions (≥99%) → {'✅' if g_feat else '❌'}")
    verdict = g_dmg and g_loc and g_rho and g_feat
    print(f"[saliency] {'✅ GATES PASSÉS → phase B (branchement opt-in) licenciée' if verdict else '❌ GATE ÉCHOUÉ → diagnostiquer sur trace AVANT tout re-train (budget : 1)'}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "form": "saliency_v1", "thr": SAL_THR,
                "rho_hat": rho, "tau": tau, "auc_cv": auc, "recall_green": recall,
                "flag_nongreen": flag_ng, "onset_med": med, "onset_q95": q95,
                "frac_lens_same": frac_same, "runs": list(args.runs),
                "gates_pass": bool(verdict)},
               out / "saliency_best.pt")
    print(f"[saliency] sauvé → {out / 'saliency_best.pt'}")


def _hausdorff(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return float("inf")

    def _one(p, q):
        return max(min(math.hypot(px - qx, pz - qz) for qx, qz in q) for px, pz in p)

    return max(_one(a, b), _one(b, a))


def selfcheck() -> None:
    assert N_RAY * 4 == 144 and RETINA_RANGE_M == 10.0
    torch.manual_seed(0)
    m = DangerSaliency().eval()
    # g strictement décroissante en distance ; ρ̂ = croisement 0.5
    d = torch.tensor([0.0, 1.0, 2.0, 5.0])
    g = m.g(d)
    assert all(g[i] > g[i + 1] for i in range(3))
    assert abs(float(m.g(torch.tensor([m.rho_hat()]))) - 0.5) < 1e-5
    # lecture déployée ≡ green_points quand l'apparence est forcée : s = σ(20·canal_vert − 10)
    # → s(vert)≈1 > SAL_THR > s(rouge)≈0, seuil par défaut, zéro bricolage.
    with torch.no_grad():
        m.app[0].weight.zero_()
        m.app[0].bias.zero_()
        m.app[0].weight[:, 1] = 1.0                  # unités = relu(g) = g
        m.app[2].weight.zero_()
        m.app[2].weight[0, 0] = 20.0
        m.app[2].bias.fill_(-10.0)
    ret = [1.0, 0.0, 0.0, 0.0] * N_RAY
    ret[4 * 9:4 * 9 + 4] = [0.2, 0.0, 1.0, 0.0]      # vert à 2 m, 90° droite
    ret[4 * 0:4 * 0 + 4] = [0.1, 1.0, 0.0, 0.0]      # rouge à 1 m devant
    s = m.s(torch.tensor(ret, dtype=torch.float32).view(N_RAY, 4)[:, 1:])
    assert float(s[9]) > SAL_THR > float(s[0]), "s doit séparer vert/rouge sous poids forcés"
    # tick_logits : shape et insensibilité aux rayons vides (d=1.0)
    x = torch.tensor([ret, [1.0, 0.0, 0.0, 0.0] * N_RAY], dtype=torch.float32)
    lg = m.tick_logits(x)
    assert lg.shape == (2,) and abs(float(lg[1]) - float(m.bias)) < 1e-5
    # parité géométrique de la lecture : EXACTEMENT les points de green_points
    pts = saliency_points(m, ret)
    gp = green_points(ret)
    assert len(pts) == len(gp) == 1 and abs(pts[0][0] - gp[0][0]) < 1e-6 \
        and abs(pts[0][1] - gp[0][1]) < 1e-6, (pts, gp)
    assert _hausdorff([(0.0, 0.0)], [(0.0, 0.0)]) == 0.0
    assert _hausdorff([(0.0, 0.0)], []) == float("inf")

    # Intégration waypoint_layer : OFF = règle verte + marges main (bit-identique) ; ON avec
    # s≡vert ET ρ̂=1.0 → décision EXACTEMENT identique (tangent 1.0+0.4 = défaut 1.4) ; ρ̂=0.5
    # → l'intrusion suit la marge APPRISE ((ρ̂−0)⁺ sur un vert posé sur la ligne).
    import os
    import tempfile

    from sylvan.control.waypoint_layer import WaypointLayer
    target = (0.0, 4.0)
    scene = [1.0, 0.0, 0.0, 0.0] * N_RAY
    scene[0:4] = [0.2, 0.0, 1.0, 0.0]                # vert à 2 m droit devant → direct bloqué
    base = WaypointLayer()
    rec0 = base.decide("food", target, scene)
    assert rec0["intr_direct"] > 0.5, rec0
    with tempfile.TemporaryDirectory() as td:
        try:
            for rho, name in ((1.0, "same"), (0.5, "short")):
                with torch.no_grad():
                    m.rho.fill_(rho)
                torch.save({"state_dict": m.state_dict(), "thr": SAL_THR, "rho_hat": rho},
                           Path(td) / f"{name}.pt")
            os.environ["SYLVAN_WP_SALIENCY"] = str(Path(td) / "same.pt")
            lay = WaypointLayer()
            rec = lay.decide("food", target, scene)
            assert (rec["choice"], rec["cost_direct"], rec["cost_best_wp"], rec["intr_direct"],
                    rec["greens"]) == (rec0["choice"], rec0["cost_direct"], rec0["cost_best_wp"],
                                       rec0["intr_direct"], rec0["greens"]), (rec, rec0)
            os.environ["SYLVAN_WP_SALIENCY"] = str(Path(td) / "short.pt")
            lay = WaypointLayer()
            rec = lay.decide("food", target, scene)
            assert abs(rec["intr_direct"] - 0.5) < 1e-6, rec       # (0.5 − 0)⁺, un seul leg
        finally:
            os.environ.pop("SYLVAN_WP_SALIENCY", None)
    print("[selfcheck] OK — g décroissante, ρ̂=g⁻¹(0.5), lecture ≡ green_points, intégration "
          "waypoint (OFF bit-identique, ρ̂=1.0 identique, ρ̂ court suivi)")


if __name__ == "__main__":
    main()
