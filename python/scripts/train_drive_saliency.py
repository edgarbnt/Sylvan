"""Entraîne une tête de saillance de PULSION sur la CONSÉQUENCE VÉCUE. Zéro couleur codée-main.

Jumeau de `train_danger_saliency.py` (juge PASS 41/9, seul module de perception `pur`), pour
la FAIM. L'étiquette est `wm.ate` — ce que l'entité a réellement consommé. Aucune couleur,
aucune position d'oracle n'entre dans l'entraînement.

    P(je consomme bientôt | rétine) = σ( b + max_k  s_faim(rgb_k) · g_faim(d_k) )

But du chantier : MODULARITÉ, pas performance (`docs/design_perception_pure_faim.md`).
Ajouter un fruit bleu ⇒ elle le mange ⇒ elle apprend qu'il nourrit ⇒ zéro ligne touchée.

GARDE-FOUS (chacun a déjà coûté un faux verdict sur ce projet) :
  · Découpe PAR ÉPISODE, jamais aléatoire — à 0,05 m/tick les ticks voisins sont quasi
    identiques (mesuré : 0,42 m en split aléatoire contre 2,58 m par épisode, facteur 6).
  · `food_rel0` et la règle-couleur sont des ORACLES D'ÉVAL, jamais d'entraînement.
  · WM GELÉ — on n'entraîne que cette tête (81 paramètres).
  · Le corpus est `foret_v1{,b,c}_{planner,babble,explore}` en ENTIER (260 k ticks), pas le
    seul `gate_foret_cl` (9 k ticks) — l'erreur de la session du 2026-07-30.

GATES pré-enregistrés (§6 du design), CV PAR ÉPISODE :
  G-cons  AUC(P̂, tick de consommation) > 0,75
  G-gis   gisement médian du slot ≤ 23,1° (le cosinus servi) — DÉCISIF, vérité non ambiguë
  🛑 KILL gisement > 30° ou AUC < 0,65 (= activement nuisible)

G-loc du design est mesuré et rapporté mais N'EST PAS décisif : il compare à la règle-couleur
qu'on veut justement remplacer, et le G0 du 2026-08-02 a montré que la vérité per-rayon
disponible ici est faussée dans les deux sens (autres fruits comptés négatifs, occulteurs
comptés positifs). On tranche sur G-gis, de bout en bout.

Usage :
    PYTHONPATH=python SYLVAN_RETINA_FOV_DEG=120 env_pytorch_3.12/bin/python \
        -m scripts.train_drive_saliency --drive food
    ... --selfcheck    # vérifications sans données
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from torch import nn

from sylvan.models.drive_saliency import (
    LAMBDA_S,
    N_RAY,
    RETINA_DIM,
    RETINA_RANGE_M,
    SAL_THR,
    TOUCH_MAX,
    DriveSaliency,
    save_drive_saliency,
)

DEFAULT_RUNS = [
    f"data/replay_buffer/foret_v1{s}_{k}"
    for s in ("", "b", "c")
    for k in ("planner", "babble", "explore")
]
LIVE_WM = "data/checkpoints/wm_foret_v2_slot/wm_best.pt"

# Gates pré-enregistrés (design §6).
BAR_AUC = 0.75
KILL_AUC = 0.65
BAR_BEARING_DEG = 23.1  # le gisement du cosinus servi — la barre à ne pas casser
KILL_BEARING_DEG = 30.0

DEPTH_OFFSET = 0.35  # slot_head.DEPTH_OFFSET

# G-mod — LE gate du chantier. Palette SERVIE (world.FORET_V1.food_type_hues) et palette
# CONTREFACTUELLE : on repeint la nourriture du corpus, on ré-entraîne la tête sur la MÊME
# conséquence vécue, et on regarde qui survit. La règle-couleur codée-main doit s'effondrer
# (elle cherche du rouge) ; une perception apprise de la conséquence doit se retrouver.
# La palette n'est utilisée QUE pour fabriquer le contrefactuel — jamais comme cible.
FOOD_HUES = ((0.9, 0.12, 0.1), (0.9, 0.55, 0.08), (0.85, 0.1, 0.45), (0.8, 0.42, 0.42))
FOOD_HUES_ALT = ((0.1, 0.35, 0.9), (0.08, 0.75, 0.8), (0.45, 0.2, 0.9), (0.2, 0.5, 0.85))


# --------------------------------------------------------------------------- corpus


def _drive_label(rec: dict, drive: str) -> bool:
    """La CONSÉQUENCE VÉCUE, telle que le corps l'a enregistrée. Aucune couleur."""
    wm = rec.get("wm") or {}
    if drive == "food":
        return float(wm.get("ate", 0.0)) > 0.5
    if drive == "water":
        return bool(rec.get("_drank"))
    raise ValueError(f"pulsion inconnue : {drive}")


def scan_run(run: Path, drive: str, ep_base: int, keep_neg: int,
             max_eps: int | None) -> tuple[list, list, list, list]:
    """Un run -> (rétines, étiquettes, id d'épisode, food_rel0 apparié [ÉVAL ONLY])."""
    X: list[list[float]] = []
    y: list[float] = []
    eps: list[int] = []
    truth: list[list[float]] = []
    files = sorted(run.glob("*.jsonl"))
    if max_eps is not None:
        files = files[:max_eps]
    ep = ep_base
    for f in files:
        used = False
        i = 0
        prev_thirst: float | None = None
        with f.open() as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                wm = rec.get("wm")
                if wm is None:
                    continue
                ret = wm.get("retina0") or (rec.get("obs") or {}).get("retina")
                if ret is None or len(ret) != RETINA_DIM:
                    i += 1
                    continue
                if drive == "water":
                    th = float((rec.get("obs") or {}).get("thirst", 0.0))
                    rec["_drank"] = prev_thirst is not None and (th - prev_thirst) > 20.0
                    prev_thirst = th
                lab = _drive_label(rec, drive)
                # Tous les positifs ; les négatifs 1 sur keep_neg, DÉTERMINISTE par index.
                if lab or i % keep_neg == 0:
                    X.append(ret)
                    y.append(float(lab))
                    eps.append(ep)
                    truth.append(wm.get("food_rel0") or [0.0, 0.0, 0.0])
                    used = True
                i += 1
        if used:
            ep += 1
    return X, y, eps, truth


def load_corpus(runs: list[str], drive: str, keep_neg: int,
                max_eps: int | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X: list = []
    y: list = []
    eps: list = []
    truth: list = []
    base = 0
    for r in runs:
        d = Path(r)
        if not d.is_dir():
            print(f"  ⚠️  run absent, ignoré : {r}")
            continue
        a, b, c, t = scan_run(d, drive, base, keep_neg, max_eps)
        X += a
        y += b
        eps += c
        truth += t
        base = (max(c) + 1) if c else base
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(eps, dtype=torch.long),
        torch.tensor(truth, dtype=torch.float32).view(-1, 3),
    )


# --------------------------------------------------------------------------- mesure


def recolor(X: torch.Tensor, tol: float = 0.98) -> tuple[torch.Tensor, int]:
    """Repeint la nourriture du corpus avec une palette CONTREFACTUELLE (G-mod).

    Un rayon dont la teinte correspond à la palette servie prend la teinte alternative de
    même indice, en conservant sa LUMINANCE (l'ombrage du monde est préservé, seule la teinte
    change). Le reste du décor est intact. C'est l'équivalent hors-ligne, gratuit et
    reproductible, de « j'ajoute un fruit d'une nouvelle couleur ».
    """
    Xr = X.clone().view(-1, N_RAY, 4)
    rgb = Xr[..., 1:]
    norm = rgb.norm(dim=-1, keepdim=True)
    rgbn = rgb / (norm + 1e-6)
    pal = torch.tensor(FOOD_HUES)
    pal = pal / pal.norm(dim=-1, keepdim=True)
    alt = torch.tensor(FOOD_HUES_ALT)
    alt = alt / alt.norm(dim=-1, keepdim=True)
    cos = rgbn @ pal.T
    best, idx = cos.max(dim=-1)
    hit = best > tol
    Xr[..., 1:][hit] = alt[idx[hit]] * norm[hit]
    return Xr.view(X.shape), int(hit.sum())


def _auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    allv = torch.cat([pos, neg])
    rank = allv.argsort().argsort().float() + 1.0
    n1, n2 = float(pos.numel()), float(neg.numel())
    return float((rank[: pos.numel()].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def ray_angles() -> torch.Tensor:
    """Copie EXACTE de slot_head : FOV via SYLVAN_RETINA_FOV_DEG."""
    fov = math.radians(float(os.environ.get("SYLVAN_RETINA_FOV_DEG", "360")))
    return torch.tensor(
        [(k if k <= N_RAY // 2 else k - N_RAY) * fov / N_RAY for k in range(N_RAY)]
    )


def slot_from_scores(retina: torch.Tensor, score: torch.Tensor,
                     thr: float) -> tuple[torch.Tensor, torch.Tensor]:
    """LA GÉOMÉTRIE DU SLOT, inchangée — seule la SÉLECTION vient d'ailleurs.

    Reproduit `slot_head._attend` + `positions` dans sa branche multi-ressource : masque dur
    sur l'affinité, logit = log(sal·aff·prox) − 4·dist, softmax, puis découplage
    direction/distance. `score` remplace le cosinus ; tout le reste est identique.

    Renvoie (position [B,2], a-t-on un candidat [B]).
    """
    th = ray_angles()
    r = retina.view(-1, N_RAY, 4)
    d = r[..., 0]
    touch = d < TOUCH_MAX
    dist = d * RETINA_RANGE_M + DEPTH_OFFSET
    aff = (score - thr).clamp(min=0.0) * touch.float()
    prox = ((1.0 - d).clamp(min=0.0)) ** 2
    logit = torch.log(score.clamp(min=0.0) * aff * prox + 1e-8) - 4.0 * dist
    logit = torch.where(aff > 0.0, logit, torch.full_like(logit, -1e9))
    w = torch.softmax(logit, dim=-1)
    px = (w * dist * torch.sin(th)).sum(-1)
    pz = (w * dist * torch.cos(th)).sum(-1)
    n = torch.sqrt(px * px + pz * pz) + 1e-6
    rad = (w * dist).sum(-1)
    return torch.stack([px / n * rad, pz / n * rad], dim=-1), aff.any(dim=-1)


def bearing_error_deg(retina: torch.Tensor, truth: torch.Tensor, score: torch.Tensor,
                      thr: float) -> tuple[float, float, int]:
    """G-gis : erreur de GISEMENT du slot, contre la position vraie de la cible.

    Vérité NON AMBIGUË (une position, pas une étiquette par rayon) — c'est pourquoi c'est
    elle qui tranche. On ne juge que les ticks où la cible est visible ET dans le cône,
    exactement comme `diag_bilan.load_perception_pairs`.
    """
    fov = float(os.environ.get("SYLVAN_RETINA_FOV_DEG", "360"))
    tb = torch.atan2(truth[:, 0], truth[:, 1])
    keep = (truth[:, 2] > 0.5) & (tb.abs() <= math.radians(fov / 2.0))
    if int(keep.sum()) == 0:
        return float("nan"), float("nan"), 0
    pos, has = slot_from_scores(retina[keep], score[keep], thr)
    tb = tb[keep]
    ok = has
    if int(ok.sum()) == 0:
        return float("nan"), float("nan"), 0
    pb = torch.atan2(pos[ok, 0], pos[ok, 1])
    err = (pb - tb[ok]).abs()
    err = torch.minimum(err, 2 * math.pi - err) * 180.0 / math.pi
    dist_err = (pos[ok].norm(dim=-1) - truth[keep][ok, :2].norm(dim=-1)).abs()
    return float(err.median()), float(dist_err.median()), int(ok.sum())


# --------------------------------------------------------------------------- train


def fit(X: torch.Tensor, y: torch.Tensor, iters: int, seed: int) -> DriveSaliency:
    torch.manual_seed(seed)
    m = DriveSaliency()
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    pos = torch.nonzero(y > 0.5).squeeze(-1)
    neg = torch.nonzero(y <= 0.5).squeeze(-1)
    for _ in range(iters):
        # Batch ÉQUILIBRÉ : la faim n'a que 0,18 % de positifs (20x moins que le danger).
        # Sans cet équilibrage la BCE est minimisée en prédisant « jamais » et s s'effondre.
        if len(pos) and len(neg):
            bi = torch.cat([
                pos[torch.randint(0, len(pos), (1024,))],
                neg[torch.randint(0, len(neg), (3072,))],
            ])
        else:
            bi = torch.randint(0, len(X), (4096,))
        logits, s, touch = m.parts(X[bi])
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y[bi])
        if bool(touch.any()):
            loss = loss + LAMBDA_S * s[touch].mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    return m.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="food", choices=("food", "water"))
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--out", default=None)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--keep-neg", type=int, default=10)
    ap.add_argument("--max-eps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gmod", action="store_true", help="gate de MODULARITÉ (monde repeint)")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return

    out = args.out or f"data/checkpoints/drive_saliency_{args.drive}"
    print(f"pulsion : {args.drive}   corpus : {len(args.runs)} runs")
    X, y, eps, truth = load_corpus(args.runs, args.drive, args.keep_neg, args.max_eps)
    n_ep = int(eps.max()) + 1 if len(eps) else 0
    print(f"  {len(X)} ticks retenus · {int(y.sum())} positifs "
          f"({float(y.mean()) * 100:.2f} %) · {n_ep} épisodes")
    if int(y.sum()) < 50 or n_ep < args.folds:
        print("❌ corpus insuffisant")
        return

    # ---- G-cons : CV PAR ÉPISODE (jamais aléatoire)
    aucs = []
    for k in range(args.folds):
        te = (eps % args.folds) == k
        tr = ~te
        if int(y[tr].sum()) == 0 or int(y[te].sum()) == 0:
            continue
        m = fit(X[tr], y[tr], args.iters, args.seed + k)
        with torch.no_grad():
            lg = m.tick_logits(X[te])
        a = _auc(lg[y[te] > 0.5], lg[y[te] <= 0.5])
        aucs.append(a)
        print(f"  pli {k} : AUC {a:.3f}  ρ̂ {m.rho_hat():.2f} m")
    auc = sum(aucs) / len(aucs) if aucs else float("nan")

    # ---- modèle final sur tout le corpus
    model = fit(X, y, args.iters, args.seed)
    with torch.no_grad():
        s_all, _ = model.ray_scores(X)
    print(f"\n  modèle final : ρ̂ = {model.rho_hat():.2f} m   biais = {float(model.bias):.2f}")

    # ---- G-gis : le gate DÉCISIF, appris contre le cosinus servi, même géométrie
    b_learn, d_learn, n_learn = bearing_error_deg(X, truth, s_all, SAL_THR)
    payload = torch.load(LIVE_WM, map_location="cpu", weights_only=False)
    q = payload["model"]["slot_encoder.color_queries"][int(payload["meta"].get("food_idx", 0))]
    q_thr = float(payload["meta"]["query_thr"][int(payload["meta"].get("food_idx", 0))])
    rgb = X.view(-1, N_RAY, 4)[..., 1:]
    cos = (rgb / (rgb.norm(dim=-1, keepdim=True) + 1e-6)) @ q
    b_rule, d_rule, n_rule = bearing_error_deg(X, truth, cos, q_thr)

    print("\n" + "=" * 74)
    print("GATES")
    print("=" * 74)
    print(f"  G-cons : AUC {auc:.3f}   (barre {BAR_AUC}, kill {KILL_AUC})")
    print(f"  G-gis  : APPRIS  gisement {b_learn:.1f}°  distance {d_learn:.2f} m  (n={n_learn})")
    print(f"           TENANT  gisement {b_rule:.1f}°  distance {d_rule:.2f} m  (n={n_rule})")
    print(f"           barre {BAR_BEARING_DEG}°, kill {KILL_BEARING_DEG}°")

    # ---- G-mod : LE gate du chantier. On repeint la nourriture et on ré-entraîne SANS
    #      toucher une ligne de code. La règle codée-main doit tomber, l'apprise se retrouver.
    if args.gmod:
        Xr, n_hit = recolor(X)
        print("\n" + "=" * 74)
        print("G-mod — MODULARITÉ : la nourriture change de couleur, le code ne change pas")
        print("=" * 74)
        print(f"  {n_hit} rayons repeints (palette contrefactuelle)")
        m_r = fit(Xr, y, args.iters, args.seed)
        with torch.no_grad():
            s_r, _ = m_r.ray_scores(Xr)
        b_mod, d_mod, n_mod = bearing_error_deg(Xr, truth, s_r, SAL_THR)
        rgb_r = Xr.view(-1, N_RAY, 4)[..., 1:]
        cos_r = (rgb_r / (rgb_r.norm(dim=-1, keepdim=True) + 1e-6)) @ q
        b_rmod, d_rmod, n_rmod = bearing_error_deg(Xr, truth, cos_r, q_thr)
        print(f"  APPRIS ré-entraîné : gisement {b_mod:.1f}°  distance {d_mod:.2f} m  "
              f"(n={n_mod})   ρ̂ {m_r.rho_hat():.2f} m")
        print(f"  TENANT codé-main   : gisement {b_rmod:.1f}°  distance {d_rmod:.2f} m  "
              f"(n={n_rmod})")
        mod_ok = b_mod == b_mod and b_mod <= KILL_BEARING_DEG
        print(f"  {'✅' if mod_ok else '❌'} G-mod : l'appris "
              f"{'se retrouve' if mod_ok else 'ne se retrouve pas'} dans un monde repeint")
        if n_rmod == 0:
            print("      → le tenant ne trouve PLUS AUCUNE cible : la clé-apparence est morte")
            print("        avec le monde qu'elle supposait. C'est exactement ce qu'on retire.")

    killed = (auc == auc and auc < KILL_AUC) or (b_learn == b_learn and b_learn > KILL_BEARING_DEG)
    passed = (auc == auc and auc > BAR_AUC) and (b_learn == b_learn and b_learn <= BAR_BEARING_DEG)
    verdict = "KILL" if killed else ("PASS" if passed else "PARTIEL")
    print(f"\n  VERDICT : {verdict}")

    save_drive_saliency(
        model, f"{out}/saliency_best.pt", args.drive,
        auc_cv=auc, bearing_learned=b_learn, bearing_rule=b_rule,
        dist_learned=d_learn, dist_rule=d_rule, n_eval=n_learn,
        n_ticks=len(X), n_pos=int(y.sum()), n_episodes=n_ep,
        runs=list(args.runs), verdict=verdict,
    )
    print(f"  → {out}/saliency_best.pt")


# --------------------------------------------------------------------------- selfcheck


def selfcheck() -> None:
    """Vérifications sans données — le gabarit du danger."""
    m = DriveSaliency()
    d = torch.linspace(0.0, 5.0, 50)
    g = m.g(d)
    assert bool((g[1:] <= g[:-1] + 1e-6).all()), "g doit décroître avec la distance"
    assert abs(float(m.g(torch.tensor(m.rho_hat()))) - 0.5) < 1e-5, "g(ρ̂) doit valoir 0,5"

    # s ne voit QUE la couleur : même rgb à deux profondeurs -> même s.
    r = torch.zeros(2, RETINA_DIM)
    r[0, 0], r[1, 0] = 0.1, 0.8
    r[0, 1:4] = torch.tensor([0.9, 0.1, 0.1])
    r[1, 1:4] = torch.tensor([0.9, 0.1, 0.1])
    s, _ = m.ray_scores(r)
    assert abs(float(s[0, 0]) - float(s[1, 0])) < 1e-6, "s ne doit PAS dépendre de la distance"

    # une rétine vide -> logit == biais (aucun rayon ne touche)
    empty = torch.ones(1, RETINA_DIM)
    assert abs(float(m.tick_logits(empty)) - float(m.bias)) < 1e-5, "rétine vide -> biais"

    # la géométrie du slot reproduit un rayon unique planté à un angle connu
    th = ray_angles()
    k = 5
    ret = torch.ones(1, RETINA_DIM)
    ret[0, 4 * k] = 0.2
    ret[0, 4 * k + 1: 4 * k + 4] = torch.tensor([0.9, 0.1, 0.1])
    sc = torch.zeros(1, N_RAY)
    sc[0, k] = 1.0
    pos, has = slot_from_scores(ret, sc, SAL_THR)
    assert bool(has[0]), "un rayon saillant doit produire un candidat"
    exp = 0.2 * RETINA_RANGE_M + DEPTH_OFFSET
    assert abs(float(pos[0, 0]) - exp * math.sin(th[k])) < 1e-3, "x du slot"
    assert abs(float(pos[0, 1]) - exp * math.cos(th[k])) < 1e-3, "z du slot"

    # aucun rayon au-dessus du seuil -> pas de candidat
    _, has0 = slot_from_scores(ret, torch.zeros(1, N_RAY), SAL_THR)
    assert not bool(has0[0]), "sous le seuil, aucun candidat"

    print("✅ selfcheck OK")


if __name__ == "__main__":
    main()
