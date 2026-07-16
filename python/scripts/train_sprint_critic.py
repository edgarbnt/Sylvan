"""CRITIQUE-SPRINT de l'étage waypoint — forme IC+TC (docs/design_critique_sprint.md).

Forme D1 (tranchée owner 2026-07-16) : au déploiement, le scoreur ANALYTIQUE reste le socle et
    score(c) = leg1 + leg2 + (W − g(s,c)) · intrusion(c),   g = W · p(s,c) ∈ [0, W]
p(s,c) = P(la traversée PAIE | état, candidat). La correction ne touche que les candidats qui
croisent le vert et ne peut qu'ADOUCIR la pénalité (jamais l'aggraver) : g=0 ⇒ bit-identique à
l'analytique — le plancher de perf est le bras géométrie. Elle n'apprend QUE la licence de sprint,
ce que la géométrie ignore (drives, santé, douleur prédite).

Label PINNÉ (Phase 0, diag_sprint_corpus sur g24×4) : y = 1[U > 0],
U = gain_observé/drain − κ_data·dégâts_de_poursuite (LINÉAIRE — plancher-mort non retenu, 3 % <
10 %) ; drain et κ_data MESURÉS du corpus (Phase 0 : 0.05 et 9.5), jamais devinés.

GATES OFFLINE PRÉ-ENREGISTRÉS (design §gates — opérationnalisés ici, écrits AVANT le train) :
  1. G-rank  : AUC(p ordonne les traversées payantes > non-payantes) > 0.70, CV-4 par VIE ;
  2. G-res   : précision du choix simulé (traverser/refuser, hystérésis incluse) vs l'action
               empiriquement meilleure du bucket (santé×énergie×dist) : corrigé ≥ analytique
               + 10 pts, sur décisions TENUES (modèles des plis) ;
  3. G-consist : replay des séquences intra-poursuite — taux de bascule du choix simulé corrigé
               ≤ 1.2× celui de l'analytique (le gate anti-flottement que v2/v3 n'avaient pas) ;
  4. G-mono   : (correction owner du volet blessés G0, 2026-07-16 — le monde montre un GRADIENT,
               pas une inversion au seuil oracle) p̂ moyen des traversées STRICTEMENT croissant par
               bande de santé [0,30)/[30,60)/[60,100] ET strictement décroissant par tercile de
               profondeur d'intrusion.
Échec d'un gate → NE PAS brancher (négatif commité). Le juge reste le closed-loop (2×24 vies).

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python -m scripts.train_sprint_critic \
      [--runs data/replay_buffer/critic_kin_g24as1 ... critic_kin_spx3 ...] \
      [--out data/checkpoints/sprint_critic] [--selfcheck]
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import statistics as st
from pathlib import Path

import torch
from torch import nn

from scripts.train_waypoint_pain import (PainCritic, _drives_series, _open_text, _text_path,
                                         health_series, pursuit_end)
from sylvan.control.waypoint_layer import WP_FEAT_DIM, WaypointConfig

SPRINT_IN_DIM = WP_FEAT_DIM + 4   # + énergie/100, soif/100, santé/100, douleur prédite (/100)
_CFG = WaypointConfig()           # W (block_weight), hysteresis, green_margin du scoreur vivant
_LEN_CAP = 20.0                   # cap de la feature longueur (candidate_features idx 6)
_INTR_EPS = 0.02                  # marge d'arrondi (costs round(3), feats round(4))
DEFAULT_PAIN = "data/checkpoints/waypoint_pain_v3/pain_best.pt"
DEFAULT_RUNS = ["data/replay_buffer/critic_kin_g24as1", "data/replay_buffer/critic_kin_g24as2",
                "data/replay_buffer/critic_kin_g24bs1", "data/replay_buffer/critic_kin_g24bs2",
                "data/replay_buffer/critic_kin_spx3", "data/replay_buffer/critic_kin_spx4"]


class SprintCritic(nn.Module):
    """entrées [B, 14] → p = P(la traversée paie) ∈ (0, 1). Déploiement : g = W·p."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(SPRINT_IN_DIM, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def p(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x).squeeze(-1))


def sprint_inputs(feats: list[list[float]], drives: tuple[float, float, float],
                  pain: list[float]) -> torch.Tensor:
    """Assemble les entrées 14-d PAR CANDIDAT — LE point de parité train/déploiement.

    feats = `candidate_features` (déjà miroir-canoniques — la symétrie s'impose, ne se fitte pas) ;
    drives = (énergie, soif, santé) 0-100 (échelle payload/BC) ; pain = sortie BRUTE de
    PainCritic.pain (unités /100) sur les MÊMES feats — le savoir douleur bankée devient une entrée."""
    e, t, h = drives
    rows = [list(f) + [e / 100.0, t / 100.0, h / 100.0, float(p)] for f, p in zip(feats, pain)]
    x = torch.tensor(rows, dtype=torch.float32)
    assert x.shape[-1] == SPRINT_IN_DIM, x.shape
    return x


def make_checkpoint(critic: SprintCritic, pain_ckpt: str, **meta) -> dict:
    """Format de checkpoint unique (déploiement : waypoint_layer recharge pain_ckpt d'ici —
    la parité de la feature douleur est portée par le chemin bankée, pas par une convention)."""
    return {"state_dict": critic.state_dict(), "in_dim": SPRINT_IN_DIM,
            "pain_ckpt": pain_ckpt, **meta}


# ------------------------------------------------------------------ corpus (partagé avec le diag)

def route_intrusions(d: dict) -> list[float]:
    """Intrusion EXACTE par candidat, reconstruite de costs − longueur (corpus à coûts analytiques).

    NaN si la longueur sature le cap de la feature (rare : spawns 2-8 m). Ne PAS utiliser sur un
    corpus collecté avec un scoreur appris (costs ≠ analytique) — les corpus post-Phase-A loguent
    `intr` explicitement, préféré quand présent."""
    out = []
    for c, f in zip(d["costs"], d["feats"]):
        length = f[6] * _LEN_CAP
        if f[6] >= 0.9995:
            out.append(float("nan"))
        else:
            out.append(max(0.0, (c - length) / _CFG.block_weight))
    return out


def load_sprint_decisions(run: Path, life_base: int = 0) -> list[dict]:
    """→ une entrée par décision labellisable : état (e,t,h), classe cross/refuse/clear, issue de
    POURSUITE (conventions v3 partagées via pursuit_end), candidats complets (simulation offline)."""
    df, gl = run / "decisions.jsonl", run / "godot.log"
    if _text_path(df) is None or _text_path(gl) is None:
        print(f"[sprint] ⚠️ {run} incomplet (decisions.jsonl/godot.log) — ignoré")
        return []
    decs = [json.loads(line) for line in _open_text(df)]
    es, ts, hs = _drives_series(run)
    gticks, gvals, gstarts = health_series(gl)
    ep_bounds = gstarts[1:] + [gticks[-1] + 10]
    bc_health = not all(math.isnan(h) for h in hs)

    def h_at(t: int) -> float:
        if bc_health and t < len(hs):
            return hs[t]
        i = bisect.bisect_left(gticks, t)          # fallback log Godot (échantillonné /10)
        return gvals[min(i, len(gvals) - 1)]

    out = []
    for i, d in enumerate(decs):
        t0 = d["tick"]
        if t0 >= len(es):                           # queue de log au-delà du BC : rare, ignoré
            continue
        b = bisect.bisect_right(ep_bounds, t0)
        end = ep_bounds[b] if b < len(ep_bounds) else gticks[-1] + 10
        drv = es if d["target"] == "food" else ts
        t1 = pursuit_end(decs, i, drv, end)
        if t1 <= t0 + 20:                           # fenêtre vide (parité trainer douleur) : sautée
            continue
        gain, got = 0.0, False
        for t in range(t0 + 1, min(t1 + 1, len(drv))):
            if drv[t] > drv[t - 1] + 5.0:
                gain, got = drv[t] - drv[t - 1], True
                break
        h0 = h_at(t0)                               # baseline dégâts (jointure exacte)
        hmin = min(h_at(t) for t in range(t0, min(t1 + 1, len(es)))) if bc_health else \
            min(h_at(t0), h_at(t1))
        # corpus post-Phase-A : intrusion exacte + drives VUS par l'étage sont loggés (additif) ;
        # anciens corpus : reconstruction costs−longueur + jointure tick.
        intr = d.get("intr") or route_intrusions(d)
        e0, t0v, h0v = d["drives"] if d.get("drives") else (es[t0], ts[t0], h0)
        chosen_i, direct_i = intr[d["chosen"]], intr[0]
        cls = ("cross" if chosen_i == chosen_i and chosen_i > _INTR_EPS else
               "refuse" if direct_i == direct_i and direct_i > _INTR_EPS else "clear")
        out.append({
            "run": run.name, "tick": t0, "target": d["target"], "explore": bool(d["explore"]),
            "cls": cls, "intr_chosen": chosen_i, "intr_direct": direct_i, "intr_all": intr,
            "e": e0, "t": t0v, "h": h0v, "d_tg": d["feats"][0][3] * 10.0,
            "got": got, "gain": gain, "dmg": max(0.0, h0 - hmin),
            "died": t1 >= end - 2, "left": max(end - 1 - t0, 0), "steps": t1 - t0,
            "life": life_base + b, "chosen": d["chosen"], "feats_all": d["feats"],
            "costs": d["costs"],
        })
    return out


def load_corpus(runs: list[str]) -> list[dict]:
    rows = []
    for k, run in enumerate(runs):
        rows += load_sprint_decisions(Path(run), life_base=1000 * k)
    return rows


def measured_drain(runs: list[str | Path]) -> float:
    """Drain énergie/pas MESURÉ : médiane des baisses tick-à-tick (les remontées = repas/respawns)."""
    drops = []
    for run in runs:
        es, _, _ = _drives_series(Path(run))
        drops += [es[t - 1] - es[t] for t in range(1, len(es))
                  if 0.0 < es[t - 1] - es[t] < 1.0]
    return st.median(drops) if drops else float("nan")


def net_utility(r: dict, kappa: float, drain: float) -> float:
    """U en pas de vie (label PINNÉ Phase 0 : LINÉAIRE — la mort est prix par la santé perdue)."""
    return r["gain"] / drain - kappa * r["dmg"]


# ------------------------------------------------------------------ simulation offline du choix

def _pain_of(pain_model: PainCritic, feats: list[list[float]]) -> list[float]:
    with torch.no_grad():
        return pain_model.pain(torch.tensor(feats, dtype=torch.float32)).tolist()


def simulate_choice(r: dict, critic: SprintCritic | None,
                    pain_model: PainCritic | None) -> bool:
    """Rejoue la règle de choix de decide() (argmin + hystérésis pro-direct) sur les candidats
    loggés → True si le choix TRAVERSE (intr>ε). critic=None ⇒ scoreur analytique pur."""
    costs = list(r["costs"])
    if critic is not None:
        x = sprint_inputs(r["feats_all"], (r["e"], r["t"], r["h"]),
                          _pain_of(pain_model, r["feats_all"]))
        with torch.no_grad():
            p = critic.p(x)
        costs = [c - _CFG.block_weight * float(p[i]) * (r["intr_all"][i] or 0.0)
                 if r["intr_all"][i] == r["intr_all"][i] else c
                 for i, c in enumerate(costs)]
    best_i = min(range(1, len(costs)), key=lambda i: costs[i])
    chosen = best_i if costs[best_i] < costs[0] * (1.0 - _CFG.hysteresis) else 0
    intr_c = r["intr_all"][chosen]
    return bool(intr_c == intr_c and intr_c > _INTR_EPS)


def _bucket_key(r: dict) -> tuple[bool, bool, bool]:
    return (r["h"] > 60.0, r["e"] < 50.0, r["d_tg"] < 3.0)


def _auc(score: torch.Tensor, label: torch.Tensor) -> float:
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = pos.unsqueeze(1) - neg.unsqueeze(0)
    return float(((d > 0).float() + 0.5 * (d == 0).float()).mean())


# ------------------------------------------------------------------ entraînement + gates

def train(args: argparse.Namespace) -> None:
    rows = load_corpus(args.runs)
    if not rows:
        raise SystemExit("[sprint] aucun corpus lisible")
    drain = measured_drain(args.runs)
    kappa = st.median([r["left"] for r in rows]) / 100.0
    cross = [r for r in rows if r["cls"] == "cross"]
    print(f"[sprint] corpus : {len(rows)} décisions ({len(cross)} traversées, "
          f"{sum(1 for r in rows if r['explore'])} ε) | drain={drain:.4f} κ={kappa:.1f} pas/dégât")

    pain_model = PainCritic()
    _pk = torch.load(args.pain, map_location="cpu", weights_only=True)
    pain_model.load_state_dict(_pk["state_dict"])
    pain_model.eval()

    X = torch.cat([sprint_inputs([r["feats_all"][r["chosen"]]], (r["e"], r["t"], r["h"]),
                                 _pain_of(pain_model, [r["feats_all"][r["chosen"]]]))
                   for r in cross])
    y = torch.tensor([net_utility(r, kappa, drain) > 0.0 for r in cross], dtype=torch.float32)
    life = torch.tensor([r["life"] for r in cross])
    print(f"[sprint] label : {int(y.sum())}/{len(y)} traversées payantes ({100 * float(y.mean()):.0f}%)")

    def fit(mask: torch.Tensor) -> SprintCritic:
        torch.manual_seed(args.seed)
        c = SprintCritic()
        opt = torch.optim.Adam(c.parameters(), 2e-3, weight_decay=1e-4)
        Xt, yt = X[mask], y[mask]
        for _ in range(args.iters):
            bi = torch.randint(0, len(Xt), (256,))
            logits = c.net(Xt[bi]).squeeze(-1)
            nn.functional.binary_cross_entropy_with_logits(logits, yt[bi]).backward()
            opt.step()
            opt.zero_grad()
        return c.eval()

    # GATE 1 — G-rank : AUC(p, traversée payante) en CV 4 plis PAR VIE (le gate décisionnel :
    # ranger les traversées payantes au-dessus des non-payantes, dans le set sprint-pertinent).
    aucs, fold_models = [], {}
    for k in range(4):
        te = (life % 4 == k)
        if int(y[te].sum()) == 0 or int((1 - y[te]).sum()) == 0 or int(te.sum()) == 0:
            print(f"[sprint]   pli {k} : classe vide, sauté")
            continue
        c_k = fit(~te)
        fold_models[k] = c_k
        with torch.no_grad():
            aucs.append(_auc(c_k.p(X[te]), y[te].bool()))
        print(f"[sprint]   pli {k} : AUC={aucs[-1]:.3f} (n_te={int(te.sum())}, pay={int(y[te].sum())})")
    auc = sum(aucs) / max(len(aucs), 1)

    # GATE 2 — G-res : le choix SIMULÉ (hystérésis incluse) matche l'action empiriquement meilleure
    # du bucket (santé×énergie×dist, référence = tout le corpus bloqué) mieux que l'analytique seul.
    blocked = [r for r in rows if r["intr_direct"] == r["intr_direct"]
               and r["intr_direct"] > _INTR_EPS]
    better: dict[tuple, bool] = {}
    for key in {_bucket_key(r) for r in blocked}:
        cr = [net_utility(r, kappa, drain) for r in blocked
              if _bucket_key(r) == key and r["cls"] == "cross"]
        rf = [net_utility(r, kappa, drain) for r in blocked
              if _bucket_key(r) == key and r["cls"] == "refuse"]
        if len(cr) >= 10 and len(rf) >= 10:
            better[key] = st.mean(cr) > st.mean(rf)   # True = traverser est l'action meilleure
    evalable = [r for r in blocked if _bucket_key(r) in better]
    acc_ana = acc_cor = n_eval = 0
    for r in evalable:
        k = int(r["life"] % 4)
        if k not in fold_models:
            continue
        n_eval += 1
        want = better[_bucket_key(r)]
        acc_ana += simulate_choice(r, None, None) == want
        acc_cor += simulate_choice(r, fold_models[k], pain_model) == want
    acc_ana = acc_ana / max(n_eval, 1)
    acc_cor = acc_cor / max(n_eval, 1)
    print(f"[sprint] G-res : buckets jugés={len(better)} n_eval={n_eval} | "
          f"analytique {100 * acc_ana:.0f}% vs corrigé {100 * acc_cor:.0f}%")

    # GATE 3 — G-consist : bascules du choix simulé entre décisions CONSÉCUTIVES d'une même
    # poursuite (même run, même cible, gap ≤ 60 ticks). Le flottement des notes MC par état a tué
    # v2/v3 — la correction ne doit pas re-faire flotter le socle analytique.
    critic = fit(torch.ones(len(X), dtype=torch.bool))
    flips_ana = flips_cor = n_pairs = 0
    by_run: dict[str, list[dict]] = {}
    for r in rows:
        by_run.setdefault(r["run"], []).append(r)
    for seq in by_run.values():
        seq.sort(key=lambda r: r["tick"])
        for a, b_ in zip(seq, seq[1:]):
            if a["target"] != b_["target"] or b_["tick"] - a["tick"] > 60:
                continue
            n_pairs += 1
            flips_ana += simulate_choice(a, None, None) != simulate_choice(b_, None, None)
            flips_cor += (simulate_choice(a, critic, pain_model)
                          != simulate_choice(b_, critic, pain_model))
    rate_ana = flips_ana / max(n_pairs, 1)
    rate_cor = flips_cor / max(n_pairs, 1)
    print(f"[sprint] G-consist : {n_pairs} paires | bascule analytique {100 * rate_ana:.1f}% "
          f"vs corrigé {100 * rate_cor:.1f}%")

    # GATE 4 — G-mono (correction owner G0) : le gradient appris doit suivre le gradient vécu —
    # p̂ croît avec la santé (bandes 0-30/30-60/60+) et décroît avec la profondeur d'intrusion.
    with torch.no_grad():
        p_all = critic.p(X)
    h_bands = [(lo, hi) for lo, hi in ((0.0, 30.0), (30.0, 60.0), (60.0, 101.0))]
    p_by_h = []
    for lo, hi in h_bands:
        m = torch.tensor([lo <= r["h"] < hi for r in cross])
        p_by_h.append(float(p_all[m].mean()) if m.any() else float("nan"))
    depths = [r["intr_chosen"] for r in cross]
    cuts = st.quantiles(depths, n=3)
    p_by_d = []
    for lo, hi in ((0.0, cuts[0]), (cuts[0], cuts[1]), (cuts[1], 1e9)):
        m = torch.tensor([lo <= r["intr_chosen"] < hi for r in cross])
        p_by_d.append(float(p_all[m].mean()) if m.any() else float("nan"))
    g_mono = (p_by_h[0] < p_by_h[1] < p_by_h[2]) and (p_by_d[0] > p_by_d[1] > p_by_d[2])
    print(f"[sprint] G-mono : p̂ par santé {['%.2f' % v for v in p_by_h]} (croissant ?) | "
          f"p̂ par profondeur {['%.2f' % v for v in p_by_d]} (décroissant ?)")

    g_rank = auc > 0.70
    g_res = acc_cor >= acc_ana + 0.10
    g_consist = rate_cor <= 1.2 * rate_ana + 1e-9
    print(f"\n[sprint] === GATES OFFLINE (pré-enregistrés) ===")
    print(f"[sprint] G-rank    : AUC CV-4 par vie = {auc:.3f} (gate > 0.70) → {'✅' if g_rank else '❌'} "
          f"[{', '.join(f'{a:.3f}' for a in aucs)}]")
    print(f"[sprint] G-res     : {100 * acc_cor:.0f}% ≥ {100 * acc_ana:.0f}% + 10 pts → {'✅' if g_res else '❌'}")
    print(f"[sprint] G-consist : {100 * rate_cor:.1f}% ≤ 1.2×{100 * rate_ana:.1f}% → {'✅' if g_consist else '❌'}")
    print(f"[sprint] G-mono    : santé {'↑' if p_by_h[0] < p_by_h[1] < p_by_h[2] else '✗'} "
          f"profondeur {'↓' if p_by_d[0] > p_by_d[1] > p_by_d[2] else '✗'} → {'✅' if g_mono else '❌'}")
    verdict = g_rank and g_res and g_consist and g_mono
    print(f"[sprint] {'✅ GATES PASSÉS → juge closed-loop (2×24 vies seeds 1+2)' if verdict else '❌ GATE ÉCHOUÉ → ne pas brancher, commiter le négatif'}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(make_checkpoint(critic, args.pain, auc_cv=auc, acc_ana=acc_ana, acc_cor=acc_cor,
                               flip_ana=rate_ana, flip_cor=rate_cor, p_by_health=p_by_h,
                               p_by_depth=p_by_d, kappa_data=kappa,
                               drain=drain, label="linear_pursuit", runs=list(args.runs),
                               gates_pass=bool(verdict)),
               out / "sprint_best.pt")
    print(f"[sprint] sauvé → {out / 'sprint_best.pt'}")


def selfcheck() -> None:
    """Contrat 14-d + intégration waypoint_layer : g≡0 ⇒ bit-identique à l'analytique ;
    g≡W ⇒ la licence ouvre le direct bloqué ; exclusivité des modes de scoring."""
    import os
    import tempfile

    from sylvan.control.waypoint_layer import WaypointLayer

    x = sprint_inputs([[0.1] * WP_FEAT_DIM] * 3, (30.0, 70.0, 100.0), [0.1, 0.2, 0.3])
    assert tuple(x.shape) == (3, SPRINT_IN_DIM) and abs(float(x[1, -1]) - 0.2) < 1e-6

    # scène synthétique : cible à 4 m droit devant, vert SUR la ligne à 2 m → direct bloqué.
    retina = [1.0, 0.0, 0.0, 0.0] * 36
    retina[0:4] = [0.2, 0.0, 1.0, 0.0]           # rayon k=0 (droit devant), d=0.2 → vert à 2 m
    target = (0.0, 4.0)
    if not Path(DEFAULT_PAIN).exists():
        print("[selfcheck] ⚠️ pain_v3 absent — intégration sautée (contrat 14-d seul vérifié)")
        return
    base = WaypointLayer()
    rec0 = base.decide("food", target, retina)
    assert rec0["intr_direct"] > 0.5, rec0        # la scène bloque bien le direct

    with tempfile.TemporaryDirectory() as td:
        for bias, name in ((-20.0, "g0"), (+20.0, "gW")):
            c = SprintCritic()
            with torch.no_grad():
                c.net[-1].weight.zero_()
                c.net[-1].bias.fill_(bias)        # σ(∓20) → p ≈ 0 / 1 quelle que soit l'entrée
            torch.save(make_checkpoint(c, DEFAULT_PAIN), Path(td) / f"{name}.pt")
        try:
            os.environ["SYLVAN_WP_SPRINT_CRITIC"] = str(Path(td) / "g0.pt")
            lay = WaypointLayer()
            lay.maybe_decide("food", target, retina, drives=(30.0, 70.0, 100.0))
            rec = lay.decide("food", target, retina)
            assert (rec["choice"], rec["cost_direct"], rec["cost_best_wp"]) == \
                (rec0["choice"], rec0["cost_direct"], rec0["cost_best_wp"]), (rec, rec0)

            os.environ["SYLVAN_WP_SPRINT_CRITIC"] = str(Path(td) / "gW.pt")
            lay = WaypointLayer()
            lay.maybe_decide("food", target, retina, drives=(30.0, 70.0, 100.0))
            rec = lay.decide("food", target, retina)
            assert rec["choice"] == "direct", rec  # pénalité verte licenciée → direct ≈ 4 m gagne
            assert abs(rec["cost_direct"] - math.hypot(*target)) < 0.2, rec

            os.environ["SYLVAN_WP_ORACLE_SPRINT"] = "1"
            try:
                WaypointLayer()
                raise AssertionError("exclusivité sprint-critic/oracle non levée")
            except ValueError:
                pass
        finally:
            os.environ.pop("SYLVAN_WP_SPRINT_CRITIC", None)
            os.environ.pop("SYLVAN_WP_ORACLE_SPRINT", None)
    print("[selfcheck] OK — 14-d, g≡0 bit-identique, g≡W licencie le direct, exclusivité")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--pain", default=DEFAULT_PAIN)
    ap.add_argument("--out", default="data/checkpoints/sprint_critic")
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    train(args)


if __name__ == "__main__":
    main()
