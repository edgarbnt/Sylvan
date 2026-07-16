"""CRITIQUE-SPRINT de l'étage waypoint — forme IC+TC (docs/design_critique_sprint.md).

Forme D1 (tranchée owner 2026-07-16), REPRISE v2 (owner, après le négatif n°1 — le SIGNE de U
jetait le signal risque, y≡repas 97.6 %) : le critique RÉGRESSE LA MAGNITUDE et au déploiement
    remise(c) = min(W·intrusion(c), 0.02 · max(0, Q̂(s,c)·100)) ;  score(c) = route_cost(c) − remise
Q̂(s,c) = U prédit de la traversée (unités U/100, signé). 0.02 m/pas = constante CALIBRÉE du corps
(waypoint_layer k_fwd), κ_data et drain MESURÉS du corpus — zéro constante libre. La remise ne
touche que les candidats à intrusion>0 et est capée à la pénalité verte (jamais d'aggravation,
jamais d'inversion) : Q̂≤0 ⇒ bit-identique à l'analytique — le plancher de perf est le bras
géométrie. Elle n'apprend QUE la licence de sprint, ce que la géométrie ignore.

Label PINNÉ (Phase 0) : U = gain_observé/drain − κ_data·dégâts_de_poursuite (LINÉAIRE —
plancher-mort non retenu, 3 % < 10 %). Le négatif n°1 a prouvé que la magnitude porte la santé
(U̅|repas 557/591/716 par bande) là où le signe ne porte que « repas obtenu ».

REPRISE n°2 — TÊTES COMPOSÉES (owner 2026-07-16, `--form composed`, DÉFAUT) : après les 2 négatifs
(signe = risque jeté ; magnitude = variance bimodale), SÉPARER les liens appris :
    remise(c) = min(W·intr(c), 0.02 · max(0, P̂(repas|s,c)·bénéfice(drive) − κ_data·douleur̂_v3(c)·100))
P̂ = seule tête entraînée (BCE sur `got`, TOUTES les décisions — la santé y entre par
mourir-avant-de-manger) ; bénéfice = min(restore_mesuré, 100−drive)/drain (satiété EXACTE) ;
douleur̂ = pain_v3 GELÉ. U reste le CRITÈRE des gates (y = U>0), plus le label d'entraînement.
La forme q (régression U) reste disponible (`--form q`) pour reproduire le négatif n°2.

GATES OFFLINE PRÉ-ENREGISTRÉS (design §gates — opérationnalisés ici, écrits AVANT le re-train) :
  1. G-rank  : AUC(Q̂ ordonne les traversées payantes > non-payantes) > 0.70, CV-4 par VIE ;
  2. G-res   : précision du choix simulé (traverser/refuser, hystérésis incluse) vs l'action
               empiriquement meilleure du bucket (santé×énergie×dist) : corrigé ≥ analytique
               + 10 pts, sur décisions TENUES (modèles des plis) ;
  3. G-consist : replay des séquences intra-poursuite — taux de bascule du choix simulé corrigé
               ≤ 1.2× celui de l'analytique (le gate anti-flottement que v2/v3 n'avaient pas) ;
  4. G-mono v2 : (owner 2026-07-16, CONDITIONNÉ où le risque vit — le volet non conditionné était
               confondu avec la proximité) Q̂ moyen STRICTEMENT croissant par bande de santé
               [0,30)/[30,60)/[60,100] PARMI les traversées PROFONDES (intr > médiane), ET
               strictement décroissant par tercile de profondeur PARMI les BLESSÉS (h<60).
Échec d'un gate → NE PAS brancher (négatif commité, budget re-train ÉPUISÉ). Le juge reste le
closed-loop (2×24 vies).

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
# corpus élargi de la tête P(mort) (P2-bis) : + les bras juge/pure (riches en morts-danger).
# ⚠️ leurs `costs` loggés ne sont PAS analytiques (leur forme de scoring) → train de têtes
# seulement ; les replays de gates (simulate_choice analytique) restent sur DEFAULT_RUNS.
DEATH_RUNS = DEFAULT_RUNS + [
    "data/replay_buffer/critic_kin_judge1", "data/replay_buffer/critic_kin_judge2",
    "data/replay_buffer/critic_kin_pure1", "data/replay_buffer/critic_kin_pure2"]


class SprintCritic(nn.Module):
    """entrées [B, 14] → Q̂ = U prédit de la traversée (unités U/100, SIGNÉ — la magnitude porte
    le risque, leçon du négatif n°1). Déploiement : remise = min(W·intr, 2·max(0, Q̂))."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(SPRINT_IN_DIM, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def q(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def p(self, x: torch.Tensor) -> torch.Tensor:
        """Lecture sigmoïde (forme composée : P̂(repas) ∈ (0,1))."""
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


def make_checkpoint(critic: SprintCritic, pain_ckpt: str, form: str = "q_regression",
                    **meta) -> dict:
    """Format de checkpoint unique (déploiement : waypoint_layer recharge pain_ckpt d'ici —
    la parité de la feature douleur est portée par le chemin bankée, pas par une convention).
    form ∈ {q_regression, composed_v1} — décide la lecture (q vs p) et l'algèbre de remise."""
    return {"state_dict": critic.state_dict(), "in_dim": SPRINT_IN_DIM,
            "form": form, "pain_ckpt": pain_ckpt, **meta}


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

    # santé FINALE par épisode (dernier échantillon Godot de l'épisode) — robuste à la dérive des
    # frontières approximées (+10/ép), contrairement à h_at(end−1) qui peut déborder sur la vie
    # suivante (santé reset 100). Sémantique parse_lives : mort-danger ⇔ santé finale < 15.
    ep_last_h: dict[int, float] = {}
    for t, v in zip(gticks, gvals):
        ep_last_h[bisect.bisect_right(gstarts, t) - 1] = v

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
            "died": t1 >= end - 2,
            # mort-DANGER = la vie se termine dans la fenêtre ET santé finale <15 (sémantique
            # parse_lives ; faim/soif gardent leur santé) — label de P̂mort (P2-bis)
            "died_danger": t1 >= end - 2 and ep_last_h.get(b, 100.0) < 15.0,
            "left": max(end - 1 - t0, 0), "steps": t1 - t0,
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


def simulate_choice(r: dict, critic: SprintCritic | None, pain_model: PainCritic | None, *,
                    composed: bool = False, pure: bool = False, kappa: float = 0.0,
                    drain: float = 0.05, restore: float = 40.0) -> bool:
    """Rejoue la règle de choix de decide() (argmin + hystérésis pro-direct) sur les candidats
    loggés → True si le choix TRAVERSE (intr>ε). critic=None ⇒ scoreur analytique pur.
    composed=False : remise = min(W·intr, 2·max(0, Q̂)) ; composed=True : remise =
    min(W·intr, 0.02·max(0, P̂·bénéfice(drive) − κ·douleur̂·100)) — parité déploiement.
    pure=True (P2, docs/design_purete_hjepa.md) : REMPLACEMENT — score = longueur +
    0.02·max(0, κ·douleur̂·100 − P̂·bénéfice) ; W/green_margin sortent du chemin décisionnel."""
    costs = list(r["costs"])
    if critic is not None:
        pains = _pain_of(pain_model, r["feats_all"])
        x = sprint_inputs(r["feats_all"], (r["e"], r["t"], r["h"]), pains)
        drive = r["e"] if r["target"] == "food" else r["t"]
        ben = min(restore, 100.0 - drive) / drain
        with torch.no_grad():
            if pure:
                p = critic.p(x)
                costs = [f[6] * _LEN_CAP
                         + 0.02 * max(0.0, kappa * pains[i] * 100.0 - float(p[i]) * ben)
                         for i, f in enumerate(r["feats_all"])]
                intr = r["intr_all"]
                best_i = min(range(1, len(costs)), key=lambda i: costs[i])
                chosen = best_i if costs[best_i] < costs[0] * (1.0 - _CFG.hysteresis) else 0
                intr_c = intr[chosen]
                return bool(intr_c == intr_c and intr_c > _INTR_EPS)
            if composed:
                p = critic.p(x)
                vals = [0.02 * max(0.0, float(p[i]) * ben - kappa * pains[i] * 100.0)
                        for i in range(len(costs))]
            else:
                q = critic.q(x)
                vals = [2.0 * max(0.0, float(q[i])) for i in range(len(costs))]
        new_costs = []
        for i, c in enumerate(costs):
            intr_i = r["intr_all"][i]
            if intr_i == intr_i and intr_i > 0.0:
                c = c - min(_CFG.block_weight * intr_i, vals[i])
            new_costs.append(c)
        costs = new_costs
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

    composed = args.form == "composed"
    got_gains = [r["gain"] for r in rows if r["got"]]
    restore = st.median(got_gains) if got_gains else 40.0   # valeur d'un repas MESURÉE (≈40 pts)

    def inputs_of(rs: list[dict]) -> torch.Tensor:
        return torch.cat([sprint_inputs([r["feats_all"][r["chosen"]]], (r["e"], r["t"], r["h"]),
                                        _pain_of(pain_model, [r["feats_all"][r["chosen"]]]))
                          for r in rs])

    # éval G-rank : toujours sur les TRAVERSÉES (le set sprint-pertinent) ; y = « la traversée
    # a payé » (signe de U, inchangé — c'est le CRITÈRE, plus le label d'entraînement).
    X = inputs_of(cross)
    u = torch.tensor([net_utility(r, kappa, drain) / 100.0 for r in cross], dtype=torch.float32)
    y = (u > 0.0)
    life = torch.tensor([r["life"] for r in cross])
    # entraînement : forme COMPOSÉE = P̂(repas) BCE sur `got`, TOUTES les décisions labellisées
    # (cible binaire propre, la santé entre par mourir-avant-de-manger) ; forme q = MSE sur U/100.
    if composed:
        Xtr = inputs_of(rows)
        ttr = torch.tensor([float(r["got"]) for r in rows])
        life_tr = torch.tensor([r["life"] for r in rows])
        print(f"[sprint] forme COMPOSÉE : P̂(repas) sur {len(rows)} décisions "
              f"(got {int(ttr.sum())} = {100 * float(ttr.mean()):.0f}%) | "
              f"bénéfice = min({restore:.0f}, 100−drive)/{drain:.4f} | douleur = pain_v3 GELÉ")
    else:
        Xtr, ttr, life_tr = X, u, life
        print(f"[sprint] forme q : label U/100 méd={float(u.median()):.2f} "
              f"q1/q3={float(u.quantile(0.25)):.2f}/{float(u.quantile(0.75)):.2f} | "
              f"payantes {int(y.sum())}/{len(y)} ({100 * float(y.float().mean()):.0f}%)")

    def fit(mask: torch.Tensor) -> SprintCritic:
        torch.manual_seed(args.seed)
        c = SprintCritic()
        opt = torch.optim.Adam(c.parameters(), 2e-3, weight_decay=1e-4)
        Xt, tt = Xtr[mask], ttr[mask]
        for _ in range(args.iters):
            bi = torch.randint(0, len(Xt), (256,))
            out = c.net(Xt[bi]).squeeze(-1)
            loss = (nn.functional.binary_cross_entropy_with_logits(out, tt[bi]) if composed
                    else nn.functional.mse_loss(out, tt[bi]))
            loss.backward()
            opt.step()
            opt.zero_grad()
        return c.eval()

    # score décisionnel par traversée (en pas de vie) — l'objet que les gates jugent.
    ben_c = torch.tensor([min(restore, 100.0 - (r["e"] if r["target"] == "food" else r["t"])) / drain
                          for r in cross])
    pain_c = torch.tensor([_pain_of(pain_model, [r["feats_all"][r["chosen"]]])[0] for r in cross])

    def crossing_scores(model: SprintCritic) -> torch.Tensor:
        with torch.no_grad():
            if composed:
                return model.p(X) * ben_c - kappa * pain_c * 100.0
            return model.q(X) * 100.0

    # GATE 1 — G-rank : AUC(score, traversée payante) en CV 4 plis PAR VIE (le gate décisionnel).
    aucs, fold_models = [], {}
    for k in range(4):
        te_c = (life % 4 == k)
        if int(y[te_c].sum()) == 0 or int((~y[te_c]).sum()) == 0:
            print(f"[sprint]   pli {k} : classe vide, sauté")
            continue
        c_k = fit(~(life_tr % 4 == k))
        fold_models[k] = c_k
        aucs.append(_auc(crossing_scores(c_k)[te_c], y[te_c]))
        print(f"[sprint]   pli {k} : AUC={aucs[-1]:.3f} (n_te={int(te_c.sum())}, pay={int(y[te_c].sum())})")
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
        acc_cor += simulate_choice(r, fold_models[k], pain_model, composed=composed,
                                   kappa=kappa, drain=drain, restore=restore) == want
    acc_ana = acc_ana / max(n_eval, 1)
    acc_cor = acc_cor / max(n_eval, 1)
    print(f"[sprint] G-res : buckets jugés={len(better)} n_eval={n_eval} | "
          f"analytique {100 * acc_ana:.0f}% vs corrigé {100 * acc_cor:.0f}%")

    # GATE 3 — G-consist : bascules du choix simulé entre décisions CONSÉCUTIVES d'une même
    # poursuite (même run, même cible, gap ≤ 60 ticks). Le flottement des notes MC par état a tué
    # v2/v3 — la correction ne doit pas re-faire flotter le socle analytique.
    critic = fit(torch.ones(len(Xtr), dtype=torch.bool))
    flips_ana = flips_cor = n_pairs = 0
    by_run: dict[str, list[dict]] = {}
    for r in rows:
        by_run.setdefault(r["run"], []).append(r)
    kw = dict(composed=composed, kappa=kappa, drain=drain, restore=restore)
    for seq in by_run.values():
        seq.sort(key=lambda r: r["tick"])
        for a, b_ in zip(seq, seq[1:]):
            if a["target"] != b_["target"] or b_["tick"] - a["tick"] > 60:
                continue
            n_pairs += 1
            flips_ana += simulate_choice(a, None, None) != simulate_choice(b_, None, None)
            flips_cor += (simulate_choice(a, critic, pain_model, **kw)
                          != simulate_choice(b_, critic, pain_model, **kw))
    rate_ana = flips_ana / max(n_pairs, 1)
    rate_cor = flips_cor / max(n_pairs, 1)
    print(f"[sprint] G-consist : {n_pairs} paires | bascule analytique {100 * rate_ana:.1f}% "
          f"vs corrigé {100 * rate_cor:.1f}%")

    # GATE 4 — G-mono v2 (owner, CONDITIONNÉ où le risque vit — le volet non conditionné était
    # confondu proximité) : Q̂ croissant en santé PARMI les traversées PROFONDES (intr > médiane) ;
    # Q̂ décroissant en profondeur PARMI les BLESSÉS (h<60).
    q_all = crossing_scores(critic)

    def _mean_at(idx: list[int]) -> float:
        return float(q_all[torch.tensor(idx)].mean()) if len(idx) >= 15 else float("nan")

    med_d = st.median([r["intr_chosen"] for r in cross])
    deep = [i for i, r in enumerate(cross) if r["intr_chosen"] > med_d]
    q_by_h = [_mean_at([i for i in deep if lo <= cross[i]["h"] < hi])
              for lo, hi in ((0.0, 30.0), (30.0, 60.0), (60.0, 101.0))]
    wounded = [i for i, r in enumerate(cross) if r["h"] < 60.0]
    if len(wounded) >= 45:
        cuts = st.quantiles([cross[i]["intr_chosen"] for i in wounded], n=3)
        q_by_d = [_mean_at([i for i in wounded if lo <= cross[i]["intr_chosen"] < hi])
                  for lo, hi in ((0.0, cuts[0]), (cuts[0], cuts[1]), (cuts[1], 1e9))]
    else:
        q_by_d = [float("nan")] * 3
    g_mono = (q_by_h[0] < q_by_h[1] < q_by_h[2]) and (q_by_d[0] > q_by_d[1] > q_by_d[2])
    print(f"[sprint] G-mono v2 : Q̂|profond par santé {['%.2f' % v for v in q_by_h]} (croissant ?) | "
          f"Q̂|blessé par profondeur {['%.2f' % v for v in q_by_d]} (décroissant ?)")

    g_rank = auc > 0.70
    g_res = acc_cor >= acc_ana + 0.10
    g_consist = rate_cor <= 1.2 * rate_ana + 1e-9
    print(f"\n[sprint] === GATES OFFLINE (pré-enregistrés) ===")
    print(f"[sprint] G-rank    : AUC CV-4 par vie = {auc:.3f} (gate > 0.70) → {'✅' if g_rank else '❌'} "
          f"[{', '.join(f'{a:.3f}' for a in aucs)}]")
    print(f"[sprint] G-res     : {100 * acc_cor:.0f}% ≥ {100 * acc_ana:.0f}% + 10 pts → {'✅' if g_res else '❌'}")
    print(f"[sprint] G-consist : {100 * rate_cor:.1f}% ≤ 1.2×{100 * rate_ana:.1f}% → {'✅' if g_consist else '❌'}")
    print(f"[sprint] G-mono v2 : santé|profond {'↑' if q_by_h[0] < q_by_h[1] < q_by_h[2] else '✗'} "
          f"profondeur|blessé {'↓' if q_by_d[0] > q_by_d[1] > q_by_d[2] else '✗'} → {'✅' if g_mono else '❌'}")
    verdict = g_rank and g_res and g_consist and g_mono
    print(f"[sprint] {'✅ GATES PASSÉS → juge closed-loop (2×24 vies seeds 1+2)' if verdict else '❌ GATE ÉCHOUÉ → ne pas brancher, commiter le négatif'}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(make_checkpoint(critic, args.pain,
                               form="composed_v1" if composed else "q_regression",
                               auc_cv=auc, acc_ana=acc_ana, acc_cor=acc_cor,
                               flip_ana=rate_ana, flip_cor=rate_cor, q_by_health_deep=q_by_h,
                               q_by_depth_wounded=q_by_d, kappa_data=kappa, drain=drain,
                               restore=restore,
                               label="got_bce_composed" if composed else "linear_pursuit_magnitude",
                               runs=list(args.runs), gates_pass=bool(verdict)),
               out / "sprint_best.pt")
    print(f"[sprint] sauvé → {out / 'sprint_best.pt'}")


def _fit_bce(X: torch.Tensor, y: torch.Tensor, iters: int, seed: int) -> SprintCritic:
    """Fit BCE générique (têtes sigmoïdes 14-d) — mêmes hyperparamètres que le trainer sprint."""
    torch.manual_seed(seed)
    c = SprintCritic()
    opt = torch.optim.Adam(c.parameters(), 2e-3, weight_decay=1e-4)
    for _ in range(iters):
        bi = torch.randint(0, len(X), (256,))
        nn.functional.binary_cross_entropy_with_logits(c.net(X[bi]).squeeze(-1), y[bi]).backward()
        opt.step()
        opt.zero_grad()
    return c.eval()


def train_death(args: argparse.Namespace) -> None:
    """P2-bis (docs/design_purete_hjepa.md) : P̂mort(s,c) — la prime de risque non-linéaire que
    W=25 encode, apprise des morts-danger vécues. Gates G-death pré-enregistrés : AUC CV-4 par
    vie > 0.80 ET monotonie santé (P̂mort décroissant en h sur les traversées profondes)."""
    rows = load_corpus(args.runs)
    if not rows:
        raise SystemExit("[death] aucun corpus lisible")
    pain_model = PainCritic()
    _pk = torch.load(args.pain, map_location="cpu", weights_only=True)
    pain_model.load_state_dict(_pk["state_dict"])
    pain_model.eval()
    X = torch.cat([sprint_inputs([r["feats_all"][r["chosen"]]], (r["e"], r["t"], r["h"]),
                                 _pain_of(pain_model, [r["feats_all"][r["chosen"]]]))
                   for r in rows])
    y = torch.tensor([float(r["died_danger"]) for r in rows])
    life = torch.tensor([r["life"] for r in rows])
    print(f"[death] corpus : {len(rows)} décisions ({len(args.runs)} runs) | "
          f"positifs died_danger = {int(y.sum())} ({100 * float(y.mean()):.1f}%)")

    # GATE G-death (1/2) — AUC en CV 4 plis PAR VIE.
    aucs = []
    for k in range(4):
        te = (life % 4 == k)
        if int(y[te].sum()) == 0 or int((1 - y[te]).sum()) == 0:
            print(f"[death]   pli {k} : classe vide, sauté")
            continue
        c_k = _fit_bce(X[~te], y[~te], args.iters, args.seed)
        with torch.no_grad():
            aucs.append(_auc(c_k.p(X[te]), y[te].bool()))
        print(f"[death]   pli {k} : AUC={aucs[-1]:.3f} (n_te={int(te.sum())}, pos={int(y[te].sum())})")
    auc = sum(aucs) / max(len(aucs), 1)

    # GATE G-death (2/2) — monotonie santé sur les traversées PROFONDES (là où le risque vit).
    death = _fit_bce(X, y, args.iters, args.seed)
    cross_i = [i for i, r in enumerate(rows) if r["cls"] == "cross"]
    med_d = st.median([rows[i]["intr_chosen"] for i in cross_i])
    deep = [i for i in cross_i if rows[i]["intr_chosen"] > med_d]
    with torch.no_grad():
        p_all = death.p(X)
    p_by_h = []
    for lo, hi in ((0.0, 30.0), (30.0, 60.0), (60.0, 101.0)):
        idx = [i for i in deep if lo <= rows[i]["h"] < hi]
        p_by_h.append(float(p_all[torch.tensor(idx)].mean()) if len(idx) >= 15 else float("nan"))
    mono = p_by_h[0] > p_by_h[1] > p_by_h[2]
    print(f"[death] P̂mort|profond par santé {['%.3f' % v for v in p_by_h]} (décroissant en h ?)")

    g_death = auc > 0.80 and mono
    print(f"\n[death] === GATE G-death (pré-enregistré) ===")
    print(f"[death] AUC CV-4 par vie : {auc:.3f} (gate > 0.80) [{', '.join(f'{a:.3f}' for a in aucs)}]")
    print(f"[death] monotonie santé : {'OUI' if mono else 'NON'}")
    print(f"[death] {'✅ G-death PASSÉ → assemblage composed_pure_v2 + gates de forme' if g_death else '❌ G-death ÉCHOUÉ → ne pas assembler, négatif commité'}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": death.state_dict(), "in_dim": SPRINT_IN_DIM, "form": "death_v1",
                "auc_cv": auc, "mono_health": mono, "p_by_health_deep": p_by_h,
                "n_pos": int(y.sum()), "runs": list(args.runs), "gates_pass": bool(g_death)},
               out / "death_best.pt")
    print(f"[death] sauvé → {out / 'death_best.pt'}")
    if g_death:
        # assemblage du ckpt combiné : tête repas (juge PASS) + tête mort + constantes mesurées.
        base = torch.load("data/checkpoints/sprint_critic/sprint_best.pt",
                          map_location="cpu", weights_only=True)
        combined = dict(base)
        combined["form"] = "composed_pure_v2"
        combined["death_state_dict"] = death.state_dict()
        combined["death_auc_cv"] = auc
        combined["note"] = ("P2-bis: pure pricing + learned death premium "
                            "(docs/design_purete_hjepa.md)")
        torch.save(combined, out / "sprint_pure_v2.pt")
        print(f"[death] assemblé → {out / 'sprint_pure_v2.pt'} (repas + mort + constantes mesurées)")


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
                c.net[-1].bias.fill_(bias)        # Q̂ ≡ ∓20 → remise 0 / capée à W·intr, ∀ entrée
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

            # forme COMPOSÉE : P̂≡0 ⇒ remise 0 (bit-identique) ; P̂≡1 ⇒ remise EXACTE recalculée ici
            for bias, name in ((-20.0, "c0"), (+20.0, "c1")):
                c = SprintCritic()
                with torch.no_grad():
                    c.net[-1].weight.zero_()
                    c.net[-1].bias.fill_(bias)
                torch.save(make_checkpoint(c, DEFAULT_PAIN, form="composed_v1", kappa_data=9.2,
                                           drain=0.05, restore=40.0), Path(td) / f"{name}.pt")
            os.environ["SYLVAN_WP_SPRINT_CRITIC"] = str(Path(td) / "c0.pt")
            lay = WaypointLayer()
            lay.maybe_decide("food", target, retina, drives=(30.0, 70.0, 100.0))
            rec = lay.decide("food", target, retina)
            assert (rec["choice"], rec["cost_direct"]) == (rec0["choice"], rec0["cost_direct"]), (rec, rec0)

            os.environ["SYLVAN_WP_SPRINT_CRITIC"] = str(Path(td) / "c1.pt")
            lay = WaypointLayer()
            lay.maybe_decide("food", target, retina, drives=(30.0, 70.0, 100.0))
            rec = lay.decide("food", target, retina)
            from sylvan.control.waypoint_layer import candidate_features, green_points
            pain_m = PainCritic()
            _pk = torch.load(DEFAULT_PAIN, map_location="cpu", weights_only=True)
            pain_m.load_state_dict(_pk["state_dict"])
            pain_m.eval()
            greens = green_points(retina)
            with torch.no_grad():
                pain0 = float(pain_m.pain(torch.tensor(
                    [candidate_features(target, target, greens)], dtype=torch.float32))[0])
            ben = min(40.0, 100.0 - 30.0) / 0.05          # drives e=30 → 800 pas
            exp_rebate = min(_CFG.block_weight * rec0["intr_direct"],
                             0.02 * max(0.0, 1.0 * ben - 9.2 * pain0 * 100.0))
            assert abs(rec["cost_direct"] - (rec0["cost_direct"] - exp_rebate)) < 0.06, \
                (rec["cost_direct"], rec0["cost_direct"], exp_rebate)

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
    ap.add_argument("--form", choices=("composed", "q"), default="composed",
                    help="composed = P̂(repas)·bénéfice − κ·douleur̂ (reprise n°2, owner) ; "
                         "q = régression U (négatif n°2, gardée pour reproduction)")
    ap.add_argument("--head", choices=("sprint", "death"), default="sprint",
                    help="death = tête P(mort|s,c) (P2-bis) sur le corpus élargi DEATH_RUNS")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    if args.head == "death":
        if args.runs == DEFAULT_RUNS:
            args.runs = DEATH_RUNS       # corpus élargi par défaut (juge/pure = riches en morts)
        train_death(args)
        return
    train(args)


if __name__ == "__main__":
    main()
