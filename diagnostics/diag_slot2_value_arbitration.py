"""diag_slot2_value_arbitration — GATE B0 du pivot Mode-2 : la valeur-survie ARBITRE-t-elle
quand la perception est EXPLICITE 2-ressources ?

CONTEXTE (2026-07-02). G2 a réfuté la valeur-survie sur LATENT BRUT avec un confondant identifié :
le rêve open-loop du WM est direction-AVEUGLE (|Δvaleur| 0.03 vs spread 0.18) → on ne sait pas si
c'est l'approche valeur/labels qui échoue ou la représentation. Le fallback SLOT-2 suppose que
c'était la représentation. B0 isole cette hypothèse SANS rien construire : on entraîne la même tête
de valeur sur des features EXPLICITES 2-ressources — (profondeur, azimut) du rayon ROUGE le plus
proche + idem BLEU (color-gating exact de obs._color_gated_depths) + drives — dérivées GRATUITEMENT
de la rétine des buffers gate2/2b/2c déjà sur disque. Ces features = exactement ce qu'un slot-2
appris fournirait. Labels = mêmes que G2 (G = 1-γ^(restant), surv100 depuis le flag de fin
d'épisode). NB gate2c a un reward pain-shapé : sans effet ici (on ne lit ni reward ni command).

CRITÈRES PRÉ-ENREGISTRÉS (écrits AVANT le run, CLAUDE.md §1 ; référence = G2 latent brut) :
  1. Prédiction  : AUC(V, surv100) held-out  PASS >= 0.85 | KILL < 0.75   (G2 : 0.88, à matcher)
  2. Équilibre   : ΔV contrefactuel NÉGATIF pour CHAQUE drive bas, ratio sensibilités >= 0.5
                   (G2 : corr soif 0.30 vs énergie 0.65, Δbas-soif -0.04 = biais énergie)
  3. ARBITRAGE (décisif) : contrefactuel swap rouge<->bleu sur états à UNE pulsion basse —
                   V préfère la config où la ressource de la pulsion basse est la plus proche.
                   PASS >= 0.70 | KILL < 0.55 (hasard)     (G2 : 0.19-0.30 = anti-arbitre)
  4. Bonus planner-side : sur les morts 'voit-mais-n'approche-pas', transport géométrique des
                   coords + drain analytique -> argmax_V pointe vers la ressource qui tue >= 0.70

CAVEAT (§2, noté avant de lancer) : labels issus de la politique MYOPE (verrou off-policy
historique), atténué par la variance réelle (épisodes qui jonglent/survivent vs campeurs/morts).
Conséquence : un FAIL est quasi décisif (même perception parfaite n'arbitre pas -> approche
valeur/labels réfutée) ; un PASS est nécessaire-mais-pas-suffisant (juge final = closed-loop).

Lancer :  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_slot2_value_arbitration.py \
              [--globs 'data/checkpoints/mode1_ppo_gate2*/iter_*/buffer'] [--seed N] [--selfcheck]

RÉSULTAT (2026-07-02, seeds 0/1/2 — 230 épisodes, 228 morts, 42.7k transitions) :
  1. AUC 0.999/0.981/0.996 = PASS mais NON-DISCRIMINANT (ablation drives-seuls ~1.0 : le countdown
     des drives sature le test).
  2. Équilibre : ΔV négatif pour les DEUX drives, corr e/t ~0.52/0.46 -> le biais-énergie de G2 est
     RÉSORBÉ dès que la soif est une ENTRÉE explicite (elle n'est pas une entrée du WM -> cause du
     biais G2 localisée).
  3. ARBITRAGE statique = ÉCHEC FRANC : 0.68/0.37/0.44 = hasard au multi-seed. Le caveat off-policy
     s'est matérialisé : des labels issus de la politique myope ne peuvent pas enseigner à une valeur
     PER-STATE « ressource-de-la-pulsion-basse proche = mieux ».
  4. Planner-side = PASS ROBUSTE : 0.92/0.90/0.96 -> et l'ablation V drives-seuls fait PAREIL
     (0.86/0.94/1.00) -> TOUT l'arbitrage est porté par le ROLLOUT qui modélise l'événement-
     consommation (approche -> contact -> refill -> drives hauts -> V haute), PAS par la valeur.

LECTURE (localisation, pas un simple pass/fail) : la « valeur-survie statique apprise » est réfutée
2× (G2 latent brut ; B0 explicite) — ce qui arbitre = LOOK-AHEAD sur coordonnées explicites par
ressource + DYNAMIQUE DES DRIVES avec refill (= le 3e verrou historique) + valeur HOMÉOSTATIQUE
simple (drives-seuls suffisent). Le slot-2 reste justifié comme FOURNISSEUR des coords à transporter
(le rêve latent est direction-aveugle) ; la sophistication de la valeur ne l'est pas.
"""

from __future__ import annotations

import argparse
import glob as globmod
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_mode1_death_cause import (  # noqa: E402  (reuse, anti-duplication)
    CLOSE_DEPTH, DRAIN_PER_STEP, STEPS_PER_MACRO, VISIBLE,
    _read_lines, classify_death, closing_rate,
)
from sylvan.control.mode1.obs import BLUE, N_RAYS, RED, _color_gated_depths  # noqa: E402
from sylvan.control.mode1.rollout_mode1 import _split_episodes  # noqa: E402

SEED = 0
GAMMA_GODOT = 0.99                                # G2 gamma, per Godot step
GAMMA_MACRO = GAMMA_GODOT ** STEPS_PER_MACRO      # buffer transitions are macro-steps (10 Godot)
H_SURV_MACRO = 100 // STEPS_PER_MACRO             # G2's 'alive in 100 Godot steps' in macro units
LOW, HIGH = 0.30, 0.50                            # one-drive-low selection (scale 0-1), G2 spirit
DEPTH_GAP_MIN = 0.05                              # swap test: require a meaningful nearer resource
FEAT_DIM = 10                                     # [dR,cosR,sinR,visR, dB,cosB,sinB,visB, e, t]
PLAN_H = 40                                       # planner-side sim horizon (macro-steps)
PLAN_PROBE_BACK = 10                              # probe frame = 10 macro before death (S205 lead)

# Pre-registered pass/kill thresholds (see docstring — do NOT move the goalposts after the run).
CRIT = {"auc_pass": 0.85, "auc_kill": 0.75, "balance_ratio": 0.5,
        "arb_pass": 0.70, "arb_kill": 0.55, "plan_pass": 0.70}


# ---- features ----------------------------------------------------------------------------------
def nearest_feat(retina: list[float], color: str) -> tuple[float, float, float, float]:
    """(depth, cos az, sin az, visible) of the nearest ray of that color. Ray k points at angle
    2π·k/N_RAYS, ray 0 = forward, increasing to the RIGHT (perception.gd convention)."""
    depths = _color_gated_depths(retina, color)
    d = min(depths)
    if d >= VISIBLE:
        return 1.0, 0.0, 0.0, 0.0
    k = depths.index(d)
    b = 2.0 * math.pi * k / N_RAYS
    return d, math.cos(b), math.sin(b), 1.0


def transition_features(tr: dict) -> list[float]:
    r = nearest_feat(tr["retina"], RED)
    b = nearest_feat(tr["retina"], BLUE)
    return [*r, *b, float(tr["energy"]) / 100.0, float(tr["thirst"]) / 100.0]


def swap_resources(x: np.ndarray) -> np.ndarray:
    """Counterfactual: exchange the red and blue feature blocks (drives untouched)."""
    y = x.copy()
    y[..., 0:4], y[..., 4:8] = x[..., 4:8].copy(), x[..., 0:4].copy()
    return y


# ---- labels (same operationalization as diag_mode1_survival_value, macro-step units) -----------
def survival_labels(ep: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """G_t = 1 - γ_macro^(L-t) and surv100 (0 within 100 Godot steps of a TRUE death end).
    Truncated episodes: surv100=1 throughout (flagged bias, same as G2)."""
    L = len(ep)
    t = np.arange(L)
    G = (1.0 - GAMMA_MACRO ** (L - t).astype(np.float64)).astype(np.float32)
    surv = np.ones(L, dtype=np.float32)
    if bool(ep[-1].get("done")) and not bool(ep[-1].get("truncated")):
        surv[(L - 1 - t) <= H_SURV_MACRO] = 0.0
    return G, surv


def episode_cause(ep: list[dict]) -> str:
    last = ep[-1]
    if bool(last.get("truncated")) and not bool(last.get("done")):
        return "trunc"
    return "energy" if float(last["energy"]) <= float(last["thirst"]) else "thirst"


# ---- split by episode, stratified by death cause (no train/test leakage) -----------------------
def split_episodes(episodes: list[list[dict]], seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.RandomState(seed)
    by_cause: dict[str, list[int]] = {}
    for i, ep in enumerate(episodes):
        by_cause.setdefault(episode_cause(ep), []).append(i)
    test: set[int] = set()
    for ids in by_cause.values():
        ids = list(ids)
        rng.shuffle(ids)
        k = max(1, int(round(0.3 * len(ids)))) if len(ids) > 1 else 0
        test.update(ids[:k])
    return [i for i in range(len(episodes)) if i not in test], sorted(test)


# ---- value head --------------------------------------------------------------------------------
class ValueMLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sd", torch.ones(in_dim))
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mu) / self.sd).squeeze(-1)


def train_value(X: np.ndarray, G: np.ndarray, tr: np.ndarray, seed: int,
                steps: int = 1500) -> ValueMLP:
    torch.manual_seed(seed)
    head = ValueMLP(X.shape[1])
    Xt, Gt = torch.tensor(X[tr]), torch.tensor(G[tr])
    head.mu.copy_(Xt.mean(0))
    head.sd.copy_(Xt.std(0) + 1e-6)                 # normalization from TRAIN only
    opt = torch.optim.Adam(head.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(steps):
        head.train()
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(head.value(Xt), Gt)
        loss.backward()
        opt.step()
    head.eval()
    return head


# ---- metrics (mirror diag_mode1_survival_value.r2/auc — tiny, copied for standalone robustness)
def r2(y: np.ndarray, p: np.ndarray) -> float:
    y = y.reshape(-1); p = p.reshape(-1)
    return float(1.0 - ((y - p) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12))


def auc(score: np.ndarray, label: np.ndarray) -> float:
    s = np.asarray(score).reshape(-1); l = np.asarray(label).reshape(-1)
    o = np.argsort(s); rk = np.empty_like(s); rk[o] = np.arange(1, len(s) + 1)
    npos, nneg = l.sum(), (1 - l).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((rk[l == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


# ---- test 2: drive-balance (counterfactual intervention on the drive features) -----------------
def drive_balance(head: ValueMLP, X: np.ndarray) -> dict:
    xs = torch.tensor(X)
    with torch.no_grad():
        v = head.value(xs).numpy()
        out = {"corr_energy": float(np.corrcoef(v, X[:, 8])[0, 1]),
               "corr_thirst": float(np.corrcoef(v, X[:, 9])[0, 1])}
        for name, idx in (("energy", 8), ("thirst", 9)):
            lo, hi = xs.clone(), xs.clone()
            lo[:, idx] = 0.15
            hi[:, idx] = 0.85
            out[f"dv_{name}"] = float((head.value(lo) - head.value(hi)).mean())  # want < 0
    return out


# ---- test 3: counterfactual swap arbitration ----------------------------------------------------
def arbitration_swap(head: ValueMLP, X: np.ndarray) -> dict:
    """On states with EXACTLY ONE drive low and BOTH resources visible, does V prefer the config
    where the depleted drive's resource is the NEARER one? (as-is vs red<->blue swapped)"""
    e, t = X[:, 8], X[:, 9]
    vis = (X[:, 3] > 0.5) & (X[:, 7] > 0.5)
    gap = np.abs(X[:, 0] - X[:, 4]) > DEPTH_GAP_MIN
    elow = (e < LOW) & (t > HIGH) & vis & gap
    tlow = (t < LOW) & (e > HIGH) & vis & gap
    sel = elow | tlow
    if not sel.any():
        return {"n": 0, "frac": float("nan")}
    Xs = X[sel]
    with torch.no_grad():
        v_orig = head.value(torch.tensor(Xs)).numpy()
        v_swap = head.value(torch.tensor(swap_resources(Xs))).numpy()
    dep_is_red = elow[sel]                       # energy low -> depleted resource = RED (food)
    red_nearer = Xs[:, 0] < Xs[:, 4]
    # config with depleted's resource nearer = orig iff (dep is red) == (red is nearer)
    orig_good = dep_is_red == red_nearer
    correct = np.where(orig_good, v_orig > v_swap, v_swap > v_orig)
    n_e = int(dep_is_red.sum())
    return {"n": int(sel.sum()), "frac": float(correct.mean()),
            "n_energy_low": n_e, "n_thirst_low": int(sel.sum()) - n_e,
            "frac_elow": float(correct[dep_is_red].mean()) if n_e else float("nan"),
            "frac_tlow": float(correct[~dep_is_red].mean()) if n_e < sel.sum() else float("nan"),
            "val_gap": float(np.abs(v_orig - v_swap).mean()),
            "val_std": float(np.concatenate([v_orig, v_swap]).std())}


# ---- test 4 (bonus): planner-side geometric rollout on real death states ------------------------
def plan_rollout(head: ValueMLP, x0: np.ndarray, pursue_red: bool, rate: float) -> float:
    """Score (mean V over PLAN_H) of 'pursue that resource': its depth closes at the empirical
    rate and its azimuth turns to dead-ahead; the OTHER resource stays put (flagged simplification);
    drives drain analytically; contact refills the pursued drive; a drive at 0 = dead (V=0 after)."""
    x = x0.copy()
    p, drive_idx = (0, 8) if pursue_red else (4, 9)
    vals, drain = [], DRAIN_PER_STEP * STEPS_PER_MACRO / 100.0   # scaled drive units per macro
    for _ in range(PLAN_H):
        x[8] -= drain
        x[9] -= drain
        if x[p + 3] > 0.5:                       # pursued resource visible -> close in on it
            x[p] = max(x[p] - rate, 0.05)
            x[p + 1], x[p + 2] = 1.0, 0.0        # heading toward it
            if x[p] < CLOSE_DEPTH:
                x[drive_idx] = 1.0               # contact -> refill; stay camped on it
        if x[8] <= 0.0 or x[9] <= 0.0:
            vals.append(0.0)                     # dead: no survival value from here on
            continue
        with torch.no_grad():
            vals.append(float(head.value(torch.tensor(x[None, :], dtype=torch.float32))[0]))
    return float(np.mean(vals))


def planner_side(head: ValueMLP, episodes: list[list[dict]], test_ids: list[int],
                 rate: float, window: int = 10) -> dict:
    n, correct = 0, 0
    for i in test_ids:
        ep = episodes[i]
        if not (bool(ep[-1].get("done")) and not bool(ep[-1].get("truncated"))):
            continue
        if classify_death(ep, window) not in ("erre", "campe_sur_autre"):
            continue                             # decision deaths only (the wall we probe)
        if len(ep) <= PLAN_PROBE_BACK + 1:
            continue
        tr = ep[-1 - PLAN_PROBE_BACK]
        x0 = np.asarray(transition_features(tr), dtype=np.float32)
        kill_is_red = float(ep[-1]["energy"]) <= float(ep[-1]["thirst"])
        if x0[0 if kill_is_red else 4] >= VISIBLE:
            continue                             # killer resource not visible at probe frame
        v_red = plan_rollout(head, x0, pursue_red=True, rate=rate)
        v_blue = plan_rollout(head, x0, pursue_red=False, rate=rate)
        n += 1
        if (v_red > v_blue) == kill_is_red:
            correct += 1
    return {"n": n, "frac": (correct / n) if n else float("nan")}


# ---- selfcheck ----------------------------------------------------------------------------------
def selfcheck() -> None:
    retina = [1.0, 0.0, 0.0, 0.0] * N_RAYS
    retina[0:4] = [0.2, 0.9, 0.1, 0.1]           # red dead-ahead, near
    retina[9 * 4:9 * 4 + 4] = [0.5, 0.1, 0.1, 0.9]  # blue at ray 9 = 90° right
    dr, cr, sr, vr = nearest_feat(retina, RED)
    db, cb, sb, vb = nearest_feat(retina, BLUE)
    assert (dr, cr, sr, vr) == (0.2, 1.0, 0.0, 1.0), (dr, cr, sr, vr)
    assert abs(db - 0.5) < 1e-9 and abs(cb) < 1e-9 and abs(sb - 1.0) < 1e-9 and vb == 1.0
    x = np.asarray(transition_features({"retina": retina, "energy": 40.0, "thirst": 80.0}),
                   dtype=np.float32)
    assert x.shape == (FEAT_DIM,) and abs(x[8] - 0.4) < 1e-6 and abs(x[9] - 0.8) < 1e-6
    y = swap_resources(x)
    assert np.allclose(y[0:4], x[4:8]) and np.allclose(y[4:8], x[0:4])
    assert np.allclose(swap_resources(y), x), "swap must be an involution"
    # labels: death episode -> G monotonically increasing backwards from 0-ish, surv100 tail = 0
    ep = [{"energy": 50.0, "thirst": 50.0, "done": False, "truncated": False}] * 30
    ep = ep[:-1] + [{"energy": 1.0, "thirst": 50.0, "done": True, "truncated": False}]
    G, s = survival_labels(ep)
    assert G[0] > G[-1] and abs(float(G[-1]) - (1 - GAMMA_MACRO)) < 1e-6
    assert s[-1] == 0.0 and s[-H_SURV_MACRO - 1] == 0.0 and s[0] == 1.0
    assert episode_cause(ep) == "energy"
    # drain arithmetic: full drive budget = 200 macro-steps in scaled units
    assert abs(100.0 / (DRAIN_PER_STEP * STEPS_PER_MACRO) - 200.0) < 1e-9
    print("[selfcheck] OK — features/azimut, swap involution, labels macro, cause, drain")


# ---- main ---------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--globs", nargs="+",
                    default=["data/checkpoints/mode1_ppo_gate2*/iter_*/buffer"])
    ap.add_argument("--seed", type=int, default=SEED,
                    help="split/train seed (multi-seed robustness of the SAME criteria)")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return

    seed = args.seed
    np.random.seed(seed)
    episodes: list[list[dict]] = []
    n_pending = 0
    for g in args.globs:
        for d in sorted(globmod.glob(g)):
            for p in sorted(Path(d).glob("part-*.jsonl")):
                for ep in _split_episodes(_read_lines(p)):
                    if len(ep) < 12:
                        continue
                    if not (bool(ep[-1].get("done")) or bool(ep[-1].get("truncated"))):
                        n_pending += 1           # unfinished episode: future unknown -> no labels
                        continue
                    episodes.append(ep)
    deaths = sum(1 for ep in episodes if bool(ep[-1].get("done")) and not bool(ep[-1].get("truncated")))
    print(f"globs={args.globs}")
    print(f"épisodes labellisables={len(episodes)} (morts={deaths}, tronqués={len(episodes)-deaths}) "
          f"| pending exclus={n_pending}")
    if deaths < 20:
        print("KILL-DATA : trop peu de morts pour des labels de survie fiables.")
        return

    X_l, G_l, S_l, EID_l = [], [], [], []
    for eid, ep in enumerate(episodes):
        G, s = survival_labels(ep)
        for t, tr in enumerate(ep):
            X_l.append(transition_features(tr))
            G_l.append(G[t]); S_l.append(s[t]); EID_l.append(eid)
    X = np.asarray(X_l, dtype=np.float32)
    G = np.asarray(G_l, dtype=np.float32)
    S = np.asarray(S_l, dtype=np.float32)
    EID = np.asarray(EID_l, dtype=np.int64)
    print(f"transitions={len(X)} | features={X.shape[1]} "
          f"(rouge visible {100*(X[:,3]>.5).mean():.0f}%, bleu {100*(X[:,7]>.5).mean():.0f}%)")

    train_ids, test_ids = split_episodes(episodes, seed)
    tr = np.isin(EID, train_ids)
    te = np.isin(EID, test_ids)
    assert not (tr & te).any(), "train/test episode leakage"
    print(f"split épisodes : train={len(train_ids)} test={len(test_ids)}")

    head = train_value(X, G, tr, seed)
    head_drv = train_value(X[:, 8:10].copy(), G, tr, seed)   # drives-only ablation (honesty)
    with torch.no_grad():
        v_te = head.value(torch.tensor(X[te])).numpy()
        v_te_drv = head_drv.value(torch.tensor(X[te, 8:10])).numpy()

    auc_full = auc(v_te, S[te])
    auc_drv = auc(v_te_drv, S[te])
    bal = drive_balance(head, X[te])
    arb = arbitration_swap(head, X[te])
    arb_drv_gap = float(np.abs(
        v_te_drv - head_drv.value(torch.tensor(swap_resources(X[te])[:, 8:10])).detach().numpy()
    ).mean())                                    # must be 0: drives-only is blind to the swap
    rate, dist = closing_rate(episodes)
    plan = planner_side(head, episodes, test_ids, rate)

    class _DrivesOnlyView:
        """Adapter: evaluate the drives-only head on full 10-dim planner states (ablation:
        if rollout+refill alone arbitrates, the perception features inside V are not the carrier)."""
        def value(self, x: torch.Tensor) -> torch.Tensor:
            return head_drv.value(x[:, 8:10])
    plan_drv = planner_side(_DrivesOnlyView(), episodes, test_ids, rate)

    print("\n=== TABLE BUT vs proxy (référence G2 latent brut entre parenthèses) ===")
    print(f"proxy   R² held-out (G)              : {r2(G[te], v_te):+.3f}")
    print(f"1. BUT  AUC(V, surv100) held-out     : {auc_full:.3f}   (G2: 0.88)  "
          f"[ablation drives-seuls: {auc_drv:.3f}]")
    print(f"2. BUT  équilibre drives             : corr e {bal['corr_energy']:+.2f} / "
          f"t {bal['corr_thirst']:+.2f} ; ΔV(e bas) {bal['dv_energy']:+.3f} / "
          f"ΔV(t bas) {bal['dv_thirst']:+.3f}   (G2: 0.65/0.30, Δ -0.04)")
    print(f"3. BUT  ARBITRAGE swap (décisif)     : {arb['frac']:.2f} sur n={arb['n']} "
          f"(e-bas {arb.get('frac_elow', float('nan')):.2f}/n={arb.get('n_energy_low', 0)}, "
          f"t-bas {arb.get('frac_tlow', float('nan')):.2f}/n={arb.get('n_thirst_low', 0)}) "
          f"(G2: 0.19-0.30) | |ΔV| {arb.get('val_gap', float('nan')):.3f} "
          f"vs spread {arb.get('val_std', float('nan')):.3f} | sanity drives-seuls |ΔV|={arb_drv_gap:.4f}")
    print(f"4. bonus planner-side (transport géo): {plan['frac']:.2f} sur n={plan['n']} morts-décision "
          f"(closing rate empirique {rate:.4f} depth/macro, dist méd {dist:.3f}) "
          f"[ablation V drives-seuls: {plan_drv['frac']:.2f}]")

    print("\n--- VERDICT (critères pré-enregistrés) ---")
    ok1 = auc_full >= CRIT["auc_pass"]
    both_neg = bal["dv_energy"] < 0 and bal["dv_thirst"] < 0
    mags = sorted([abs(bal["dv_energy"]), abs(bal["dv_thirst"])])
    ok2 = both_neg and (mags[0] / (mags[1] + 1e-12)) >= CRIT["balance_ratio"]
    ok3 = arb["n"] >= 30 and arb["frac"] >= CRIT["arb_pass"]
    kill3 = arb["n"] >= 30 and arb["frac"] < CRIT["arb_kill"]
    print(f"1. prédiction : {'PASS' if ok1 else ('KILL' if auc_full < CRIT['auc_kill'] else 'PARTIEL')}")
    print(f"2. équilibre  : {'PASS' if ok2 else 'FAIL (biais un-seul-drive, le défaut G2)'}")
    print(f"3. arbitrage  : "
          f"{'PASS' if ok3 else ('KILL (~hasard)' if kill3 else ('PARTIEL (sous le seuil)' if arb['n'] >= 30 else 'n-insuffisant'))}")
    print(f"4. planner    : {'PASS (bonus)' if plan['n'] and plan['frac'] >= CRIT['plan_pass'] else 'non-concluant (bonus)'}")
    plan_ok = plan["n"] >= 30 and plan["frac"] >= CRIT["plan_pass"]
    if ok1 and ok2 and ok3:
        print("\nGATE B0 = PASS -> l'hypothèse slot-2 tient : la valeur apprise ARBITRE dès que la")
        print("perception est explicite -> payer l'étape suivante (slot-2 perception sur WM GELÉ).")
        print("Rappel §2 : PASS = nécessaire-mais-pas-suffisant (labels off-policy) ; juge = closed-loop.")
    elif not ok3 and plan_ok:
        print("\nGATE B0 = valeur STATIQUE réfutée, ROLLOUT+refill validé (cf RÉSULTAT en docstring) :")
        print("l'arbitrage vient du look-ahead qui modélise l'événement-consommation sur des coords")
        print("explicites par ressource — pas d'une valeur per-state sophistiquée (drives-seuls suffisent).")
        print("-> slot-2 justifié comme fournisseur de coords ; la dynamique drives+refill dans le rollout")
        print("   est l'ingrédient décisif (3e verrou ; analytique flaggée d'abord, tête apprise = pureté).")
    elif kill3:
        print("\nGATE B0 = KILL -> même avec perception explicite parfaite, ni la valeur statique NI le")
        print("rollout n'arbitrent : approche valeur/labels réfutée -> NE PAS construire slot-2 ; escalade.")
    else:
        print("\nGATE B0 = PARTIEL -> lire les sous-scores avant de payer quoi que ce soit.")


if __name__ == "__main__":
    main()
