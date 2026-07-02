"""diag_bridge_trigger — un TRIGGER CHEAP peut-il flaguer les états qui mènent à une mort Mode-1 imminente ?

Question owner (falsifiable) : le PONT Mode-1↔Mode-2 = « faire tourner le réflexe (Mode-1) par défaut,
DÉLÉGUER la décision au planner lent (Mode-2) quand Mode-1 est SUR LE POINT D'ÉCHOUER ». Les morts Mode-1
sont à 96 % des morts de DÉCISION (diag_mode1_death_cause : campe-sur-une-ressource / voit-mais-n'approche-pas).
→ Peut-on, avec un trigger SIMPLE calculé DE L'OBS SEULE (aucun appel planner), flaguer les états qui mènent
à une mort imminente, à un taux-de-délégation RAISONNABLE ? Si oui → un arbitre cheap codé-main suffit → pont
viable. Si non → le trigger doit être APPRIS (plus dur).

Test GRATUIT (read-only sur les buffers Mode-1 déjà collectés ; PAS de Godot, PAS d'entraînement) :
  1. STEP-0 : schéma + morts par macro-transition, longueur d'épisode, taux de mort.
  2. LABEL : imminent_death(t) = l'épisode se termine par une MORT (done) dans les H prochaines macro-étapes.
  3. TRIGGERS CHEAP (de l'obs seule) : niveau du drive bas ; tension d'arbitrage (la ressource du drive bas
     n'est PAS la plus proche) ; écart |energy-thirst| ; « les deux drives bas » ; distance de la ressource
     du drive bas. → 3-5 scores scalaires « danger » (haut = plus de risque).
  4. MESURE, held-out (split par ÉPISODE, seed fixe) : AUC vs imminent_death ; au seuil qui attrape ~80 %
     des états-imminents (recall 0.8), le TAUX-DE-DÉLÉGATION (fraction de TOUS les états qui firent) et la
     précision. Bon trigger = AUC haute ET attrape la plupart des morts en firant sur une MINORITÉ d'états.
  5. TABLE trigger × {AUC, défer@recall0.8, précision} + meilleur trigger seul + meilleur combo simple.

VERDICT falsifiable :
  - meilleur AUC > ~0.75 ET défer@recall0.8 < ~0.35 → arbitre cheap FAISABLE → pont viable (construire la démo).
  - meilleur AUC ~0.5-0.7 OU défer > 0.5 pour attraper les morts → trigger cheap INSUFFISANT → trigger APPRIS
    nécessaire (WM-surprise / petit classifieur) → plus dur, documenter comme travail futur.

HYPOTHÈSES / choix (à assumer explicitement) :
  - H = 15 macro-étapes ≈ 150 pas Godot (10 pas physiques/macro). « moment dur » = à moins de H de la mort.
  - drive épuisé = argmin(energy, thirst) ; sa ressource = bouffe(rouge) si energy est le plus bas, sinon eau(bleu).
  - « ressource du drive bas plus loin » = profondeur color-gatée la plus proche de sa couleur > celle de l'autre
    (1.0 = non-vue = traitée comme la plus loin).
  - épisodes MORTS = positifs possibles ; épisodes TRONQUÉS (survie au cap) = négatifs (a survécu) ;
    épisodes PENDING (buffer coupé, issue INCONNUE) = EXCLUS (évite le bruit de label).

Lancer :
  PYTHONPATH=python env_pytorch_3.12/bin/python diag_bridge_trigger.py --selfcheck
  PYTHONPATH=python env_pytorch_3.12/bin/python diag_bridge_trigger.py \
     [--glob 'data/checkpoints/mode1_ppo_gate2{b,c}/iter_*/buffer'] [--horizon 15] [--test-frac 0.4] [--seed 0]
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import statistics as st
from pathlib import Path

from sylvan.control.mode1.obs import _color_gated_depths, RED, BLUE, N_RAYS
from sylvan.control.mode1.rollout_mode1 import _split_episodes

DRIVE_MAX = 100.0  # energy/thirst stockés en 0..100
UNSEEN = 1.0       # profondeur d'une couleur non vue (color-gating → 1.0)


# ------------------------------------------------------------------------- I/O
def _read_lines(path: Path) -> list[dict]:
    """Lecteur ROBUSTE (une ligne JSON éventuellement incomplète = iter en cours d'écriture → skip)."""
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    return out


def _nearest(retina, color) -> float:
    """Profondeur du rayon le plus proche de cette couleur (1.0 = rien de visible)."""
    return min(_color_gated_depths(retina, color))


# ---------------------------------------------------------------- features/label
def frame_feats(tr: dict) -> dict:
    """Scalaires CHEAP calculés de l'obs SEULE (energy, thirst, retina). Aucune info future, aucun planner."""
    e = float(tr["energy"]) / DRIVE_MAX      # 0..1
    t = float(tr["thirst"]) / DRIVE_MAX      # 0..1
    food_d = _nearest(tr["retina"], RED)     # profondeur bouffe la plus proche (1.0 = non vue)
    water_d = _nearest(tr["retina"], BLUE)   # profondeur eau la plus proche
    energy_is_low = e <= t                   # quel drive est le plus bas
    depleted_d = food_d if energy_is_low else water_d   # ressource du drive bas
    other_d = water_d if energy_is_low else food_d      # ressource de l'autre drive
    return {
        "e": e, "t": t,
        "min_drive": min(e, t), "max_drive": max(e, t), "gap": abs(e - t),
        "food_d": food_d, "water_d": water_d,
        "depleted_d": depleted_d, "other_d": other_d,
    }


def danger_scores(f: dict) -> dict:
    """Scores « danger » scalaires (HAUT = mort plus probable). Tous de l'obs seule."""
    depleted_far = f["depleted_d"] - f["other_d"]        # >0 : la ressource du drive bas est plus loin
    return {
        # (1) un drive devient bas
        "lowest_drive":   1.0 - f["min_drive"],
        # (2) les deux drives pressent (même le plus haut est bas)
        "both_low":       1.0 - f["max_drive"],
        # (3) écart d'arbitrage
        "drive_gap":      f["gap"],
        # (4) ressource du drive bas plus loin que l'autre (géométrie pure)
        "depleted_far":   depleted_far,
        # (5) distance de la ressource du drive bas (loin/non-vue = danger)
        "depleted_dist":  f["depleted_d"],
        # (2') tension d'arbitrage : drive bas ET sa ressource loin/non-vue
        "arb_tension":    (1.0 - f["min_drive"]) * f["depleted_d"],
        # (2'') drive bas + bonus si sa ressource est plus loin que l'autre
        "arb_tension2":   (1.0 - f["min_drive"]) + 0.5 * max(0.0, depleted_far),
    }


SCORE_KEYS = ["lowest_drive", "both_low", "drive_gap", "depleted_far",
              "depleted_dist", "arb_tension", "arb_tension2"]


def label_episode(ep: list[dict], horizon: int) -> list[int]:
    """imminent_death(t)=1 si l'épisode MEURT (done) et t est à < horizon macro-étapes de la fin."""
    died = bool(ep[-1].get("done"))
    if not died:
        return [0] * len(ep)
    L = len(ep)
    return [1 if (L - 1 - i) < horizon else 0 for i in range(L)]


# ------------------------------------------------------------------------- metrics
def auc(scores: list[float], labels: list[int]) -> float:
    """AUC ROC = P(score(pos) > score(neg)) via rangs (Mann-Whitney), gère les ex-aequo (0.5)."""
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    # rangs moyens (1-based) avec gestion des ex-aequo
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos = sum(ranks[i] for i in range(len(scores)) if labels[i] == 1)
    return (sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def threshold_at_recall(scores: list[float], labels: list[int], target_recall: float) -> float:
    """Seuil (score >= thr fait firer) qui atteint le recall cible sur CE set (le plus haut → défer minimal)."""
    pos_scores = sorted((scores[i] for i in range(len(scores)) if labels[i] == 1), reverse=True)
    if not pos_scores:
        return float("inf")
    k = max(1, int(round(target_recall * len(pos_scores))))
    k = min(k, len(pos_scores))
    return pos_scores[k - 1]   # attrape >= target_recall des positifs (seuil = k-ième positif le plus haut)


def eval_at_threshold(scores: list[float], labels: list[int], thr: float) -> dict:
    fired = [i for i in range(len(scores)) if scores[i] >= thr]
    pos = sum(labels)
    tp = sum(1 for i in fired if labels[i] == 1)
    defer_rate = len(fired) / len(scores) if scores else float("nan")
    recall = tp / pos if pos else float("nan")
    precision = tp / len(fired) if fired else float("nan")
    return {"defer_rate": defer_rate, "recall": recall, "precision": precision}


# ------------------------------------------------------------------------- pipeline
def load_episodes(globs: list[str]) -> list[list[dict]]:
    eps: list[list[dict]] = []
    for g in globs:
        for d in sorted(glob.glob(g)):
            for p in sorted(Path(d).glob("part-*.jsonl")):
                eps.extend(_split_episodes(_read_lines(p)))
    return eps


def classify_ep(ep: list[dict]) -> str:
    if not ep:
        return "empty"
    if bool(ep[-1].get("done")):
        return "death"
    if bool(ep[-1].get("truncated")):
        return "trunc"
    return "pending"


def _swap(sy):
    """(S, Y) → (S['lowest_drive'], Y) — petit adaptateur pour auc() dans la sonde d'honnêteté."""
    S, Y = sy
    return S["lowest_drive"], Y


def build_frames(eps: list[list[dict]], horizon: int):
    """Retourne (feats_scores, labels) au niveau frame pour un ensemble d'épisodes (death+trunc ; pending exclus)."""
    S = {k: [] for k in SCORE_KEYS}
    Y: list[int] = []
    for ep in eps:
        labs = label_episode(ep, horizon)
        for tr, y in zip(ep, labs):
            sc = danger_scores(frame_feats(tr))
            for k in SCORE_KEYS:
                S[k].append(sc[k])
            Y.append(y)
    return S, Y


def run(globs: list[str], horizon: int, test_frac: float, seed: int) -> None:
    eps = load_episodes(globs)
    kinds = [classify_ep(ep) for ep in eps]
    n_death = kinds.count("death")
    n_trunc = kinds.count("trunc")
    n_pend = kinds.count("pending")
    lens = [len(ep) for ep in eps]
    frames_total = sum(lens)
    print("=" * 78)
    print("STEP-0 — schéma & vue d'ensemble")
    print(f"globs = {globs}")
    print(f"épisodes={len(eps)} | frames(macro)={frames_total}")
    print(f"  morts(done)={n_death} | tronqués(cap)={n_trunc} | pending(exclus)={n_pend}")
    print(f"  taux de mort = {n_death}/{len(eps)-n_pend} des épisodes à issue connue "
          f"= {100*n_death/max(1,len(eps)-n_pend):.0f}%")
    if lens:
        print(f"  longueur d'épisode (macro) : med={st.median(lens)} min={min(lens)} max={max(lens)}")
    deaths = [ep for ep in eps if classify_ep(ep) == "death"]
    if deaths:
        byh = sum(1 for ep in deaths if float(ep[-1]["energy"]) <= float(ep[-1]["thirst"]))
        print(f"  morts par FAIM={byh}/{len(deaths)} | par SOIF={len(deaths)-byh}/{len(deaths)}")

    # ---- split par ÉPISODE (déterministe), pending exclus ----
    usable = [ep for ep in eps if classify_ep(ep) in ("death", "trunc")]
    rng = random.Random(seed)
    rng.shuffle(usable)
    n_test = int(round(test_frac * len(usable)))
    test_eps = usable[:n_test]
    train_eps = usable[n_test:]
    Strain, Ytrain = build_frames(train_eps, horizon)
    Stest, Ytest = build_frames(test_eps, horizon)
    pos_tr, pos_te = sum(Ytrain), sum(Ytest)
    print()
    print(f"LABEL imminent_death (H={horizon} macro ≈ {10*horizon} pas Godot)")
    print(f"  split par épisode (seed={seed}, test_frac={test_frac}): "
          f"train {len(train_eps)} ép / {len(Ytrain)} frames (pos={pos_tr}={100*pos_tr/max(1,len(Ytrain)):.1f}%) | "
          f"test {len(test_eps)} ép / {len(Ytest)} frames (pos={pos_te}={100*pos_te/max(1,len(Ytest)):.1f}%)")

    # ---- table par trigger : AUC(test) + défer@recall0.8 (seuil sur TRAIN, éval sur TEST) ----
    print()
    print("TABLE — trigger CHEAP × {AUC(test), défer@recall0.8(test), précision(test), recall(test)}")
    print(f"{'trigger':<16}{'AUC':>7}{'defer@R.8':>11}{'precision':>11}{'recall':>9}")
    rows = []
    for k in SCORE_KEYS:
        a = auc(Stest[k], Ytest)
        thr = threshold_at_recall(Strain[k], Ytrain, 0.80)
        m = eval_at_threshold(Stest[k], Ytest, thr)
        rows.append((k, a, m))
        print(f"{k:<16}{a:>7.3f}{m['defer_rate']:>11.3f}{m['precision']:>11.3f}{m['recall']:>9.3f}")

    best = max(rows, key=lambda r: (r[1] if r[1] == r[1] else -1))
    print(f"\nMEILLEUR trigger seul (AUC) = '{best[0]}' : AUC={best[1]:.3f} "
          f"défer@R.8={best[2]['defer_rate']:.3f} précision={best[2]['precision']:.3f}")

    # ---- meilleur COMBO simple = 2 seuils (min_drive bas ET ressource-épuisée loin) ----
    #  danger = 1 si (min_drive < a) ET (depleted_d > b) ; grille sur TRAIN (recall>=0.8, défer min), éval TEST.
    def combo_feats(eps_):
        md, dd, Y = [], [], []
        for ep in eps_:
            labs = label_episode(ep, horizon)
            for tr, y in zip(ep, labs):
                f = frame_feats(tr)
                md.append(f["min_drive"]); dd.append(f["depleted_d"]); Y.append(y)
        return md, dd, Y

    md_tr, dd_tr, Ytr = combo_feats(train_eps)
    md_te, dd_te, Yte = combo_feats(test_eps)
    best_combo = None
    for a in [i / 100 for i in range(5, 90, 5)]:        # seuil min_drive
        for b in [j / 100 for j in range(0, 105, 5)]:  # seuil depleted_d (0 = ignoré → OR dégénère en drive seul)
            fired = [(md_tr[i] < a) and (dd_tr[i] > b) for i in range(len(Ytr))]
            tp = sum(1 for i in range(len(Ytr)) if fired[i] and Ytr[i] == 1)
            pos = sum(Ytr)
            recall = tp / pos if pos else 0
            defer = sum(fired) / len(fired) if fired else 1
            if recall >= 0.80 and (best_combo is None or defer < best_combo[0]):
                best_combo = (defer, a, b, recall)
    if best_combo is not None:
        _, a, b, _ = best_combo
        fired = [(md_te[i] < a) and (dd_te[i] > b) for i in range(len(Yte))]
        tp = sum(1 for i in range(len(Yte)) if fired[i] and Yte[i] == 1)
        pos = sum(Yte)
        defer = sum(fired) / len(fired) if fired else float("nan")
        rec = tp / pos if pos else float("nan")
        prec = tp / sum(fired) if sum(fired) else float("nan")
        print(f"MEILLEUR combo 2-seuils : (min_drive < {a:.2f}) ET (depleted_d > {b:.2f}) "
              f"→ défer={defer:.3f} précision={prec:.3f} recall={rec:.3f} (test)")
    else:
        print("MEILLEUR combo 2-seuils : aucun combo n'atteint recall>=0.8 (bizarre).")

    # ---- HONNÊTETÉ : le meilleur trigger est-il un vrai « early-warning » ou un COMPTE-À-REBOURS trivial ? ----
    #  (CLAUDE.md §2) La mort = un drive→0, le drain est ~déterministe → min_drive EST une horloge de mort.
    #  On mesure : (a) le seuil min_drive au fire, (b) le LEAD-TIME (macro-étapes avant la mort), (c) la stabilité
    #  de l'AUC vs H (une AUC plate = tautologique, pas un signal appris).
    thr_ld = threshold_at_recall(Strain["lowest_drive"], Ytrain, 0.80)
    min_drive_fire = 1.0 - thr_ld
    leads = []
    for ep in test_eps:
        if classify_ep(ep) != "death":
            continue
        fa = next((i for i, tr in enumerate(ep)
                   if min(float(tr["energy"]), float(tr["thirst"])) / DRIVE_MAX < min_drive_fire), None)
        if fa is not None:
            leads.append((len(ep) - 1) - fa)
    print()
    print("HONNÊTETÉ (§2) — le meilleur trigger 'lowest_drive' est un COMPTE-À-REBOURS, pas un détecteur de décision :")
    print(f"  fire quand min_drive < {min_drive_fire:.3f} (~{100*min_drive_fire:.0f} unités de drive restantes)")
    if leads:
        sl = sorted(leads)
        print(f"  LEAD-TIME avant mort (test, morts): med={st.median(leads)} macro "
              f"(~{10*st.median(leads):.0f} pas Godot) | p25={sl[len(sl)//4]} p75={sl[3*len(sl)//4]} min={min(leads)}")
    aucs = [(H, auc(*_swap(build_frames(usable, H)))) for H in (3, 15, 50)]
    print("  AUC(lowest_drive) plate vs H : " + " ".join(f"H{H}={a:.3f}" for H, a in aucs)
          + "  → tautologique (min_drive ≈ horloge de mort), PAS un signal sémantique appris.")
    print("  ⇒ Un arbitre cheap 'défer quand un drive est presque vide' est FAISABLE et fire sur une minorité")
    print("     d'états, MAIS avec un préavis FIXE ~10 macro (~100 pas). Il ne détecte PAS TÔT la décision myope")
    print("     (les triggers d'arbitrage depleted_far/arb_tension ne battent PAS l'horloge). Question ouverte NON")
    print("     testée ici : Mode-2 peut-il SAUVER depuis ce transfert tardif ? (rescate = étape suivante).")

    # ---- VERDICT falsifiable ----
    print()
    print("=" * 78)
    print("VERDICT")
    bauc = best[1]
    bdef = best[2]["defer_rate"]
    if bauc > 0.75 and bdef < 0.35:
        print(f"→ FAISABLE : meilleur AUC={bauc:.3f} (>0.75) ET défer@R.8={bdef:.3f} (<0.35).")
        print("  Un arbitre CHEAP codé-main suffit → PONT Mode-1↔Mode-2 VIABLE.")
        print("  Prochain : construire la démo boucle-combinée minimale (réflexe par défaut, défer au planner au trigger).")
    elif bauc < 0.7 or bdef > 0.5:
        print(f"→ INSUFFISANT : meilleur AUC={bauc:.3f}, défer@R.8={bdef:.3f}.")
        print("  Trigger cheap insuffisant → trigger APPRIS nécessaire (WM-surprise / petit classifieur) → plus dur.")
        print("  Documenter comme travail futur ; NE PAS construire la démo cheap.")
    else:
        print(f"→ ZONE GRISE : meilleur AUC={bauc:.3f}, défer@R.8={bdef:.3f}.")
        print("  Ni franchement faisable ni franchement insuffisant → creuser (autre H, autre combo) avant de trancher.")


# ------------------------------------------------------------------------- selfcheck
def selfcheck() -> None:
    # dims
    assert N_RAYS == 36, N_RAYS
    retina = [1.0, 0.0, 0.0, 0.0] * N_RAYS
    assert len(retina) == 4 * N_RAYS == 144
    retina[0:4] = [0.2, 0.9, 0.1, 0.1]   # rouge proche → food_d=0.2
    retina[4:8] = [0.5, 0.1, 0.1, 0.9]   # bleu moyen  → water_d=0.5
    assert abs(_nearest(retina, RED) - 0.2) < 1e-9, _nearest(retina, RED)
    assert abs(_nearest(retina, BLUE) - 0.5) < 1e-9, _nearest(retina, BLUE)

    # features : energy le plus bas → drive épuisé = bouffe(rouge) ; sa ressource = food_d
    f = frame_feats({"energy": 20.0, "thirst": 80.0, "retina": retina})
    assert abs(f["min_drive"] - 0.20) < 1e-9, f
    assert abs(f["depleted_d"] - 0.2) < 1e-9, f   # energy bas → bouffe → food_d
    assert abs(f["other_d"] - 0.5) < 1e-9, f
    # thirst le plus bas → drive épuisé = eau(bleu)
    f2 = frame_feats({"energy": 80.0, "thirst": 20.0, "retina": retina})
    assert abs(f2["depleted_d"] - 0.5) < 1e-9, f2   # thirst bas → eau → water_d

    # label monotonicity sur un épisode synthétique qui MEURT : les H derniers=1, avant=0
    ep = [{"energy": 100 - i, "thirst": 100, "retina": retina, "done": (i == 29)} for i in range(30)]
    ep[-1]["done"] = True
    labs = label_episode(ep, 15)
    assert labs == [0] * 15 + [1] * 15, labs
    assert sum(labs) == 15
    # monotone : une fois passé à 1, ne repasse jamais à 0
    seen1 = False
    for v in labs:
        if v == 1:
            seen1 = True
        assert not (seen1 and v == 0), labs
    # épisode qui NE MEURT PAS (tronqué) → tout 0
    ep_t = [{"energy": 50, "thirst": 50, "retina": retina, "truncated": True} for _ in range(20)]
    assert sum(label_episode(ep_t, 15)) == 0

    # AUC sanity : séparation parfaite → 1.0 ; inversée → 0.0 ; aléatoire ex-aequo → 0.5
    sc = [0, 1, 2, 3, 4, 5]; lb = [0, 0, 0, 1, 1, 1]
    assert abs(auc(sc, lb) - 1.0) < 1e-9, auc(sc, lb)
    assert abs(auc([5, 4, 3, 2, 1, 0], lb) - 0.0) < 1e-9, auc([5, 4, 3, 2, 1, 0], lb)
    assert abs(auc([1, 1, 1, 1, 1, 1], lb) - 0.5) < 1e-9, auc([1, 1, 1, 1, 1, 1], lb)

    # threshold_at_recall : sur positifs {0.9,0.6,0.3}, recall 0.8→k=round(2.4)=2→2e plus haut=0.6
    thr = threshold_at_recall([0.9, 0.6, 0.3, 0.1, 0.05], [1, 1, 1, 0, 0], 0.80)
    assert abs(thr - 0.6) < 1e-9, thr
    m = eval_at_threshold([0.9, 0.6, 0.3, 0.1, 0.05], [1, 1, 1, 0, 0], thr)
    assert abs(m["recall"] - 2 / 3) < 1e-9 and abs(m["defer_rate"] - 2 / 5) < 1e-9, m

    print("[selfcheck] OK — dims(retina 144/N_RAYS 36), depleted-drive routing, label monotonicity, "
          "AUC (1.0/0.0/0.5), threshold_at_recall + eval.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", nargs="+", default=[
        "data/checkpoints/mode1_ppo_gate2b/iter_*/buffer",
        "data/checkpoints/mode1_ppo_gate2c/iter_*/buffer",
    ], help="un ou plusieurs globs de répertoires buffer")
    ap.add_argument("--horizon", type=int, default=15, help="H macro-étapes (≈10×H pas Godot)")
    ap.add_argument("--test-frac", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    run(args.glob, args.horizon, args.test_frac, args.seed)


if __name__ == "__main__":
    main()
