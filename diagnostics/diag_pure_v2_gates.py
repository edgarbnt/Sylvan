"""GATES OFFLINE P2-bis — forme pure + prime de mort apprise (docs/design_purete_hjepa.md).

Forme v2 : score = longueur + 0.02·max(0, κ·douleur̂·100 + P̂mort·κ·100 − P̂·bénéfice).
G-death est jugé au train (AUC 0.839 ✓) ; ici les gates de FORME pré-enregistrés :
  1. G-kill-decisions (le cœur) : sur les décisions died_danger de classe cross, TENUES
     (têtes-mort de plis CV), v2 refuse la traversée ≥ +30 pts plus souvent que pure-v1 ;
  2. G-res-v2  : précision choix-vs-bucket-best ≥ analytique (runs à coûts analytiques) ;
  3. G-consist-v2 : bascule intra-poursuite ≤ 1.2× analytique.
⚠️ Caveat disclosé : P̂(repas) = modèle final (in-sample sur g24/spx — identique pour v1/v2,
comparaison homogène) ; la tête mort est TENUE par plis pour G-kill.

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_pure_v2_gates.py
"""

from __future__ import annotations

import statistics as st

import torch

from scripts.train_sprint_critic import (DEATH_RUNS, DEFAULT_PAIN, DEFAULT_RUNS, _INTR_EPS,
                                         PainCritic, SprintCritic, _bucket_key, _fit_bce,
                                         _pain_of, load_corpus, measured_drain, net_utility,
                                         simulate_choice, sprint_inputs)

CKPT = "data/checkpoints/sprint_critic/sprint_pure_v2.pt"


def main() -> None:
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    meal = SprintCritic()
    meal.load_state_dict(ck["state_dict"])
    meal.eval()
    death = SprintCritic()
    death.load_state_dict(ck["death_state_dict"])
    death.eval()
    pain = PainCritic()
    pk = torch.load(DEFAULT_PAIN, map_location="cpu", weights_only=True)
    pain.load_state_dict(pk["state_dict"])
    pain.eval()

    rows_all = load_corpus(DEATH_RUNS)          # tête mort : corpus élargi (10 runs)
    rows_ana = [r for r in rows_all if not ("judge" in r["run"] or "pure" in r["run"])]
    drain = measured_drain(DEFAULT_RUNS)
    kappa = st.median([r["left"] for r in rows_all]) / 100.0
    kw = dict(kappa=kappa, drain=drain, restore=float(ck["restore"]))
    print(f"[p2bis] corpus {len(rows_all)} déc (analytiques : {len(rows_ana)}) | κ={kappa:.1f} "
          f"| tête mort AUC={ck['death_auc_cv']:.3f}")

    # GATE 1 — G-kill-decisions : les décisions qui ont RÉELLEMENT tué (classe cross), tenues.
    pain_model = pain
    Xd = torch.cat([sprint_inputs([r["feats_all"][r["chosen"]]], (r["e"], r["t"], r["h"]),
                                  _pain_of(pain_model, [r["feats_all"][r["chosen"]]]))
                    for r in rows_all])
    yd = torch.tensor([float(r["died_danger"]) for r in rows_all])
    lifed = torch.tensor([r["life"] for r in rows_all])
    fold_death = {}
    for k in range(4):
        tr = ~(lifed % 4 == k)
        if int(yd[tr].sum()) > 0:
            fold_death[k] = _fit_bce(Xd[tr], yd[tr], 4000, 0)
    killers = [r for r in rows_all if r["died_danger"] and r["cls"] == "cross"]
    n_v1 = n_v2 = n_k = 0
    for r in killers:
        k = int(r["life"] % 4)
        if k not in fold_death:
            continue
        n_k += 1
        n_v1 += not simulate_choice(r, meal, pain, pure=True, **kw)
        n_v2 += not simulate_choice(r, meal, pain, pure=True, death_model=fold_death[k], **kw)
    ref_v1 = n_v1 / max(n_k, 1)
    ref_v2 = n_v2 / max(n_k, 1)
    g_kill = ref_v2 >= ref_v1 + 0.30
    print(f"[p2bis] G-kill-decisions : {n_k} décisions-tueuses | refus v1 {100 * ref_v1:.0f}% "
          f"→ v2 {100 * ref_v2:.0f}% (gate ≥ +30 pts) → {'✅' if g_kill else '❌'}")

    # GATE 2 — G-res-v2 (runs analytiques seulement : replay du bras analytique fiable).
    blocked = [r for r in rows_ana if r["intr_direct"] == r["intr_direct"]
               and r["intr_direct"] > _INTR_EPS]
    better: dict[tuple, bool] = {}
    for key in {_bucket_key(r) for r in blocked}:
        cr = [net_utility(r, kappa, drain) for r in blocked
              if _bucket_key(r) == key and r["cls"] == "cross"]
        rf = [net_utility(r, kappa, drain) for r in blocked
              if _bucket_key(r) == key and r["cls"] == "refuse"]
        if len(cr) >= 10 and len(rf) >= 10:
            better[key] = st.mean(cr) > st.mean(rf)
    evalable = [r for r in blocked if _bucket_key(r) in better]
    acc_a = acc_2 = 0
    for r in evalable:
        want = better[_bucket_key(r)]
        acc_a += simulate_choice(r, None, None) == want
        acc_2 += simulate_choice(r, meal, pain, pure=True, death_model=death, **kw) == want
    acc_a = acc_a / max(len(evalable), 1)
    acc_2 = acc_2 / max(len(evalable), 1)
    g_res = acc_2 >= acc_a
    print(f"[p2bis] G-res-v2 : analytique {100 * acc_a:.0f}% | v2 {100 * acc_2:.0f}% "
          f"(gate ≥ analytique) → {'✅' if g_res else '❌'}  (n={len(evalable)})")

    # GATE 3 — G-consist-v2.
    by_run: dict[str, list[dict]] = {}
    for r in rows_ana:
        by_run.setdefault(r["run"], []).append(r)
    f_a = f_2 = n_pairs = 0
    for seq in by_run.values():
        seq.sort(key=lambda r: r["tick"])
        for a, b in zip(seq, seq[1:]):
            if a["target"] != b["target"] or b["tick"] - a["tick"] > 60:
                continue
            n_pairs += 1
            f_a += simulate_choice(a, None, None) != simulate_choice(b, None, None)
            f_2 += (simulate_choice(a, meal, pain, pure=True, death_model=death, **kw)
                    != simulate_choice(b, meal, pain, pure=True, death_model=death, **kw))
    rate_a = f_a / max(n_pairs, 1)
    rate_2 = f_2 / max(n_pairs, 1)
    g_consist = rate_2 <= 1.2 * rate_a + 1e-9
    print(f"[p2bis] G-consist-v2 : {n_pairs} paires | analytique {100 * rate_a:.1f}% | "
          f"v2 {100 * rate_2:.1f}% (gate ≤ 1.2×) → {'✅' if g_consist else '❌'}")

    verdict = g_kill and g_res and g_consist
    print(f"[p2bis] {'✅ GATES DE FORME PASSÉS → juge closed-loop (≥40 repas ET ≤10 morts poolés)' if verdict else '❌ GATE ÉCHOUÉ → remise conservée, négatif commité'}")


if __name__ == "__main__":
    main()
