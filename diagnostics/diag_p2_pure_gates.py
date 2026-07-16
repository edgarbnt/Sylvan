"""GATES OFFLINE P2 — tarification du vert 100 % apprise (docs/design_purete_hjepa.md §P2).

Forme PURE (remplacement) : score = longueur + 0.02·max(0, κ·douleur̂·100 − P̂·bénéfice) —
W=25/green_margin sortent du chemin décisionnel. ZÉRO entraînement : mêmes têtes que le juge
PASS (sprint_best.pt + pain_v3). Le quantum de ranking (P̂·ben − κ·pain̂) étant inchangé,
G-rank/G-mono portent ; ici les 3 gates PRÉ-ENREGISTRÉS spécifiques au remplacement :
  1. G-res-pure   : précision choix-vs-bucket-best ≥ analytique (parité 72 %) ;
  2. G-consist-pure : bascule intra-poursuite ≤ 1.2× analytique (le tueur historique) ;
  3. G-safe       : traversées simulées sur bloqués BLESSÉS-PROFONDS (h<30, direct>médiane)
                    ≤ forme-remise + 10 pts.
⚠️ Caveat disclosé : P̂ = modèle final (in-sample pour les deux formes apprises — comparaison
homogène ; le juge closed-loop reste le critère).

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_p2_pure_gates.py
"""

from __future__ import annotations

import statistics as st

import torch

from scripts.train_sprint_critic import (DEFAULT_PAIN, DEFAULT_RUNS, _INTR_EPS, PainCritic,
                                         SprintCritic, _bucket_key, load_corpus, measured_drain,
                                         net_utility, simulate_choice)

CKPT = "data/checkpoints/sprint_critic/sprint_best.pt"


def main() -> None:
    rows = load_corpus(DEFAULT_RUNS)
    drain = measured_drain(DEFAULT_RUNS)
    kappa = st.median([r["left"] for r in rows]) / 100.0
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    critic = SprintCritic()
    critic.load_state_dict(ck["state_dict"])
    critic.eval()
    pain = PainCritic()
    pk = torch.load(DEFAULT_PAIN, map_location="cpu", weights_only=True)
    pain.load_state_dict(pk["state_dict"])
    pain.eval()
    kw = dict(kappa=kappa, drain=drain, restore=float(ck["restore"]))
    print(f"[p2] corpus {len(rows)} décisions | κ={kappa:.1f} drain={drain:.4f} "
          f"restore={ck['restore']:.0f} | têtes = juge PASS (45/8)")

    # référence empirique : action meilleure par bucket (même règle que le trainer)
    blocked = [r for r in rows if r["intr_direct"] == r["intr_direct"]
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

    # GATE 1 — G-res-pure : parité avec l'analytique au minimum.
    acc = {"analytique": 0, "remise": 0, "pure": 0}
    for r in evalable:
        want = better[_bucket_key(r)]
        acc["analytique"] += simulate_choice(r, None, None) == want
        acc["remise"] += simulate_choice(r, critic, pain, composed=True, **kw) == want
        acc["pure"] += simulate_choice(r, critic, pain, pure=True, **kw) == want
    for k in acc:
        acc[k] = acc[k] / max(len(evalable), 1)
    g_res = acc["pure"] >= acc["analytique"]
    print(f"[p2] G-res-pure   : analytique {100 * acc['analytique']:.0f}% | remise "
          f"{100 * acc['remise']:.0f}% | pure {100 * acc['pure']:.0f}% (gate ≥ analytique) "
          f"→ {'✅' if g_res else '❌'}  (n={len(evalable)})")

    # GATE 2 — G-consist-pure : bascule intra-poursuite.
    by_run: dict[str, list[dict]] = {}
    for r in rows:
        by_run.setdefault(r["run"], []).append(r)
    flips = {"analytique": 0, "pure": 0}
    n_pairs = 0
    for seq in by_run.values():
        seq.sort(key=lambda r: r["tick"])
        for a, b in zip(seq, seq[1:]):
            if a["target"] != b["target"] or b["tick"] - a["tick"] > 60:
                continue
            n_pairs += 1
            flips["analytique"] += (simulate_choice(a, None, None)
                                    != simulate_choice(b, None, None))
            flips["pure"] += (simulate_choice(a, critic, pain, pure=True, **kw)
                              != simulate_choice(b, critic, pain, pure=True, **kw))
    rate_a = flips["analytique"] / max(n_pairs, 1)
    rate_p = flips["pure"] / max(n_pairs, 1)
    g_consist = rate_p <= 1.2 * rate_a + 1e-9
    print(f"[p2] G-consist-pure : {n_pairs} paires | analytique {100 * rate_a:.1f}% | "
          f"pure {100 * rate_p:.1f}% (gate ≤ 1.2×) → {'✅' if g_consist else '❌'}")

    # GATE 3 — G-safe : bloqués blessés-profonds (h<30, direct plus profond que la médiane).
    med_d = st.median([r["intr_direct"] for r in blocked])
    danger_zone = [r for r in blocked if r["h"] < 30.0 and r["intr_direct"] > med_d]
    cr_reb = sum(simulate_choice(r, critic, pain, composed=True, **kw) for r in danger_zone)
    cr_pur = sum(simulate_choice(r, critic, pain, pure=True, **kw) for r in danger_zone)
    n_dz = max(len(danger_zone), 1)
    g_safe = cr_pur / n_dz <= cr_reb / n_dz + 0.10
    print(f"[p2] G-safe       : blessés-profonds n={len(danger_zone)} | traverse remise "
          f"{100 * cr_reb / n_dz:.0f}% | pure {100 * cr_pur / n_dz:.0f}% (gate ≤ +10 pts) "
          f"→ {'✅' if g_safe else '❌'}")

    verdict = g_res and g_consist and g_safe
    print(f"[p2] {'✅ GATES OFFLINE PASSÉS → juge closed-loop (≥40 repas ET ≤10 morts poolés)' if verdict else '❌ GATE ÉCHOUÉ → forme-remise conservée, négatif commité'}")


if __name__ == "__main__":
    main()
