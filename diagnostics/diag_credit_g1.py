"""G1 (viabilité du monde rendu) du chantier ATTRIBUTION DE CRÉDIT baie-buisson.

Doc : docs/design_attribution_credit.md §Gates G1. Mesure sur le corpus RÉEL rendu (buisson Godot
ON, monde food-only) les 3 conditions de viabilité PRÉ-ENREGISTRÉES — AUCUN entraînement :
  (a) CO-OCCURRENCE : au CONTACT d'une baie, un buisson est perçu comme voulu (≈ _bush_p) ;
  (b) DÉCORRÉLATION : il existe des baies SEULES (sans buisson) ET des buissons SEULS (sans baie)
      → identifiabilité (sinon la contingence ne peut pas trancher, cf design) ;
  (c) SÉPARABILITÉ : la rétine sépare baie et buisson (2 clusters distincts, écart inter <
      affinité intra) — sinon la reconnaissance est impossible, on ajuste le MONDE (teinte déclarée).

Si un gate échoue → ajuster la PROPRIÉTÉ DU MONDE déclarée (teinte/co-loc/nb buissons), JAMAIS le gate.

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_credit_g1.py [run_dir]
"""

from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "python")

from scripts.build_typed_slots import CONTACT_M, RELIEF, stage_a_cluster  # noqa: E402
from scripts.train_danger_saliency import LIFE_JUMP  # noqa: E402
from sylvan.models.slot_head import DEPTH_OFFSET, NRAY, RANGE  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else "data/replay_buffer/critic_kin_g1"
ASSIGN_COS = 0.90


def _scan(run: Path) -> list[dict]:
    p = run / "ep_0000.jsonl"
    op = open(p, errors="ignore") if p.exists() else gzip.open(run / "ep_0000.jsonl.gz",
                                                               "rt", errors="ignore")
    recs = []
    for line in op:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        recs.append((r["wm"]["retina0"], float(r["obs"]["energy"])))
    out = []
    for i in range(len(recs) - 1):
        ret, e = recs[i]
        e1 = recs[i + 1][1]
        rgbns = []
        for k in range(NRAY):
            d = ret[4 * k]
            if d >= 0.999 or d * RANGE + DEPTH_OFFSET >= CONTACT_M:
                continue
            v = np.array(ret[4 * k + 1:4 * k + 4], dtype=np.float64)
            if v.max() - v.min() <= 0.15:
                continue
            n = np.linalg.norm(v)
            if n > 1e-9:
                rgbns.append(v / n)
        out.append({"rgbns": rgbns, "meal": float(e1 - e > RELIEF and e1 - e < LIFE_JUMP)})
    return out


def main() -> None:
    rng = np.random.default_rng(0)
    ticks = _scan(Path(RUN))
    pooled = np.array([v for tk in ticks for v in tk["rgbns"]])
    meals = int(sum(tk["meal"] for tk in ticks))
    print(f"[g1] {len(ticks)} ticks-contact, {len(pooled)} rayons colorés, repas={meals}")
    if len(pooled) < 200:
        print("[g1] ❌ trop peu de rayons colorés — collecte insuffisante")
        return

    A = stage_a_cluster(pooled, rng)
    C, K = A["C"], A["K"]
    # identifier baie (R dominant net) et buisson (le cluster NON-rouge le plus fréquent)
    berry_j = int(max(range(K), key=lambda j: C[j][0] - max(C[j][1], C[j][2])))
    assign = np.argmax(pooled @ C.T, axis=1)
    counts = np.bincount(assign, minlength=K)
    bush_j = int(max((j for j in range(K) if j != berry_j), key=lambda j: counts[j]))
    print(f"[g1] Étape A : K={K} ; prototypes :")
    for j in range(K):
        tag = "BAIE" if j == berry_j else ("BUISSON" if j == bush_j else "autre")
        print(f"[g1]   cl{j} ({tag:7s}) = ({C[j][0]:.2f},{C[j][1]:.2f},{C[j][2]:.2f})  n={counts[j]}")

    # présence par tick (multi-rayon) de baie / buisson au contact
    berry_p = np.zeros(len(ticks)); bush_p = np.zeros(len(ticks))
    for i, tk in enumerate(ticks):
        for v in tk["rgbns"]:
            cs = C @ v
            j = int(np.argmax(cs))
            if cs[j] < ASSIGN_COS:
                continue
            if j == berry_j:
                berry_p[i] = 1.0
            elif j == bush_j:
                bush_p[i] = 1.0

    # (a) co-occurrence au contact d'une baie
    berry_frames = berry_p > 0
    cooc = float(bush_p[berry_frames].mean()) if berry_frames.any() else float("nan")
    meal_arr = np.array([tk["meal"] for tk in ticks])
    meal_frames = (meal_arr > 0)
    cooc_meal = float(bush_p[meal_frames].mean()) if meal_frames.any() else float("nan")
    # (b) décorrélation
    berry_alone = int(((berry_p > 0) & (bush_p == 0)).sum())
    bush_alone = int(((bush_p > 0) & (berry_p == 0)).sum())
    # (c) séparabilité : cos entre prototypes baie/buisson vs affinité intra (q05)
    inter = float(C[berry_j] @ C[bush_j])
    own_berry = float(np.quantile(pooled[assign == berry_j] @ C[berry_j], 0.05))
    own_bush = float(np.quantile(pooled[assign == bush_j] @ C[bush_j], 0.05))

    print(f"\n[g1] === MESURES ===")
    print(f"[g1] (a) co-occurrence AU REPAS P(buisson | repas) = {cooc_meal:.2f} (n_repas={meals}) "
          f"[réf : P(buisson|baie au contact)={cooc:.2f}]")
    print(f"[g1] (b) décorrélation : baies-seules={berry_alone}, buissons-seuls={bush_alone}")
    print(f"[g1] (c) séparabilité : cos(baie,buisson)={inter:.2f} vs intra q05 "
          f"baie={own_berry:.2f} buisson={own_bush:.2f}")

    # verdict (seuils pré-enregistrés) — (a) sur la co-occurrence AU REPAS (mesure pré-enregistrée,
    # design_attribution_credit.md §G1 « co-occurrence rendue au repas »), pas au simple contact
    # (le simple contact inclut des baies vues de loin, buisson hors portée → sous-compte).
    ga = cooc_meal >= 0.6 and meals >= 20
    gb = berry_alone >= 20 and bush_alone >= 20
    gc = inter < min(own_berry, own_bush)          # inter < intra → séparables
    print(f"\n[g1] === VERDICT G1 ===")
    print(f"[g1] (a) co-occurrence AU REPAS ≥ 0.6 (et ≥20 repas) : {'✅' if ga else '❌'}")
    print(f"[g1] (b) décorrélation (baies-seules ET buissons-seuls ≥ 20) : {'✅' if gb else '❌'}")
    print(f"[g1] (c) séparabilité (inter < intra) : {'✅' if gc else '❌'}")
    ok = ga and gb and gc
    print(f"\n[g1] {'✅✅ G1 PASSÉ' if ok else '❌ G1 ÉCHOUÉ'} — "
          f"{'monde viable → licencie G2 (mesure offline sur corpus réel)' if ok else 'ajuster la PROPRIÉTÉ DU MONDE déclarée (teinte/co-loc/nb), pas le gate'}")


if __name__ == "__main__":
    main()
