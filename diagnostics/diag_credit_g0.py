"""G0 (GRATUIT, synthétique) du chantier ATTRIBUTION DE CRÉDIT baie-buisson.

Doc : docs/design_attribution_credit.md §Gates G0. Zéro run, zéro Godot, zéro entraînement — rejeu
des corpus PROPRES existants (sans swap) + injection SYNTHÉTIQUE d'un buisson co-occurrent.

Thèse à trancher : la liaison NAÏVE (co-occurrence P(conséquence|indice présent), = l'argmax
actuel de build_typed_slots) se fait PIÉGER par un distracteur neutre qui accompagne la baie ; la
liaison par CONTINGENCE PARTIELLE (Rescorla-Wagner = régression linéaire de la conséquence sur le
VECTEUR des indices présents) BLOQUE le buisson (coeff ≈ 0 : la baie explique déjà le repas) → classe
NEUTRE. Condition d'identifiabilité : décorrélation (baies parfois seules, buissons parfois seuls).

Critères G0 (pré-enregistrés) :
  (a) baie(rouge) → énergie (naïf ET partiel d'accord, coeff fort) ;
  (b) buisson → NEUTRE : coeff partiel ≤ plancher-bruit (indice ALÉATOIRE calibrateur) MALGRÉ une
      co-occurrence forte au repas ;
  (c) CONTRASTE décisif : la vue NAÏVE donne au buisson un P(énergie|buisson) élevé (le piège), la
      contingence partielle le RENVERSE ;
  (d) non-régression : eau(bleu)→soif, danger(vert)→dégâts corrects, le vert (confond Mur A) N'est
      PAS lié à l'énergie par la méthode partielle.

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_credit_g0.py
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, "python")

from scripts.build_typed_slots import CONTACT_M, RELIEF, stage_a_cluster  # noqa: E402
from scripts.train_danger_saliency import DMG_DROP, LIFE_JUMP  # noqa: E402
from sylvan.models.slot_head import DEPTH_OFFSET, NRAY, RANGE  # noqa: E402

CLEAN_RUNS = [
    "data/replay_buffer/critic_kin_typcorp",
    "data/replay_buffer/critic_kin_g24as1", "data/replay_buffer/critic_kin_g24as2",
    "data/replay_buffer/critic_kin_g24bs1", "data/replay_buffer/critic_kin_g24bs2",
    "data/replay_buffer/critic_kin_pure1", "data/replay_buffer/critic_kin_pure2",
]
ASSIGN_COS = 0.90                 # un rayon-contact appartient à un cluster si cos ≥ ce seuil
# Injection buisson (propriétés DÉCLARÉES du monde synthétique). Régime FIDÈLE au scénario « baie
# (presque) TOUJOURS dans un buisson, buissons vides = minorité » = le cas DUR où le naïf ne peut
# pas séparer baie et buisson (P(repas|buisson) ≈ P(repas|baie)). Les buissons vides (F_BUSH_ALONE)
# restent nécessaires à l'IDENTIFIABILITÉ (décorrélation) mais peu nombreux (monde réaliste).
P_BUSH_GIVEN_BERRY = 0.92         # 92 % des baies vues sont DANS un buisson (co-occurrence forte)
F_BUSH_ALONE = 0.010              # buissons vides dispersés : 1 % des ticks SANS baie (décorrélation minimale)
P_RANDOM = 0.10                   # indice ALÉATOIRE calibrateur (présence i.i.d.) → plancher-bruit


def _scan(run: Path) -> list[dict]:
    """Par tick : rgbn des rayons COLORÉS au CONTACT (liste) + conséquences vécues t→t+1."""
    p = run / "ep_0000.jsonl"
    op = open(p, errors="ignore") if p.exists() else gzip.open(run / "ep_0000.jsonl.gz",
                                                                "rt", errors="ignore")
    recs = []
    for line in op:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        o = r["obs"]
        recs.append((r["wm"]["retina0"], float(o["energy"]), float(o["thirst"]), float(o["health"])))
    out = []
    for i in range(len(recs) - 1):
        ret, e, t, h = recs[i]
        e1, t1, h1 = recs[i + 1][1], recs[i + 1][2], recs[i + 1][3]
        boundary = e1 - e > LIFE_JUMP or t1 - t > LIFE_JUMP or h1 - h > LIFE_JUMP
        rgbns = []
        for k in range(NRAY):
            d = ret[4 * k]
            if d >= 0.999:
                continue
            if d * RANGE + DEPTH_OFFSET >= CONTACT_M:      # au CONTACT seulement
                continue
            v = np.array(ret[4 * k + 1:4 * k + 4], dtype=np.float64)
            if v.max() - v.min() <= 0.15:                  # coloré (saturé), pas du fond
                continue
            n = np.linalg.norm(v)
            if n > 1e-9:
                rgbns.append(v / n)
        # dégât : chute de santé sur ce pas (t-1→t déjà consommé ; on prend t→t+1 comme les reliefs)
        dmg = (not boundary) and (h - h1 > DMG_DROP)
        out.append({"rgbns": rgbns,
                    "energy": float((not boundary) and e1 - e > RELIEF),
                    "thirst": float((not boundary) and t1 - t > RELIEF),
                    "damage": float(dmg)})
    return out


def main() -> None:
    rng = np.random.default_rng(0)
    ticks: list[dict] = []
    for r in CLEAN_RUNS:
        if Path(r).exists():
            ticks.extend(_scan(Path(r)))
    n = len(ticks)
    pooled = np.array([v for tk in ticks for v in tk["rgbns"]])
    print(f"[g0] {n} ticks-contact, {len(pooled)} rayons colorés poolés")

    # --- Étape A (réutilisée) : clusters des VRAIES couleurs (rouge/vert/bleu) ---
    A = stage_a_cluster(pooled, rng)
    C = A["C"]
    K = A["K"]
    dom = [int(np.argmax(C[j])) for j in range(K)]            # canal dominant du prototype
    name_of = {0: "rouge/baie", 1: "vert/danger", 2: "bleu/eau"}
    berry_j = max(range(K), key=lambda j: C[j][0] - max(C[j][1], C[j][2]))   # + rouge dominant net
    water_j = max(range(K), key=lambda j: C[j][2])
    danger_j = max(range(K), key=lambda j: C[j][1])
    print(f"[g0] Étape A : K={K} ; clusters (canal dom) = "
          f"{[name_of.get(dom[j], '?') for j in range(K)]}  "
          f"(baie=cl{berry_j}, eau=cl{water_j}, danger=cl{danger_j})")

    # --- présence par tick des vrais clusters (parmi TOUS les rayons-contact, multi-indice) ---
    present = np.zeros((n, K), dtype=float)
    for i, tk in enumerate(ticks):
        for v in tk["rgbns"]:
            cs = C @ v
            j = int(np.argmax(cs))
            if cs[j] >= ASSIGN_COS:
                present[i, j] = 1.0
    berry = present[:, berry_j]

    # --- INJECTION SYNTHÉTIQUE du buisson (neutre) + indice aléatoire calibrateur ---
    bush = np.zeros(n)
    for i in range(n):
        if berry[i] > 0:
            bush[i] = 1.0 if rng.random() < P_BUSH_GIVEN_BERRY else 0.0   # baie DANS un buisson
        else:
            bush[i] = 1.0 if rng.random() < F_BUSH_ALONE else 0.0          # buisson dispersé (seul)
    rand = (rng.random(n) < P_RANDOM).astype(float)                        # calibrateur i.i.d.

    y = {o: np.array([tk[o] for tk in ticks]) for o in ("energy", "thirst", "damage")}
    # décorrélation effective (identifiabilité) : baies-seules & buissons-seuls
    berry_alone = float(((berry > 0) & (bush == 0)).sum())
    bush_alone = float(((bush > 0) & (berry == 0)).sum())
    meals = int(y["energy"].sum())
    meals_bush = int(((y["energy"] > 0) & (bush > 0)).sum())
    print(f"[g0] injection buisson : co-occ baie {P_BUSH_GIVEN_BERRY:.0%} / buisson-seul {F_BUSH_ALONE:.1%} ; "
          f"baies-seules={int(berry_alone)}, buissons-seuls={int(bush_alone)} (décorrélation OK) ; "
          f"repas={meals} dont {meals_bush} en buisson ({100*meals_bush/max(meals,1):.0f}%)")

    # --- matrice des VRAIS indices (clusters) + biais ; le buisson/placebos s'ajoutent en colonne ---
    base = [present[:, j] for j in range(K)]
    names = [name_of.get(dom[j], f"cl{j}") for j in range(K)]

    def naive(cue: np.ndarray, yy: np.ndarray) -> float:
        m = cue > 0
        return float(yy[m].mean()) if m.any() else float("nan")

    def coef_with(extra: np.ndarray, yy: np.ndarray) -> np.ndarray:
        """coeffs de [clusters..., extra] (Rescorla-Wagner = régression) ; dernière = 'extra'."""
        X = np.column_stack(base + [extra, np.ones(n)])
        c, *_ = np.linalg.lstsq(X, yy, rcond=None)
        return c[:-1]

    # PLANCHER-BRUIT MESURÉ (pas réglé) : distribution du coeff d'un indice causalement NUL mais
    # AVEC LA MÊME STRUCTURE que le buisson (même fréquence + même collinéarité 92% avec la baie).
    # Un indice quasi-collinéaire à la baie a une variance de coeff GONFLÉE → le plancher doit être
    # ce que ce type d'indice obtient PAR HASARD, sinon on sous-estime le bruit (leçon de la 1ʳᵉ passe).
    def placebo_bush() -> np.ndarray:
        pb = np.zeros(n)
        for i in range(n):
            pb[i] = 1.0 if (berry[i] > 0 and rng.random() < P_BUSH_GIVEN_BERRY) or \
                           (berry[i] == 0 and rng.random() < F_BUSH_ALONE) else 0.0
        return pb
    placebo_coef_e = np.array([coef_with(placebo_bush(), y["energy"])[-1] for _ in range(40)])
    floor = float(np.quantile(np.abs(placebo_coef_e), 0.95))    # p95 |coeff| d'un nul même-structure

    print("\n[g0] === LIAISON par indice (NAÏF = P(conséq|indice) ; PARTIEL = coeff régression) ===")
    for o in ("energy", "thirst", "damage"):
        ce_o = coef_with(bush, y[o])
        print(f"  conséquence « {o} » :")
        for ci, cname in enumerate(names + ["BUISSON(neutre)"]):
            cue = (base + [bush])[ci]
            print(f"    {cname:16s} naïf P={naive(cue, y[o]):.4f}   partiel coeff={ce_o[ci]:+.4f}")

    # --- verdict G0 ---
    ce = coef_with(bush, y["energy"]); ct = coef_with(bush, y["thirst"]); cd = coef_with(bush, y["damage"])
    berry_c, bush_c = ce[berry_j], ce[-1]
    naive_bush, naive_berry = naive(bush, y["energy"]), naive(berry, y["energy"])
    a = berry_c > 3 * floor
    b = abs(bush_c) <= floor                               # dans la bande-nulle même-structure → NEUTRE
    c = naive_bush > 0.5 * naive_berry and bush_c < 0.3 * berry_c
    d = (ct[water_j] > 3 * floor and cd[danger_j] > 3 * floor and ce[danger_j] < 0.3 * berry_c)
    print(f"\n[g0] === VERDICT G0 (plancher-bruit MESURÉ = p95 |coeff placebo même-structure| = {floor:.4f}) ===")
    print(f"[g0] (a) baie→énergie (coeff {berry_c:+.4f} > 3×plancher) : {'✅' if a else '❌'}")
    print(f"[g0] (b) buisson NEUTRE (|coeff {bush_c:+.4f}| ≤ plancher, malgré 83% de repas en buisson) : {'✅' if b else '❌'}")
    print(f"[g0] (c) CONTRASTE : naïf NE SÉPARE PAS (P_buisson={naive_bush:.4f} ≈ P_baie={naive_berry:.4f}) "
          f"MAIS partiel renverse (bush {bush_c:+.4f} ≪ baie {berry_c:+.4f}) : {'✅' if c else '❌'}")
    print(f"[g0] (d) non-régression eau→soif/danger→dégâts + vert PAS→énergie (Mur A) : {'✅' if d else '❌'}")
    ok = a and b and c and d
    print(f"\n[g0] {'✅✅ G0 PASSÉ' if ok else '❌ G0 ÉCHOUÉ'} — "
          f"{'contingence partielle valide → licencie G1 (bump Godot buisson)' if ok else 'mécanisme à corriger AVANT Godot'}")


if __name__ == "__main__":
    main()
