"""G-pré (GRATUIT, 0 train) — reconnaissance des types en 2 étapes, sur corpus SYNTHÉTIQUEMENT varié.

Gate le travail Godot (docs/design_perception_types.md §Prochain-pas) : la MACHINERIE de
reconnaissance marche-t-elle avant qu'on rende quoi que ce soit de varié ? On perturbe les rétines
RÉELLES (teinte+texture+désat, module appearance_synth) pour simuler un monde à apparences variées,
puis on exécute les 2 étapes :
  - Étape A (regrouper) : k-means sur la couleur du rayon le plus proche → K découvert + prototypes
    + MARGE mesurée de l'écart entre groupes (remplace le seuil global 0.55 → lève le Mur B) ;
  - Étape B (lier) : liaison groupe→drive par CONTINGENCE forward P(outcome | groupe le plus proche)
    (Rescorla-Wagner : l'INFORMATION, pas la contiguïté) → le vert, présent en permanence près du
    danger mais rarement suivi d'un repas, n'est PAS lié à l'énergie (lève le Mur A). On montre en
    contraste la vue NAÏVE P(groupe | repas) qui, elle, était dominée par le vert (le piège P6).

Oracle d'ÉVAL SEULEMENT (licite monde-jouet) : couleurs vraies = médiane rgbn des rayons NON
perturbés par canal dominant. ⚠️ Synthétique ≠ rendu réel : gate la machinerie, pas la difficulté
réelle (celle-ci se mesure au vrai bump).

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_pretypes_recognition.py [--selfcheck]
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from scripts.appearance_synth import NRAY, perturb
from scripts.train_danger_saliency import DMG_DROP, LIFE_JUMP
from sylvan.models.perception_head import RETINA_DIM

RELIEF = 5.0                 # remontée de drive/tick = consommation (convention partagée)
SAT = 0.15                   # saturation min pour classer une couleur (oracle)
RUNS = ["data/replay_buffer/critic_kin_g24as1", "data/replay_buffer/critic_kin_g24bs1",
        "data/replay_buffer/critic_kin_judge1", "data/replay_buffer/critic_kin_pure2"]
CLASSES = ("rouge", "vert", "bleu")           # canal dominant 0/1/2 (oracle)
OUTCOMES = ("energy", "thirst", "damage")


def _nearest_touch(ret: torch.Tensor) -> int | None:
    """Indice du rayon touchant (depth<0.999) le plus proche, ou None si aucun objet en vue."""
    d = ret.view(NRAY, 4)[:, 0]
    touch = d < 0.999
    if not bool(touch.any()):
        return None
    dd = torch.where(touch, d, torch.full_like(d, 2.0))
    return int(dd.argmin())


def _rgbn(ret: torch.Tensor, k: int) -> np.ndarray:
    rgb = ret.view(NRAY, 4)[k, 1:4].numpy().astype(np.float64)
    n = np.linalg.norm(rgb)
    return rgb / (n + 1e-9)


def _dominant(rgb: np.ndarray) -> int | None:
    if rgb.max() - rgb.min() <= SAT:
        return None
    return int(rgb.argmax())


def scan(runs: list[str], gen: torch.Generator) -> dict:
    """Par tick usable : rgbn du rayon le plus proche (perturbé ET propre), classe dominante propre,
    et les 3 outcomes vécus (relief énergie/soif à t+1, dégâts à t). Frontières de vie = saut vital."""
    pert = {"rgbn": [], "y": {o: [] for o in OUTCOMES}}
    clean_by_class: dict[int, list[np.ndarray]] = {0: [], 1: [], 2: []}
    for run in runs:
        recs = []
        p = Path(run) / "ep_0000.jsonl.gz"
        op = gzip.open(p, "rt", errors="ignore") if p.exists() else open(
            Path(run) / "ep_0000.jsonl", errors="ignore")
        for line in op:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            recs.append((torch.tensor(r["wm"]["retina0"], dtype=torch.float32),
                         float(r["obs"]["energy"]), float(r["obs"]["thirst"]),
                         float(r["obs"]["health"])))
        for i in range(len(recs) - 1):
            ret, e, t, h = recs[i]
            e1, t1, h1 = recs[i + 1][1], recs[i + 1][2], recs[i + 1][3]
            boundary = (e1 - e > LIFE_JUMP or t1 - t > LIFE_JUMP or h1 - h > LIFE_JUMP)
            k = _nearest_touch(ret)
            if k is None:
                continue
            pr = perturb(perturb(perturb(ret.view(1, RETINA_DIM), "hue", 20.0, gen),
                         "texture", 0.05, gen), "desat", 0.4, gen)     # combiné modéré réaliste
            rp = _rgbn(pr.view(RETINA_DIM), k)
            pert["rgbn"].append(rp)
            pert["y"]["energy"].append(float(not boundary and e1 - e > RELIEF))
            pert["y"]["thirst"].append(float(not boundary and t1 - t > RELIEF))
            dmg = (i > 0 and recs[i - 1][3] - h > DMG_DROP
                   and not (e - recs[i - 1][1] > LIFE_JUMP or t - recs[i - 1][2] > LIFE_JUMP
                            or h - recs[i - 1][3] > LIFE_JUMP))
            pert["y"]["damage"].append(float(dmg))
            rc = _rgbn(ret.view(RETINA_DIM), k)
            dom = _dominant(rc)
            if dom is not None:
                clean_by_class[dom].append(rc)
    return {"X": np.array(pert["rgbn"]), "y": {o: np.array(pert["y"][o]) for o in OUTCOMES},
            "oracle": {c: np.median(np.array(clean_by_class[c]), axis=0)
                       for c in range(3) if clean_by_class[c]}}


def _kmeans(X: np.ndarray, k: int, rng: np.random.Generator, iters: int = 50):
    c = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = np.min(((X[:, None, :] - np.array(c)[None]) ** 2).sum(-1), axis=1)
        c.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    C = np.array(c)
    a = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        a = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        newC = np.array([X[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return C, a


def _silhouette(X: np.ndarray, a: np.ndarray, k: int) -> float:
    if k < 2:
        return -1.0
    D = np.sqrt(((X[:, None, :] - X[None]) ** 2).sum(-1) + 1e-12)
    s = []
    for i in range(len(X)):
        same = a == a[i]
        same[i] = False
        ai = D[i, same].mean() if same.any() else 0.0
        bi = min((D[i, a == j].mean() for j in range(k) if j != a[i] and (a == j).any()),
                 default=0.0)
        s.append((bi - ai) / (max(ai, bi) + 1e-9))
    return float(np.mean(s))


def _cos(u: np.ndarray, v: np.ndarray) -> float:
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=RUNS)
    ap.add_argument("--cluster-cap", type=int, default=1500)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return

    gen = torch.Generator().manual_seed(0)
    rng = np.random.default_rng(0)
    d = scan(args.runs, gen)
    X, Y, oracle = d["X"], d["y"], d["oracle"]
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)         # directions (teinte)
    print(f"[g-pre] {len(Xn)} ticks avec objet ; outcomes "
          f"{ {o: int(Y[o].sum()) for o in OUTCOMES} } ; oracle classes {sorted(oracle)}")

    # ÉTAPE A — regrouper : découvrir K par silhouette (2..5), sur un sous-échantillon déterministe.
    idx = rng.choice(len(Xn), size=min(args.cluster_cap, len(Xn)), replace=False)
    Xs = Xn[idx]
    best = {}
    for k in range(2, 6):
        C, a = _kmeans(Xs, k, np.random.default_rng(k))
        best[k] = (_silhouette(Xs, a, k), C)
        print(f"[g-pre] K={k} : silhouette={best[k][0]:.3f}")
    K = max(best, key=lambda k: best[k][0])
    C = best[K][1]
    # MARGE mesurée = moitié du plus petit écart angulaire entre centres (remplace 0.55 global)
    cos_between = [[_cos(C[i], C[j]) for j in range(K) if j != i] for i in range(K)]
    margin = 0.5 * (1.0 - max(max(row) for row in cos_between))
    # apparier les groupes découverts aux classes oracle (rouge/vert/bleu) par cosinus max
    match = {}
    for cl, ctr in oracle.items():
        j = int(np.argmax([_cos(C[j], ctr) for j in range(K)]))
        match[cl] = (j, _cos(C[j], ctr))
    print(f"\n[g-pre] ÉTAPE A : K découvert = {K} (marge mesurée = {margin:.3f})")
    for cl in sorted(match):
        j, cs = match[cl]
        print(f"[g-pre]   {CLASSES[cl]:6s} → groupe {j} (cos au vrai = {cs:.4f})")

    # ÉTAPE B — lier par CONTINGENCE forward P(outcome | groupe le plus proche).
    assign = np.argmin(((Xn[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
    print(f"\n[g-pre] ÉTAPE B : contingence P(outcome | groupe) — le lien appris")
    print(f"[g-pre]   {'classe':<8}{'P(energy)':>11}{'P(thirst)':>11}{'P(damage)':>11}{'  → lié à':>12}")
    cont = {}
    for cl in sorted(match):
        j = match[cl][0]
        m = assign == j
        row = {o: float(Y[o][m].mean()) if m.any() else float("nan") for o in OUTCOMES}
        cont[cl] = row
        bound = max(row, key=row.get)
        print(f"[g-pre]   {CLASSES[cl]:<8}{row['energy']:>11.3f}{row['thirst']:>11.3f}"
              f"{row['damage']:>11.3f}{bound:>12}")
    # CONTRASTE : vue naïve P(groupe | repas énergie) — le piège P6 (dominé par le vert)
    er = Y["energy"] > 0.5
    naive = {cl: float((assign[er] == match[cl][0]).mean()) for cl in sorted(match)}
    print(f"[g-pre]   NAÏVE P(groupe | repas énergie) = "
          f"{ {CLASSES[c]: round(naive[c], 2) for c in naive} } (le vert dominait en P6)")

    # === GATES G-pré (pré-enregistrés §Prochain-pas ; synthétique = machinerie, pas difficulté réelle) ===
    cos_min = min(match[cl][1] for cl in match) if len(match) == 3 else 0.0
    g_a = K == 3 and len(match) == 3 and cos_min >= 0.95
    g_b = (0 in cont and 2 in cont
           and max(cont[0], key=cont[0].get) == "energy"      # rouge → énergie
           and max(cont[2], key=cont[2].get) == "thirst")     # bleu → soif
    g_c = (1 in cont and max(cont[1], key=cont[1].get) == "damage"   # vert → dégâts, PAS énergie
           and cont[1]["energy"] < 0.5 * cont[0]["energy"]
           and naive.get(1, 0) > naive.get(0, 0))             # naïf aurait mis le vert (contraste)
    print(f"\n[g-pre] === GATES G-pré ===")
    print(f"[g-pre] G-pré-A regrouper : K=3 ✔ ET cos_min={cos_min:.3f}≥0.95 → {'✅' if g_a else '❌'}")
    print(f"[g-pre] G-pré-B lier      : rouge→énergie ET bleu→soif → {'✅' if g_b else '❌'}")
    print(f"[g-pre] G-pré-C blocage   : vert→dégâts (PAS énergie) ET P(en|vert)<½P(en|rouge) ET "
          f"naïf=vert → {'✅' if g_c else '❌'}")
    ok = g_a and g_b and g_c
    print(f"[g-pre] {'✅ MACHINERIE VALIDÉE → bump Godot licencié' if ok else '❌ machinerie fautive → corriger AVANT Godot (zéro coût)'}")


def selfcheck() -> None:
    assert NRAY == 36 and RETINA_DIM == 144
    ret = torch.tensor([1.0, 0.0, 0.0, 0.0] * NRAY)
    ret[4 * 5:4 * 5 + 4] = torch.tensor([0.3, 0.0, 0.0, 1.0])           # bleu à 3 m
    ret[4 * 2:4 * 2 + 4] = torch.tensor([0.1, 0.9, 0.1, 0.1])           # rouge à 1 m (plus proche)
    assert _nearest_touch(ret) == 2                                     # le plus proche
    assert _dominant(_rgbn(ret, 2)) == 0 and _dominant(_rgbn(ret, 5)) == 2
    assert _dominant(np.array([0.4, 0.4, 0.4])) is None                 # gris insaturé
    # k-means récupère 2 amas nets ; silhouette 2 > silhouette dégénéré
    rng = np.random.default_rng(0)
    A = np.array([1.0, 0.0, 0.0]) + 0.02 * rng.standard_normal((80, 3))
    B = np.array([0.0, 1.0, 0.0]) + 0.02 * rng.standard_normal((80, 3))
    X = np.vstack([A, B])
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    C, a = _kmeans(X, 2, np.random.default_rng(1))
    assert _silhouette(X, a, 2) > 0.7 and len(set(a.tolist())) == 2
    assert abs(_cos(np.array([1.0, 0, 0]), np.array([1.0, 0, 0])) - 1.0) < 1e-6
    print("[selfcheck] OK — plus-proche/dominant/rgbn, k-means 2 amas, silhouette, cos")


if __name__ == "__main__":
    main()
