"""BUILD — WM à slots TYPÉS APPRIS du vécu (P6-reopen, docs/design_perception_types.md).

Dissout la dernière connaissance du monde codée-main : les requêtes-couleur ET le lien slot→drive.
Zéro gradient, zéro retrain (leçon 4× confirmée : une géométrie se MESURE) :

  Étape A — REGROUPER (corpus VARIÉ seulement) : k-means sur la couleur normalisée du rayon
    touchant le plus proche → K DÉCOUVERT (silhouette) ; prototype = centroïde normalisé ;
    **MARGE PAR TYPE MESURÉE** : thr_k = milieu entre le cos intra-groupe (q05) et le cos
    inter-groupe (q99.5) — la séparation émerge des données (fix Mur B, plus de 0.55 imposé).
  Étape B — LIER (corpus POOLÉ : 10 runs plats + varié — les couleurs plates vivent DANS les
    clusters appris → ~10× plus d'événements-relief, zéro collecte) : contingence à
    PORTÉE-CONTACT P(conséquence | groupe le plus proche ET à portée) ; le blocage
    Rescorla-Wagner est porté par la forme forward (G-pré : renverse le confond vert 73 %).
  Émission : slots ORDONNÉS par drive découvert (food=0, water=1, danger=2 — la convention meta
    des consommateurs est préservée ; le LIEN, lui, est appris) ; `color_queries` ← prototypes ;
    meta["query_thr"] ← marges mesurées (le serveur les charge). WM GELÉ, poids intacts.

GATES PRÉ-ENREGISTRÉS (§gates du design — échec → PAS d'émission, négatif commité) :
  G-sep     : monde séparable — cos intra q05 > cos inter q99.5 pour chaque groupe (sinon
              ajuster les DISTRIBUTIONS du monde, jamais les gates) ;
  G-cluster : K découvert = 3 ET cos(prototype, couleur rendue vraie par classe) ≥ 0.98
              (oracle d'ÉVAL, licite monde-jouet) ;
  G-bind    : bijection groupe→conséquence correcte (food→énergie, water→soif, vert→dégâts) ;
  G-slot    : position du slot typé vs oracle couleur-rendue ≤ 0.5 m méd sur visibles (varié).

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python -m scripts.build_typed_slots [--selfcheck]
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np
import torch

from scripts.train_danger_saliency import DMG_DROP, LIFE_JUMP
from scripts.train_sprint_critic import DEATH_RUNS
from sylvan.models.slot_head import DEPTH_OFFSET, NRAY, RANGE

RELIEF = 5.0            # remontée de drive/tick = consommation (convention partagée)
CONTACT_M = 1.5         # « à portée » : reliefs/morsures arrivent au contact (reach 1.2 / morsure 1.3)
VARIED_RUN = "data/replay_buffer/critic_kin_typcorp"
SRC_WM = "data/checkpoints/wm_objcentric_kin_haz/wm_best.pt"
OUT_DIR = "data/checkpoints/wm_objcentric_kin_typed"
OUTCOMES = ("energy", "thirst", "damage")
PURE_CLASS = {0: "rouge", 1: "vert", 2: "bleu"}     # canal dominant (oracle d'éval)


# ------------------------------------------------------------------ scan (une passe par run)

def scan_run(run: Path) -> dict:
    """Par tick avec ≥1 rayon touchant : rgbn + distance du PLUS PROCHE, classe dominante
    (oracle éval), les 3 conséquences vécues, et la rétine brute (échantillonnée, pour G-slot)."""
    recs = []
    p = run / "ep_0000.jsonl.gz"
    op = gzip.open(p, "rt", errors="ignore") if p.exists() else open(run / "ep_0000.jsonl",
                                                                     errors="ignore")
    for line in op:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        recs.append((r["wm"]["retina0"], float(r["obs"]["energy"]),
                     float(r["obs"]["thirst"]), float(r["obs"]["health"])))
    rgbn, dist, dom, ys = [], [], [], {o: [] for o in OUTCOMES}
    ret_sample = []
    for i in range(len(recs) - 1):
        ret, e, t, h = recs[i]
        e1, t1, h1 = recs[i + 1][1], recs[i + 1][2], recs[i + 1][3]
        boundary = e1 - e > LIFE_JUMP or t1 - t > LIFE_JUMP or h1 - h > LIFE_JUMP
        best_k, best_d = -1, 2.0
        for k in range(NRAY):
            d = ret[4 * k]
            if d < 0.999 and d < best_d:
                best_d, best_k = d, k
        if best_k < 0:
            continue
        v = np.array(ret[4 * best_k + 1:4 * best_k + 4], dtype=np.float64)
        n = np.linalg.norm(v)
        if n < 1e-9:
            continue
        rgbn.append(v / n)
        dist.append(best_d * RANGE + DEPTH_OFFSET)
        # oracle éval : canal dominant, même seuil de saturation ABSOLU que la règle mur-vert
        dom.append(int(v.argmax()) if v.max() - v.min() > 0.15 else -1)
        ys["energy"].append(float(not boundary and e1 - e > RELIEF))
        ys["thirst"].append(float(not boundary and t1 - t > RELIEF))
        dmg = (i > 0 and recs[i - 1][3] - h > DMG_DROP
               and not (e - recs[i - 1][1] > LIFE_JUMP or t - recs[i - 1][2] > LIFE_JUMP
                        or h - recs[i - 1][3] > LIFE_JUMP))
        ys["damage"].append(float(dmg))
        if i % 10 == 0:
            ret_sample.append(ret)
    return {"rgbn": np.array(rgbn), "dist": np.array(dist), "dom": np.array(dom),
            "y": {o: np.array(ys[o]) for o in OUTCOMES}, "retinas": ret_sample}


# ------------------------------------------------------------------ k-means / silhouette (numpy, mesure)

def kmeans(X: np.ndarray, k: int, rng: np.random.Generator, iters: int = 60):
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
    return C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12), a


def silhouette(X: np.ndarray, a: np.ndarray, k: int) -> float:
    if k < 2 or len(set(a.tolist())) < k:
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


# ------------------------------------------------------------------ étapes A/B (mesure pure)

def stage_a_cluster(rgbn: np.ndarray, rng: np.random.Generator,
                    fit_cap: int = 8000, sil_cap: int = 1500) -> dict:
    fit_idx = rng.choice(len(rgbn), size=min(fit_cap, len(rgbn)), replace=False)
    Xf = rgbn[fit_idx]
    best = {}
    for k in range(2, 6):
        C, a = kmeans(Xf, k, np.random.default_rng(k))
        si = rng.choice(len(Xf), size=min(sil_cap, len(Xf)), replace=False)
        best[k] = (silhouette(Xf[si], a[si], k), C)
    K = max(best, key=lambda k: best[k][0])
    C = best[K][1]
    assign = np.argmax(rgbn @ C.T, axis=1)                    # cos (tout est normalisé)
    # MARGE MESURÉE par groupe : milieu entre intra-q05 et inter-q99.5 (fix Mur B)
    thr, own_q05, cross_q995 = [], [], []
    for j in range(K):
        own = rgbn[assign == j] @ C[j]
        cross = rgbn[assign != j] @ C[j]
        o5 = float(np.quantile(own, 0.05)) if len(own) else float("nan")
        c995 = float(np.quantile(cross, 0.995)) if len(cross) else 0.0
        own_q05.append(o5)
        cross_q995.append(c995)
        thr.append(0.5 * (o5 + c995))
    return {"K": K, "sil": {k: round(best[k][0], 3) for k in best}, "C": C, "assign": assign,
            "thr": np.array(thr), "own_q05": np.array(own_q05), "cross_q995": np.array(cross_q995)}


def stage_b_bind(C: np.ndarray, pooled: dict) -> dict:
    """Contingence à portée-contact sur le corpus POOLÉ → lien groupe→conséquence (bijection)."""
    assign = np.argmax(pooled["rgbn"] @ C.T, axis=1)
    contact = pooled["dist"] < CONTACT_M
    K = C.shape[0]
    table = np.zeros((K, len(OUTCOMES)))
    n_k = np.zeros(K, dtype=int)
    for j in range(K):
        m = (assign == j) & contact
        n_k[j] = int(m.sum())
        for oi, o in enumerate(OUTCOMES):
            table[j, oi] = float(pooled["y"][o][m].mean()) if m.any() else float("nan")
    bound = {j: OUTCOMES[int(np.nanargmax(table[j]))] for j in range(K)}
    return {"table": table, "bound": bound, "n_contact": n_k}


# ------------------------------------------------------------------ gates + émission

def g_slot_positions(C_ordered: np.ndarray, thr_ordered: np.ndarray,
                     retinas: list, cls_of_slot: list[int]) -> dict:
    """Position du slot TYPÉ vs oracle couleur-rendue (rayon le plus proche de la classe),
    par type, sur les rétines variées échantillonnées."""
    from sylvan.models.slot_head import SelfSupervisedSlotHead
    payload = torch.load(SRC_WM, map_location="cpu", weights_only=False)
    head = SelfSupervisedSlotHead(n_resources=3)
    head.load_state_dict({k.removeprefix("slot_encoder."): v for k, v in payload["model"].items()
                          if k.startswith("slot_encoder.")})
    head.eval()
    with torch.no_grad():
        head.color_queries.copy_(torch.tensor(C_ordered, dtype=torch.float32))
        head.query_thr.copy_(torch.tensor(thr_ordered, dtype=torch.float32))
    X = torch.tensor(retinas, dtype=torch.float32)
    with torch.no_grad():
        pos = head.positions(X)                                # [N, 3, 2]
        vis = head.visibility(X) > 1e-6
    errs = {s: [] for s in range(3)}
    for n, ret in enumerate(retinas):
        for s in range(3):
            if not bool(vis[n, s]):
                continue
            want = cls_of_slot[s]
            best_k, best_d = -1, 2.0
            for k in range(NRAY):
                d, R, G, B = ret[4 * k:4 * k + 4]
                if d >= 0.999 or d >= best_d:
                    continue
                v = [R, G, B]
                if max(v) - min(v) > 0.15 and int(np.argmax(v)) == want:
                    best_d, best_k = d, k
            if best_k < 0:
                continue
            dm = best_d * RANGE + DEPTH_OFFSET
            th = 2.0 * math.pi * best_k / NRAY
            ox, oz = dm * math.sin(th), dm * math.cos(th)
            errs[s].append(math.hypot(float(pos[n, s, 0]) - ox, float(pos[n, s, 1]) - oz))
    return {s: (float(np.median(e)), float(np.quantile(e, 0.9)), len(e)) if e else
            (float("nan"),) * 2 + (0,) for s, e in errs.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--varied", default=VARIED_RUN)
    ap.add_argument("--pool", nargs="+", default=DEATH_RUNS)
    ap.add_argument("--src", default=SRC_WM)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return

    rng = np.random.default_rng(0)
    varied = scan_run(Path(args.varied))
    print(f"[typed] corpus VARIÉ : {len(varied['rgbn'])} ticks-objet, "
          f"reliefs E={int(varied['y']['energy'].sum())} T={int(varied['y']['thirst'].sum())}")
    pools = [varied] + [scan_run(Path(r)) for r in args.pool]
    pooled = {"rgbn": np.concatenate([p["rgbn"] for p in pools]),
              "dist": np.concatenate([p["dist"] for p in pools]),
              "y": {o: np.concatenate([p["y"][o] for p in pools]) for o in OUTCOMES}}
    print(f"[typed] corpus POOLÉ (lien) : {len(pooled['rgbn'])} ticks, reliefs "
          f"E={int(pooled['y']['energy'].sum())} T={int(pooled['y']['thirst'].sum())} "
          f"dgr={int(pooled['y']['damage'].sum())}")

    # Étape A (varié seulement)
    A = stage_a_cluster(varied["rgbn"], rng)
    print(f"[typed] Étape A : K={A['K']} (silhouettes {A['sil']}) ; marges mesurées "
          f"{[round(float(v), 3) for v in A['thr']]}")
    g_sep = bool(np.all(A["own_q05"] > A["cross_q995"]))
    # oracle éval : couleur rendue vraie par classe dominante (médiane par canal, varié)
    proto_cos = {}
    cls_of_group = {}
    for j in range(A["K"]):
        mask = A["assign"] == j
        doms = varied["dom"][mask]
        cl = int(np.bincount(doms[doms >= 0], minlength=3).argmax()) if (doms >= 0).any() else -1
        cls_of_group[j] = cl
        members = varied["rgbn"][(varied["dom"] == cl)]
        true_c = np.median(members, axis=0)
        true_c /= (np.linalg.norm(true_c) + 1e-12)
        proto_cos[j] = float(A["C"][j] @ true_c)
        print(f"[typed]   groupe {j} ≈ {PURE_CLASS.get(cl, '?'):6s} cos(vrai rendu)={proto_cos[j]:.4f} "
              f"intra-q05={A['own_q05'][j]:.3f} inter-q99.5={A['cross_q995'][j]:.3f}")
    g_cluster = A["K"] == 3 and len(set(cls_of_group.values())) == 3 \
        and min(proto_cos.values()) >= 0.98

    # Étape B (poolé)
    B = stage_b_bind(A["C"], pooled)
    print(f"[typed] Étape B : contingences (contact<{CONTACT_M}m, n={list(B['n_contact'])})")
    for j in range(A["K"]):
        row = " ".join(f"P({o})={B['table'][j, oi]:.4f}" for oi, o in enumerate(OUTCOMES))
        print(f"[typed]   groupe {j} ({PURE_CLASS.get(cls_of_group[j], '?')}) : {row} → {B['bound'][j]}")
    want = {0: "energy", 2: "thirst", 1: "damage"}            # rouge→E, bleu→T, vert→D (oracle éval)
    g_bind = (sorted(B["bound"].values()) == sorted(OUTCOMES)
              and all(B["bound"][j] == want.get(cls_of_group[j]) for j in range(A["K"])))

    # ordre des slots par drive DÉCOUVERT (convention consommateurs : food=0, water=1, danger=2)
    order = {o: j for j, o in B["bound"].items()}
    slot_order = [order["energy"], order["thirst"], order["damage"]]
    C_ord = A["C"][slot_order]
    thr_ord = A["thr"][slot_order]
    cls_of_slot = [cls_of_group[j] for j in slot_order]

    # G-slot (varié, oracle couleur-rendue)
    gs = g_slot_positions(C_ord, thr_ord, varied["retinas"], cls_of_slot)
    names = ("food", "water", "danger")
    for s in range(3):
        med, p90, n = gs[s]
        print(f"[typed] G-slot {names[s]:6s} : méd={med:.3f} m p90={p90:.3f} m (n={n})")
    g_slot = all(gs[s][2] > 0 and gs[s][0] <= 0.5 for s in range(3))

    print(f"\n[typed] === GATES (pré-enregistrés) ===")
    print(f"[typed] G-sep     : intra-q05 > inter-q99.5 ∀groupe → {'✅' if g_sep else '❌ (monde non séparable → ajuster le MONDE)'}")
    print(f"[typed] G-cluster : K=3, classes distinctes, cos≥0.98 → {'✅' if g_cluster else '❌'} "
          f"(cos_min={min(proto_cos.values()):.4f})")
    print(f"[typed] G-bind    : bijection correcte → {'✅' if g_bind else '❌'} ({B['bound']})")
    print(f"[typed] G-slot    : méd ≤ 0.5 m ∀type → {'✅' if g_slot else '❌'}")
    verdict = g_sep and g_cluster and g_bind and g_slot
    if not verdict:
        print("[typed] ❌ GATE ÉCHOUÉ → PAS d'émission (négatif à commiter, diagnostiquer sur trace)")
        return
    # Émission : WM gelé, seuls color_queries (buffer) + meta changent
    payload = torch.load(args.src, map_location="cpu", weights_only=False)
    state = dict(payload["model"])
    old_q = state["slot_encoder.color_queries"]
    new_q = torch.tensor(C_ord, dtype=torch.float32)
    state["slot_encoder.color_queries"] = new_q
    meta = dict(payload["meta"])
    meta.update({"query_thr": [float(v) for v in thr_ord],
                 "queries": "typed_learned_from_consequence_P6reopen",
                 "queries_cos_to_hand": [round(float(new_q[i] @ old_q[i]), 4) for i in range(3)],
                 "bind_table": B["table"][slot_order].tolist(),
                 "varied_run": str(args.varied), "pool_runs": list(args.pool)})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state, "meta": meta}, out / "wm_best.pt")
    print(f"[typed] ✅ GATES PASSÉS → WM TYPÉ émis : {out / 'wm_best.pt'}")
    print(f"[typed]   requêtes apprises (cos vs main : {meta['queries_cos_to_hand']}), "
          f"marges {meta['query_thr']} — zéro couleur codée-main, lien slot→drive DÉCOUVERT.")


def selfcheck() -> None:
    rng = np.random.default_rng(0)
    # monde synthétique 2 types : amas rouge→energy, amas bleu→thirst, contact partout
    A_ = np.array([0.9, 0.2, 0.1]) + 0.03 * rng.standard_normal((300, 3))
    B_ = np.array([0.1, 0.3, 0.9]) + 0.03 * rng.standard_normal((300, 3))
    X = np.abs(np.vstack([A_, B_]))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    res = stage_a_cluster(X, rng, fit_cap=600, sil_cap=400)
    assert res["K"] == 2, res["sil"]
    assert bool(np.all(res["own_q05"] > res["cross_q995"])), "amas nets → G-sep doit tenir"
    y_e = np.array([1.0 if i < 300 and i % 30 == 0 else 0.0 for i in range(600)])
    y_t = np.array([1.0 if i >= 300 and i % 30 == 0 else 0.0 for i in range(600)])
    pooled = {"rgbn": X, "dist": np.full(600, 1.0),
              "y": {"energy": y_e, "thirst": y_t, "damage": np.zeros(600)}}
    Bb = stage_b_bind(res["C"], pooled)
    red_j = int(np.argmax(res["C"][:, 0]))
    assert Bb["bound"][red_j] == "energy" and Bb["bound"][1 - red_j] == "thirst", Bb["bound"]
    print("[selfcheck] OK — K découvert, G-sep sur amas nets, lien par contingence correct")


if __name__ == "__main__":
    main()
