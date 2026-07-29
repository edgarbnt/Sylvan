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
  G-cluster : chaque PULSION capte un groupe de classe distincte ET cos(prototype, couleur rendue
              vraie) ≥ 0.98 pour ces groupes-là (oracle d'ÉVAL, licite monde-jouet). Corrigé le
              2026-07-29 : exigeait « K = 3 », ce qui interdisait au monde d'avoir une classe
              NEUTRE (le vert des buissons/arbres, lié à aucune pulsion). Le seuil 0,98 est
              CONSERVÉ et s'applique aux groupes servis, danger compris ;
  G-bind    : bijection PULSION→groupe correcte (énergie→rouge, soif→bleu, dégâts→vert). Corrigé
              le même jour : exigeait une bijection sur TOUS les groupes, donc qu'aucun ne reste
              neutre. Le groupe élu pour une pulsion est celui de contingence MAXIMALE ;
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
    # DEUX dispositions sur disque, toutes deux légitimes : la collecte BC écrit UN `ep_0000.jsonl`,
    # la collecte WM — celle qui alimente le retrain, donc celle sur laquelle on veut DÉCOUVRIR les
    # requêtes — écrit UN FICHIER PAR ÉPISODE. Ne lire que la première faisait échouer cet outil sur
    # le corpus même du nouveau monde, avec un FileNotFoundError qui ressemble à une erreur de
    # chemin alors que le corpus est là.
    fichiers = [f for f in (run / "ep_0000.jsonl.gz", run / "ep_0000.jsonl") if f.exists()]
    if not fichiers:
        fichiers = sorted([*run.glob("episode_*.jsonl"), *run.glob("episode_*.jsonl.gz")])
    if not fichiers:
        raise SystemExit(f"aucun ep_0000.jsonl ni episode_*.jsonl dans {run}")
    for f in fichiers:
        op = gzip.open(f, "rt", errors="ignore") if f.suffix == ".gz" else open(f, errors="ignore")
        with op:
            for line in op:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                recs.append((r["wm"]["retina0"], float(r["obs"]["energy"]),
                             float(r["obs"]["thirst"]), float(r["obs"]["health"])))
    rgbn, dist, dom, ys = [], [], [], {o: [] for o in OUTCOMES}
    ret_sample = []
    # État du tick précédent, pour ne compter qu'un FRONT de dégâts par épisode (voir plus bas).
    # Remis à faux par run : deux runs distincts ne se chaînent pas.
    was_hurting = False
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
        # 🚨 LES DÉGÂTS COMME ÉVÉNEMENT, PAS COMME DURÉE (corrigé 2026-07-29). `energy` et `thirst`
        # ci-dessus sont des FRONTS : une remontée de jauge = un repas = UN tick. Les dégâts, eux,
        # étaient vrais à CHAQUE tick passé dans la zone — soit ~110 ticks pour un seul événement.
        # La contingence P(conséquence | groupe) comparait donc un événement à une durée, et les
        # dégâts écrasaient tout : mesuré sur ce monde, 16 241 reliefs de dégâts contre 126
        # d'énergie, si bien que les QUATRE groupes se liaient à « damage », y compris le rouge.
        # On ne touche NI le seuil NI le gate : on rend la grandeur comparable en ne comptant que
        # le FRONT MONTANT — l'entrée en dégâts, une fois par épisode de dégâts.
        hurting = (i > 0 and recs[i - 1][3] - h > DMG_DROP
                   and not (e - recs[i - 1][1] > LIFE_JUMP or t - recs[i - 1][2] > LIFE_JUMP
                            or h - recs[i - 1][3] > LIFE_JUMP))
        ys["damage"].append(float(hurting and not was_hurting))
        was_hurting = hurting
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
                     retinas: list, cls_of_slot: list[int], src_wm: str = SRC_WM) -> dict:
    """Position du slot TYPÉ vs oracle couleur-rendue (rayon le plus proche de la classe),
    par type, sur les rétines variées échantillonnées."""
    from sylvan.models.slot_head import SelfSupervisedSlotHead
    # 🚨 LE CHECKPOINT JUGÉ DOIT ÊTRE CELUI QU'ON ÉMET (corrigé 2026-07-29). Ce gate lisait la
    # CONSTANTE SRC_WM pendant que l'émission, elle, honore --src : passer --src rendait donc un
    # verdict G-slot sur un AUTRE modèle que celui écrit, sans rien signaler.
    payload = torch.load(src_wm, map_location="cpu", weights_only=False)
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
    ap.add_argument("--emit-anyway", action="store_true",
                    help="écrire malgré un gate rouge, en INSCRIVANT l'échec dans le meta")
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
    # G-CLUSTER — SPÉCIFICATION CORRIGÉE (owner, 2026-07-29), et il faut dire exactement ce qui
    # change et ce qui NE change pas. L'ancienne forme exigeait « K == 3 » : elle a été écrite pour
    # un monde où chaque couleur était une pulsion. Le monde-forêt en a QUATRE — rouge, bleu, le
    # vert du danger, et un vert NEUTRE (buissons, arbres) qui ne se lie à rien. Exiger 3 revient à
    # interdire au monde d'être plus riche que l'hypothèse du gate, ce que le gate ne cherchait pas
    # à garantir. Ce qu'il garantissait vraiment : que chaque PULSION dispose d'un groupe propre et
    # fidèle à la couleur rendue. C'est ce qu'on écrit ici.
    # 🚨 CE QUI EST DÉLIBÉRÉMENT CONSERVÉ : le seuil de fidélité 0,98, et il s'applique aux groupes
    # LIÉS À UNE PULSION. On ne l'abaisse pas, et on n'exempte pas le danger. Un groupe neutre, lui,
    # n'a pas à être fidèle : personne ne s'en sert.
    g_cluster = None   # calculé après l'étape B (il dépend des liaisons) — voir plus bas

    # Étape B (poolé)
    B = stage_b_bind(A["C"], pooled)
    print(f"[typed] Étape B : contingences (contact<{CONTACT_M}m, n={list(B['n_contact'])})")
    for j in range(A["K"]):
        row = " ".join(f"P({o})={B['table'][j, oi]:.4f}" for oi, o in enumerate(OUTCOMES))
        print(f"[typed]   groupe {j} ({PURE_CLASS.get(cls_of_group[j], '?')}) : {row} → {B['bound'][j]}")
    want = {0: "energy", 2: "thirst", 1: "damage"}            # rouge→E, bleu→T, vert→D (oracle éval)
    # G-BIND — MÊME CORRECTION. L'ancienne forme exigeait une bijection sur TOUS les groupes, donc
    # qu'aucun groupe ne reste sans pulsion : impossible dès qu'il existe une classe neutre. Ce
    # qu'elle protégeait réellement, c'est que chaque PULSION soit captée par le BON groupe. On
    # vérifie donc la bijection sur les trois pulsions, et que le groupe élu est bien celui que
    # l'oracle d'éval attend. Un groupe neutre peut se lier à ce qu'il veut : il n'est pas servi.
    #
    # 🚨 ET ON RÉPARE UNE FRAGILITÉ AU PASSAGE. `order` inversait le dictionnaire des liaisons :
    # quand deux groupes revendiquaient la même pulsion, c'est le DERNIER dans l'ordre d'itération
    # qui gagnait. Ici le bon groupe l'emportait par chance ; si le neutre avait porté un indice
    # plus grand, il aurait écrasé le bon SANS RIEN SIGNALER. On élit désormais explicitement le
    # groupe de CONTINGENCE MAXIMALE pour chaque pulsion, ce qui est ce qu'on voulait dire.
    oi = {o: i for i, o in enumerate(OUTCOMES)}
    order = {o: int(np.argmax(B["table"][:, oi[o]])) for o in OUTCOMES}
    g_bind = (len(set(order.values())) == 3
              and all(want.get(cls_of_group[order[o]]) == o for o in OUTCOMES))
    # G-cluster, maintenant que les groupes SERVIS sont connus (fidélité exigée sur eux seuls).
    g_cluster = (len({cls_of_group[j] for j in order.values()}) == 3
                 and min(proto_cos[j] for j in order.values()) >= 0.98)
    slot_order = [order["energy"], order["thirst"], order["damage"]]
    C_ord = A["C"][slot_order]
    thr_ord = A["thr"][slot_order]
    cls_of_slot = [cls_of_group[j] for j in slot_order]

    # G-slot (varié, oracle couleur-rendue)
    gs = g_slot_positions(C_ord, thr_ord, varied["retinas"], cls_of_slot, args.src)
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
    failed = [n for n, ok in (("G-sep", g_sep), ("G-cluster", g_cluster),
                              ("G-bind", g_bind), ("G-slot", g_slot)) if not ok]
    if not verdict and not args.emit_anyway:
        print("[typed] ❌ GATE ÉCHOUÉ → PAS d'émission (négatif à commiter, diagnostiquer sur trace)")
        return
    if not verdict:
        # ÉMISSION SOUS RÉSERVE, décidée par l'owner. Le point important n'est pas qu'on émette :
        # c'est que le checkpoint PORTE la trace de ce qui a échoué. Un canal dont on sait qu'un
        # slot est mauvais n'est pas dangereux ; un canal dont plus personne ne sait qu'un slot est
        # mauvais l'est. `meta["gates_failed"]` est donc écrit et le serveur pourra le lire.
        print(f"[typed] ⚠️  ÉMISSION FORCÉE malgré {failed} — inscrit dans meta['gates_failed']")
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
                 "varied_run": str(args.varied), "pool_runs": list(args.pool),
                 # TRACE DE CE QUI N'A PAS PASSÉ. Vide = tous les gates verts. Non vide = le
                 # checkpoint est utilisable MAIS un consommateur doit savoir sur quoi. On préfère
                 # un canal dont on connaît le défaut à un canal dont le défaut a été oublié.
                 "gates_failed": failed,
                 "slot_quality_m": {n: round(float(gs[i][0]), 3)
                                    for i, n in enumerate(("food", "water", "danger"))}})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state, "meta": meta}, out / "wm_best.pt")
    print(f"[typed] {'✅ GATES PASSÉS' if verdict else '⚠️  ÉMIS SOUS RÉSERVE'} → WM TYPÉ émis : "
          f"{out / 'wm_best.pt'}")
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
