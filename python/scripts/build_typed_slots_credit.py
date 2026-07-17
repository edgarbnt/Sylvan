"""G2 — BUILD du WM CRÉDIT-TYPÉ (chantier attribution de crédit, docs/design_attribution_credit.md §G2).

Étend build_typed_slots avec la LEÇON du buisson (G0/G1) : la reconnaissance apprend à DISTINGUER un
distracteur NEUTRE (le buisson) d'une ressource, et à EXCLURE le buisson des marges de slot (fix
« espace couleur encombré » — la marge du voisin se resserre autour du cluster buisson).

LIAISON — leçon G2 (négatif informatif) : la contingence ΔP sur la PRÉSENCE multi-indice (G0
synthétique) RÉINTRODUIT le Mur A sur données réelles (le rouge, présent quand on se fait mordre au
cœur du danger, hérite d'un gros coeff DÉGÂTS > son coeff énergie rare). Le mécanisme VIVANT
(build_typed_slots) l'évite en conditionnant sur le PLUS PROCHE au contact (on est le plus proche de
ce qu'on CONSOMME → rouge-proche → énergie). G2 combine donc : contingence au PLUS PROCHE (proven) +
test ΔP-avec-SIGNIFICATIVITÉ pour le NEUTRE (le buisson, rarement le plus proche à un événement et
jamais co-présent au danger dans le corpus food-only G1, n'a AUCUN lift significatif → NEUTRE).

Zéro gradient, zéro retrain, WM GELÉ. Clustering AVEC le buisson → K découvert + marges mesurées
avec le buisson (fix encombrement). Émission : 3 slots (food/water/danger) = clusters non-neutres
(dominant par présence si split) ; le buisson n'est PAS un slot mais a resserré les marges.

GATES G2 : (a) baie→énergie, eau→soif, danger→dégâts (non-régression, lift significatif) ; (b)
buisson NEUTRE (≥1 cluster sans lift significatif) ; (c) TRANSFERT : sur les trames BAIE-SEULE (G1),
le slot food appris localise la baie ≤ 0.5 m méd. Échec → PAS d'émission (négatif commité).

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python -m scripts.build_typed_slots_credit
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from scripts.build_typed_slots import (CONTACT_M, OUTCOMES, scan_run, stage_a_cluster)
from scripts.train_sprint_critic import DEFAULT_RUNS
from sylvan.models.slot_head import DEPTH_OFFSET, NRAY, RANGE, SelfSupervisedSlotHead

BUSH_RUN = "data/replay_buffer/critic_kin_g1"                       # G1 : baie + buisson réel
TYPCORP = "data/replay_buffer/critic_kin_typcorp"                   # varié : food/eau/danger
SRC_WM = "data/checkpoints/wm_objcentric_kin_haz/wm_best.pt"        # WM gelé source (mêmes poids)
OUT_DIR = "data/checkpoints/wm_objcentric_kin_typed_credit"
ASSIGN_COS = 0.90
SIG_K = 3.0              # lift ΔP significatif si > SIG_K écarts-types (test de neutralité)


def bind_nearest_dp(C: np.ndarray, pooled: dict) -> dict:
    """Contingence au PLUS PROCHE + test de SIGNIFICATIVITÉ (ΔP > SIG_K·SE) → lien ou NEUTRE.
    ΔP(o|j) = P(o | j est le plus proche au contact) − P(o | un AUTRE cluster est le plus proche) :
    positif-significatif = j PRÉDIT o (lift réel) ; aucun lift significatif = cluster NEUTRE."""
    assign = np.argmax(pooled["rgbn"] @ C.T, axis=1)
    contact = pooled["dist"] < CONTACT_M
    K = C.shape[0]
    table = np.zeros((K, len(OUTCOMES)))       # P(o | j plus proche)
    dp = np.zeros((K, len(OUTCOMES)))
    se = np.zeros((K, len(OUTCOMES)))
    n_in = np.zeros(K, dtype=int)
    for j in range(K):
        m_in = (assign == j) & contact
        m_out = (assign != j) & contact
        ni, no = int(m_in.sum()), int(m_out.sum())
        n_in[j] = ni
        for oi, o in enumerate(OUTCOMES):
            p_in = float(pooled["y"][o][m_in].mean()) if ni else 0.0
            p_out = float(pooled["y"][o][m_out].mean()) if no else 0.0
            table[j, oi] = p_in
            dp[j, oi] = p_in - p_out
            se[j, oi] = math.sqrt(p_in * (1 - p_in) / max(ni, 1) + p_out * (1 - p_out) / max(no, 1))
    bound: dict[int, str] = {}
    for j in range(K):
        sig = [oi for oi in range(len(OUTCOMES)) if dp[j, oi] > SIG_K * se[j, oi]]
        bound[j] = OUTCOMES[max(sig, key=lambda oi: dp[j, oi])] if sig else "neutral"
    return {"table": table, "dp": dp, "se": se, "bound": bound, "n_in": n_in}


def transfer_error(C_slot: np.ndarray, thr_slot: np.ndarray, retinas: list,
                   C: np.ndarray, berry_j: int, bush_js: list[int]) -> tuple[float, float, int]:
    """Transfert : sur les trames BAIE-SEULE (rayon assigné au cluster BAIE présent, AUCUN rayon
    assigné à un cluster BUISSON au contact), le slot food (requête baie apprise) localise-t-il la
    baie ? Erreur vs le rayon assigné-baie le plus proche (détection par assignation de cluster)."""
    head = SelfSupervisedSlotHead(n_resources=3)
    payload = torch.load(SRC_WM, map_location="cpu", weights_only=False)
    head.load_state_dict({k.removeprefix("slot_encoder."): v for k, v in payload["model"].items()
                          if k.startswith("slot_encoder.")})
    head.eval()
    with torch.no_grad():
        head.color_queries.copy_(torch.tensor(C_slot, dtype=torch.float32))
        head.query_thr.copy_(torch.tensor(thr_slot, dtype=torch.float32))
    errs = []
    for ret in retinas:
        has_berry = has_bush = False
        best_k, best_d = -1, 2.0
        for k in range(NRAY):
            d = ret[4 * k]
            if d >= 0.999 or d * RANGE + DEPTH_OFFSET >= CONTACT_M:
                continue
            v = np.array(ret[4 * k + 1:4 * k + 4], dtype=np.float64)
            if v.max() - v.min() <= 0.15:
                continue
            nv = v / (np.linalg.norm(v) + 1e-12)
            cs = C @ nv
            cj = int(np.argmax(cs))
            if cs[cj] < ASSIGN_COS:
                continue
            if cj == berry_j:
                has_berry = True
                if d < best_d:
                    best_d, best_k = d, k
            elif cj in bush_js:
                has_bush = True
        if not has_berry or has_bush or best_k < 0:      # baie-seule uniquement
            continue
        rt = torch.tensor(ret, dtype=torch.float32)
        with torch.no_grad():
            pos = head.positions(rt)[0]                  # slot 0 = food
            vis = head.visibility(rt)[0]
        if float(vis) <= 1e-6:
            continue
        dm = best_d * RANGE + DEPTH_OFFSET
        th = 2.0 * math.pi * best_k / NRAY
        errs.append(math.hypot(float(pos[0]) - dm * math.sin(th), float(pos[1]) - dm * math.cos(th)))
    if not errs:
        return float("nan"), float("nan"), 0
    return float(np.median(errs)), float(np.quantile(errs, 0.9)), len(errs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bush", default=BUSH_RUN)
    ap.add_argument("--pool", nargs="+", default=[TYPCORP] + DEFAULT_RUNS)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    bush = scan_run(Path(args.bush))
    pools = [bush] + [scan_run(Path(r)) for r in args.pool if Path(r).exists()]
    pooled = {"rgbn": np.concatenate([p["rgbn"] for p in pools]),
              "dist": np.concatenate([p["dist"] for p in pools]),
              "y": {o: np.concatenate([p["y"][o] for p in pools]) for o in OUTCOMES}}
    print(f"[credit] pooled {len(pooled['rgbn'])} ticks (nearest), reliefs "
          f"E={int(pooled['y']['energy'].sum())} T={int(pooled['y']['thirst'].sum())} "
          f"dgr={int(pooled['y']['damage'].sum())} (buisson={Path(args.bush).name}, {len(pools)} runs)")

    # Étape A — clustering AVEC le buisson
    A = stage_a_cluster(pooled["rgbn"], rng)
    C, K = A["C"], A["K"]
    print(f"[credit] Étape A : K={K} (sil {A['sil']}) ; marges {[round(float(v), 3) for v in A['thr']]}")
    for j in range(K):
        print(f"[credit]   cl{j} = ({C[j][0]:.2f},{C[j][1]:.2f},{C[j][2]:.2f})")

    # Étape B — liaison au plus proche + significativité
    B = bind_nearest_dp(C, pooled)
    print(f"[credit] Étape B : contingence au plus proche + ΔP significatif (>{SIG_K}·SE)")
    for j in range(K):
        row = " ".join(f"{o}:P={B['table'][j, oi]:.4f} ΔP={B['dp'][j, oi]:+.4f}(±{B['se'][j, oi]:.4f})"
                       for oi, o in enumerate(OUTCOMES))
        print(f"[credit]   cl{j} (n={B['n_in'][j]}) {row} → {B['bound'][j].upper()}")

    # slots = clusters non-neutres, dominant par présence-plus-proche si split ; ordre food/water/danger
    by_out: dict[str, int] = {}
    for o in OUTCOMES:
        cands = [j for j in range(K) if B["bound"][j] == o]
        if cands:
            by_out[o] = max(cands, key=lambda j: B["n_in"][j])
    neutral = [j for j in range(K) if B["bound"][j] == "neutral"]
    have = all(o in by_out for o in OUTCOMES)
    if not have:
        print(f"[credit] ❌ liaison incomplète (manque {[o for o in OUTCOMES if o not in by_out]}) → PAS d'émission")
        return
    slot_order = [by_out["energy"], by_out["thirst"], by_out["damage"]]
    C_slot = A["C"][slot_order]
    thr_slot = A["thr"][slot_order]
    berry_j = by_out["energy"]

    # (c) transfert baie-seule (rétines échantillonnées de G1)
    med, p90, ntr = transfer_error(C_slot, thr_slot, bush["retinas"], C, berry_j, neutral)
    print(f"[credit] (c) transfert baie-seule : erreur méd={med:.3f} m p90={p90:.3f} m (n={ntr})")

    g_bind = have
    g_neutral = len(neutral) >= 1
    g_transfer = ntr > 0 and med <= 0.5
    print(f"\n[credit] === GATES G2 ===")
    print(f"[credit] (a) non-régression baie→énergie, eau→soif, danger→dégâts : {'✅' if g_bind else '❌'}")
    print(f"[credit] (b) buisson NEUTRE (≥1 cluster neutre) : {'✅' if g_neutral else '❌'} (neutres {neutral})")
    print(f"[credit] (c) transfert baie-seule méd ≤ 0.5 m : {'✅' if g_transfer else '❌'}")
    if not (g_bind and g_neutral and g_transfer):
        print("[credit] ❌ GATE ÉCHOUÉ → PAS d'émission (négatif à commiter, diagnostiquer sur trace)")
        return

    payload = torch.load(SRC_WM, map_location="cpu", weights_only=False)
    state = dict(payload["model"])
    old_q = state["slot_encoder.color_queries"]
    new_q = torch.tensor(C_slot, dtype=torch.float32)
    state["slot_encoder.color_queries"] = new_q
    meta = dict(payload["meta"])
    meta.update({"query_thr": [float(v) for v in thr_slot],
                 "queries": "credit_typed_bush_neutral_G2",
                 "queries_cos_to_hand": [round(float(new_q[i] @ old_q[i]), 4) for i in range(3)],
                 "bind_table": B["table"][slot_order].tolist(),
                 "neutral_clusters": [A["C"][j].tolist() for j in neutral],
                 "bush_run": str(args.bush), "pool_runs": list(args.pool)})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state, "meta": meta}, out / "wm_best.pt")
    print(f"\n[credit] ✅✅ GATES PASSÉS → WM CRÉDIT-TYPÉ émis : {out / 'wm_best.pt'}")
    print(f"[credit]   marges {meta['query_thr']} (resserrées avec le buisson) ; "
          f"{len(neutral)} cluster(s) NEUTRE(s) = buisson (pas de slot).")


if __name__ == "__main__":
    main()
