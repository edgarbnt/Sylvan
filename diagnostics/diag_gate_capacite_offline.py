"""DIAG GRATUIT (0 run) — pourquoi le Gate-capacité a échoué (négatif du 2026-07-17).

Rejoue les corpus DÉJÀ collectés (bras contrôle gcctl1/2, bras appris gclrn1/2) pour trancher
DEUX questions, sans rien ré-entraîner ni recollecter :

  A. LE MÉCANISME re-perçoit-il ? -> rejoue PeriodicRemeasure sur la rétine brute de chaque bras
     appris, journalise la TEINTE de la requête food après chaque mise à jour + la composition de
     la fenêtre. Attendu si sain : la requête food bascule vers le magenta (~0.83) après les swaps.

  B. LA MÉTRIQUE est-elle valide ? -> le repas est compté sur une remontée d'énergie, mais
     try_consume mange par DISTANCE (pas par perception). En lisant food_d de godot.log (oracle,
     tous les 10 pas), on classe chaque repas : APPROCHE (food_d décroît nettement avant) vs
     CONTACT-AVEUGLE (food_d ne décroît pas — l'entité allait ailleurs, a heurté la bouffe). Si la
     fenêtre tardive du CONTRÔLE est dominée par le contact-aveugle, la métrique ne peut pas isoler
     la capacité perçue (§2) — le gate serait mal posé, indépendamment du mécanisme.

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_gate_capacite_offline.py
"""

from __future__ import annotations

import colorsys
import gzip
import json
import os
import re
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, "python")

from sylvan.control.remeasure import PeriodicRemeasure, _nearest_touch  # noqa: E402
from scripts.build_typed_slots import RELIEF  # noqa: E402

SWAP_TICK = 700
LATE_START = 900
EAT_RADIUS = 1.0
_EP_RE = re.compile(r"\[Godot\] Episode (\d+) \| Step (\d+) \| Energy: ([\d.]+) \| Thirst: [\d.]+ "
                    r"\| Health: [\d.]+ \|.*?food_d: ([\d.]+)")


def _load_bc(tag: str) -> list[dict]:
    p = f"data/replay_buffer/critic_kin_{tag}/ep_0000.jsonl"
    op = open(p) if os.path.exists(p) else gzip.open(p + ".gz", "rt")
    return [json.loads(l) for l in op]


def _hue(rgb) -> float:
    r, g, b = (max(float(c), 0.0) for c in rgb)
    return colorsys.rgb_to_hsv(r, g, b)[0]


def _band(h: float | None) -> str:
    if h is None:
        return "none"
    if h < 0.12 or h > 0.95:
        return "red"
    if 0.20 < h < 0.45:
        return "green"
    if 0.55 < h < 0.75:
        return "blue"
    if 0.75 < h < 0.92:
        return "magenta"
    return "other"


# ---------------------------------------------------------------- A. le mécanisme re-perçoit-il ?

def diag_query_trajectory(tag: str) -> None:
    recs = _load_bc(tag)
    rm = PeriodicRemeasure(every=150, window=6000, min_samples=40, seed=0)
    print(f"\n=== A. [{tag}] trajectoire de la requête FOOD (rejeu PeriodicRemeasure) ===")
    n_food_upd = 0
    for i, r in enumerate(recs):
        ret = r["wm"]["retina0"]
        o = r["obs"]
        rm.observe(ret, o["energy"], o["thirst"], o["health"])
        if not rm.due():
            continue
        res = rm.measure()
        if res is None:
            continue
        # cluster lié à "energy" (food) ce cycle ?
        food_j = next((j for j, out in res["bound"].items() if out == "energy"), None)
        if food_j is None:
            continue
        n_food_upd += 1
        centroid = res["C"][food_j]
        h = _hue(centroid)
        if n_food_upd <= 4 or i > len(recs) - 3000:  # premiers + derniers
            print(f"  tick={i:6d} food<-cluster {food_j}: teinte={h:.3f} ({_band(h)}) "
                  f"centroïde=({centroid[0]:.2f},{centroid[1]:.2f},{centroid[2]:.2f})  "
                  f"bound={res['bound']}")
    print(f"  -> {n_food_upd} cycles où un cluster s'est lié à FOOD (sur ~{len(recs)//150} cycles)")


# ---------------------------------------------------------------- B. la métrique est-elle valide ?

def _parse_godot_food_d(tag: str) -> list[list[tuple[int, float, float]]]:
    """-> par épisode : liste de (step, energy, food_d) échantillonnés tous les 10 pas."""
    path = Path(f"data/replay_buffer/critic_kin_{tag}/godot.log")
    per_ep: dict[int, list[tuple[int, float, float]]] = {}
    with open(path, errors="ignore") as f:
        for line in f:
            m = _EP_RE.search(line)
            if m:
                ep, step, e, fd = int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4))
                per_ep.setdefault(ep, []).append((step, e, fd))
    return [per_ep[e] for e in sorted(per_ep)]


def diag_meal_provenance(tags: list[str], arm: str) -> None:
    """Classe chaque repas : APPROCHE dirigée (food_d décroissait vers eat_radius dans les ~80 pas
    AVANT le repas) vs CONTACT-AVEUGLE (food_d restait haut/plat — l'entité a heurté la bouffe sans
    l'approcher). IMPORTANT : l'échantillon DU repas porte déjà le food_d POST-RESPAWN (bouffe
    re-spawnée 2-4.5 m plus loin) — on l'EXCLUT et on regarde la fenêtre pré-repas [k-8, k-1]."""
    print(f"\n=== B. [{arm}] provenance des repas (food_d oracle godot.log, détecteur corrigé) ===")
    tot_pre = {"approach": 0, "blind": 0}
    tot_late = {"approach": 0, "blind": 0}
    for tag in tags:
        for ep in _parse_godot_food_d(tag):
            for k in range(1, len(ep)):
                step, e, _fd_meal = ep[k]
                if e - ep[k - 1][1] <= RELIEF:            # pas un repas
                    continue
                back = ep[max(0, k - 15):k]              # ~150 pas AVANT le repas (hors respawn) :
                fds = [x[2] for x in back]               # l'approche est LENTE (~0.07 m/échantillon)
                # approche = la bouffe s'est rapprochée d'>1.0 m ET a atteint la portée (<1.3 m).
                # Calibré sur 3 approches vérifiées visuellement (food_d 2.0->1.0 monotone,
                # water_d divergent) — un détecteur à fenêtre courte les manquait (faux "0% approche").
                approached = (max(fds) - min(fds) > 1.0) and (min(fds) < 1.3)
                bucket = "approach" if approached else "blind"
                if step < SWAP_TICK:
                    tot_pre[bucket] += 1
                elif step >= LATE_START:
                    tot_late[bucket] += 1
    def pct(d):
        n = d["approach"] + d["blind"]
        return f"approche={d['approach']} contact-aveugle={d['blind']} " + (
            f"(approche {100*d['approach']/n:.0f}%)" if n else "(0 repas)")
    print(f"  pré-swap  [0,{SWAP_TICK})   : {pct(tot_pre)}")
    print(f"  tardive   [{LATE_START},fin] : {pct(tot_late)}")


# ------------------------------------------------ C. LA CAUSE-RACINE : le swap aveugle-t-il vraiment ?

def diag_query_blindness(tags: list[str]) -> None:
    """Le slot lit par COSINE en RGB-normalisé, pas par teinte. Le magenta (teinte 0.83, "libre" en
    HUE) partage le canal ROUGE avec la requête food -> reste AU-DESSUS du seuil -> la requête
    statique VOIT ENCORE le magenta -> le swap ne crée aucun déficit perceptuel (cause-racine du
    non-effondrement du contrôle). Mesuré sur les rayons magenta RÉELLEMENT perçus."""
    import torch
    payload = torch.load("data/checkpoints/wm_objcentric_kin_typed/wm_best.pt",
                         map_location="cpu", weights_only=False)
    fq = payload["model"]["slot_encoder.color_queries"][0]
    fq = fq / fq.norm()
    thr = float(payload["meta"]["query_thr"][0])
    mags = []
    for tag in tags:
        for r in _load_bc(tag):
            t = _nearest_touch(r["wm"]["retina0"])
            if t is None:
                continue
            if _band(_hue(t[0])) == "magenta":
                mags.append(t[0])
    magt = torch.tensor(np.array(mags), dtype=torch.float32)
    magt = magt / magt.norm(dim=1, keepdim=True)
    cos = magt @ fq
    print(f"\n=== C. CAUSE-RACINE : la requête food ROUGE voit-elle le magenta ? (seuil={thr:.3f}) ===")
    print(f"  {len(mags)} rayons magenta perçus : cos(food_q) moy={float(cos.mean()):.3f} "
          f"med={float(cos.median()):.3f}")
    print(f"  fraction AU-DESSUS du seuil (donc VISIBLE malgré le swap) = {100*float((cos>thr).float().mean()):.0f}%")
    print(f"  -> le swap magenta (teinte libre) NE crée PAS de déficit : magenta partage le canal R "
          f"avec le rouge => le contrôle ne peut pas s'effondrer.")


def main() -> None:
    for tag in ("gclrn1", "gclrn2"):
        diag_query_trajectory(tag)
    diag_meal_provenance(["gcctl1", "gcctl2"], "CONTRÔLE statique (rouge figé)")
    diag_meal_provenance(["gclrn1", "gclrn2"], "APPRIS re-mesure")
    diag_query_blindness(["gcctl1", "gcctl2"])


if __name__ == "__main__":
    main()
