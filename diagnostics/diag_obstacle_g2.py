"""G2-0 (GRATUIT, décisif) du chantier CANAL OBSTACLE — le label est-il PROPRE et APPRENABLE ?

Doc : docs/design_obstacle_affordance.md §Gates G2. Zéro entraînement. GATE l'entraînement du
prédicteur d'affordance (voie B). Sur les corpus RÉELS (obstacle ON, torso loggé), on :
 1. bootstrappe le label COMMANDÉ-vs-RÉEL (auto-supervisé, appearance-agnostic) :
    « bloqué » = commandé en avant (vx>seuil) MAIS déplacement réalisé ≈ 0, APRÈS avoir bougé
    (exclut le délai de reset : l'agent stationnaire au spawn n'a jamais bougé) ;
 2. vérifie que la rétine PORTE le signal — AUC de « profondeur du rayon le plus proche DEVANT »
    (géométrie SEULE, aucune couleur) prédisant bloqué. AUC haute → un prédicteur MIL l'apprendra ;
 3. confirme (sans l'utiliser pour le label) que les ticks bloqués ont bien l'OBSTACLE devant
    (couleur cyan) → le prédicteur découvrira « cyan devant ⇒ bloque » depuis le label MOTEUR.

Verdict : label propre (assez de bloqués ET de libres) ET AUC ≥ 0.90 → licencie l'entraînement.

Usage : PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_obstacle_g2.py
"""

from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "python")

from sylvan.models.slot_head import DEPTH_OFFSET, NRAY, RANGE  # noqa: E402

RUNS = ["data/replay_buffer/obstacle_g2ds", "data/replay_buffer/obstacle_g2nav"]
OBSTACLE_RGB = np.array([0.05, 0.7, 0.95]); CYAN_N = OBSTACLE_RGB / np.linalg.norm(OBSTACLE_RGB)
FRONT_RAYS = [NRAY - 1, 0, 1]           # -10°, 0°, +10° droit devant
VX_MIN = 0.30                           # on ne juge que les ticks commandés « avance »
STEP_BLOCKED = 0.0015                   # déplacement réalisé/tick sous ce seuil = arrêté (m)
STEP_FREE = 0.0030                       # au-dessus = libre
MOVED_MIN = 0.20                         # cumul déplacé dans le segment avant de compter un blocage (exclut reset)
TELEPORT = 0.5                           # saut de torse > ça = frontière d'épisode (respawn)


def _load(run: str) -> list[dict]:
    out = []
    for fp in sorted(glob.glob(str(Path(run) / "ep_*.jsonl"))):
        for line in open(fp, errors="ignore"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def front_min_depth(ret: list[float]) -> float:
    """Profondeur (m) du rayon le plus proche DROIT DEVANT, TOUTES couleurs (géométrie seule)."""
    best = np.inf
    for k in FRONT_RAYS:
        d = ret[4 * k]
        if d < 0.999:
            best = min(best, d * RANGE + DEPTH_OFFSET)
    return best


def front_is_obstacle(ret: list[float]) -> bool:
    for k in FRONT_RAYS:
        d = ret[4 * k]
        if d >= 0.999:
            continue
        v = np.array(ret[4 * k + 1:4 * k + 4], dtype=np.float64)
        n = np.linalg.norm(v)
        if n > 1e-9 and float(v @ CYAN_N / n) >= 0.95:
            return True
    return False


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels > 0.5; npos = pos.sum(); nneg = (~pos).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main() -> None:
    ticks = []  # (front_depth, front_obstacle, blocked, free)
    for run in RUNS:
        recs = _load(run)
        if not recs:
            print(f"[g2] (absent) {run}")
            continue
        # segmentation par téléportation du torse (respawn)
        tor = [r["wm"].get("torso0") for r in recs]
        cmd = [r["wm"].get("cmd") for r in recs]
        moved = 0.0
        for i in range(len(recs) - 1):
            a, b = tor[i], tor[i + 1]
            if not a or not b:
                moved = 0.0
                continue
            step = math.hypot(b[0] - a[0], b[1] - a[1])
            if step > TELEPORT:                     # frontière d'épisode → reset du cumul
                moved = 0.0
                continue
            vx = cmd[i][0] if cmd[i] else 0.0
            if vx <= VX_MIN:
                moved += step
                continue
            blocked = step < STEP_BLOCKED and moved > MOVED_MIN
            free = step > STEP_FREE
            if blocked or free:
                ret = recs[i]["wm"]["retina0"]
                fd = front_min_depth(ret)
                ticks.append((fd if np.isfinite(fd) else RANGE, front_is_obstacle(ret),
                              float(blocked), float(free)))
            moved += step
        print(f"[g2] {run} : {len(recs)} ticks lus")

    if not ticks:
        print("[g2] ❌ aucun tick exploitable — corpus/torse absent ?")
        return
    fd = np.array([t[0] for t in ticks]); obst = np.array([t[1] for t in ticks])
    blk = np.array([t[2] for t in ticks]); fre = np.array([t[3] for t in ticks])
    n_blk, n_fre = int(blk.sum()), int(fre.sum())
    print(f"\n[g2] label commandé-vs-réel : BLOQUÉS={n_blk}  LIBRES={n_fre}")
    # (2) AUC : profondeur DEVANT (géométrie seule, sans couleur) prédit bloqué. Score = proximité = -depth.
    mask = (blk > 0.5) | (fre > 0.5)
    auc = _auc(-fd[mask], blk[mask])
    # (3) appearance : fraction des bloqués qui ont l'obstacle (cyan) devant (label NE l'utilise PAS)
    obst_when_blocked = float(obst[blk > 0.5].mean()) if n_blk else float("nan")
    obst_when_free = float(obst[fre > 0.5].mean()) if n_fre else float("nan")
    med_blk = float(np.median(fd[blk > 0.5])) if n_blk else float("nan")
    med_fre = float(np.median(fd[fre > 0.5])) if n_fre else float("nan")

    print(f"[g2] profondeur DEVANT méd : bloqués {med_blk:.2f} m vs libres {med_fre:.2f} m")
    print(f"[g2] AUC(proximité-devant → bloqué) = {auc:.3f}  (géométrie SEULE, appearance-agnostic)")
    print(f"[g2] obstacle (cyan) devant : bloqués {obst_when_blocked:.0%} vs libres {obst_when_free:.0%} "
          f"(le label NE regarde PAS la couleur → le prédicteur découvrira cyan⇒bloque)")

    # --- VERDICT G2-0 ---
    ga = n_blk >= 30 and n_fre >= 100          # label propre : assez des deux classes
    gb = np.isfinite(auc) and auc >= 0.90      # la rétine porte le signal → apprenable
    print("\n[g2] === VERDICT G2-0 (gate l'entraînement) ===")
    print(f"[g2] (a) label propre (≥30 bloqués, ≥100 libres) : {'✅' if ga else '❌'} ({n_blk}/{n_fre})")
    print(f"[g2] (b) rétine porte le signal (AUC ≥ 0.90) : {'✅' if gb else '❌'} ({auc:.3f})")
    ok = ga and gb
    print(f"\n[g2] {'✅✅ G2-0 PASSÉ' if ok else '❌ G2-0 ÉCHOUÉ'} — "
          f"{'label commandé-vs-réel propre + apprenable → licencie train_obstacle_affordance' if ok else 'corriger le label/corpus AVANT d’entraîner'}")


if __name__ == "__main__":
    main()
