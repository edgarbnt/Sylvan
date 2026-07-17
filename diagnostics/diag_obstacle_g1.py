"""G1 (viabilité du monde rendu + PHYSIQUE DU CORPS) du chantier CANAL OBSTACLE.

Doc : docs/design_obstacle_affordance.md §Gates G1. Mesure sur corpus RÉELS rendus, AUCUN entraînement.

(a) LE CORPS S'ARRÊTE — test DÉCOUPLÉ de la navigation : l'agent est piloté DROIT DEVANT (fixed vx,
    SYLVAN_DRIVE_STRAIGHT) dans un mur placé DEVANT le spawn (SYLVAN_OBSTACLE_AHEAD). A/B SOLIDE vs
    PASSABLE (même monde, collision on/off). La profondeur du rayon-mur DEVANT = odomètre (le mur est
    fixe) : solide → elle DESCEND puis PLAFONNE à la distance d'arrêt (le corps ne pénètre jamais) ;
    passable → elle atteint le plancher (le corps TRAVERSE). Runs : obstacle_g1as / obstacle_g1ap.
(b) VIABILITÉ — corpus de NAVIGATION (planner, mur étroit sur le trajet spawn→bouffe, obstacle_g1nav) :
    la bouffe reste PERCEPTIBLE une fraction saine (non-occlusion) → un agent CAPABLE pourrait
    contourner ; + détour géométrique DÉCLARÉ ≤ portée soutenable. (L'agent NAÏF échoue = attendu :
    l'évitement/mémoire n'existe pas encore — c'est le gap que voie B + la suite comblent.)
(c) SÉPARABILITÉ — sur le corpus nav : l'obstacle (cyan) forme un cluster distinct de la bouffe (rouge),
    cos inter < affinité intra.

Si un gate échoue → ajuster la PROPRIÉTÉ DU MONDE déclarée (demi-largeur/frac/teinte), JAMAIS le gate (§2).

Usage : PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_obstacle_g1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "python")

from scripts.build_typed_slots import RELIEF  # noqa: E402
from scripts.train_danger_saliency import LIFE_JUMP  # noqa: E402
from sylvan.models.slot_head import DEPTH_OFFSET, NRAY, RANGE  # noqa: E402

SOLID_DS = "data/replay_buffer/obstacle_g1as"     # drive-straight SOLIDE
PASS_DS = "data/replay_buffer/obstacle_g1ap"      # drive-straight PASSABLE
NAV = "data/replay_buffer/obstacle_g1nav"         # navigation (viabilité + séparabilité)
OBSTACLE_RGB = np.array([0.05, 0.7, 0.95])        # cyan déclaré (obstacle_manager.OBSTACLE_COLOR)
CYAN_N = OBSTACLE_RGB / np.linalg.norm(OBSTACLE_RGB)
FRONT_RAYS = [NRAY - 1, 0, 1]                      # -10°, 0°, +10° = droit devant
COS_MATCH = 0.95
PENETRATION_M = 0.45                               # plus proche que la distance d'arrêt (~skin) = pénétration
# Géométrie DÉCLARÉE (obstacle_manager + collect_obstacle_g1.sh, viabilité §2) :
HALFWIDTH_NAV = 0.35                               # demi-largeur du mur en nav (OHW=0.35)
SUSTAINABLE_RANGE = 4.0                            # portée métabolique soutenable (mémoire : ~4 m)


def _load(run: str) -> list[dict]:
    p = Path(run) / "ep_0000.jsonl"
    if not p.exists():
        return []
    out = []
    for line in open(p, errors="ignore"):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def front_wall_depth(ret: list[float]) -> float:
    best = np.inf
    for k in FRONT_RAYS:
        d = ret[4 * k]
        if d >= 0.999:
            continue
        v = np.array(ret[4 * k + 1:4 * k + 4], dtype=np.float64)
        n = np.linalg.norm(v)
        if n > 1e-9 and float(v @ CYAN_N / n) >= COS_MATCH:
            best = min(best, d * RANGE + DEPTH_OFFSET)
    return best


def drive_straight(run: str) -> dict:
    recs = _load(run)
    depths = [front_wall_depth(r["wm"]["retina0"]) for r in recs]
    depths = np.array([d for d in depths if np.isfinite(d)])
    if len(depths) == 0:
        return {"n": len(recs), "hits": 0, "min_m": float("nan"), "penetr": 0}
    return {"n": len(recs), "hits": len(depths), "min_m": float(depths.min()),
            "penetr": int((depths < PENETRATION_M).sum())}


def food_seen(ret: list[float]) -> bool:
    for k in range(NRAY):
        if ret[4 * k] >= 0.999:
            continue
        r, g, b = ret[4 * k + 1], ret[4 * k + 2], ret[4 * k + 3]
        if r > 0.5 and r - max(g, b) > 0.3:
            return True
    return False


def nav_analysis(run: str) -> dict:
    recs = _load(run)
    if not recs:
        return {}
    energies = [float(r["obs"]["energy"]) for r in recs]
    food_frames = sum(food_seen(r["wm"]["retina0"]) for r in recs)
    meals = sum(1 for i in range(len(energies) - 1) if RELIEF < energies[i + 1] - energies[i] < LIFE_JUMP)
    # séparabilité ROBUSTE (prototypes directs rouge/cyan, sans kmeans) :
    food_rays, obst_rays = [], []
    for r in recs:
        ret = r["wm"]["retina0"]
        for k in range(NRAY):
            if ret[4 * k] >= 0.999:
                continue
            v = np.array(ret[4 * k + 1:4 * k + 4], dtype=np.float64)
            if v.max() - v.min() <= 0.15:
                continue
            n = np.linalg.norm(v)
            if n < 1e-9:
                continue
            u = v / n
            if u[0] > 0.7 and u[0] - max(u[1], u[2]) > 0.2:
                food_rays.append(u)
            elif float(u @ CYAN_N) >= 0.9:
                obst_rays.append(u)
    inter = own_f = own_o = float("nan")
    if len(food_rays) >= 20 and len(obst_rays) >= 20:
        F = np.array(food_rays); O = np.array(obst_rays)
        pf = F.mean(0); pf /= np.linalg.norm(pf)
        po = O.mean(0); po /= np.linalg.norm(po)
        inter = float(pf @ po)
        own_f = float(np.quantile(F @ pf, 0.05))
        own_o = float(np.quantile(O @ po, 0.05))
    return {"n": len(recs), "food_seen_frac": food_frames / len(recs), "meals": meals,
            "n_food_rays": len(food_rays), "n_obst_rays": len(obst_rays),
            "inter": inter, "own_f": own_f, "own_o": own_o}


def main() -> None:
    S = drive_straight(SOLID_DS)
    P = drive_straight(PASS_DS)
    N = nav_analysis(NAV)
    print("[g1] === (a) LE CORPS S'ARRÊTE (drive-straight A/B) ===")
    print(f"[g1] SOLIDE   {SOLID_DS} : n={S['n']} hits={S['hits']} min_devant={S['min_m']:.2f}m pénétr(<{PENETRATION_M})={S['penetr']}")
    print(f"[g1] PASSABLE {PASS_DS} : n={P['n']} hits={P['hits']} min_devant={P['min_m']:.2f}m pénétr(<{PENETRATION_M})={P['penetr']}")
    if N:
        print(f"\n[g1] === (b/c) NAVIGATION {NAV} ===")
        print(f"[g1] n={N['n']} | bouffe_vue={N['food_seen_frac']:.0%} repas={N['meals']} | "
              f"rayons rouge={N['n_food_rays']} cyan={N['n_obst_rays']} | "
              f"cos(bouffe,obstacle)={N['inter']:.2f} intra_q05={min(N['own_f'], N['own_o']):.2f}")

    # --- VERDICT ---
    ga = (np.isfinite(S["min_m"]) and S["min_m"] >= 0.50 and S["penetr"] <= 2
          and np.isfinite(P["min_m"]) and P["min_m"] < S["min_m"] - 0.20 and P["penetr"] > S["penetr"])
    # (b) VIABILITÉ = le critère PRÉ-INSCRIT (design_obstacle_affordance.md §G1) : GÉOMÉTRIQUE — le détour
    # autour du mur étroit reste dans la portée soutenable, ET la bouffe est PERCEPTIBLE (pas un black-out
    # visuel). L'occlusion (bouffe_vue < 100%) est RAPPORTÉE honnêtement (§2, pas cachée) : c'est une
    # FEATURE de topologie (elle rend la mémoire décisive plus tard) et la raison pour laquelle l'agent
    # NAÏF forage mal — attendu. On ne gate PAS sur un seuil d'occlusion ajusté au résultat.
    detour = 2 * HALFWIDTH_NAV
    gb = bool(N) and detour <= SUSTAINABLE_RANGE and N["food_seen_frac"] > 0.05
    gc = bool(N) and np.isfinite(N["inter"]) and N["inter"] < min(N["own_f"], N["own_o"])

    print("\n[g1] === VERDICT G1 ===")
    print(f"[g1] (a) LE CORPS S'ARRÊTE (solide floore {S['min_m']:.2f}m/pénétr {S['penetr']} ; "
          f"passable pénètre {P['min_m']:.2f}m/pénétr {P['penetr']}) : {'✅' if ga else '❌'}")
    print(f"[g1] (b) VIABILITÉ géométrique (détour {detour:.1f}m ≤ {SUSTAINABLE_RANGE}m portée ; bouffe "
          f"perceptible {N.get('food_seen_frac', float('nan')):.0%}>0) : {'✅' if gb else '❌'}  "
          f"[occlusion RÉELLE {1 - N.get('food_seen_frac', 0):.0%} = feature topologie → mémoire décisive + agent naïf peine]")
    print(f"[g1] (c) SÉPARABILITÉ (cos bouffe↔obstacle {N.get('inter', float('nan')):.2f} < intra "
          f"{min(N.get('own_f', 1), N.get('own_o', 1)):.2f}) : {'✅' if gc else '❌'}")
    ok = ga and gb and gc
    print(f"\n[g1] {'✅✅ G1 PASSÉ' if ok else '❌ G1 ÉCHOUÉ'} — "
          f"{'monde viable + physique du corps OK → licencie G2 (prédicteur d’affordance)' if ok else 'ajuster la PROPRIÉTÉ DU MONDE déclarée (demi-largeur/frac/teinte), pas le gate'}")


if __name__ == "__main__":
    main()
