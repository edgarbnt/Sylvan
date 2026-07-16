"""SONDE GRATUITE — l'odométrie-par-COMMANDE peut-elle porter le waypoint de l'étage haut ?

CONTEXTE (chantier « petit H-JEPA », docs/recherche_hjepa_waypoint.md §6). L'étage waypoint COMMIT
une cible spatiale pendant ~150-200 pas ; le serveur doit suivre sa position EGO entre deux décisions.
Le corps cinématique OBÉIT exactement à (vx, ω) (sylvan_agent.gd:_kinematic_step : yaw += kin_turn·ω·dt,
pos += fwd·kin_speed·vx·dt) → intégrer la commande ÉMISE = odométrie exacte par construction… si les
constantes effectives (dt effectif par tick serveur) et la CONVENTION DE SIGNE (ω>0 = droite ou gauche
dans le repère ego rétine ?) sont les bonnes. Cette sonde les CALIBRE et les VALIDE sur données réelles.

MÉTHODE (aucun entraînement, corpus existant) : dans data/replay_buffer/critic_kin_a (BC log : retina0
brute + cmd par tick), la BOUFFE est un POINT-MONDE FIXE entre deux repas → son track ego observé
(rayon rouge le plus proche, label-free) = vérité-terrain de l'odométrie. On dead-reckonne sa position
ego depuis t0 avec les seules commandes, et on mesure l'écart aux observations fraîches.

  update ego d'un point-monde fixe sous (dfwd, dyaw) : v=(x, z-dfwd) puis rotation du REPÈRE de dyaw
  (dyaw>0 = vire à droite, convention bearing atan2(x_right, z_fwd) du planner) :
      x' = cos(dyaw)·vx − sin(dyaw)·vz ;  z' = sin(dyaw)·vx + cos(dyaw)·vz
  avec dfwd = KIN_SPEED·vx_cmd·dt_eff et dyaw = sign·KIN_TURN·ω_cmd·dt_eff.
  Inconnues calibrées : dt_eff (grille fine) × sign (±1). KIN_SPEED=0.8, KIN_TURN=1.5 (collecte).

CRITÈRES (écrits AVANT) :
  PASS  : erreur médiane < 0.5 m sur fenêtres de 150 pas (rayon d'atteinte waypoint = 0.9 m)
          avec UNE config (dt_eff, sign) stable entre calibration et validation.
  KILL  : > 0.5 m partout → l'odométrie-commande ne peut pas ancrer le waypoint → re-designer
          l'ancrage (ré-ancrage sur cible fraîche + ego-motion) AVANT de construire l'étage.
  Proxy trompeur affiché à côté : l'erreur à 10 pas (toujours petite, même avec de mauvaises
  constantes) — juger sur err@150, pas sur err@10.

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_waypoint_deadreckon.py --selfcheck
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_waypoint_deadreckon.py \
      --corpus data/replay_buffer/critic_kin_a
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import math
import statistics as st

from sylvan.control.mode1.obs import RED, _color_gated_depths

KIN_SPEED = 0.8          # SYLVAN_KIN_SPEED de la collecte critic_kin_* (défaut script)
KIN_TURN = 1.5           # SYLVAN_KIN_TURN idem
RETINA_DIM = 144
N_RAY = 36
RANGE_M = 10.0           # MAX_RANGE (perception.gd)
HORIZONS = (10, 50, 100, 150)
WINDOW = 150
WINDOW_STRIDE = 50
SEG_SKIP = 40            # pas ignorés en début de segment (reset_timer : le corps ne bouge pas encore)


def red_pos(retina: list[float]) -> tuple[float, float] | None:
    """Position ego (x_right, z_fwd) du rayon ROUGE le plus proche — même code que le serveur
    (_retina_food_pos, serve_planner_command.py). None si aucun rayon rouge ne touche."""
    d = _color_gated_depths(retina, RED)
    m = min(d)
    if m >= 0.999:
        return None
    k = d.index(m)
    b = 2.0 * math.pi * k / N_RAY
    return (m * RANGE_M * math.sin(b), m * RANGE_M * math.cos(b))


def step_point(x: float, z: float, vx_cmd: float, om_cmd: float,
               dt_eff: float, sign: float) -> tuple[float, float]:
    """Un tick d'odométrie : où passe un point-monde fixe dans le repère ego après (vx, ω)·dt."""
    dfwd = KIN_SPEED * vx_cmd * dt_eff
    dyaw = sign * KIN_TURN * om_cmd * dt_eff
    vx_, vz_ = x, z - dfwd
    c, s = math.cos(dyaw), math.sin(dyaw)
    return (c * vx_ - s * vz_, s * vx_ + c * vz_)


def load_ticks(corpus: str) -> list[dict]:
    """[{food: (x,z)|None, cmd: (vx, om), energy: float}] dans l'ordre du log."""
    ticks: list[dict] = []
    for path in sorted(globmod.glob(f"{corpus}/ep_*.jsonl")):
        for line in open(path, errors="ignore"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            ret = r.get("wm", {}).get("retina0")
            cmd = r.get("wm", {}).get("cmd")
            if not ret or len(ret) != RETINA_DIM or not cmd:
                continue
            ticks.append({"food": red_pos(ret), "cmd": (float(cmd[0]), float(cmd[1])),
                          "energy": float(r.get("obs", {}).get("energy", 0.0))})
    return ticks


def segments(ticks: list[dict]) -> list[list[int]]:
    """Indices contigus où la bouffe est LE MÊME point-monde : coupe sur remontée d'énergie (>+5 =
    repas → respawn ailleurs ; le reset d'épisode remonte aussi) et sur saut du track observé
    (> 1.5 m entre 2 observations = identité changée / téléport)."""
    segs: list[list[int]] = []
    cur: list[int] = []
    prev_e: float | None = None
    prev_obs: tuple[float, float] | None = None
    for i, t in enumerate(ticks):
        cut = prev_e is not None and t["energy"] > prev_e + 5.0
        if t["food"] is not None and prev_obs is not None and not cut:
            if math.hypot(t["food"][0] - prev_obs[0], t["food"][1] - prev_obs[1]) > 1.5:
                cut = True
        if cut:
            if len(cur) > SEG_SKIP + WINDOW:
                segs.append(cur)
            cur = []
            prev_obs = None
        cur.append(i)
        prev_e = t["energy"]
        if t["food"] is not None:
            prev_obs = t["food"]
    if len(cur) > SEG_SKIP + WINDOW:
        segs.append(cur)
    return segs


def window_errors(ticks: list[dict], segs: list[list[int]], dt_eff: float, sign: float,
                  ) -> dict[int, list[float]]:
    """Pour chaque fenêtre (départ = bouffe visible), dead-reckon depuis t0 et erreur vs
    observation fraîche à chaque horizon (obs la plus proche à ±5 ticks, sinon horizon sauté)."""
    errs: dict[int, list[float]] = {h: [] for h in HORIZONS}
    for seg in segs:
        body = seg[SEG_SKIP:]
        for w0 in range(0, len(body) - WINDOW, WINDOW_STRIDE):
            i0 = body[w0]
            obs0 = ticks[i0]["food"]
            if obs0 is None:
                continue
            x, z = obs0
            preds: dict[int, tuple[float, float]] = {}
            for k in range(1, WINDOW + 1):
                vx_c, om_c = ticks[body[w0 + k - 1]]["cmd"]
                x, z = step_point(x, z, vx_c, om_c, dt_eff, sign)
                if k in HORIZONS:
                    preds[k] = (x, z)
            for h, (px, pz) in preds.items():
                best: float | None = None
                for dk in range(-5, 6):
                    j = w0 + h + dk
                    if 0 <= j < len(body) and ticks[body[j]]["food"] is not None:
                        ox, oz = ticks[body[j]]["food"]
                        e = math.hypot(px - ox, pz - oz)
                        best = e if best is None or e < best else best
                if best is not None:
                    errs[h].append(best)
    return errs


def selfcheck() -> None:
    # Rotation pure à droite (dyaw>0) : un point pile devant doit passer À GAUCHE (x' < 0).
    x, z = step_point(0.0, 2.0, 0.0, 1.0, 1.0, +1.0)   # dyaw = 1.5 rad
    assert x < 0.0 and abs(math.hypot(x, z) - 2.0) < 1e-9, (x, z)
    # Avance pure : le point devant se rapproche, x inchangé.
    x, z = step_point(0.3, 2.0, 1.0, 0.0, 0.1, +1.0)
    assert abs(x - 0.3) < 1e-9 and abs(z - (2.0 - 0.08)) < 1e-9, (x, z)
    # Aller-retour de rotation = identité.
    x, z = step_point(*step_point(1.0, 1.0, 0.0, 1.0, 0.5, +1.0), 0.0, -1.0, 0.5, +1.0)
    assert abs(x - 1.0) < 1e-6 and abs(z - 1.0) < 1e-6, (x, z)
    # Rétine synthétique : rouge pur à ray 9 (=droite, bearing +90°), depth 0.5 → (5, ~0).
    ret = [1.0, 0.0, 0.0, 0.0] * N_RAY
    ret[9 * 4:9 * 4 + 4] = [0.5, 0.9, 0.1, 0.1]
    p = red_pos(ret)
    assert p is not None and abs(p[0] - 5.0) < 1e-6 and abs(p[1]) < 1e-6, p
    print("selfcheck OK (géométrie ego + parsing rétine)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/replay_buffer/critic_kin_a")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return

    ticks = load_ticks(args.corpus)
    segs = segments(ticks)
    n_win = sum(max(0, (len(s) - SEG_SKIP - WINDOW) // WINDOW_STRIDE) for s in segs)
    print(f"corpus={args.corpus} ticks={len(ticks)} segments={len(segs)} fenêtres≈{n_win}")
    if n_win < 20:
        print("KILL(données) : trop peu de fenêtres exploitables — recollecter avant de conclure.")
        return

    # Calibration (moitié des segments) → validation (l'autre moitié). dt fin × signe.
    half = max(1, len(segs) // 2)
    cal, val = segs[:half], segs[half:]
    dt_grid = [round(0.008 + 0.002 * i, 4) for i in range(25)]          # 8 ms → 56 ms
    best_cfg, best_med = None, float("inf")
    for sign in (+1.0, -1.0):
        for dt in dt_grid:
            e = window_errors(ticks, cal, dt, sign)[150]
            if len(e) >= 10 and st.median(e) < best_med:
                best_med, best_cfg = st.median(e), (dt, sign)
    if best_cfg is None:
        print("KILL(données) : aucune fenêtre avec observation à 150 pas.")
        return
    dt, sign = best_cfg
    print(f"calibration : dt_eff={dt:.4f}s (≈{1 / dt:.0f} Hz) sign={sign:+.0f} "
          f"→ err@150 médiane {best_med:.2f} m (moitié calibration)")

    ev = window_errors(ticks, val, dt, sign)
    print("\n  BUT (validation, config calibrée)          | proxy trompeur")
    print(f"  {'horizon':>8} {'n':>5} {'méd (m)':>8} {'p90 (m)':>8} | err@10 toujours flatteuse")
    for h in HORIZONS:
        e = ev[h]
        if e:
            p90 = st.quantiles(e, n=10)[8] if len(e) >= 10 else max(e)
            print(f"  {h:>8} {len(e):>5} {st.median(e):>8.2f} {p90:>8.2f} |")
    med150 = st.median(ev[150]) if ev[150] else float("inf")
    if med150 < 0.5:
        print(f"\nPASS : err@150 médiane {med150:.2f} m < 0.5 → l'odométrie-commande porte le waypoint "
              f"(constantes : dt_eff={dt:.4f}, sign={sign:+.0f}).")
    else:
        print(f"\nKILL : err@150 médiane {med150:.2f} m ≥ 0.5 → NE PAS ancrer le waypoint sur "
              f"l'odométrie-commande ; re-designer l'ancrage avant de construire l'étage.")


if __name__ == "__main__":
    main()
