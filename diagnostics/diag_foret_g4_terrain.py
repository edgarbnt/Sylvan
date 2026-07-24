"""G4 GRATUIT — LE TERRAIN QUI RALENTIT : le déplacement dépend-il enfin de OÙ l'on est ?

PÉRIMÈTRE. Aucune collecte de corpus retenue, aucun entraînement. On lance Godot en babillage,
on LIT les positions écrites, on jette le run. Coût : quelques minutes.

POURQUOI CETTE BRIQUE (design_foret_complete.md §2.3, item 4 du §5). L'audit JEPA a mesuré que le
déplacement prédit par le WM est reconstructible à **R² 0,985 depuis la COMMANDE SEULE** : son
prédicteur n'a jamais rien appris d'autre que sa propre cinématique, parce que le corps était
parfaitement obéissant. Le terrain qui ralentit est le fix DIRECT : dans un sous-bois dense on
avance moins, donc le déplacement dépend enfin de la POSITION, pas seulement de la commande.

CE QUE MESURE LA SONDE, ET POURQUOI C'EST DÉCISIF. Pour chaque décision, le corpus WM écrit la
commande (vx, omega) et les positions du torse AVANT et APRÈS (wm.cmd / wm.torso0 / wm.torso1). On
ajuste le déplacement réalisé par une régression sur la commande — disp ~ vx + |omega| — et on
regarde le R². C'est EXACTEMENT la grandeur de l'audit :
  * si le déplacement ne dépend que de la commande, R² ≈ 1 (le monde plat, la cinématique pure) ;
  * si la POSITION se met à compter (sous-bois), la même commande produit des déplacements
    différents selon l'endroit → la commande n'explique plus tout → R² CHUTE.
Un R² qui chute est donc la preuve directe que le fix vise juste — mesurée, pas déduite.

CE QU'ON LIT EXACTEMENT. Le déplacement PAR FENÊTRE reconstruit depuis les positions du snapshot est
biaisé (les positions du torse LAGGENT derrière le glissement cinématique — mesuré : disp constant
~0,003 m quel que soit vx, aberrant). On lit donc la grandeur EXACTE et sans lag : la VITESSE
réalisée, stockée dans la proprioception (torso.linear_velocity, dims 1-3), qui vaut par
construction vitesse_effective x vx. C'est la même chose au facteur durée près, et c'est fiable.

⚠️ CE QUE LA VITESSE ISOLE, ET POURQUOI C'EST UN AVANTAGE. La vitesse stockée est la vitesse
VOULUE (commande x terrain) : une COLLISION plafonne le déplacement mais ne touche pas la vitesse
stockée. La mesure isole donc le TERRAIN de la collision — exactement ce qu'on veut. On garde trois
conditions à babillage identique :
  A — ARÈNE PLATE (aucun arbre)            → R² de référence, la cinématique pure : doit valoir ~1
  B — FORÊT, terrain OFF                    → doit RESTER ~1 (les collisions n'entament pas la vitesse)
  C — FORÊT, terrain ON                     → R² CHUTE si le sous-bois rend la vitesse position-dépendante
Le fix est validé si C < A d'une marge claire (le terrain fait chuter le R²), si B reste proche de A
(preuve que la vitesse isole bien la collision), ET si le [terrain] log prouve un ralentissement
réellement servi (mesuré, pas demandé).

CE QU'ELLE NE PEUT PAS DIRE : que l'entité NAVIGUE mieux, ni que la survie tienne — cela se mesure
en vies après collecte et retrain (§6quinquies E : le terrain change la dynamique que le WM doit
apprendre). Ici on prouve seulement que la dynamique a bien changé, et de la bonne façon.

CLI :
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_foret_g4_terrain.py
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_foret_g4_terrain.py --selfcheck
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from sylvan.world import BOSQUETS_V2  # noqa: E402

GODOT = os.path.join(ROOT, "tools", "godot", "godot")

TREES = 45
STANDS = 6
CLEARINGS = 3
TERRAIN_STRENGTH = 0.6      # pente du ralentissement par arbre proche
TERRAIN_RADIUS = 2.5
TERRAIN_FLOOR = 0.25

R2_DROP_MIN = 0.05          # C doit être sous B d'au moins ça (le terrain ajoute au-delà des collisions)
SLOWED_FRAC_MIN = 10.0      # % de ticks réellement ralentis pour que la mesure ait du sens

# `[terrain] episode : facteur de vitesse moyen MESURE 0.812 | ralenti 34.0% des ticks (sur 900)`
RE_TERRAIN = re.compile(
    r"\[terrain\] episode : facteur de vitesse moyen MESURE ([\d.]+) \| ralenti ([\d.]+)% des ticks")


def _run(label: str, trees: int, terrain: float, episodes: int, steps: int, seed: int) -> tuple[str, str]:
    """Lance Godot en babillage, renvoie (chemin du jsonl, stdout). NE supprime PAS le run : on le lit."""
    run_dir = f"/tmp/foret_g4_{label}"
    os.system(f"rm -rf {run_dir}")
    e = dict(os.environ)
    e.update(BOSQUETS_V2.to_env())
    e.update({
        "SYLVAN_COLLECT": "1", "SYLVAN_WM_COLLECT": "1", "SYLVAN_COLLECTOR_MODE": "babbling",
        "SYLVAN_CPG": "1", "SYLVAN_RESIDUAL_GAIN": "0.0", "SYLVAN_TURN_FADE": "0",
        # Babillage NATUREL : on lit la VITESSE (proprio), dont la norme vaut eff_speed x vx quel que
        # soit le cap, donc le virage ne confond RIEN. Les lignes droites comme les virages traversent
        # des zones de densité variées — c'est ce contraste de position qu'on veut échantillonner.
        "SYLVAN_WM_VX_MIN": "0.55", "SYLVAN_WM_VX_MAX": "0.75", "SYLVAN_WM_WMAX": "0.6",
        "SYLVAN_POLICY_EXPLORATION_STD_INITIAL": "0", "SYLVAN_POLICY_EXPLORATION_STD_FINAL": "0",
        "SYLVAN_REFLEX_STRENGTH": "0", "SYLVAN_ASSIST_RATIO": "0",
        "SYLVAN_NUM_EPISODES": str(episodes), "SYLVAN_MAX_EPISODE_STEPS": str(steps),
        "SYLVAN_SEED": str(seed), "SYLVAN_RUN_DIR": run_dir,
        "SYLVAN_DISABLE_HOMEOSTASIS": "1",
        "SYLVAN_FOREST_COUNT": str(trees),
        "SYLVAN_FOREST_STANDS": str(STANDS if trees else 0),
        "SYLVAN_FOREST_CLEARINGS": str(CLEARINGS if trees else 0),
    })
    if terrain > 0.0:
        e["SYLVAN_TERRAIN_SLOW"] = str(terrain)
        e["SYLVAN_TERRAIN_RADIUS"] = str(TERRAIN_RADIUS)
        e["SYLVAN_TERRAIN_FLOOR"] = str(TERRAIN_FLOOR)
    p = subprocess.run([GODOT, "--path", os.path.join(ROOT, "godot"), "--headless"],
                       env=e, capture_output=True, text=True, timeout=600)
    out = p.stdout + p.stderr
    for fatal in ("Parse Error", "Failed to load script"):
        if fatal in out:
            first = next((ln for ln in out.splitlines() if fatal in ln), fatal)
            raise SystemExit(f"[{label}] Godot n'a PAS chargé le script — mesure invalide.\n  {first}")
    files = sorted(glob.glob(os.path.join(run_dir, "*.jsonl")))
    if not files:
        raise SystemExit(f"[{label}] aucun jsonl écrit dans {run_dir} — la collecte n'a rien produit")
    return files[0], out


def _r2_command_to_speed(jsonl: str) -> tuple[float, int, float]:
    """R² d'un modèle vitesse_réalisée ~ vx, ajusté sur les décisions MOBILES.

    La vitesse réalisée = norme horizontale de la vitesse du torse (proprio dims 1 et 3), qui vaut
    par construction vitesse_effective x vx. En arène plate elle est parfaitement linéaire en vx
    (R²≈1) ; le sous-bois la rend position-dépendante (même vx, vitesses différentes) → R² chute.
    On ne dépend PAS de l'orientation : |vitesse| = eff_speed x vx quel que soit le cap, donc pas
    besoin de forcer la ligne droite. On écarte seulement les fenêtres immobiles (phase de reset).
    """
    xs, ys = [], []
    with open(jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            obs = json.loads(line).get("obs", {})
            wm = json.loads(line).get("wm", {})
            proprio = obs.get("proprio", [])
            cmd = wm.get("cmd", [0.0, 0.0])
            if len(proprio) < 4 or cmd[0] <= 0.0:
                continue
            speed = math.hypot(proprio[1], proprio[3])   # vitesse horizontale RÉALISÉE (sans lag)
            if speed <= 0.001:                           # fenêtre immobile (reset) — hors sujet
                continue
            xs.append(cmd[0])
            ys.append(speed)
    if len(ys) < 20:
        raise SystemExit(f"{jsonl} : {len(ys)} décisions mobiles — trop peu pour un R² fiable")
    x = np.array(xs)
    y = np.array(ys)
    a = np.column_stack([x, np.ones(len(x))])          # [vx, 1]
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    resid = y - a @ coef
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return r2, len(ys), float(y.mean())


def _terrain_measured(out: str) -> tuple[float, float]:
    """(facteur de vitesse moyen, % de ticks ralentis) rapportés par le log [terrain]."""
    m = RE_TERRAIN.findall(out)
    if not m:
        return 1.0, 0.0
    return float(m[0][0]), float(m[0][1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()

    print(f"MONDE : {BOSQUETS_V2.name} | forêt {TREES} arbres | terrain pente {TERRAIN_STRENGTH} "
          f"rayon {TERRAIN_RADIUS} m plancher {TERRAIN_FLOOR} | {a.steps} ticks | graine {a.seed}")
    conds = [
        ("A_plat", 0, 0.0, "ARÈNE PLATE (référence audit)"),
        ("B_foret", TREES, 0.0, "FORÊT, terrain OFF (collisions seules)"),
        ("C_terrain", TREES, TERRAIN_STRENGTH, "FORÊT, terrain ON (collisions + sous-bois)"),
    ]
    res = {}
    for label, trees, terr, desc in conds:
        jsonl, out = _run(label, trees, terr, 1, a.steps, a.seed)
        r2, n, spd = _r2_command_to_speed(jsonl)
        fac, slowed = _terrain_measured(out)
        res[label] = {"r2": r2, "n": n, "spd": spd, "fac": fac, "slowed": slowed}
        os.system(f"rm -rf /tmp/foret_g4_{label}")
        extra = f" | terrain: facteur {fac:.3f}, ralenti {slowed:.1f}% des ticks" if terr > 0 else ""
        print(f"\n  {desc}")
        print(f"    R²(commande→vitesse) = {r2:.3f}  ({n} décisions, vitesse moyenne {spd:.3f}){extra}")

    a_, b_, c_ = res["A_plat"], res["B_foret"], res["C_terrain"]
    drop_terrain = a_["r2"] - c_["r2"]
    drift_collision = a_["r2"] - b_["r2"]
    print("\n=== LECTURE ===")
    print(f"  arène plate     R² {a_['r2']:.3f}  (cinématique pure — doit valoir ~1)")
    print(f"  + collisions    R² {b_['r2']:.3f}  (doit RESTER ~1 : la vitesse ignore la collision, écart {drift_collision:+.3f})")
    print(f"  + sous-bois     R² {c_['r2']:.3f}  (chute due au TERRAIN : {-drop_terrain:+.3f})")

    fails = []
    if a_["r2"] < 0.95:
        fails.append(f"la référence plate R²={a_['r2']:.3f} n'approche pas 1 — la méthode elle-même "
                     "ne reproduit pas la cinématique pure, mesure suspecte")
    if abs(drift_collision) > 0.05:
        fails.append(f"la forêt SANS terrain fait déjà bouger le R² de {drift_collision:+.3f} — la "
                     "vitesse ne devrait PAS voir la collision ; confondant non contrôlé")
    if drop_terrain < R2_DROP_MIN:
        fails.append(f"le terrain ne fait chuter le R² que de {drop_terrain:.3f} (< {R2_DROP_MIN}) — "
                     "la vitesse reste prédictible depuis la commande, le fix ne mord pas")
    if c_["slowed"] < SLOWED_FRAC_MIN:
        fails.append(f"le terrain n'a ralenti que {c_['slowed']:.1f}% des ticks (< {SLOWED_FRAC_MIN}) — "
                     "le sous-bois n'est presque jamais rencontré, augmenter densité/rayon")
    if c_["fac"] <= TERRAIN_FLOOR + 0.01:
        fails.append(f"facteur moyen {c_['fac']:.3f} au plancher — TOUTE l'arène est lente, ce n'est "
                     "plus un terrain contrasté mais un corps uniformément ralenti")

    print("\n=== VERDICT ===")
    if fails:
        for f in fails:
            print(f"  ✗ {f}")
        print("  G4 TERRAIN = ÉCHEC")
        return 1
    print(f"  G4 TERRAIN = PASS — la commande explique {c_['r2']*100:.0f}% de la vitesse en sous-bois "
          f"contre {a_['r2']*100:.0f}% en plat : la POSITION compte désormais.")
    print("  ⚠️ NON MESURÉ ICI : que l'entité navigue mieux ou survive — cela se mesure en vies,")
    print("     après collecte et retrain (le terrain change la dynamique que le WM doit apprendre).")
    return 0


def _fake(cmd_vx: float, speed: float) -> dict:
    """Une transition minimale : proprio dims 1..3 = vitesse du torse, wm.cmd = commande."""
    proprio = [0.0] * 4
    proprio[1] = speed        # vitesse horizontale portée par le seul axe x, hypot = speed
    return {"obs": {"proprio": proprio}, "wm": {"cmd": [cmd_vx, 0.0]}}


def selfcheck() -> int:
    import tempfile
    # R² : une vitesse parfaitement linéaire en vx doit donner ~1 (le terrain OFF).
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf:
        for i in range(200):
            vx = 0.55 + 0.2 * (i % 3) / 2.0
            tf.write(json.dumps(_fake(vx, 0.8 * vx)) + "\n")   # vitesse = kin_speed x vx, cinématique pure
        clean = tf.name
    r2, n, _ = _r2_command_to_speed(clean)
    os.unlink(clean)
    assert r2 > 0.999 and n == 200, (r2, n)
    print(f"  [ok] R²=1 sur une vitesse exactement proportionnelle à la commande ({n} décisions)")

    # Vitesse rendue INDÉPENDANTE de la commande (facteur terrain aléatoire) → R²≈0.
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf:
        rng = np.random.default_rng(0)
        for _ in range(200):
            tf.write(json.dumps(_fake(0.65, float(rng.uniform(0.1, 0.5)))) + "\n")
        noisy = tf.name
    r2n, _, _ = _r2_command_to_speed(noisy)
    os.unlink(noisy)
    assert r2n < 0.2, r2n
    print(f"  [ok] R²≈0 ({r2n:.3f}) quand la vitesse ne dépend PAS de la commande — le test discrimine")

    line = "[terrain] episode : facteur de vitesse moyen MESURE 0.812 | ralenti 34.0% des ticks (sur 900)"
    fac, slowed = _terrain_measured(line)
    assert abs(fac - 0.812) < 1e-9 and abs(slowed - 34.0) < 1e-9, (fac, slowed)
    print("  [ok] le parseur lit la ligne [terrain] émise par main.gd")

    assert os.path.exists(GODOT), GODOT
    print("  [ok] binaire Godot présent")
    print("SELFCHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
