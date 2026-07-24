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
CTRL_CORR_MAX = 0.30        # |corr(facteur terrain, commande)| maxi : au-delà, l'obéissance est corrompue
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


def _r2(y: np.ndarray, feats: np.ndarray) -> float:
    a = np.column_stack([feats, np.ones(len(feats))])
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    resid = y - a @ coef
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


KIN_SPEED = 0.8   # sylvan_agent : vitesse = kin_speed x terrain_scale x vx (kin_speed=0.8, mesuré)


def _r2_command_to_speed(jsonl: str) -> tuple[float, float, float, float, int, float]:
    """Vitesse réalisée = |vitesse horizontale du torse| (proprio dims 1,3) = kin x terrain x vx.

    Renvoie (R²(vx seul), corr(facteur_terrain, vx), eff_min, eff_max, n, vitesse_moyenne).
      * R²(vx seul) : chute en sous-bois (la position se met à compter) = le but recherché.
      * CONTRÔLABILITÉ (retour de pair) — au lieu de logger terrain_scale (capturé un tick trop tôt,
        donc désaligné), on reconstruit le facteur terrain EFFECTIF = vitesse / (kin_speed x vx),
        aligné avec la vitesse PAR DÉFINITION. Si le corps obéit, ce facteur ne dépend QUE de la
        position, PAS de la commande : corr(facteur, vx) ≈ 0. S'il dépendait de vx, l'obéissance du
        corps serait corrompue (corps cassé). Et son étendue [eff_min, eff_max] prouve que le terrain
        engage vraiment. C'est non-circulaire : « vitesse ∝ vx à position fixe » sans supposer le
        résultat. On ne dépend pas de l'orientation ; on écarte les fenêtres immobiles (reset).
    """
    vxs, effs, ys = [], [], []
    with open(jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            proprio = rec.get("obs", {}).get("proprio", [])
            cmd = rec.get("wm", {}).get("cmd", [0.0, 0.0])
            if len(proprio) < 4 or cmd[0] <= 0.0:
                continue
            speed = math.hypot(proprio[1], proprio[3])   # vitesse horizontale RÉALISÉE (sans lag)
            if speed <= 0.001:                           # fenêtre immobile (reset) — hors sujet
                continue
            vxs.append(cmd[0])
            effs.append(speed / (KIN_SPEED * cmd[0]))    # facteur terrain EFFECTIF, aligné par def.
            ys.append(speed)
    if len(ys) < 20:
        raise SystemExit(f"{jsonl} : {len(ys)} décisions mobiles — trop peu pour un R² fiable")
    y, vx, eff = np.array(ys), np.array(vxs), np.array(effs)
    r2_cmd = _r2(y, vx)
    corr = 0.0 if eff.std() < 1e-9 else float(np.corrcoef(eff, vx)[0, 1])
    return r2_cmd, abs(corr), float(eff.min()), float(eff.max()), len(ys), float(y.mean())


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
        r2, ctrl_corr, eff_lo, eff_hi, n, spd = _r2_command_to_speed(jsonl)
        fac, slowed = _terrain_measured(out)
        res[label] = {"r2": r2, "corr": ctrl_corr, "eff_lo": eff_lo, "eff_hi": eff_hi,
                      "n": n, "spd": spd, "fac": fac, "slowed": slowed}
        os.system(f"rm -rf /tmp/foret_g4_{label}")
        extra = f" | terrain: facteur {fac:.3f}, ralenti {slowed:.1f}% des ticks" if terr > 0 else ""
        print(f"\n  {desc}")
        print(f"    R²(commande→vitesse) = {r2:.3f} | facteur terrain effectif {eff_lo:.2f}..{eff_hi:.2f}, "
              f"corr(facteur,commande) = {ctrl_corr:.3f}  ({n} décisions){extra}")

    a_, b_, c_ = res["A_plat"], res["B_foret"], res["C_terrain"]
    drop_terrain = a_["r2"] - c_["r2"]
    drift_collision = a_["r2"] - b_["r2"]
    print("\n=== LECTURE ===")
    print(f"  arène plate     R²(cmd) {a_['r2']:.3f}  (cinématique pure — doit valoir ~1)")
    print(f"  + collisions    R²(cmd) {b_['r2']:.3f}  (doit RESTER ~1 : la vitesse ignore la collision, écart {drift_collision:+.3f})")
    print(f"  + sous-bois     R²(cmd) {c_['r2']:.3f}  (chute due au TERRAIN : {-drop_terrain:+.3f})")
    print(f"  CONTRÔLABILITÉ  facteur terrain {c_['eff_lo']:.2f}..{c_['eff_hi']:.2f} (engage) et "
          f"INDÉPENDANT de la commande (corr {c_['corr']:.3f} ≈ 0) → le corps OBÉIT, le terrain module.")

    fails = []
    if a_["r2"] < 0.95:
        fails.append(f"la référence plate R²={a_['r2']:.3f} n'approche pas 1 — la méthode elle-même "
                     "ne reproduit pas la cinématique pure, mesure suspecte")
    if abs(drift_collision) > 0.05:
        fails.append(f"la forêt SANS terrain fait déjà bouger le R² de {drift_collision:+.3f} — la "
                     "vitesse ne devrait PAS voir la collision ; confondant non contrôlé")
    if drop_terrain < R2_DROP_MIN:
        fails.append(f"le terrain ne fait chuter le R²(cmd) que de {drop_terrain:.3f} (< {R2_DROP_MIN}) — "
                     "la vitesse reste prédictible depuis la commande, le fix ne mord pas")
    if c_["corr"] > CTRL_CORR_MAX:
        # RETOUR DE PAIR : un corps qui n'obéit plus est aussi inutile qu'un corps trop obéissant.
        fails.append(f"le facteur terrain DÉPEND de la commande (corr {c_['corr']:.3f} > {CTRL_CORR_MAX}) — "
                     "l'obéissance du corps est corrompue, le terrain ne le module pas proprement (corps cassé)")
    if c_["eff_hi"] - c_["eff_lo"] < 0.1:
        fails.append(f"le facteur terrain effectif ne varie que de {c_['eff_hi']-c_['eff_lo']:.2f} — "
                     "le sous-bois n'engage pas (attendu une étendue nette entre plancher et 1)")
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
    print(f"  G4 TERRAIN = PASS — la commande SEULE n'explique plus que {c_['r2']*100:.0f}% de la vitesse "
          f"en sous-bois (contre {a_['r2']*100:.0f}% en plat : la POSITION compte), MAIS le facteur "
          f"terrain est indépendant de la commande (corr {c_['corr']:.3f}) : le corps OBÉIT, le terrain module.")
    print("  ⚠️ NON MESURÉ ICI : que l'entité navigue mieux ou survive — cela se mesure en vies,")
    print("     après collecte et retrain (le terrain change la dynamique que le WM doit apprendre).")
    return 0


def _fake(cmd_vx: float, speed: float, ts: float = 1.0) -> dict:
    """Une transition minimale : proprio dims 1..3 = vitesse du torse, wm.cmd + terrain_scale."""
    proprio = [0.0] * 4
    proprio[1] = speed        # vitesse horizontale portée par le seul axe x, hypot = speed
    return {"obs": {"proprio": proprio}, "wm": {"cmd": [cmd_vx, 0.0], "terrain_scale": ts}}


def selfcheck() -> int:
    import tempfile
    # R² : une vitesse parfaitement linéaire en vx doit donner ~1 (le terrain OFF).
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf:
        for i in range(200):
            vx = 0.55 + 0.2 * (i % 3) / 2.0
            tf.write(json.dumps(_fake(vx, 0.8 * vx)) + "\n")   # vitesse = kin_speed x vx, cinématique pure
        clean = tf.name
    r2, corr, lo, hi, n, _ = _r2_command_to_speed(clean)
    os.unlink(clean)
    assert r2 > 0.999 and n == 200 and abs(hi - 1.0) < 1e-6, (r2, n, hi)
    print(f"  [ok] R²(cmd)=1 sur une vitesse proportionnelle à la commande, facteur terrain 1.0 ({n} décisions)")

    # Terrain fort qui MODULE proprement : vitesse = kin × vx × ts, ts tiré INDÉPENDAMMENT de vx.
    # → R²(cmd) chute (position domine) MAIS facteur terrain ⊥ commande (corr≈0) = corps OBÉIT.
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf:
        rng = np.random.default_rng(0)
        for _ in range(400):
            vx = 0.55 + 0.2 * rng.random()
            ts = float(rng.uniform(0.25, 1.0))
            tf.write(json.dumps(_fake(vx, 0.8 * vx * ts, ts)) + "\n")
        good = tf.name
    r2g, corrg, log, hig, _, _ = _r2_command_to_speed(good)
    os.unlink(good)
    assert r2g < 0.6 and corrg < CTRL_CORR_MAX and (hig - log) > 0.3, (r2g, corrg, log, hig)
    print(f"  [ok] terrain qui module : R²(cmd)={r2g:.3f} (position compte), facteur {log:.2f}..{hig:.2f} "
          f"⊥ commande (corr {corrg:.3f}) — corps OBÉIT (contrôlabilité, retour de pair)")

    # Corps CASSÉ : le facteur terrain DÉPEND de la commande (ts corrélé à vx) → corr élevé, refusé.
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf:
        rng = np.random.default_rng(1)
        for _ in range(400):
            vx = 0.55 + 0.2 * rng.random()
            ts = 0.25 + 0.7 * (vx - 0.55) / 0.2      # facteur LIÉ à vx = obéissance corrompue
            tf.write(json.dumps(_fake(vx, 0.8 * vx * ts, ts)) + "\n")
        broken = tf.name
    _, corrb, _, _, _, _ = _r2_command_to_speed(broken)
    os.unlink(broken)
    assert corrb > CTRL_CORR_MAX, corrb
    print(f"  [ok] corps cassé : facteur terrain corrélé à la commande (corr {corrb:.3f} > {CTRL_CORR_MAX}) "
          "— le test le REFUSE")

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
