"""G8 GRATUIT — APPARENCE VARIABLE DU NON-NOURRITURE : les troncs varient, sans jamais lire « bouffe ».

PÉRIMÈTRE. Aucune collecte retenue, aucun entraînement. Une courte babillage, on LIT, on jette.

POURQUOI (design_foret_complete.md §2.8). « Faire varier TOUT, pas seulement la nourriture. » Un
encodeur entraîné sur des troncs TOUS identiques n'alloue aucune capacité à représenter l'apparence
des troncs — puis on recrée la cécité pour la classe suivante. En variant la teinte PAR ARBRE
(stable dans une vie, re-tirée par épisode → variable entre objets ET entre épisodes, la règle
§2.8), on force l'encodeur à représenter aussi l'apparence du non-nourriture.

🚨 LA GARDE §3 (tronc-brun) EST LE CŒUR DU TEST. Le slot détecte une ressource par le COSINUS de la
couleur (rouge=bouffe, seuil 0,55). Un tronc dont la teinte jitterée dériverait vers le rouge serait
lu comme de la NOURRITURE et corromprait le foraging. La variation doit donc rester HORS des cônes.

CE QUE VÉRIFIE LA SONDE (log [forest] apparence, §6bis + rétine) :
  T1 VARIE ......... l'étendue cos-rouge des arbres (hi - lo) >= 0,10 : l'apparence varie vraiment.
  T2 HORS DU CÔNE .. cos-rouge max des arbres < 0,55 : AUCUN tronc n'est lu comme de la bouffe (§3).
  T3 PERÇU ......... des rayons rétine voient des troncs (hors cônes, colorés) → l'entité les perçoit.
  T4 ENTRE ÉPISODES  les deux épisodes ont des étendues DIFFÉRENTES → la teinte est re-tirée (variable
                     entre épisodes, pas seulement entre arbres) — la règle §2.8.
  T5 DÉFAUT ........ sans SYLVAN_FOREST_APPEARANCE_VAR, aucun log apparence (bit-identique).

CE QUE LA SONDE NE DIT PAS : que l'ENCODEUR apprenne à représenter l'apparence des troncs — gate
post-retrain (comme la palette G5, « séparable » est nécessaire pas suffisant). Elle établit que le
monde PRODUIT la variation, hors des cônes, perceptible.

CLI :
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_foret_g8_apparence.py
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_foret_g8_apparence.py --selfcheck
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

QUERY_RED = np.array([1.0, 0.0, 0.0])
QUERY_BLUE = np.array([0.0, 0.0, 1.0])
SLOT_THRESHOLD = 0.55
APP_VAR = 0.15
SPREAD_MIN = 0.10     # étendue cos-rouge mini pour que « ça varie » soit vrai

# `[forest] apparence : var 0.15 | cos-rouge des arbres MESURE 0.207..0.499 (hi < 0.55 = ...)`
RE_APP = re.compile(r"\[forest\] apparence : var ([\d.]+) \| cos-rouge des arbres MESURE ([\d.]+)\.\.([\d.]+)")


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _run(app_var: float, steps: int, seed: int) -> tuple[str, str]:
    run_dir = "/tmp/foret_g8"
    os.system(f"rm -rf {run_dir}")
    e = dict(os.environ)
    e.update(BOSQUETS_V2.to_env())
    e.update({
        "SYLVAN_FOREST_COUNT": "45", "SYLVAN_FOREST_STANDS": "6", "SYLVAN_FOREST_CLEARINGS": "3",
        "SYLVAN_COLLECT": "1", "SYLVAN_WM_COLLECT": "1", "SYLVAN_COLLECTOR_MODE": "babbling",
        "SYLVAN_CPG": "1", "SYLVAN_RESIDUAL_GAIN": "0.0", "SYLVAN_TURN_FADE": "0",
        "SYLVAN_WM_VX_MIN": "0.55", "SYLVAN_WM_VX_MAX": "0.75", "SYLVAN_WM_WMAX": "0.6",
        "SYLVAN_POLICY_EXPLORATION_STD_INITIAL": "0", "SYLVAN_POLICY_EXPLORATION_STD_FINAL": "0",
        "SYLVAN_REFLEX_STRENGTH": "0", "SYLVAN_ASSIST_RATIO": "0",
        "SYLVAN_NUM_EPISODES": "2", "SYLVAN_MAX_EPISODE_STEPS": str(steps),
        "SYLVAN_SEED": str(seed), "SYLVAN_RUN_DIR": run_dir, "SYLVAN_DISABLE_HOMEOSTASIS": "1",
    })
    if app_var > 0.0:
        e["SYLVAN_FOREST_APPEARANCE_VAR"] = str(app_var)
    else:
        e.pop("SYLVAN_FOREST_APPEARANCE_VAR", None)
    p = subprocess.run([GODOT, "--path", os.path.join(ROOT, "godot"), "--headless"],
                       env=e, capture_output=True, text=True, timeout=600)
    out = p.stdout + p.stderr
    for fatal in ("Parse Error", "Failed to load script"):
        if fatal in out:
            first = next((ln for ln in out.splitlines() if fatal in ln), fatal)
            raise SystemExit(f"[var={app_var}] Godot n'a PAS chargé — mesure invalide.\n  {first}")
    files = sorted(glob.glob(os.path.join(run_dir, "*.jsonl")))
    return (files[0] if files else ""), out


def _tree_rays(jsonl: str) -> tuple[int, float, float]:
    """Rayons rétine voyant un objet HORS des cônes ressource (troncs/non-nourriture) : compte + cos-rouge min/max."""
    n, lo, hi = 0, 1.0, 0.0
    with open(jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            retina = json.loads(line).get("wm", {}).get("retina0", [])
            for k in range(0, len(retina), 4):
                rgb = np.array(retina[k + 1:k + 4])
                if rgb.sum() <= 1e-6:
                    continue
                cr, cb = _cos(rgb, QUERY_RED), _cos(rgb, QUERY_BLUE)
                if cr < SLOT_THRESHOLD and cb < SLOT_THRESHOLD:   # hors cônes = tronc/non-nourriture
                    n += 1
                    lo, hi = min(lo, cr), max(hi, cr)
    return n, lo, hi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()

    print(f"MONDE : {BOSQUETS_V2.name} + 45 arbres, apparence variable (var {APP_VAR})")

    jsonl, out = _run(APP_VAR, a.steps, a.seed)
    eps = RE_APP.findall(out)   # une ligne par épisode
    if not eps:
        raise SystemExit("aucune ligne [forest] apparence — le mécanisme n'a pas tourné (mesure invalide)")
    spreads = [(float(lo), float(hi)) for _, lo, hi in eps]
    lo0, hi0 = spreads[0]
    spread = hi0 - lo0
    print(f"\n  SERVI (log [forest] apparence, §6bis) : {len(eps)} épisodes")
    for i, (lo, hi) in enumerate(spreads):
        print(f"    épisode {i} : cos-rouge des arbres {lo:.3f}..{hi:.3f} (étendue {hi-lo:.3f})")

    ray_n, ray_lo, ray_hi = _tree_rays(jsonl) if jsonl else (0, 1.0, 0.0)
    print(f"  PERÇU (rétine) : {ray_n} rayons voient un tronc (hors cônes) | cos-rouge {ray_lo:.3f}..{ray_hi:.3f}")

    _, off_out = _run(0.0, a.steps, a.seed)
    off_clean = "[forest] apparence" not in off_out

    # variable ENTRE ÉPISODES : les étendues des 2 épisodes doivent différer (teinte re-tirée).
    cross_ep = len(spreads) >= 2 and spreads[0] != spreads[1]

    fails = []
    if spread < SPREAD_MIN:
        fails.append(f"T1 varie : étendue cos-rouge {spread:.3f} < {SPREAD_MIN} — l'apparence ne varie pas assez")
    if hi0 >= SLOT_THRESHOLD or ray_hi >= SLOT_THRESHOLD:
        fails.append(f"T2 hors cône : cos-rouge max {max(hi0, ray_hi):.3f} >= {SLOT_THRESHOLD} — un tronc "
                     "serait lu comme de la BOUFFE (§3, tronc-brun)")
    if ray_n <= 0:
        fails.append("T3 perçu : aucun rayon rétine ne voit de tronc hors cône — non perçu")
    if not cross_ep:
        fails.append("T4 entre épisodes : les 2 épisodes ont la MÊME étendue — la teinte n'est pas re-tirée")
    if not off_clean:
        fails.append("T5 défaut : le mode OFF émet quand même un log apparence (pas bit-identique)")

    print("\n=== VERDICT ===")
    if fails:
        for f in fails:
            print(f"  ✗ {f}")
        print("  G8 APPARENCE = ÉCHEC")
        return 1
    print(f"  G8 APPARENCE = PASS — les troncs varient (cos-rouge {lo0:.2f}..{hi0:.2f}), tous HORS du "
          f"cône bouffe (max {max(hi0, ray_hi):.2f} < 0,55), perçus ({ray_n} rayons), re-tirés entre épisodes.")
    print("  ⚠️ NON MESURÉ ICI : que l'ENCODEUR représente l'apparence des troncs (gate post-retrain, "
          "comme G5 : séparable est nécessaire pas suffisant).")
    print("  ⛔ RESTE : buissons (perceptibles, mineurs) ; le SOL n'est PAS sur la couche rétine → invisible.")
    return 0


def selfcheck() -> int:
    line0 = "[forest] apparence : var 0.15 | cos-rouge des arbres MESURE 0.207..0.499 (hi < 0.55 = ...)"
    line1 = "[forest] apparence : var 0.15 | cos-rouge des arbres MESURE 0.206..0.493 (hi < 0.55 = ...)"
    m = RE_APP.findall(line0 + "\n" + line1)
    assert len(m) == 2 and m[0] == ("0.15", "0.207", "0.499"), m
    print("  [ok] le parseur lit les lignes [forest] apparence (une par épisode)")

    # _tree_rays : un rayon vert (hors cône) compte comme tronc ; un rayon rouge (bouffe) non.
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf:
        ret = [0.5, 0.13, 0.35, 0.13] + [0.5, 0.9, 0.05, 0.05] + [1.0, 0.0, 0.0, 0.0]
        tf.write(json.dumps({"wm": {"retina0": ret}}) + "\n")
        p = tf.name
    n, lo, hi = _tree_rays(p)
    os.unlink(p)
    assert n == 1 and hi < 0.55, (n, lo, hi)
    print(f"  [ok] rétine synthétique : {n} rayon tronc (le vert), cos-rouge {hi:.3f} < 0.55 — le rouge exclu")

    # un vert franc est hors cône ; un brun est DANS le cône rouge (contrôle §3).
    assert _cos(np.array([0.13, 0.35, 0.13]), QUERY_RED) < 0.55
    assert _cos(np.array([0.36, 0.25, 0.15]), QUERY_RED) > 0.55
    print("  [ok] vert (0.13,0.35,0.13) hors cône, brun (0.36,0.25,0.15) DANS le cône — la garde §3 discrimine")

    assert os.path.exists(GODOT), GODOT
    print("  [ok] binaire Godot présent")
    print("SELFCHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
