"""G7 GRATUIT — LES DISTRACTEURS : perceptibles, mobiles, et INVISIBLES au slot ressource.

PÉRIMÈTRE. Aucune collecte retenue, aucun entraînement. Une courte babillage, on LIT, on jette.

POURQUOI (design_foret_complete.md §2.9). Aujourd'hui les seules choses qui BOUGENT sont les proies,
et elles sont TOUTES de la nourriture. « ça bouge donc c'est de la nourriture » est un raccourci
GRATUIT : l'agent n'a jamais à discriminer, et le prédicteur du WM peut apprendre « mouvement→repas »
au lieu de « apparence→repas ». Des animaux qui bougent et qu'on ne peut PAS manger cassent ce
raccourci — même famille que les types arbitraires, la seule à avoir passé le filtre §1.

LE FILTRE §1, APPLIQUÉ AU DISTRACTEUR :
  T1 PERCEPTIBLE ......... il apparaît dans la rétine (rayons à sa couleur). Sinon rien ne l'apprend.
  T2 HORS DES CÔNES ...... 🚨 la garde §3 (tronc-brun). Le slot détecte une ressource par le COSINUS
     de la couleur du rayon (rouge=bouffe, bleu=eau, seuil 0,55) et EXCLUT en dur les rayons sous le
     seuil (slot_head.py). Si le distracteur a cos-rouge > 0,55 il serait lu comme de la NOURRITURE et
     corromprait la localisation. On EXIGE cos-rouge ET cos-bleu < 0,55 → le slot l'ignore, le foraging
     n'est pas corrompu. C'est une PREUVE (le masque dur du slot), pas une heuristique.
  T3 MOBILE .............. il vague (distance mesurée > 0). Sans mouvement, ce n'est pas un distracteur.
  T4 DÉFAUT .............. sans SYLVAN_DISTRACTOR_COUNT, aucun log, aucun changement (bit-identique).

CE QUE LA SONDE NE DIT PAS : que l'agent APPRENNE « mouvement ≠ nourriture » — c'est un gate
post-retrain (l'encodeur doit lire l'apparence, pas le mouvement). La sonde établit la condition
NÉCESSAIRE : le distracteur est présent, perçu, mobile, et n'active PAS le slot ressource.

CLI :
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_foret_g7_distracteurs.py
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_foret_g7_distracteurs.py --selfcheck
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

DISTRACTOR_HUE = (0.30, 0.70, 0.25)   # défaut de distractor_manager.gd (couleur « bestiole » verte)
QUERY_RED = np.array([1.0, 0.0, 0.0])
QUERY_BLUE = np.array([0.0, 0.0, 1.0])
SLOT_THRESHOLD = 0.55
COUNT = 6
OCCLUSION_KEEP_MIN = 0.60   # les distracteurs doivent laisser >= 60% des rayons bouffe (sinon ils cassent)

# `[distractor] episode : distance MESUREE 6.03 m sur 201 ticks (0.03000 m/tick moyen)`
RE_DIST = re.compile(r"\[distractor\] episode : distance MESUREE ([\d.]+) m sur (\d+) ticks")


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _run(count: int, steps: int, seed: int) -> tuple[str, str]:
    run_dir = "/tmp/foret_g7"
    os.system(f"rm -rf {run_dir}")
    e = dict(os.environ)
    e.update(BOSQUETS_V2.to_env())
    e.update({
        "SYLVAN_COLLECT": "1", "SYLVAN_WM_COLLECT": "1", "SYLVAN_COLLECTOR_MODE": "babbling",
        "SYLVAN_CPG": "1", "SYLVAN_RESIDUAL_GAIN": "0.0", "SYLVAN_TURN_FADE": "0",
        "SYLVAN_WM_VX_MIN": "0.55", "SYLVAN_WM_VX_MAX": "0.75", "SYLVAN_WM_WMAX": "0.6",
        "SYLVAN_POLICY_EXPLORATION_STD_INITIAL": "0", "SYLVAN_POLICY_EXPLORATION_STD_FINAL": "0",
        "SYLVAN_REFLEX_STRENGTH": "0", "SYLVAN_ASSIST_RATIO": "0",
        "SYLVAN_NUM_EPISODES": "2", "SYLVAN_MAX_EPISODE_STEPS": str(steps),
        "SYLVAN_SEED": str(seed), "SYLVAN_RUN_DIR": run_dir,
        "SYLVAN_DISABLE_HOMEOSTASIS": "1",   # les distracteurs bougent hors homéostasie ; ticks constants
    })
    if count > 0:
        e["SYLVAN_DISTRACTOR_COUNT"] = str(count)
    else:
        e.pop("SYLVAN_DISTRACTOR_COUNT", None)
    p = subprocess.run([GODOT, "--path", os.path.join(ROOT, "godot"), "--headless"],
                       env=e, capture_output=True, text=True, timeout=600)
    out = p.stdout + p.stderr
    for fatal in ("Parse Error", "Failed to load script"):
        if fatal in out:
            first = next((ln for ln in out.splitlines() if fatal in ln), fatal)
            raise SystemExit(f"[count={count}] Godot n'a PAS chargé — mesure invalide.\n  {first}")
    files = sorted(glob.glob(os.path.join(run_dir, "*.jsonl")))
    jsonl = files[0] if files else ""
    return jsonl, out


def _retina_stats(jsonl: str) -> tuple[int, int, int, int]:
    """Compte les rayons rétine : (a) touchés-colorés, (b) « distracteur » (cos>0.9 à sa teinte),
    (c) parmi ces distracteur-rayons, ceux lus comme bouffe (cos-rouge > 0.55), (d) rayons BOUFFE
    (cos-rouge > 0.55). Le (d) sert à l'A/B d'occlusion : distracteurs ON vs OFF à trajectoire égale."""
    hue = np.array(DISTRACTOR_HUE)
    hits, dist_rays, dist_as_food, food_rays = 0, 0, 0, 0
    with open(jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            retina = json.loads(line).get("wm", {}).get("retina0", [])
            if not retina:
                continue
            for k in range(0, len(retina), 4):
                rgb = np.array(retina[k + 1:k + 4])
                if rgb.sum() <= 1e-6:            # rayon manqué (pas de hit coloré)
                    continue
                hits += 1
                if _cos(rgb, QUERY_RED) > SLOT_THRESHOLD:
                    food_rays += 1
                if _cos(rgb, hue) > 0.9:         # ce rayon voit un distracteur
                    dist_rays += 1
                    if _cos(rgb, QUERY_RED) > SLOT_THRESHOLD:
                        dist_as_food += 1
    return hits, dist_rays, dist_as_food, food_rays


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()

    print(f"MONDE : {BOSQUETS_V2.name} + {COUNT} distracteurs {DISTRACTOR_HUE}")

    cr, cb = _cos(np.array(DISTRACTOR_HUE), QUERY_RED), _cos(np.array(DISTRACTOR_HUE), QUERY_BLUE)
    print(f"\n  COULEUR : distracteur {DISTRACTOR_HUE} → cos-rouge {cr:.3f}, cos-bleu {cb:.3f} "
          f"(seuil slot {SLOT_THRESHOLD}) — {'HORS des cônes' if cr < SLOT_THRESHOLD and cb < SLOT_THRESHOLD else 'DANS un cône !'}")

    jsonl, out = _run(COUNT, a.steps, a.seed)
    m = RE_DIST.findall(out)
    travel = float(m[-1][0]) if m else 0.0
    print(f"  SERVI (log [distractor]) : {'oui' if m else 'NON'} | distance mesurée {travel:.2f} m")
    hits, dist_rays, dist_as_food, food_on = _retina_stats(jsonl) if jsonl else (0, 0, 0, 0)
    print(f"  PERÇU : {dist_rays} rayons rétine voient un distracteur (sur {hits} touchés) ; "
          f"lus comme BOUFFE : {dist_as_food}")

    off_jsonl, off_out = _run(0, a.steps, a.seed)
    off_clean = "[distractor]" not in off_out
    # A/B d'OCCLUSION : trajectoire IDENTIQUE (distracteurs non bloquants + RNG dédié) → l'écart de
    # rayons BOUFFE vient UNIQUEMENT des distracteurs qui masquent la bouffe. §1-Q4 : changer le monde
    # sans le CASSER.
    _, _, _, food_off = _retina_stats(off_jsonl) if off_jsonl else (0, 0, 0, 0)
    keep = food_on / food_off if food_off > 0 else 1.0
    print(f"  OCCLUSION (A/B, trajectoire égale) : rayons BOUFFE {food_off} (sans) → {food_on} (avec) "
          f"= {keep*100:.0f}% conservés")

    fails = []
    if cr >= SLOT_THRESHOLD or cb >= SLOT_THRESHOLD:
        fails.append(f"T2 : la couleur est DANS un cône (cos-rouge {cr:.3f} / cos-bleu {cb:.3f}) — "
                     "le slot la lirait comme une ressource et corromprait le foraging (§3)")
    if not m or travel <= 0.0:
        fails.append(f"T3 : le distracteur ne bouge pas (distance {travel:.2f} m) — pas un distracteur")
    if dist_rays <= 0:
        fails.append("T1 : aucun rayon rétine ne voit de distracteur — il n'est pas perçu")
    if dist_as_food > 0:
        fails.append(f"T2 : {dist_as_food} rayons distracteur seraient lus comme BOUFFE — il corromprait "
                     "le slot (la couleur SERVIE n'est pas vraiment hors cône)")
    if keep < OCCLUSION_KEEP_MIN:
        fails.append(f"T5 occlusion : les distracteurs masquent trop la bouffe ({keep*100:.0f}% de rayons "
                     f"bouffe conservés < {OCCLUSION_KEEP_MIN*100:.0f}%) — réduire nombre/taille des distracteurs")
    if not off_clean:
        fails.append("T4 : le mode OFF émet quand même un log distracteur (pas bit-identique)")

    print("\n=== VERDICT ===")
    if fails:
        for f in fails:
            print(f"  ✗ {f}")
        print("  G7 DISTRACTEURS = ÉCHEC")
        return 1
    print(f"  G7 DISTRACTEURS = PASS — {COUNT} animaux perçus ({dist_rays} rayons), mobiles "
          f"({travel:.1f} m), HORS des cônes (0 rayon lu comme bouffe) : le slot les ignore, le")
    print("  foraging n'est pas corrompu, et « mouvement » n'est plus un synonyme gratuit de « nourriture ».")
    print("  ⚠️ NON MESURÉ ICI : que l'agent APPRENNE mouvement≠nourriture (gate post-retrain).")
    return 0


def selfcheck() -> int:
    cr = _cos(np.array(DISTRACTOR_HUE), QUERY_RED)
    cb = _cos(np.array(DISTRACTOR_HUE), QUERY_BLUE)
    assert cr < SLOT_THRESHOLD and cb < SLOT_THRESHOLD, (cr, cb)
    print(f"  [ok] la couleur par défaut {DISTRACTOR_HUE} est hors des cônes "
          f"(cos-rouge {cr:.3f}, cos-bleu {cb:.3f} < {SLOT_THRESHOLD})")

    # Une couleur DANS le cône rouge (un tronc-brun mobile) doit être REFUSÉE par le test.
    brown = np.array([0.36, 0.25, 0.15])
    assert _cos(brown, QUERY_RED) > SLOT_THRESHOLD
    print(f"  [ok] contrôle : un brun (0.36,0.25,0.15) a cos-rouge {_cos(brown, QUERY_RED):.3f} > "
          f"{SLOT_THRESHOLD} — le test le classerait DANS le cône (donc refusé)")

    line = "[distractor] episode : distance MESUREE 6.03 m sur 201 ticks (0.03000 m/tick moyen)"
    m = RE_DIST.findall(line)
    assert m and m[0] == ("6.03", "201"), m
    print("  [ok] le parseur lit la ligne [distractor] émise par distractor_manager.gd")

    # _retina_stats : un rayon vert (distracteur) compte comme distracteur et PAS comme bouffe ;
    # un rayon rouge (bouffe) ne compte pas comme distracteur.
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf:
        green = list(DISTRACTOR_HUE)
        ret = [0.5, green[0], green[1], green[2]] + [0.5, 0.9, 0.05, 0.05] + [1.0, 0.0, 0.0, 0.0]
        tf.write(json.dumps({"wm": {"retina0": ret}}) + "\n")
        p = tf.name
    hits, dr, daf, fr = _retina_stats(p)
    os.unlink(p)
    assert hits == 2 and dr == 1 and daf == 0 and fr == 1, (hits, dr, daf, fr)
    print(f"  [ok] rétine synthétique : {hits} touchés, {dr} distracteur, {daf} lu-bouffe, {fr} bouffe — "
          "le vert compte distracteur/PAS-bouffe, le rouge compte bouffe/PAS-distracteur")

    assert os.path.exists(GODOT), GODOT
    print("  [ok] binaire Godot présent")
    print("SELFCHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
