"""G5 GRATUIT — LA PALETTE : une couleur variable SURVIT-ELLE à la perception, AVANT de collecter ?

PÉRIMÈTRE. Aucune collecte, aucun entraînement, aucun Godot. Pure géométrie de couleurs sur la
perception RÉELLEMENT servie. Coût : quelques secondes. C'est LE test de 2 minutes dont l'absence a
coûté une collecte entière (~1 h) la nuit d'avant, sur « une palette qui ne passait pas l'encodeur ».

LE VERROU (audit A1, 3ᵉ mesure). Le type de proie est lisible à 82,9 % dans la rétine brute mais
tombe à 29,5 % après l'encodeur (majorité = 44,2 %). Cause retenue : la couleur de la nourriture
était CONSTANTE à l'entraînement, l'encodeur n'a alloué aucune capacité à l'apparence. Le fix (§2.8)
est de faire VARIER la couleur dans la collecte — mais « mettre plusieurs couleurs » ne suffit pas.

CE QUE LA SONDE VÉRIFIE, ET POURQUOI C'EST LE BON TEST. La perception servie (slot_head) NORMALISE
la couleur d'un rayon puis calcule son COSINUS avec des requêtes fixes (rouge=(1,0,0), bleu=(0,0,1),
seuil 0,55). Deux conséquences dures, mesurées dans le code :
  * elle ne voit que la DIRECTION de la couleur, jamais sa magnitude (le cosinus est invariant
    d'échelle) — deux couleurs qui ne diffèrent QUE par la luminosité sont, pour elle, IDENTIQUES ;
  * une couleur n'est « de la nourriture » que si son cosinus au rouge dépasse 0,55.
Donc une palette n'est exploitable QUE si ses teintes (a) tombent dans le cône bouffe (détectées),
(b) restent hors du cône eau, et (c) diffèrent en DIRECTION — pas seulement en luminosité — sinon
rien en aval ne pourra jamais les séparer, quel que soit le ré-entraînement.

🚨 CE QUE LA SONDE RÉVÈLE SUR LA PALETTE ACTUELLE (attendu). Les 4 TYPE_COLORS servies
(0.900,0.300,0.200) … (0.288,0.096,0.064) sont des MULTIPLES SCALAIRES l'une de l'autre : même
direction, luminosité décroissante. Leur cosinus mutuel vaut ~1,0 → la perception ne peut PAS les
distinguer. C'est la racine des 29,5 %, et un test de 2 minutes le dit sans rien collecter.

CE QUE LA SONDE NE PEUT PAS DIRE. Que l'encodeur APPRENDRA une bonne palette : ça exige le retrain
(hors périmètre). Elle établit la condition NÉCESSAIRE — la palette est séparable dans l'espace que
la perception utilise — qui est précisément ce qu'un test gratuit peut trancher avant de payer.

CLI :
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_foret_g5_palette.py
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_foret_g5_palette.py --palette candidate
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_foret_g5_palette.py --selfcheck
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# La perception RÉELLEMENT servie (python/sylvan/models/slot_head.py) : requêtes pure-canal
# normalisées + seuil 0,55. On les reproduit ici pour juger dans le MÊME espace, pas un espace supposé.
QUERY_RED = np.array([1.0, 0.0, 0.0])
QUERY_BLUE = np.array([0.0, 0.0, 1.0])
SLOT_THRESHOLD = 0.55

# SÉPARABILITÉ DIRECTIONNELLE : deux teintes de cosinus mutuel > ce seuil sont, pour une perception
# qui normalise, quasi indistinguables. 0.985 ≈ 10° d'écart angulaire minimal exigé entre types.
MAX_PAIRWISE_COS = 0.985
PROBE_ACC_MARGIN = 0.15    # la sonde linéaire sur la DIRECTION doit battre le hasard d'au moins ça

FOOD_MANAGER = os.path.join(ROOT, "godot", "scripts", "world", "food_manager.gd")

# Une palette CANDIDATE séparable : mêmes 4 types, mais étalés en DIRECTION dans le cône bouffe.
# Chaque teinte garde R dominant (cos rouge > 0,55) tout en variant G et B → directions distinctes.
# Ce n'est PAS « la » réponse, c'est un candidat que la sonde valide ; le monde le sert via
# SYLVAN_FOOD_TYPE_HUES (voir food_manager.gd).
CANDIDATE_DEFAULT = [
    (0.90, 0.12, 0.10),   # rouge franc
    (0.90, 0.55, 0.08),   # rouge-orangé
    (0.85, 0.10, 0.45),   # rouge-rose
    (0.80, 0.42, 0.42),   # rouge-saumon
]


def _candidate() -> list[tuple[float, float, float]]:
    """La palette candidate = SYLVAN_FOOD_TYPE_HUES si défini, sinon le défaut ci-dessus.

    Ainsi `SYLVAN_FOOD_TYPE_HUES="..." diag_foret_g5_palette.py` valide EXACTEMENT la palette qu'un
    harnais servira à la collecte — le test de pré-vol, dans l'espace que la perception utilise.
    """
    env = os.environ.get("SYLVAN_FOOD_TYPE_HUES", "")
    if not env:
        return CANDIDATE_DEFAULT
    pal = []
    for grp in env.split(";"):
        p = grp.split(",")
        if len(p) == 3:
            pal.append((float(p[0]), float(p[1]), float(p[2])))
    return pal or CANDIDATE_DEFAULT


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _parse_type_colors() -> list[tuple[float, float, float]]:
    """Lit TYPE_COLORS DIRECTEMENT dans food_manager.gd — la sonde juge ce qui est SERVI, pas une
    copie qui pourrait dériver (§6bis : ne jamais faire confiance à une valeur recopiée)."""
    with open(FOOD_MANAGER) as f:
        txt = f.read()
    m = re.search(r"const TYPE_COLORS\s*:=\s*\[(.*?)\]", txt, re.DOTALL)
    if not m:
        raise SystemExit("TYPE_COLORS introuvable dans food_manager.gd")
    cols = re.findall(r"Color\(([\d.]+),\s*([\d.]+),\s*([\d.]+)\)", m.group(1))
    return [(float(r), float(g), float(b)) for r, g, b in cols]


def _slot_cone(palette: list[tuple[float, float, float]]) -> list[dict]:
    out = []
    for c in palette:
        v = np.array(c)
        cr, cb = _cos(v, QUERY_RED), _cos(v, QUERY_BLUE)
        out.append({"c": c, "cos_red": cr, "cos_blue": cb,
                    "food": cr > SLOT_THRESHOLD, "water": cb > SLOT_THRESHOLD})
    return out


def _pairwise(palette: list[tuple[float, float, float]]) -> tuple[float, list[tuple[int, int, float]]]:
    """Cosinus mutuel entre teintes (dans l'espace DIRECTION que la perception utilise)."""
    n = len(palette)
    worst = 0.0
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            c = _cos(np.array(palette[i]), np.array(palette[j]))
            pairs.append((i, j, c))
            worst = max(worst, c)
    return worst, pairs


def _direction_probe(palette: list[tuple[float, float, float]], seed: int = 0) -> float:
    """Sonde linéaire : peut-on LIRE le type depuis la seule DIRECTION (RGB normalisé + bruit) ?

    C'est ce que voit tout lecteur en aval de la normalisation du slot. Si les teintes partagent une
    direction, la précision retombe au hasard, quoi qu'on entraîne ensuite.
    """
    rng = np.random.default_rng(seed)
    dirs = np.array([np.array(c) / (np.linalg.norm(c) + 1e-9) for c in palette])
    k, per = len(palette), 200
    x = np.repeat(dirs, per, axis=0) + rng.normal(0, 0.03, (k * per, 3))   # bruit rétine léger
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    y = np.repeat(np.arange(k), per)
    # moindres carrés multi-classes (one-hot), split moitié/moitié
    idx = rng.permutation(len(x))
    x, y = x[idx], y[idx]
    h = len(x) // 2
    onehot = np.eye(k)[y[:h]]
    a = np.column_stack([x[:h], np.ones(h)])
    w, *_ = np.linalg.lstsq(a, onehot, rcond=None)
    at = np.column_stack([x[h:], np.ones(len(x) - h)])
    pred = (at @ w).argmax(1)
    return float((pred == y[h:]).mean())


def _report(name: str, palette: list[tuple[float, float, float]]) -> tuple[bool, list[str]]:
    print(f"\n=== PALETTE : {name} ({len(palette)} types) ===")
    fails = []
    cone = _slot_cone(palette)
    print("  détection par le slot (cône bouffe cos_rouge > 0.55, PAS eau cos_bleu > 0.55) :")
    for i, r in enumerate(cone):
        tag = "OK bouffe" if r["food"] and not r["water"] else ("EAU!" if r["water"] else "INVISIBLE")
        print(f"    type {i} {tuple(round(x,3) for x in r['c'])} : cos_rouge {r['cos_red']:.3f} "
              f"cos_bleu {r['cos_blue']:.3f}  → {tag}")
    if not all(r["food"] for r in cone):
        fails.append("au moins un type n'est PAS détecté comme bouffe (cos_rouge <= 0.55)")
    if any(r["water"] for r in cone):
        fails.append("au moins un type déclenche le slot EAU (cos_bleu > 0.55)")

    worst, pairs = _pairwise(palette)
    print(f"  séparabilité DIRECTIONNELLE (cosinus mutuel ; 1.0 = même direction = indistinguable) :")
    for i, j, c in pairs:
        flag = "  ⚠️ TROP PROCHES" if c > MAX_PAIRWISE_COS else ""
        print(f"    types {i}-{j} : cos {c:.4f} ({math_angle(c):.1f}°){flag}")
    if worst > MAX_PAIRWISE_COS:
        fails.append(f"cosinus mutuel max {worst:.4f} > {MAX_PAIRWISE_COS} — au moins deux types se "
                     "distinguent seulement par la LUMINOSITÉ, que la perception (cosinus) ignore")

    acc = _direction_probe(palette)
    chance = 1.0 / len(palette)
    print(f"  sonde linéaire sur la DIRECTION : {acc*100:.0f}% de types lus (hasard {chance*100:.0f}%)")
    if acc < chance + PROBE_ACC_MARGIN:
        fails.append(f"la direction ne permet de lire le type qu'à {acc*100:.0f}% (hasard "
                     f"{chance*100:.0f}%) — rien en aval ne pourra les séparer, retrain compris")
    return (not fails), fails


def math_angle(cos_val: float) -> float:
    import math
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_val))))


def selfcheck() -> int:
    # Palette de multiples scalaires (même direction) → cos mutuel 1.0, sonde au hasard.
    same_dir = [(0.9, 0.3, 0.2), (0.45, 0.15, 0.10), (0.27, 0.09, 0.06)]
    worst, _ = _pairwise(same_dir)
    assert worst > 0.9999, worst
    acc = _direction_probe(same_dir)
    assert acc < 0.5, acc
    print(f"  [ok] multiples scalaires : cos mutuel {worst:.4f} ≈ 1, sonde {acc*100:.0f}% ≈ hasard 33% "
          "— la sonde ATTRAPE la palette qui ne passe pas")

    # Palette bien étalée en direction → séparable.
    spread = [(0.9, 0.1, 0.1), (0.1, 0.9, 0.1), (0.1, 0.1, 0.9)]
    worst2, _ = _pairwise(spread)
    acc2 = _direction_probe(spread)
    assert worst2 < 0.5 and acc2 > 0.9, (worst2, acc2)
    print(f"  [ok] directions distinctes : cos mutuel {worst2:.4f}, sonde {acc2*100:.0f}% — séparable")

    assert abs(_cos(np.array([1.0, 0, 0]), QUERY_RED) - 1.0) < 1e-6   # epsilon du dénominateur de _cos
    assert _cos(np.array([0.9, 0.3, 0.2]), QUERY_RED) > 0.55
    print("  [ok] cosinus au rouge : (1,0,0)→1.0, (0.9,0.3,0.2)→dans le cône bouffe")

    pal = _parse_type_colors()
    assert len(pal) == 4 and pal[0] == (0.9, 0.3, 0.2), pal
    print(f"  [ok] TYPE_COLORS lues depuis food_manager.gd : {len(pal)} teintes servies")
    print("SELFCHECK PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--palette", choices=["served", "candidate", "both"], default="both")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()

    print("PERCEPTION SERVIE : requêtes rouge/bleu normalisées, seuil cosinus 0.55 "
          "(python/sylvan/models/slot_head.py). La palette est jugée dans CET espace.")
    verdicts = {}
    if a.palette in ("served", "both"):
        served = _parse_type_colors()
        verdicts["servie (TYPE_COLORS)"] = _report("SERVIE — food_manager.gd", served)
    if a.palette in ("candidate", "both"):
        cand = _candidate()
        src = "SYLVAN_FOOD_TYPE_HUES" if os.environ.get("SYLVAN_FOOD_TYPE_HUES") else "défaut interne"
        verdicts["candidate séparable"] = _report(f"CANDIDATE ({src}) — étalée en direction", cand)

    print("\n=== VERDICT ===")
    rc = 0
    for name, (ok, fails) in verdicts.items():
        print(f"  {name} : {'PASS' if ok else 'ÉCHEC'}")
        for f in fails:
            print(f"      ✗ {f}")
        if not ok and name.startswith("candidate"):
            rc = 1     # seul un ÉCHEC du candidat qu'on veut promouvoir est bloquant
    print("\n  ⚠️ CONDITION NÉCESSAIRE SEULEMENT : que l'ENCODEUR apprenne la palette exige le retrain")
    print("     (hors périmètre). La sonde tranche seulement qu'elle est SÉPARABLE dans l'espace")
    print("     que la perception utilise — précisément ce qu'un test gratuit doit dire avant de payer.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
