#!/usr/bin/env python3
"""Test unitaire GRATUIT pour occlude_retina (Task 4 — mémoire spatiale).

Protocole : asserts + prints, sans pytest (convention projet Sylvan).
Lance : PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_occlude_retina.py

Géométrie retina :
    144 floats = 36 rayons × 4 canaux [depth, R, G, B]
    Ray k → angle k*10°, distance angulaire depuis avant = min(k*10, 360-k*10)
    Ray 0  = avant     (dist 0°)
    Ray 9  = 90° droite (dist 90°)
    Ray 18 = derrière  (dist 180°)
"""
import sys
sys.path.insert(0, "python")

from scripts.serve_planner_command import occlude_retina, _N_RAYS, _CHANNELS

RETINA_DIM = _N_RAYS * _CHANNELS   # 144

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_failures = 0


def check(cond: bool, msg: str) -> None:
    global _failures
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  {msg}")
        _failures += 1


def make_retina(n_rays: int = _N_RAYS) -> list[float]:
    """Rétine synthétique : depth=0.5, R=0.8, G=0.6, B=0.4 pour chaque rayon."""
    return [v for _ in range(n_rays) for v in (0.5, 0.8, 0.6, 0.4)]


def ray_is_occluded(ret: list[float], k: int) -> bool:
    base = k * _CHANNELS
    return ret[base] == 1.0 and ret[base+1] == 0.0 and ret[base+2] == 0.0 and ret[base+3] == 0.0


def ray_is_intact(ret: list[float], k: int, orig: list[float]) -> bool:
    base = k * _CHANNELS
    return ret[base:base+_CHANNELS] == orig[base:base+_CHANNELS]


# ── TEST 1 : FOV=360 → identité EXACTE ──────────────────────────────────────
print("\n[TEST 1] FOV=360 → identité exacte")
orig = make_retina()
out360 = occlude_retina(orig, 360.0)
check(out360 == orig, "FOV 360 : sortie == entrée")
check(out360 is not orig, "FOV 360 : sortie est une COPIE (pas in-place)")
check(len(out360) == RETINA_DIM, f"longueur == {RETINA_DIM}")

# Vérifier aussi FOV=361 (> 360)
out361 = occlude_retina(orig, 361.0)
check(out361 == orig, "FOV 361 : sortie == entrée (≥360 toujours identité)")

# ── TEST 2 : FOV=180 — vérification des rayons frontaux et arrière ──────────
print("\n[TEST 2] FOV=180 (cône frontal ±90°) — rayons clés")
orig = make_retina()
out180 = occlude_retina(orig, 180.0)

# Règle : dist > 90° → occlus
# Ray 0  : dist=0°   → INTACT (dans le cône)
check(ray_is_intact(out180, 0, orig),   "ray 0 (avant,   dist  0°) → INTACT")
# Ray 9  : dist=90°  → INTACT (exactement au bord, dist NOT > half = 90)
check(ray_is_intact(out180, 9, orig),   "ray 9  (droite, dist 90°) → INTACT (bord inclusif)")
# Ray 27 : dist=90°  → INTACT (symétrique gauche)
check(ray_is_intact(out180, 27, orig),  "ray 27 (gauche, dist 90°) → INTACT (bord inclusif)")
# Ray 10 : dist=100° → OCCULTÉ
check(ray_is_occluded(out180, 10),      "ray 10 (dist 100°) → OCCULTÉ")
# Ray 26 : dist=100° → OCCULTÉ
check(ray_is_occluded(out180, 26),      "ray 26 (dist 100°) → OCCULTÉ")
# Ray 18 : dist=180° → OCCULTÉ (derrière)
check(ray_is_occluded(out180, 18),      "ray 18 (derrière, dist 180°) → OCCULTÉ")

# Compter le nombre de rayons occultés : dist > 90°  → rays 10..26 sauf... non :
# dist(k) = min(k*10, 360-k*10) ; half=90 ; rays avec dist>90 :
#   k=10 dist=100, k=11 dist=110, ..., k=18 dist=180, ..., k=26 dist=100
#   → rays 10..26 = 17 rayons occultés ; rays 0..9 + 27..35 = 19 rayons intacts
occluded_count = sum(1 for k in range(_N_RAYS) if ray_is_occluded(out180, k))
intact_count   = sum(1 for k in range(_N_RAYS) if ray_is_intact(out180, k, orig))
check(occluded_count == 17, f"FOV 180 : exactement 17 rayons occultés (got {occluded_count})")
check(intact_count   == 19, f"FOV 180 : exactement 19 rayons intacts   (got {intact_count})")

# ── TEST 3 : non-mutation de l'entrée ──────────────────────────────────────
print("\n[TEST 3] Non-mutation de l'entrée")
orig = make_retina()
orig_copy = list(orig)
_ = occlude_retina(orig, 120.0)
check(orig == orig_copy, "L'entrée n'est PAS mutée après occlude_retina(FOV=120)")

# ── TEST 4 : longueur de sortie toujours 144 ───────────────────────────────
print("\n[TEST 4] Longueur de sortie")
for fov in (0.0, 90.0, 180.0, 270.0, 360.0, 400.0):
    out = occlude_retina(make_retina(), fov)
    check(len(out) == RETINA_DIM, f"FOV={fov} → len={len(out)} (attendu {RETINA_DIM})")

# ── TEST 5 : FOV=0 → TOUS les rayons occultés sauf ray 0 (dist=0 NOT > 0) ──
print("\n[TEST 5] FOV=0 (cone ultra-serré, seul ray 0 à dist 0 passe)")
orig = make_retina()
out0 = occlude_retina(orig, 0.0)
# dist 0° = 0, half = 0.0 → dist (0) NOT > half (0) → RAY 0 INTACT
check(ray_is_intact(out0, 0, orig), "ray 0 (dist 0°) intact même avec FOV=0")
# tous les autres : dist > 0 → occultés
all_others_occluded = all(ray_is_occluded(out0, k) for k in range(1, _N_RAYS))
check(all_others_occluded, f"FOV=0 : rays 1..{_N_RAYS-1} tous occultés")

# ── RÉSUMÉ ─────────────────────────────────────────────────────────────────
print()
if _failures == 0:
    print(f"\033[32m✓ ALL TESTS PASSED (0 failures)\033[0m")
    sys.exit(0)
else:
    print(f"\033[31m✗ {_failures} TEST(S) FAILED\033[0m")
    sys.exit(1)
