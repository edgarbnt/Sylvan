# Task 4 Report — Masque d'occlusion + gates mémoire spatiale

## Status: DONE (infrastructure, tests GREEN; Godot gates à lancer nativement)

---

## Ce qui a été construit

### 1. Masque d'occlusion dans `python/scripts/serve_planner_command.py`

- Fonction `occlude_retina(retina, fov_deg)` ajoutée en tête de fichier (après les imports) :
  - Ray k (k=0..35) à angle k×10°, distance angulaire = min(k×10, 360−k×10) degrés.
  - Si dist > fov_deg/2 → depth=1.0, RGB=0.0 (saillance → 0 dans slot_encoder).
  - fov_deg ≥ 360 → copie identique exacte (non-régression byte-identique).
  - L'entrée n'est PAS mutée (copie défensive).
- Variable module `_OCCLUDE_FOV_DEG` lue une fois au démarrage depuis `SYLVAN_OCCLUDE_FOV_DEG` (défaut 360.0 = OFF).
- Application dans `predict_full`, immédiatement après `retina = list(payload.get("retina") or [])`, **avant tout usage downstream** (localisation, wm_obs, SlotMemory re-grounding). Un seul site d'application.
- Log au démarrage du service : actif ou inactif selon la valeur de FOV.

**Note honnêteté (documentée dans le code) :** occluder la rétine côté serveur présente une rétine hors-distribution au WM (entraîné sur 360°). C'est une approximation acceptable pour la gate (l'objet disparaît correctement du slot_encoder → SlotMemory doit le maintenir par dead-reckoning). Un cône frontal "de production" nécessiterait un retrain WM sur données avec cone — travail différé, hors scope Task 4.

### 2. `run_forage_memory.sh` (clone de `run_forage_wmslot.sh`)

Knobs supplémentaires :
- `MEM=on|off` : `on` → ajoute `--egomotion-head $EGOMOTION_CKPT --slot-memory` au serveur ; `off` → rien.
- `SYLVAN_OCCLUDE_FOV_DEG` : exporté au process serveur (défaut 180°).
- `EGOMOTION_CKPT` overridable (défaut `data/checkpoints/egomotion_head/best.pt`).
- Tout le reste (Godot env block, parser survie) identique à `run_forage_wmslot.sh`.

### 3. `diag_nav_ab_memory.sh` (clone de `diag_nav_ab_wmslot.sh`)

Mêmes knobs `MEM` et `SYLVAN_OCCLUDE_FOV_DEG` sur le lancement serveur.
Protocole azimut, homeostasis off, `parse_nav_ab.py` — identique au parent.

### 4. `diag_occlude_retina.py` — test pur-Python (20 assertions, 5 blocs)

- TEST 1 : FOV=360 → identité exacte + copie + longueur (×2 : 360 et 361) — PASS
- TEST 2 : FOV=180 — ray 0 intact, ray 9/27 intacts (bord inclusif), ray 10/26/18 occultés, comptage 17/19 — PASS
- TEST 3 : non-mutation entrée — PASS
- TEST 4 : longueur 144 pour 6 valeurs de FOV — PASS
- TEST 5 : FOV=0 — ray 0 intact, rays 1..35 tous occultés — PASS

Résultat : **20/20 PASS** (GREEN confirmé).

---

## Commandes natives pour l'owner (Steps 2-4)

### Step 2 — Gate A→B engagement, mémoire ON vs OFF avec occlusion (FOV 180°)

```bash
# Mémoire OFF (baseline avec occlusion)
MEM=off bash diag_nav_ab_memory.sh

# Mémoire ON (SlotMemory dead-reckoning quand bouffe hors cône)
MEM=on  bash diag_nav_ab_memory.sh
```

Critère : mémoire ON ≥ mémoire OFF sur les azimuts arrière (|az| > 90°, i.e. 135/180/225/270).

### Step 3 — Gate foraging survie, mémoire ON vs OFF avec occlusion (FOV 180°)

```bash
# Mémoire OFF
MEM=off bash run_forage_memory.sh

# Mémoire ON
MEM=on  bash run_forage_memory.sh
```

Critère : survie médiane ON ≥ OFF.

### Step 4 — Non-régression (sans occlusion, mémoire ON)

```bash
MEM=on SYLVAN_OCCLUDE_FOV_DEG=360 bash run_forage_memory.sh
```

Critère : engagement ≥ 15/16, foraging médiane ≥ ~860 (= baseline wmslot).

### Variantes utiles

```bash
# Ajuster FOV (ex. 90° = cône étroit)
MEM=on SYLVAN_OCCLUDE_FOV_DEG=90 bash diag_nav_ab_memory.sh

# Sous-ensemble d'azimuts pour un test rapide
SYLVAN_NAV_ANGLES="135 180 225" MEM=on bash diag_nav_ab_memory.sh

# Override egomotion checkpoint si chemin différent
MEM=on EGOMOTION_CKPT=data/checkpoints/egomotion_head/best.pt bash diag_nav_ab_memory.sh
```

---

## Fichiers modifiés / créés

| Fichier | Action |
|---------|--------|
| `python/scripts/serve_planner_command.py` | Ajout `occlude_retina()` + `_OCCLUDE_FOV_DEG` + application dans `predict_full` + log |
| `run_forage_memory.sh` | Créé (executable) |
| `diag_nav_ab_memory.sh` | Créé (executable) |
| `diag_occlude_retina.py` | Créé (test pure-python, 20/20 PASS) |

---

## Préoccupations / points de vigilance

1. **`egomotion_head/best.pt` doit exister** : si Task 3 n'a pas produit ce checkpoint, lancer `MEM=on` affiche un avertissement du serveur et ignore silencieusement la mémoire (comportement sûr — wm.with_slot requis aussi).
2. **OOD retina** : le WM entraîné sur 360° reçoit une rétine masquée à FOV=180° — léger shift de distribution dans le rollout imaginaire. Acceptable pour la gate ; si la non-régression (Step 4) montre une dégradation inattendue, vérifier que FOV=360 restaure les chiffres wmslot.
3. **Godot ne lit pas SYLVAN_OCCLUDE_FOV_DEG** : la variable est exportée au serveur Python seulement (c'est là que le masque vit). Godot n'a aucune connaissance de l'occlusion.
