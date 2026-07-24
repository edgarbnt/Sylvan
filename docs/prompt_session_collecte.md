# PROMPT — Session COLLECTE + RETRAIN du monde-forêt Sylvan

> À coller tel quel pour démarrer la session dédiée. Écrit à la fin de la session build+vérif
> (2026-07-24) qui a bâti et consolidé le monde-forêt. Projet : `/home/edgarbrunet/Documents/PERSO/SylvanV1`.

---

## ⛔ PÉRIMÈTRE DE CETTE SESSION
Cette session **COLLECTE** un corpus du monde-forêt complet et **RE-ENTRAÎNE le WM**. Discipline dure :
- **DRY-RUN GRATUIT d'abord** (collecte COURTE + vérif), collecte réelle **SEULEMENT si c'est vert**.
- **Diagnostiquer avant de payer** : un run coûteux deviné est le chemin le plus cher (PRINCIPE N°1).
- **Pré-enregistrer succès ET kill** avant chaque run long.

## 🚫 RÈGLES DURES
- **NE RIEN METTRE SUR `main`, ne rien pousser sur origin.** `main` = dépôt public (vitrine). Si tu
  crois qu'il faut y toucher : **arrête-toi et demande**.
- Le monde-forêt est **déjà consolidé** sur `feat/perception-consequence` (@ `2b1741b`). Travailler
  là, ou dans un worktree (`git worktree add`, jamais partager un checkout — piège déjà payé).
- **PPO : `--lr 1e-4`** (le défaut 3e-4 diverge). **Tuer un run** : `/sylvan-kill` puis vérifier
  `pgrep -af serve_ppo_collect` = 0 et `pgrep -xc godot` = 0 (les workers survivent à un pkill simple).
- venv **`env_pytorch_3.12/bin/python`** (CPU obligatoire), **`PYTHONPATH=python`** depuis la racine,
  **`GODOT_BIN="$(pwd)/tools/godot/godot"`**.
- **`tools/archi_hud/architecture.json` à jour DANS LE MÊME COMMIT** que tout changement de conception
  (validateur : `env_pytorch_3.12/bin/python tools/archi_hud/validate_architecture.py`).
- Commits **anglais, Conventional Commits, sans co-author**.

## LIRE D'ABORD (specs, pas des notes)
1. `docs/design_foret_complete.md` — §2ter (liste bloquante), §5 (ordre), §6sexies (collecte mixte),
   §6ter (ordre de réparation de l'archi), §6quinquies (décisions angles morts).
2. `docs/audit_conformite_jepa.md` — A1-A5 (dont A2 = requêtes codées-main à promouvoir).
3. `docs/outils_diagnostic.md` — `diag_world_contract.py`, `diag_info_matrix.py` (outils du dry-run).
4. `CLAUDE.md` — règles ops, §1-§4 principes, corps servi par les harnais.

---

## ÉTAT — CE QUI EST FAIT ET VÉRIFIÉ (ne pas re-faire)
8 briques du monde bâties, chacune **opt-in défaut-OFF bit-identique**, chacune avec sa **sonde
gratuite qui a tourné** (`diagnostics/diag_foret_g1..g8`, toutes PASS ; selfcheck `--selfcheck`) :

| brique | flags opt-in | sonde |
|---|---|---|
| arrangement écologique (Clark-Evans) + coût calcul | `SYLVAN_FOREST_STANDS/CLEARINGS` | g1 |
| re-calibration métabolique (ANALYSE seule) | — | g2 |
| regard indépendant (proprio 132→**133**) | `SYLVAN_GAZE=1` | g3 |
| terrain qui ralentit (+ contrôlabilité : le corps obéit) | `SYLVAN_TERRAIN_SLOW=0.6` | g4 |
| palette nourriture séparable (fix A1, test PRÉ-retrain) | `SYLVAN_FOOD_TYPE_HUES=...` | g5 |
| flaques (eau qui rétrécit graduellement) | `SYLVAN_WATER_PUDDLE_PERIOD=300` | g6 |
| distracteurs (mobiles non comestibles, hors cônes) | `SYLVAN_DISTRACTOR_COUNT=6` | g7 |
| apparence variable des troncs (hors cône, garde §3) | `SYLVAN_FOREST_APPEARANCE_VAR=0.15` | g8 |

Occlusion (forêt solide + terrain) et objets mobiles (proies v6 + distracteurs) sont dans le lot.

## ⚠️ CE QUI N'EST **PAS** BÂTI (honnêteté — ne pas le supposer présent)
- **Éventail de vitesse (§2.13)** : G2 a **analysé** (le corps est trop lent, ±15 % = façade ; pour
  10-30 événements/vie il faut drain plus fort ET vitesse plus large). Le **mécanisme n'est pas
  bâti** : `vx_grid` reste `(0.55,0.65,0.75)`. **Décision à trancher** (voir plus bas).
- **Tapi / se tapir (§2.11)** : délibérément **différé** « avec l'affût » (§6quinquies E), PAS dans
  cette collecte.
- **`symmetry.py` pas à jour pour l'angle de tête** (regard) : l'angle est normalisé et change de
  signe sous miroir → le refléter AVANT tout entraînement qui utilise l'augmentation par symétrie.
- **Preset monde-complet** : `python/sylvan/world.py` n'émet **aucun** `SYLVAN_FOREST_*` — un preset
  gelé ne décrit donc pas encore la forêt, et un corpus forêt ne se décrirait pas lui-même. **À
  faire avant la collecte** (sinon `diag_world_contract` n'a rien à comparer) : étendre `world.py`
  pour émettre les flags forêt/terrain/distracteurs/regard/flaques/apparence → un preset
  `foret_v1` self-describing.
- **Palette bushes** (buissons) : couleur encore partagée (mineur, ~4-8 marqueurs).
- **Écart 0,011 vs 0,0100 m/tick** (vitesse déclarée vs formule au sommet du grid) : non résolu.

---

## ORDRE (chaque étape gatée)

### 0. Déterminisme — GATE gratuit AVANT tout (§6quater F)
Beaucoup plus d'objets = beaucoup plus de consommateurs de RNG. Re-vérifier le **rejeu bit-identique**
du monde complet (même seed + mono-thread torch + serveur frais → deux runs identiques). Les flux RNG
nouveaux ont un seed DÉDIÉ (regard +7171, distracteurs +3131) pour ne pas décaler les commandes, mais
le vérifier. Si le déterminisme tombe, **tous** les juges contrefactuels tombent → bloquant.

### 1. Assembler la config monde-complet (harnais scripté `.sh` avec `export`)
Base = preset proies+types (`bosquets_v7_types`) + les flags forêt. Réunir (valeurs de départ, à
calibrer) : `SYLVAN_FOREST_COUNT=45 SYLVAN_FOREST_STANDS=6 SYLVAN_FOREST_CLEARINGS=3
SYLVAN_FOREST_APPEARANCE_VAR=0.15 SYLVAN_TERRAIN_SLOW=? SYLVAN_TERRAIN_RADIUS=2.5 SYLVAN_TERRAIN_FLOOR=0.25
SYLVAN_GAZE=1 SYLVAN_FOOD_TYPE_HUES="0.9,0.12,0.1;0.9,0.55,0.08;0.85,0.1,0.45;0.8,0.42,0.42"
SYLVAN_DISTRACTOR_COUNT=6 SYLVAN_WATER_COUNT=? SYLVAN_WATER_PATCHES=? SYLVAN_WATER_PUDDLE_PERIOD=300
SYLVAN_THIRST_DRAIN=? SYLVAN_ENERGY_DRAIN=?`. Un `.sh` avec `export` (le piège shell multi-ligne est
déjà documenté). **Chaque flag opt-in a été vérifié bit-identique OFF** ; c'est leur COMBINAISON qui
n'a jamais tourné → d'où le dry-run.

### 2. Décisions à trancher AVANT la collecte
- **Métabolisme** : la collecte au drain actuel (0,05) reproduira la famine de données (1-2 repas/vie,
  survie saturée). G2 recommande **drain ~0,23** (+ vitesse) pour 10-30 événements. **Trancher** :
  drain recalibré au minimum. Cible mesurable : 10-30 événements/vie, survie ni saturée ni effondrée.
- **Vitesse** : collecter au `vx_grid` actuel (drain recalibré compensant en partie), OU **bâtir**
  l'éventail large (change le corps → ré-exploration + re-mesure navigabilité en forêt). Choix owner.
  ⚠️ Si on ouvre la vitesse, le terrain (g4) et la navigabilité forêt (45 arbres) sont à re-mesurer.
- **Regard** : `SYLVAN_GAZE=1` porte la proprio à **133** → le WM re-entraîné est en 133 (pas de
  compat avec les checkpoints actuels). Exploration du regard = déjà câblée dans le babillage.
- **Palette** : servir `SYLVAN_FOOD_TYPE_HUES` (validée séparable par g5). Ne PAS servir les
  `TYPE_COLORS` par défaut (multiples scalaires, illisibles — mesuré g5).

### 3. DRY-RUN (le gate qui a manqué la nuit de la palette perdue)
Collecte **COURTE** (~2-3 vies) du monde complet assemblé → `diag_world_contract.py` (le monde
sert-il ce qu'il déclare ?) + `diag_info_matrix.py` (l'information survit-elle à l'encodeur ?).
**Collecter pour de vrai UNIQUEMENT si c'est vert.** Lire les `[world]`/`[patch]`/`[forest]`/`[gaze]`/
`[terrain]`/`[puddle]`/`[distractor]` : ils rapportent le MESURÉ, pas le demandé (§6bis).

### 4. COLLECTE MIXTE (§6sexies) — seulement si dry-run vert
- **Part planner** (relevance : visiter les états qui comptent, dont le CONTACT) + **part bruit
  d'exploration ÉTENDU** au regard (§6quinquies E : toute action nouvelle DOIT être explorée) +
  **part très exploratoire**. Le bruit seul n'attrape jamais rien → WM qui connaît le vide.
- **Parallélisme** (plusieurs Godot) pour la vitesse de collecte — jamais augmenter le pas de sim.

### 5. RETRAIN WM — warm-start
🚨 **DANS LE MÊME COMMIT, promouvoir les requêtes APPRISES** (`wm_objcentric_kin_typed`, verrou **A2**)
— sinon le détecteur cos codé-main (seuil 0,55) fait qu'un tronc-brun reste « de la nourriture » et
**survit au retrain**. Le retrain est légitime parce qu'il répare une **cécité générale** (l'encodeur
n'encode pas l'apparence, A1 3ᵉ mesure), pas parce que « la bouffe compte » (§3).

### 6. GATES POST-RETRAIN (pré-enregistrés, falsifiables)
- **Perception du type** : encodeur ~30 % → **> 70 %** (le gate que g5 ne pouvait PAS trancher :
  « séparable » est nécessaire, PAS suffisant — un pair a mesuré 82,9 % rétine → 30 % encodeur).
- **Latent porte l'objet** (`diag_latent_carries_object.py` / `_type.py`).
- **Non-régression open-loop** : position/yaw/eff_rank **≥ actuel** (JAMAIS troquer robustesse contre
  pureté).
- **Dynamique** : le prédicteur bat « l'objet reste immobile » (sonde à écrire).

### 7. CRITIQUE EN DERNIER (§6ter)
Il échouait parce qu'il n'avait rien à lire (substrat aveugle). Ne le re-tenter qu'après A1 réglé.
**Juge = A/B PLEINE POLITIQUE**, JAMAIS une AUC poolée.

---

## NE PAS FAIRE
- Toucher/pousser `main` ; collecter avant que le dry-run soit vert ; supprimer les 2 fichiers
  `info_matrix.py`/`diag_info_matrix.py` (ils se résolvent seuls, un commit de nettoyage crée un
  conflit modify/delete ou une suppression silencieuse — vérifié).
- Ajouter une mécanique qui échoue au **filtre §1** (perceptible / survit-encodeur / non-dérivable /
  change-l'issue).
- **Rouvrir le risque de blessure à la chasse** (échec P2-bis mesuré, §6quinquies B) sans ancre
  d'aversion explicite.
- Juger un critique sur une AUC poolée ; masquer une lacune de capacité en relâchant un critère (§2).
