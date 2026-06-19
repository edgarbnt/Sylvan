# SCOPE GRATUIT — RÉTINE (perception apprise) — ZÉRO entraînement

> **✅ JALON ATTEINT (2026-06-18 nuit).** Étage 0 (rétine raycast) + étage 1 (tête de perception apprise)
> FAITS. A/B foraging : RÉTINE apprise médiane 860 vs ORACLE 965 = parité ~ ; l'agent navigue vers la bouffe
> avec UNIQUEMENT les rayons couleur bruts (oracle débranché). Tête : `data/checkpoints/retina_head/head_best.pt`.
> Détail + chemin (incl. le bug `predict_planner` non-sérialisé) dans `memory/sylvan-retina-decision.md`.
> Reste (optionnel) : étage 2 (le WM consomme la rétine), rétiniser l'eau, raffiner le multi mid-range.

> **STATUT (2026-06-18) : SCOPE écrit, à discuter AVANT tout code/entraînement.** Suite de
> `docs/design_retina.md` (décision) — ici = le plan fin, dims exactes, fichiers, gating, critères.
> Discipline CLAUDE.md §1 (diagnostiquer/gater gratuitement d'abord) + §2 (pas de fausse solution, pas
> d'oracle déguisé, pas de conclusion arrangeante).

---

## 0. DÉCOUVERTES DE CARTOGRAPHIE (corrigent `design_retina.md`)

1. **L'infra raycast N'EXISTE PAS.** `perception.gd::food_radar()` est **analytique** (géométrie sur les
   positions des ressources), PAS un raycast. Food/water n'ont **ni collider ni groupe de collision**
   (`food_manager.gd`, `main.gd:77`) — repérés par position seule. → la phrase « l'infra raycast existe déjà »
   du design est **inexacte** (à corriger). **DÉCISION (2026-06-18, owner) : raycast PHYSIQUE Godot**
   (`intersect_ray`), PAS analytique — parce que le monde va s'enrichir (objets, collisions, occlusion) et le
   raycast physique est le seul socle qui passe à l'échelle d'un vrai monde sans re-coder la perception. On
   construit donc l'infra colliders maintenant (voir §1bis SÉCURITÉ : couche dédiée → ne perturbe PAS le gait).
2. **WM-obs et localisation sont DÉJÀ découplés** dans `serve_planner_command.py` : le WM encode `radar`(12),
   mais la bouffe est localisée depuis `radar_ema` (fine 36 si Godot l'envoie) via `food_xz_from_radar`. La
   rétine remplacera les deux, mais ce découplage rend 🅐 plus simple (la tête remplace juste le localisateur).
3. **La policy résiduelle (hexapod_v2, BANKÉE) NE voit PAS le radar.** En mode commande (défaut,
   `main.gd::_compute_vision()` cas 5) la vision policy = `[vx, omega, 0,…]`. → **la rétine ne touche QUE le
   WM/planner, PAS la policy** → pas de retrain moteur, `obs policy 144` inchangé. (Énorme réduction de risque.)
4. **La plupart des dims Python sont POLYMORPHES** (auto-détectées depuis les données) : `obs_dim` coule de
   `wm_dataset` → `train_wm_command` → `CommandWorldModel` → `meta["obs_dim"]`. Les seuls **hardcodes** à
   toucher sont `symmetry.py` et le champ lu dans `wm_dataset._obs_at`.

---

## 1. LAYOUT EXACT DE LA RÉTINE + NOUVELLE OBS WM (à figer)

### Rétine
- **N = 36 rayons**, **FOV 360°**, espacement **10°**. Indexation : **rayon 0 = droit devant (bearing 0)**,
  sens trigonométrique, rayon k = bearing `+k·10°`, rayon 18 = arrière (180°). (Choix de l'origine = forward
  pour que la carte miroir soit propre, cf §2.)
- **4 canaux / rayon** = `[depth, R, G, B]` → **144 dims** (36×4).
- **Rendu : raycast PHYSIQUE** (`PhysicsDirectSpaceState3D.intersect_ray`). Couleur lue via
  `collider.get_meta("retina_color")` (chaque objet perceptible déclare sa couleur → futur objet = juste poser
  collider+meta, rien à re-coder côté perception). Occlusion gratuite.
- **Encodage couleur** : RGB de l'objet le plus proche touché par le rayon, dans `[0,1]` (Godot `Color` est
  déjà 0–1). Miss (rien sur le rayon) → `depth = 1.0` (max) et `RGB = (0,0,0)`.
- **Normalisation depth** : `depth_norm = clamp(dist / MAX_RANGE, 0, 1)`, `MAX_RANGE = 10.0` (= RADAR_MAX_RANGE
  actuel → cohérent avec le reste du code). Proche = 0, loin/rien = 1.
- **Sémantique apprise (PAS codée)** : rouge≈(0.9,0.3,0.2)=bouffe, bleu≈(0.2,0.5,0.95)=eau (couleurs réelles
  `food_manager.gd:46`, `main.gd:77`). L'agent ne reçoit JAMAIS « c'est de la bouffe » — il l'apprend.

### Obs WM (la seule qui change)
| Bloc | dims | indices |
|------|------|---------|
| proprio | 132 | `[0:132]` |
| **rétine** (remplace radar 12) | **144** | `[132:276]` |
| énergie (normalisée /100) | 1 | `[276]` = `[-1]` |
| **TOTAL obs WM** | **277** | (était 145) |

> `obs policy = 144` (132 proprio + 12 vision-commande) **INCHANGÉ** (cf §0.3).

### §1bis — SÉCURITÉ PHYSIQUE (ne PAS perturber le gait banké hexapod_v2)
La rétine ajoute des colliders dans la scène → risque #1 = perturber la physique que le gait a apprise.
Garde-fou : les ressources perceptibles sont des **`Area3D`** (jamais bloquantes) sur une **couche de
collision DÉDIÉE** (couche 8 = « perceptible-rétine », `1<<7`), `collision_mask=0`. La requête raycast
filtre sur cette couche seule. L'agent ne masque jamais la couche 8 → **zéro interaction physique avec la
locomotion**. Futur objet solide (mur) = StaticBody sur la même couche 8 (vu par la rétine) + sa couche
solide propre (bloque le corps). À vérifier après implémentation : le gait reste identique (test visuel/cap).

---

## 2. LISTE EXHAUSTIVE DES FICHIERS / DIMS À RESYNCHRONISER

> Légende : 🔴 changement obligatoire bloquant · 🟡 changement simple · 🟢 auto/aucun · 🔵 commentaire only.

### Côté Godot
- 🔴 `godot/scripts/agent/perception.gd` : **nouvelle fonction `retina(space_state, origin, forward)`**
  → renvoie `Array[144]` (`[depth,R,G,B]`×36). **Raycast physique** (`intersect_ray`) par rayon (mask couche 8,
  `collide_with_areas=true`), garde le hit le plus proche, lit `collider.get_meta("retina_color")` + depth.
  `food_radar()` **reste** (sonde debug + label oracle, cf §3/§4).
- 🔴 `godot/scripts/world/food_manager.gd` : par pastille, ajouter une `Area3D` (sphère collision ~0.35,
  couche 8, mask 0) + `set_meta("retina_color", _albedo)` → rend food/eau perceptibles par la rétine sans
  toucher la physique du gait (§1bis).
- 🔴 `godot/scripts/main.gd::_compute_vision()` : ajouter un mode rétine pour le **canal WM uniquement** (la
  vision policy/commande ne change pas). Le payload réseau (vers le serveur planner) doit désormais inclure
  `retina:[144]` (cf serveur). Les couleurs food/water sont déjà connues du manager → passer (position,couleur).
- 🔵 `godot/scripts/agent/observation_builder.gd` : `build_observation` renvoie un Dict ; ajouter clé `retina`.
  `PROPRIO_DIM=132` inchangé. (NB carto : la décompo réelle hexapode = 108, mais 132 est la constante de contrat
  validée partout — **ne pas y toucher dans ce scope**.)

### Côté Python — dims
- 🟡 `python/sylvan/constants.py` : ajouter `DEFAULT_RETINA_RAYS=36`, `DEFAULT_RETINA_CHANNELS=4`,
  `DEFAULT_RETINA_DIM=144`. `DEFAULT_VISION_SHAPE=(12,)` **inchangé** (c'est la vision policy).
- 🟡 `python/sylvan/config.py` : `EnvConfig` += `retina_dim:int=144` (consommé seulement par le chemin WM).
- 🔴 `python/sylvan/buffer/wm_dataset.py::_obs_at` (~ligne 52) : remplacer
  `… + list(r["wm"]["radar0"]) + …` par `… + list(r["wm"]["retina0"]) + …` (nom de champ à figer lors de la
  collecte, cf §4). C'est **le seul endroit** qui détermine `obs_dim` → tout le reste suit.
- 🟢 `python/sylvan/models/command_wm.py` : `obs_dim` polymorphe. Le slicing des pertes marche tel quel :
  proprio `[:132]`, **rétine `[132:-1]`** (était radar), énergie `[-1]`. 🔵 MAJ commentaire l.53-54.
- 🟢 `python/sylvan/models/encoders.py`, `heads.py` : `nn.Linear(obs_dim,…)` polymorphe → **aucun** changement
  fonctionnel (option 🔵 renommer `proprio_dim`→`obs_dim` pour la clarté).
- 🟢 `python/scripts/train_wm_command.py`, `eval_wm_command.py` : `obs_dim` auto-détecté / lu de `meta`. RAS.

### Côté Python — symétrie (⚠️ à conditionner, pas « ultra-critique » par défaut)
- 🟡/🟢 `python/sylvan/control/ppo/symmetry.py` : `_VISION` et `_OBS` sont des hardcodes (12 / 144).
  **MAIS** : la policy résiduelle n'est PAS retrainée et ne voit pas la rétine (§0.3). Ce fichier ne doit
  changer **QUE SI** on mirror-augmente un entraînement qui contient la rétine dans l'obs.
  - **À VÉRIFIER (gratuit)** avant de toucher : est-ce que `train_wm_command`/le WM consomment la carte obs de
    `symmetry.py` quand `SYLVAN_MIRROR_COMMAND=1` ? Si NON → ne rien changer pour le MVP (v2 a appris sans).
  - Si OUI/plus tard : carte miroir rétine = par rayon `r → (36 - r) % 36` (PAS `35-r` : il faut `(-r) mod 36`
    pour que rayon0=forward reste fixe), en gardant l'ordre des 4 canaux, **tous signes +1** (depth/RGB ne
    changent pas de signe en miroir). Puis `_VISION=144`, `_OBS=277`, et `self_check()` doit passer.

### Côté Python — planner / serveur
- 🔴 `serve_planner_command.py` : lire `retina = payload["retina"]` (144). Construire
  `wm_obs = proprio + retina + [energy/100]` (277). **Localisation** : remplacer
  `food_xz_from_radar(radar_ema)` par `head(retina) → (food_xz, water_xz, conf)` (§3). EMA : lisser la rétine
  (ou les positions sorties de la tête) au lieu du radar.
- 🟡 `command_planner.py` : `plan()` prend déjà des positions food/water en interne. **Le MPC brute-force
  (~102 candidats, rollout latent, coût `-min_dist + heading_weight·mean_align …`) n'est PAS touché** (cf
  design : Mode-1 en dernier). Adapter seulement la **source** des positions (de l'oracle → tête apprise) et,
  si on passe la tête en amont (serveur), `plan()` peut recevoir `(food_xz, water_xz)` directement au lieu des
  radars bruts. `food_xz_from_radar` **reste** dispo (sonde/label).

---

## 3. DESIGN DE LA TÊTE DE PERCEPTION APPRISE (🅐)

- **Rôle** : remplacer l'oracle géométrique `food_xz_from_radar`. C'est le SEUL endroit où « rouge=bouffe » doit
  émerger.
- **Entrée** : `retina[144]` (mêmes rayons bruts que le WM voit).
- **Sortie** : `[food_dx, food_dz, food_conf, water_dx, water_dz, water_conf]` (6) — positions en **frame agent**
  (mètres, x=droite, z=avant, cohérent avec la convention actuelle de `food_xz_from_radar`), + confiance/présence
  ∈ [0,1] (sigmoïde) qui joue le rôle du `None` actuel (conf basse = pas de ressource).
- **Archi** : petit MLP (144 → 128 → 128 → 6), CPU, négligeable (tourne 1×/replan dans le serveur).
- **Où elle vit** : `python/sylvan/models/perception_head.py`, chargée par `serve_planner_command.py` depuis un
  checkpoint dédié `data/checkpoints/retina_head/`. **Indépendante du WM** → on peut la tester seule (gating §5).
- **Entraînement (supervisé, label = oracle OFFLINE)** : cible = sortie de `food_radar`/positions vraies au
  moment de la collecte. **Honnêteté (§2)** : l'oracle ne sert QUE de label hors-ligne ; à l'éval il est
  **débranché** (`assert` aucun signal oracle dans le chemin live) et on mesure l'erreur de position. La capacité
  (localiser depuis la couleur brute) est réellement apprise et réellement testée — ce n'est pas un relâchement
  de critère.
- **Perte** : MSE position (masquée par présence) + BCE présence. Held-out par épisode.

---

## 4. PLAN DE RE-COLLECTE + RETRAIN WM (gaté)

> **Régime PROPRE hexapode obligatoire** (CLAUDE.md) : `SYLVAN_CPG=1 RESIDUAL_GAIN=0.4 TURN_FADE=0
> FOOT_FRICTION=7 CPG_SPEEDCAD=0.6 CPG_PERIOD=0.5`, vx 0.55–0.75. Garder discipline JEPA (cosine + VICReg).

1. **Logger la rétine pendant la collecte, À CÔTÉ de l'oracle.** Champ JSONL `wm.retina0`/`wm.retina1`
   (t et t+1, comme `radar0/radar1`), + garder `radar0` comme label tête + sonde. Un seul format, figé ici.
2. **Étage 1 (CHEAP) — tête seule** : à partir d'un petit set (~20–40 ép suffisent pour un MLP de localisation),
   entraîner `perception_head` (minutes). **GATE** : MAE position bouffe < ~0.5 m sur held-out, présence AUC > 0.9.
   Tant que ce gate n'est pas passé, **on ne paie pas le WM**.
3. **Étage 2 (CHER) — WM rétine** : re-collecte complète régime propre (même volume que `wm_hex_v2`, p.ex.
   2×N ép, 2 seeds), `train_wm_command` (obs_dim 277 auto) avec cosine + VICReg → `wm_command_hex_retina_v1`.
   Éval open-loop (`eval_wm_command --horizons 50 80 100 150`) AVANT tout closed-loop.
4. **Étage 3 — closed-loop** : brancher tête + WM rétine dans le serveur, lancer le jalon §5.

Artefacts à créer (sur le gabarit existant) : `collect_wm_retina.sh` (= `collect_wm_hex_v2.sh` + rétine),
`train_retina_head.py`/`.sh`, `diag_retina.sh` (sanity Godot + MAE tête), réutiliser `run_forage_hex.sh`.

---

## 5. JALON FALSIFIABLE + CRITÈRES SUCCÈS/KILL + GATING

> **Jalon** : « l'agent navigue jusqu'à la bouffe en utilisant UNIQUEMENT les rayons couleur bruts, sans qu'on
> lui dise jamais où elle est » (oracle débranché, `assert` au runtime).

| Étape | Coût | SUCCÈS (continuer) | KILL / STOP (escalade, pas de tweak à l'aveugle) |
|------|------|--------------------|--------------------------------------------------|
| 0. Rétine Godot + sanity | **gratuit** | rayon sur bouffe → R>0.7,G/B bas, depth≈vraie dist (±0.1) ; eau → bleu ; miss → depth=1,RGB=0 | un rayon ne récupère pas la bonne couleur/dist → bug géométrie, corriger avant de continuer |
| 1. Tête seule (supervisé) | cheap | MAE pos bouffe **< 0.5 m**, présence AUC **> 0.9** (held-out, oracle débranché à l'éval) | MAE > ~1 m après convergence → la rétine 36×4 ne porte pas l'info → repenser layout (densité rayons/canaux) AVANT le WM |
| 2. WM rétine open-loop | cher | open-loop pos **≤ ~0.3 m @100** (réf v2≈0.21, jepa2≈0.21) ; eff_rank latent ne s'effondre pas (>~15) | open-loop ≫ baseline OU eff_rank→effondré (perte latente vacante) → diagnostiquer (diag_jepa) avant closed-loop |
| 3. Closed-loop foraging | cher | A→B ≥ ~80 % à 2–4 m **avec rétine seule** ; foraging survie ≥ baseline radar (~990, éco de vie ~2745) | cale loin de la bouffe alors que la tête localise bien (étape 1 OK) → c'est le WM/planner, PAS la perception → localiser, ne PAS élargir eat_radius (§2 fausse solution) |

**Gating** : chaque étage chère est derrière une mesure pas-chère pré-enregistrée (0→1→2→3). On ne descend
jamais d'un cran sans le SUCCÈS du cran précédent.

**Garde-fous §2 (ne pas masquer)** : (a) oracle **débranché** à l'éval, vérifié par `assert` ; (b) critères
ci-dessus **fixes**, on ne gonfle pas eat_radius ni les tolérances pour « faire passer » ; (c) si un échec
contredit une autre mesure (ex. tête OK mais foraging KO), on ne conclut pas « plafond perception » — on
localise la vraie cause.

---

## 6. RISQUES / INCONNUES + VERSION MINIMALE VIABLE

### Risques
- **R1 — pas de colliders/raycast** (§0.1) → on les construit. Mitigation perturbation gait : couche dédiée +
  Area3D non-bloquantes (§1bis). À VÉRIFIER post-impl : cap/vitesse du gait inchangés.
- **R2 — 36 rayons trop grossiers** pour localiser finement (1 cible peut tomber entre 2 rayons à distance).
  Détecté gratuitement à l'étape 1 (MAE). Mitigation : ↑ rayons (48/72) ou ajouter canal « bearing intra-rayon ».
- **R3 — couleurs trop séparables** (rouge vs bleu purs) → la tête « triche » sur 1 canal et n'apprend pas une
  vraie représentation. Acceptable au MVP (le but est localiser), mais le noter ; complexifier le monde plus tard.
- **R4 — symmetry.py** : ne le toucher qu'après avoir vérifié qu'un chemin d'entraînement consomme l'obs rétine
  (sinon changement inutile et risque de casser le self_check). Cf §2.
- **R5 — coût MPC** : inchangé (rétine encodée 1× par replan, rollout reste en latent) → temps réel préservé.

### Version MINIMALE viable (à tester EN PREMIER, gratuite)
**Étage 0 seul** : implémenter `perception.gd::retina()` (raycast physique) + colliders/meta couleur sur
food/eau + sanity numérique (pastille épinglée droit devant à distance connue → rayon 0 lit depth≈d/10 +
couleur rouge ; miss → depth=1, RGB=0 ; gait inchangé). **Aucun entraînement.** Si ça passe → étage 1 (tête
seule, cheap) qui gate tout le reste.

---

## 7. ORDRE D'EXÉCUTION PROPOSÉ (après validation de ce scope)
0 (gratuit) rétine Godot + sanity → 1 (cheap) tête seule + gate MAE → 2 (cher) WM rétine open-loop →
3 (cher) closed-loop jalon. Aucun retrain moteur. MPC intact. Mode-1 et plan-en-latent (🅑) **plus tard**.
