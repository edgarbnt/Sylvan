# Sylvan — État des lieux (handoff propre, 2026-06-17)

> Doc de reprise. Lis-le en premier, puis la mémoire `sylvan-rearchitecture.md` (détail/journal complet)
> et `sylvan-locomotion-rl-knowledge.md` (la saga moteur). Remplace la version du 2026-06-14
> (antérieure au pivot HEXAPODE, à la re-validation du WM et au foraging hexapode).

## ⭐ ÉTAT COURANT (2026-07-02) — lire AVANT le reste
**Chantier actif = BASCULE Mode-1 → Mode-2.** Le foraging mono-drive marche ; le mur restant est
l'**arbitrage multi-pulsions**. On a construit **Mode-1** (politique réflexe apprise, model-free PPO,
branche `mode1-build`) et **mesuré** qu'une politique RÉACTIVE plafonne au niveau BC (~1900) : le mur est
le **look-ahead** (planner look-ahead 2300 vs clone réactif 1930 ; 96% des morts = décision, corps
réfuté). **→ Pivot : planifier dans le WM avec une VALEUR DE SURVIE APPRISE sur le latent** (remplace le
coût codé-main du planner) = le Mode 2 JEPA-pur. **Prochain = gate GRATUIT de faisabilité** (la
valeur-survie sépare-t-elle bon/mauvais arbitrage sur les rêves ?) avant tout retrain.
**Lire : `docs/design_mode1_pivot_mode2.md` (le pourquoi) + `memory/sylvan-mode1-build.md` (l'arc).**

---

## 0. Le but (north star)
Une **ALife émergente dans un World-Model type JEPA** : une entité qui **décide elle-même**
(a faim → cherche → va vers la bouffe → survit), par **planification dans un modèle du monde appris**.
La locomotion n'est qu'un **prérequis**. Voir `BLUEPRINT.md` / `JEPAConcept.md`.

---

## 1. L'architecture en 3 couches (tournant 2026-06-10, corps = HEXAPODE depuis 2026-06-17)
On a abandonné la locomotion par PPO reward-shapé. Le mur récurrent = le **virage contrôlé / la nav**.

```
┌─ JEPA (cerveau) : planifie en ESPACE DE COMMANDE (vx, ω) ────────┐  ← Phases 4-5 ✅ FAIT (hexapode)
│  WM imagine la nav ; planner MPC horizon-glissant → foraging      │
└──────────────────────────────────────────────────────────────────┘
            │ émet (vx, ω)
┌─ RÉSIDU PPO (borné) : performance + équilibre ──────────────────┐  ← ✅ (hexapod_v2)
└──────────────────────────────────────────────────────────────────┘
            │ s'ajoute aux cibles du CPG via le servo PD
┌─ CPG (codé main, zéro apprentissage) : marche + virage PAR CONSTRUCTION (trépied 6 pattes) ┐  ← ✅
└──────────────────────────────────────────────────────────────────┘
```
**Idée-clé** : le virage est de la CINÉMATIQUE (par construction), pas un gradient de reward. Le CPG est
un tuteur modulable. Règle permanente : **fiabilité/résultat d'abord**.

**Pourquoi hexapode** : après l'échec exhaustif du virage agile sur le quad puis la salamandre (cf §4),
on a changé de NATURE → **hexapode démarche trépied (cafard)** : 6 pattes = stabilité gratuite → l'énergie
va à la propulsion, on crank sans chute. C'est ce qui a débloqué la VITESSE (3× le quad).

---

## 2. Où on en est — BASE + CERVEAU vivants sur l'HEXAPODE ✅

| Brique | État | Checkpoint vivant |
|---|---|---|
| CPG hexapode trépied (marche + virage par construction) | ✅ | (code) |
| Résidu : gait rapide (0.49 m/s), symétrique, 0% chute | ✅ | `hexapod_v2/policy_best.pt` |
| WM commande-space (imagine la nav de l'hexapode) | ✅ | `wm_command_hex_v2/wm_best.pt` |
| Planner MPC (foraging émergent) | ✅ FONCTIONNEL (modeste) | (search, pas un réseau) |

Dims hexapode : **proprio=132, action=18, obs(policy)=144, obs(WM)=145**.
**Base motrice (`hexapod_v2`)** : marche 0.49 m/s à vx0.7, cap droit, virages symétriques, fluide, 0% chute.
Config à servir/collecter (LE régime propre) : `SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0
SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 SYLVAN_CPG_PERIOD=0.5`. Vitesse pilotable par vx (0.19@vx0.3 →
0.49@vx0.7) ; **plage propre vx~0.5-0.75** (en dessous il dérive, cf §3).

---

## 3. PHASE 4+5 RE-FAITES SUR L'HEXAPODE (2026-06-17) — forager FONCTIONNEL, re-bute sur le mur moteur

L'ancien WM (`wm_command_v2`, quad obs=107) était périmé pour l'hexapode. Refait :
- **WM re-collecté + retrain** → `wm_command_hex_v2`. **DÉCOUVERTE CLÉ** : il faut collecter le WM dans le
  **RÉGIME PROPRE du corps**. hexapod_v2 n'exécute proprement (vx,ω) qu'à **vx~0.7 + CPG_PERIOD=0.5** (cap
  droit, tourne des 2 sens) ; à vx0.3-0.6 il DÉRIVE fort. Le babbling de collecte était codé en dur vx0.3-0.6
  (le mauvais régime) → WM v1 instable. **FIX** : babbling overridable (`SYLVAN_WM_VX_MIN/MAX/WMAX`), grille
  vx planner alignée (0.55-0.75). WM v2 : éval held-out yaw @100 = 10.5° (vs 29° v1), pos @50 = 0.12m. Jalon ✅.
- **Foraging re-testé** (`run_forage_hex.sh`, planner horizon 80 + résidu hexapod_v2 + WM v2) : survie médiane
  **810 @eat_radius 1.5** (1 survivant/12 plein 1500 pas), **l'agent perçoit → planifie → navigue → MANGE**
  (mécanisme north-star intact, sans nav codée). MAIS = **~le niveau de l'ancien quad, pas au-delà.**
- **CONCLUSION (re-confirme le diagnostic historique)** : le goulot N'EST PAS le WM/perception/représentation
  (le WM propre n'a rien transformé) — c'est le **MOTEUR** : (1) approche terminale (orbite à ~1.5m sans
  refermer le dernier mètre, rayon braquage vx/ω >> eat_radius), (2) réorientation hors-axe trop lente.

---

## 4. LE MUR DU VIRAGE AGILE — le plus profond du projet (à NE PAS sous-estimer)
Avant de retenter quoi que ce soit côté virage, lire `sylvan-locomotion-rl-knowledge.md` EN ENTIER. Résumé :
le virage **agile EN AVANÇANT** (>>15°/s sans perdre la vitesse) a **résisté à TOUTES les méthodes** : PPO
conservateur (~14°/s), PPO spin, **fully-learned BC+finetune** (8-13°/s, asymétrique), **SAC off-policy**
(toutes variantes), **tous les params corps** (friction/géo/damp/cadence/servo, retrain inclus), et **2
redesigns** (salamandre, hexapode). **Le corps PEUT tourner** (couple direct → **1000°/s**, pics pattes
126-180°/s) → c'est 100% un problème de **CONTRÔLE/APPRENTISSAGE** : aucun gait appris ne SOUTIENT le lacet
en marchant en avant ; le RL échange toujours le virage contre la stabilité/vitesse. Ce qui MARCHE = le
**« pivote-puis-avance »** (CPG owns le virage, perd l'avance) via découplage (`SYLVAN_TURN_FADE` sur la
salamandre) ou le virage symétrique modéré de l'hexapode. **Leviers de fond restants (décision owner)** :
migrer sur Isaac Gym (4096 envs — mais GPU = AMD RX5700XT → CUDA impossible), OU accepter ~15°/s + le planner
compose (pivote-puis-fonce). **Levier jamais tiré SPÉCIFIQUEMENT sur l'hexapode** : un curriculum de commande
ω progressif dédié (les runs hexapode v1-v6 visaient la vitesse droite/symétrie). = le seul « coup unique
informé » avant de trancher la voie structurelle. **NE PAS enchaîner des runs aveugles (erreur du saga résidu).**

---

## 5. Comment lancer / tester (HEXAPODE)
Binaire Godot **chemin absolu** : `./tools/godot/godot`. Python : `env_pytorch_3.12/bin/python`. Headless : `--headless`.

**Voir la base motrice** : `bash voir_salamandre_cerveau.sh 0.7` (sert hexapod_v2, ω en arg ; serveur auto-kill).
**Foraging headless quantitatif** : `bash run_forage_hex.sh [eat_radius=1.0] [horizon=80] [episodes=12]`
(démarre planner+WM v2+résidu hexapod_v2, N épisodes, cleanup ; métrique = survie via `[Godot] Episode|Step|Energy`).
**Re-collecter le WM** (régime propre) : `bash collect_wm_hex_v2.sh <run-prefix> <episodes> <seed>`.
**Entraîner le WM** : `PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.train_wm_command --runs
data/replay_buffer/wm_hex_v2_a data/replay_buffer/wm_hex_v2_b --out data/checkpoints/<name> --epochs 20`
(CPU obligatoire). Éval : `scripts.eval_wm_command --checkpoint .../wm_best.pt --horizons 50 80 100 150`.
**Entraîner un résidu** (PPO) : `scripts.train_ppo` avec les env hexapode ci-dessus (§2) +
`SYLVAN_CPG_SAMPLE_CMD=1 SYLVAN_REWARD_OBJECTIVE=…`, `--init-from …/hexapod_v2/policy_best.pt --num-workers 8`.
Symétrie hexapode : `ppo/symmetry.py` (carte miroir 18/132 reconstruite, self_check OK).

⚠️ **Piège shell** : une commande avec env multi-lignes a besoin de `\` en fin de ligne (sinon chaque ligne
est une commande vide et l'env n'est pas passé au python). Préférer un script `.sh` avec `export`.

---

## 6. Pièges / leçons (à garder)
- **Lancer un train = la commande python SEULE backgroundée** (un préambule de `kill` dans la commande
  backgroundée fait exit1). **Tuer un train** : `kill -9` + `pkill -9 -f serve_ppo_collect` + `pkill -9 -f
  'godot --path godot'` + **VÉRIFIER** 0 restant (sinon serveurs orphelins busy-spinning → thrash).
- **GPU AMD = HIP crash** → entraînement **sur CPU obligatoire** (WM et PPO).
- **Collecter le WM dans le RÉGIME PROPRE du corps** (pas une plage arbitraire) — sinon le WM apprend la
  dérive et le planner pilote dans la zone instable (leçon 2026-06-17).
- **lr 1e-4** stable ; surveiller KL (<0.1) et `std` (s'il MONTE = ne converge pas → tuer).
- **Ne PAS juger un entraînement avant convergence** (WM long-horizon mauvais à mi-parcours puis se corrige) ;
  mais tuer tôt si divergence claire (fwd_vel négatif + std qui monte).
- **Mesurer le virage TOUJOURS avec un résidu servi** ; l'open-loop avec un ANCIEN résidu ne valide PAS un
  nouveau mécanisme (il l'annule) → réentraîner.
- **JAMAIS de glob large avec `rm`** (`*cr*` a matché `scripts`/`critic` → suppression massive le 2026-06-16).
- Erreur bénigne `get_locomotion_metrics Array→Array[float]` : pré-existante, ignorer.

---

## 7. Fichiers clés
- **Corps + CPG** : `godot/scripts/agent/sylvan_agent.gd` (hexapode trépied ; `cpg_reference` ; servo PD).
  Action 18-d (6 pattes × 3). Proprio 132.
- **Boucle env / commande / planner / collecte WM** : `godot/scripts/main.gd` (babbling WM overridable
  `SYLVAN_WM_VX_MIN/MAX/WMAX` ; mode planner `SYLVAN_CPG_PLANNER=1`) ; perception `agent/perception.gd`
  (`food_radar`) ; bouffe `world/food_manager.gd` (`eat_radius`, défaut 1.0).
- **Reward** : `godot/scripts/rl/reward_manager.gd` (`locomotion_omni_v1` pour l'hexapode + autres).
- **WM** : `python/sylvan/models/command_wm.py` (`CommandWorldModel`) ; dataset `buffer/wm_dataset.py`.
- **Planner** : `python/sylvan/control/planning/command_planner.py` (grille vx 0.55-0.75) ; serveur
  `scripts/serve_planner_command.py` (action_dim dynamique).
- **PPO** : `python/sylvan/control/ppo/` + `scripts/train_ppo.py` ; symétrie `ppo/symmetry.py` (18/132).
- **Scripts racine** : `voir_salamandre_cerveau.sh` (voir base), `run_forage_hex.sh` (foraging),
  `collect_wm_hex_v2.sh` (re-collecte WM), `tools/eval_ckpt.sh` (éval virage — ⚠️ défauts quad, adapter
  pour l'hexapode : pas de turn_amp/spineturn, ajouter FOOT_FRICTION=7 SPEEDCAD=0.6).
- **Mémoire** (`.claude/.../memory/`) : `sylvan-rearchitecture.md` (DÉTAIL/journal), `sylvan-jepa-stance.md`,
  `sylvan-locomotion-rl-knowledge.md` (LA saga moteur — lire avant tout retrain virage),
  `sylvan-training-perf.md`, `sylvan-blueprint-direction.md`.

---

## 8. PROCHAINE ÉTAPE (reprendre ici)

### ⭐ MAJ 2026-06-19 — 🅑 (planifier dans le latent) : WM FOOD-AWARE FAIT, MUR = LA RECHERCHE → PROCHAIN = CEM
> **Lire `memory/sylvan-retina-decision.md` (à jour) + `docs/scope_guided_search.md` (le plan du prochain chantier).**

**Parcours 🅑 de la session (tout par diagnostics GRATUITS, discipline CLAUDE.md §1/§2) :**
1. Le « feu vert 🅑 énergie (corr −0.46) » était une **fausse piste** (un proxy corrélé ≠ le BUT) — réfuté par un test
   DIRECT : le candidat argmax-énergie n'allait pas vers la bouffe mieux que le hasard.
2. **Cause-racine** (probe eat-fidelity) : le WM ne prédisait que **4 % de la bosse-repas** (23 repas / 18k lignes).
3. **Fix data** : mécanique de collecte EAT-RICHE (`SYLVAN_FOOD_HUNGER_MAX`, eat gated sur la faim → repas en basse
   énergie avec vraie marge) → **323 repas** (`retina_eat_a/b`) + retrain. → Toujours non au gate.
4. **Vrai diagnostic** : le latent **teacher-forced** PORTE la bouffe (tête valeur AUC 0.79) mais le latent **RÊVÉ**
   (open-loop, ce que le planner utilise) la PERD → le rêve propageait le mouvement, pas la nourriture.
5. **FIX FAIT** : perte auxiliaire « repas imminent » sur les **latents rêvés** (`train_wm_command --w-food`, méthode
   `dream_latents`, tête `food_head` non sauvée ; warm-start `--init-from`) → **`wm_command_hex_retina_eat_v2`** :
   le rêve garde la bouffe (food_auc 0.84), dynamique intacte (déplacement 0.010, eff_rank 27, open-loop JALON ✅).
6. **MUR FINAL localisé (≠ le latent)** : un readout latent (énergie ou valeur) ne classe les candidats que ~⅓ aussi
   bien que la géométrie, car la **grille d'arcs (vx,ω) fixe** ne produit AUCUN arc qui atteint la bouffe en un seul
   rêve (tous ~1.7 m, 0 % <1 m) → presque rien à classer. Muscler w_food n'aide pas (1.5 PIRE que 0.5). La tête de
   valeur échoue en plus (apprise sur vraies commandes, ne généralise pas aux arcs candidats hors-distribution).

**→ PROCHAIN CHANTIER = RECHERCHE GUIDÉE (CEM/gradient à travers le WM différentiable)** au lieu de la grille d'arcs.
C'est LE débloqueur de 🅑 (intermédiaire prévu avant Mode-1). Plan détaillé + critères : **`docs/scope_guided_search.md`**.

**Assets bankés** : `wm_command_hex_retina_eat_v2` (food-aware), `value_head_food`, outils `diag_latent_foodaware.py`,
`diag_value_direct.py`, `diag_critic_feasibility.py`, `train_value_head.py`, `run_food_aware_retrain.sh`,
`collect_wm_eatrich.sh`. **LIVE inchangé = planner-COORDONNÉES + `wm_command_hex_v2`** (north-star atteint, marche).
**Nettoyage repo parké** (« go nettoyage » = git init + suppr ~11 .sh obsolètes + élaguer `.claude/settings.local.json`).

---

### (antérieur) MAJ 2026-06-18 (nuit-2) : RÉTINE LIVRÉE ✅ → CAP = LATENT 🅑
> **MAJ 2026-06-18 nuit-2 — LA RÉTINE EST FAITE (perception 100 % apprise, oracle radar mort).** Détail complet
> + commandes : `docs/scope_retina.md` et `memory/sylvan-retina-decision.md` (LIRE en premier). Résumé :
> Étage 0 (rétine raycast physique 36×[depth,RGB], `perception.gd::retina`, colliders couche dédiée) → Étage 1
> (tête apprise `data/checkpoints/retina_head/`, attention géométrique, foraging à parité) → Étage 2 (le WM
> CONSOMME la rétine, obs 277, `SYLVAN_WM_USE_RETINA=1`). Latent ENRICHI : `wm_command_hex_retina_jepa_v2`
> (eff_rank 3.6→**14**, open-loop 0.22 m@100, **foraging 980 ≥ oracle 965**). **Feu vert 🅑** : énergie prédite
> FOOD-AWARE (corr −0.46, `diag_latent_foodaware.py`). **PROCHAINE = coût latent 🅑** dans `command_planner.py`
> (score = énergie future prédite + survie, SANS coordonnées ; commencer hybride). Outils : `run_forage_retina.sh`
> (`WM_CKPT=`, `--retina-head`), `collect_forage_retina.sh`, `train_retina_head.py`, `diag_latent_foodaware.py`.
> Schéma JEPA×Sylvan complet : `docs/schema_jepa_sylvan.md` (+ 11 PNG). Process tous tués (clean). — la suite ci-dessous = contexte antérieur.

### (antérieur) MAJ 2026-06-18 (nuit) : A→B + 2ᵉ PULSION + JEPA FAITS → CAP = RÉTINE
**Le vrai goulot du foraging N'ÉTAIT PAS le moteur (révise les §3/§7) — c'était l'ENGAGEMENT du planner.
Établi par diagnostic gratuit, corrigé sans entraînement. Voir `memory/sylvan-ab-navigation-fix.md`.**
- **Diagnostic A→B GRATUIT** (`diag_nav_ab.sh` : cible unique fixée à un azimut donné, homéostasie OFF, on mesure
  l'approche min ; parser `parse_nav_ab.py`) : A→B PAS solide (37 %), échec STRUCTURÉ — **devant OK, arrière/droite
  n'engagent JAMAIS le virage** (fil-du-rasoir, le bruit physique fait basculer 0.2 m ↔ 4 m). Cause PROUVÉE = ni
  moteur (arrivées à 0.04 m), ni perception (radar 360°), mais le **gradient de virage quasi-nul dans le coût du
  planner** (au poids 0 il tournait à l'opposé des cibles à droite = bug d'imagination min_dist du WM).
- **Fix task-agnostic, ZÉRO entraînement** (`command_planner.py`) : terme d'alignement-vers-la-cible au score
  `heading_weight * mean( cos(bearing) * clamp(dist/gate,0,1) )`, gate de distance = s'estompe près de la cible
  (sinon orbite). Banké : `heading_weight=2.0`, `heading_far_gate=2.0`. Env : `SYLVAN_PLANNER_HEADING_W`.
  §14 INTACT (la bouffe reste seulement dans le coût du planner).
- **Résultats** : A→B **2 m 88 % / 4 m 94 %** (plage opérationnelle solide) ; 6 m 62 % = arête connue (spirale).
  **Foraging réel A/B (`forage_ab.sh`) : survie médiane 610 → 990 (+62 %)** — sans le fix il famine 9/12 sur place.
- **2ᵉ PULSION (soif+eau) LIVRÉE — étage 1 (2026-06-18)** : arbitrage homéostatique ÉMERGENT (pas codé) validé
  11/12 au diag gratuit, **WM inchangé (145), zéro entraînement** (eau/soif = inputs planner-only, dynamique
  analytique dans le coût urgence). Voir `memory/sylvan-second-drive-arbitration.md` + `docs/design_second_drive.md`.
  Diag/parser : `diag_arbitration.sh` / `parse_arbitration.py`. Env clés : `SYLVAN_WATER_COUNT/ANGLE_DEG/...`,
  `SYLVAN_INIT_ENERGY/THIRST`, `SYLVAN_PLANNER_URGENCY_W`.
- **JEPA-IFICATION FAITE (étapes 1+2, 2026-06-18) → WM fonctionnellement JEPA.** Voir `memory/sylvan-jepa-stance.md`,
  `docs/jepa_*_criteria.md`. Diag gratuit `diag_jepa.py` : le WM `hex_v2` était EFFONDRÉ (eff_rank latent 6/128) → la
  perte latente était vacante → c'était un Dreamer. Étape 1 (`train_wm_jepa.sh`, VICReg+cosine) → `wm_command_hex_v3_jepa`
  eff_rank 6→26. Étape 2 (`train_wm_jepa2.sh`, drop proprio/radar, latent×5) → `wm_command_hex_v3_jepa2` : eff_rank tient
  21 SANS reconstruction, open-loop pos **0.21 m@100 (meilleur du projet)**, closed-loop A→B 12/16 (parité). **PAS
  PROMU live** (foraging équivalent, pas meilleur) — `hex_v2`+`wm_command_hex_v2` restent les checkpoints vivants.
- **FORAGING : vrai goulot = ÉCONOMIE DE SURVIE, PAS la précision ni le moteur** (`memory/sylvan-foraging-economy.md`).
  Deux conclusions hâtives RÉFUTÉES (cf CLAUDE.md §2 ⭐ nouveau « ne pas masquer ») : « agrandir la bouche » (fausse
  solution) et « plafond moteur » (faux : `diag_approach.sh` → il MANGE une cible bien placée 89% à eat_radius 1.0).
  Vraie cause = `passive_energy_drain=0.15` est un réglage COLLECTE-DE-DONNÉES → mort à ~660 pas → bouffe à 7 m
  inatteignable. Fix légitime : `SYLVAN_ENERGY_DRAIN`/`SYLVAN_THIRST_DRAIN` réglables ; valeur « de vie » ~0.05
  → survie 850→2745, repas 0.75→3.12 (`diag_drain.sh`). Le 0.15 reste pour la collecte WM.
- **MULTI-PULSIONS EN SURVIE LIBRE (`diag_multidrive.sh`, éco de vie)** : survie médiane **2075**, jongle faim+soif
  6/10, équilibre parfait 2/10 — **arbitrage émergent RÉEL mais pas robuste** (laisse souvent une pulsion crasher).
  Cause = planner MYOPE (horizon court, glouton sur l'inconfort immédiat, pas de foresight), PAS un manque
  d'intelligence. Vrai fix = critique appris / horizon long (plus tard), pas un patch.
- **⭐ PROCHAINE ÉTAPE CHOISIE = LA RÉTINE (perception apprise)** — voir `docs/design_retina.md` (à scoper en détail).
  Raison : le radar est un ORACLE (il souffle la position de la bouffe) → tant que c'est le cas, « JEPA » est un titre,
  pas une réalité. La rétine fait apprendre la perception depuis le brut = le vrai cœur JEPA + la plus grosse dette
  d'honnêteté. **Technique choisie : rétine RAYCAST PROFONDEUR+COULEUR (1D)** — ~36 rayons sur 360° × (depth + RGB) ≈
  144 dims, qui REMPLACENT le radar-oracle ; l'agent doit APPRENDRE rouge=bouffe/bleu=eau + localiser. Pas de pixels
  (CNN trop lourd/data-hungry, et inutile à ce stade). Sous-décision : 🅐 tête de perception APPRISE (rayons→position
  estimée, remplace `food_xz_from_radar`) d'abord, 🅑 planifier en LATENT (cost sur l'état latent) plus tard.
  **NE PAS toucher au MPC brute-force (120 itérations, déjà en latent) maintenant** : il sera remplacé par Mode-1
  (politique apprise + recherche en fallback) EN DERNIER, une fois le système mûr. ORDRE : rétine → décision
  foresighted (critique) → Mode-1. Jalon falsifiable rétine = « navigue jusqu'à la bouffe avec UNIQUEMENT les rayons
  couleur bruts, sans oracle ». Démarrer par un SCOPE GRATUIT (design, dims, ce qui change, plan de re-collecte) avant
  tout entraînement.
- (Différé) 3ᵉ pulsion (danger/évitement), tuning arbitrage (urgence+hystérésis), curiosité (pulsion intrinsèque).
