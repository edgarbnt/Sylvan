# ✅ RÉSOLU (close) — 🅑 closed-loop : l'agent CLOSE et MANGE (hypothèse TRANSFERT validée)

> **2026-06-21 : close CORRIGÉ.** L'hypothèse « transfert teacher-forced→rêvé » (hyp. 1 ci-dessous) était la
> bonne. Fix = réentraîner la value sur les latents **RÊVÉS multi-pas** (`value_head_food_dream`) + agrégat
> **`mean`**. A/B propre (même WM symétrisé, seule la value+agg changent) : teacher-forced+max = **0 repas/4 ép
> (ne close jamais, min food_d 1.5+)** → value-rêve+mean = **3 repas/4 ép (close à 1.02 m puis MANGE)**. Le WM
> n'était PAS le problème (il rêve le contact à 0.01 m), c'était bien la **couche rapide** (value), conforme à
> l'archi substrat/pulsions (CLAUDE.md §3). Config gagnante câblée dans `run_forage_latent.sh` (WM_sym +
> `value_head_food_dream` + `SYLVAN_VALUE_AGG=mean`). Outils du fix : `train_value_head.py SYLVAN_VALUE_DREAM=1`
> (entraîne sur `rollout_open_loop` profondeurs variées, commandes exécutées, labels honnêtes) ; gate du close =
> `diag_value_direct.py` avec `SYLVAN_DIAG_DMIN/DMAX` (close rang 0.65→0.08) ; `command_planner.plan_latent` a un
> switch `SYLVAN_VALUE_AGG` (max|mean).
>
> ## ⚠️ CE QUI RESTE (problèmes SÉPARÉS, pas le close — ne pas les confondre avec ce bug)
> - **Survie** : l'agent finit par mourir car taux de repas < métabolisme (drain par défaut). Knob = 
>   `SYLVAN_ENERGY_DRAIN` (~0.05 « de vie », cf `memory/sylvan-foraging-economy.md`). À régler/valider ensuite.
> - **Engagement demi-tour = MUR SUBSTRAT (diagnostiqué 2026-06-21, gratuit).** Bouffe non-frontale : front OK,
>   côté faible-mais-OK (turn gradient correct, engage lentement), **DERRIÈRE cassé** (0 repas). RACINE PROUVÉE :
>   le **rêve open-loop du WM est AVEUGLE à l'acquisition-par-virage** — quand on rêve un demi-tour, la cible
>   DISPARAÎT du latent rêvé (value=0 ET orient.ahead reste négatif alors que la géométrie amène la cible à 19°
>   devant ; vrai à TOUS horizons, même courts). Cause = données WM biaisées « cible-devant » (planner-coord fait
>   toujours face) → dynamique virage-acquisition hors-distribution. **PAS fixable par le cost/heads** (rêve aveugle).
>   FIX = enrichir le SUBSTRAT : recollecter des données riches en VIRAGES/scan près des ressources + retrain WM
>   (warm-start, + `--w-rollout` sur trajectoires tournantes) → le rêve apprend « tourner fait apparaître la cible ».
>   Capacité GÉNÉRALE visée = « rêver se déplacer vers un point dans N'IMPORTE QUELLE direction » (pas food-spécifique).
>   INFRA PRÊTE (dormante, s'activera quand le rêve saura tourner) : `OrientHead` (value_head.py) = cap latent appris
>   (devant/derrière 80%), `train_orient_head.py`, terme `SYLVAN_ORIENT_W` dans `plan_latent` (défaut 0 = off).
>   Probes du diag : engagement par azimut + `orient.ahead` vs géométrie le long d'un demi-tour.
>   • **RAFFINEMENT 2026-06-21 (gate cheap AVANT retrain — a évité un retrain inutile)** : « la donnée manque de
>     virages » = **FAUX**. `retina_forage` (planner) a DÉJÀ plus de virages/croisements derrière↔devant (|ω|~0.36,
>     ~5 crois./ép) qu'un babbling neuf (|ω|~0.13). Donc enrichir par collecte babbling N'AIDERA probablement PAS.
>     Le vrai obstacle = **fidélité du rêve OPEN-LOOP en rotation** (rêver un long demi-tour diverge ; teacher-forced
>     vs open-loop = bruité/non-concluant, corr ~0.2 les deux) = problème DUR de world-model, pas un « collecte plus ».
>     ⟹ payoff du retrain INCERTAIN. Alternative robuste = **turn-to-scan** (contourne le rêve : tourner → la rétine
>     RÉELLE ré-acquiert → pipeline 'devant' validé engage). Outils du gate : `collect_wm_turning.sh` + analyses azimut/
>     croisements/amplitude-par-fenêtre + corr orient.ahead(TF vs OL).
> - **Confiance** : A/B sur 4 ép / 1 seed. Le contraste 0 vs 3 + probes statiques (close 40/40 à 0.10 m) est
>   solide, mais plus d'épisodes/seeds le confirmeraient.
>
> ---
> ## ARCHIVE — description du bug quand il était ouvert (2026-06-19) :

## SYMPTÔME
Le foraging 🅑-pur (coût-VALEUR latent, **coordonnées débranchées**) : l'agent **navigue vers la bouffe**
(engagement OK, biais directionnel corrigé) **mais ne close pas le dernier mètre** → **0/8 mange**, meurt de
faim. `food_d` min ≈ 1.0–1.5 m (eat_radius 1.0) ; meilleur cas ep4 = **1.01 m** (frôle à 1 cm, sans manger).
Le live planner-COORDONNÉES, lui, mange (survie ~900) car il a la géométrie exacte.

## CE QUI MARCHE (établi par tests — NE PAS refaire / NE PAS soupçonner)
- **WM `data/checkpoints/wm_rich_fidele_sym/wm_best.pt`** : riche (eff_rank ~20), **fidèle** (cos rêve↔réel
  0.94@40, 0.93@250), food-aware, **SYMÉTRISÉ** (augmentation miroir). Le rêve **ATTEINT la bouffe** :
  best_possible **0.02 m** au close, **92 % des candidats atteignent <1 m** dans le rêve. → **LE WM N'EST PAS LE
  PROBLÈME.**
- **Value head `data/checkpoints/value_head_food/value_best.pt`** (K=20, AUC 0.80) : a du **signal** (écart logit
  3.95 au close, **0 % saturés**).
- **Symétrie** : biais gauche corrigé (béquille d'inférence `plan_latent` + augmentation miroir dans le WM).
  `plan_latent` tourne du bon côté des deux côtés (validé : 32/40 droite, 39/40 gauche).
- **Agrégat du score** : `logit.max` (le PIC = contact). `logit.mean` (récompense l'orbite) et `Vmax` sigmoid
  (sature) sont PIRES — déjà corrigé dans `plan_latent`.

## LE BUG (cause localisée, mais non résolue)
Au **CLOSE** (bouffe 1.0–1.5 m), le coût-valeur **ne sélectionne pas le candidat OPTIMAL** : **rang 0.38**
(top 38 %, pas le top) ; corr(logit, approche) +0.30 (faible). L'agent vise dans le bon tiers → frôle à ~1 m
sans closer. À distance moyenne le rang est 0.41 (l'engagement marche), au close il ne s'améliore pas (devrait
tendre vers 0). **Le signal de valeur sur les latents RÊVÉS n'est pas assez précis pour le close au cm.**

## HYPOTHÈSES RÉFUTÉES (NE PAS refaire)
- ❌ Saturation de la value au close → RÉFUTÉE (0 % saturés, écart logit 3.95).
- ❌ Granularité K « repas<20pas » trop large → RÉFUTÉE : **K=8 est PIRE** (close rang 0.71 vs 0.38 ; moyen
  0.62 vs 0.41). Cible courte = trop rare/déséquilibrée → généralise moins. `value_head_food_k8` = JETABLE.
- ❌ Rêve / horizon trop court → RÉFUTÉ (92 % atteignent, best 0.02 m, fidélité 0.93@250).
- ❌ Biais directionnel → CORRIGÉ (symétrie : béquille + WM symétrisé).
- ❌ Agrégat `mean`/`Vmax` → CORRIGÉ (`logit.max`).

## HYPOTHÈSES RESTANTES (à tester DEMAIN, gratuit/cheap d'abord)
1. **Transfert teacher-forced → rêvé** (le levier propre restant). La value est entraînée sur les latents
   **teacher-forced** (`train_value_head.py` utilise `rollout_open_loop(...)["predicted_latents"][:,0]` = 1 pas)
   mais le planner l'applique sur les latents **RÊVÉS multi-pas**. Test : modifier `train_value_head.py` pour
   entraîner sur les latents du **rollout multi-pas** (`dream_latents`, sous commandes variées), puis **re-mesurer
   le rang au close**. Si 0.38 → <0.2 → c'était le transfert → re-tester le foraging. Si pas d'amélioration → (2).
2. **Plafond intrinsèque de la perception apprise.** Le latent (rétine apprise) est peut-être moins précis que
   l'oracle-coordonnées pour le close au cm — c'est le prix de la pureté 🅑. Si (1) échoue, c'est probablement ça :
   acter que **🅑-pur engage bien mais ne close pas au cm**, garder le live COORDONNÉES qui mange, et capitaliser
   l'acquis archi. (NE PAS élargir eat_radius pour « faire passer » — §2.)

## COMMENT REPRENDRE (fichiers + commandes)
- **Gate du close (le test qui tranche)** = mesurer le **rang du candidat choisi par distance** (proche/moyen),
  argmax `logit.max(value(latents rêvés))`, sur `retina_forage` (frames bouffe visible). Cible : rang close <0.2.
  (Le script inline existe dans l'historique de session ; le re-coder est trivial.)
- **WM** : `wm_rich_fidele_sym` (NE PAS réentraîner — pas lui). **Value** : `value_head_food` (K=20, garder).
- **Planner 🅑** : `python/sylvan/control/planning/command_planner.py::plan_latent` (logit.max + symétrisation
  inférence, coordonnées débranchées + garde-fou). **Serveur** : `serve_planner_command --value-head`.
- **Foraging** : `bash run_forage_latent.sh 1.0 300 8` (headless, juge = survie/food_d). `voir_forage_latent.sh`
  (visuel, ~2 fps : lenteur INTRINSÈQUE du MPC latent H300, pas un bug).
- **Réentraîner la value** (hypothèse 1) : `SYLVAN_VALUE_K=` + `SYLVAN_VALUE_OUT=` (déjà câblés dans
  `train_value_head.py`) ; il faut EN PLUS basculer la source des latents de `[:,0]` vers le rollout multi-pas.

## GARDE-FOUS (acquis de la session, à respecter)
- **NE PAS réentraîner le WM** (prouvé : il atteint à 92 %). Si réentraînement → la **VALUE HEAD** (couche rapide).
- **NE PAS élargir eat_radius** ni gonfler un seuil pour « faire passer » (§2 = fausse solution).
- **Le live COORDONNÉES marche** (planner-coords + `wm_command_hex_v2`, survie ~900) → voie de secours intacte.
- **OPS** : les entraînements multi-epochs **PLANTENT dans le harness agent** (sandbox) → les lancer dans le
  **terminal du owner** (`!`). Les petits diags + train_value_head marchent côté agent.

## ACQUIS MAJEURS DE LA SESSION (à ne pas perdre)
WM **riche + à imagination fidèle** (chantier `--w-rollout` : corrige l'exposure-bias qui figeait le rêve riche) ;
**découplage validé** (le rêve fidèle porte la bouffe SANS `--w-food` → CLAUDE.md §3 substrat/pulsions) ; **WM
symétrisé** (augmentation miroir, fidélité préservée → c'était un déséquilibre de données, pas une asymétrie
physique masquée) ; **🅑 entièrement câblé en pur** (plan_latent, coordonnées débranchées). Reste **le close**.
