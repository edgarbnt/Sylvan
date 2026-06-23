# SCOPE — RECHERCHE GUIDÉE (CEM / gradient) pour 🅑 (planifier dans le latent)

> **Point de reprise après la session 2026-06-19.** Pré-requis : lire `memory/sylvan-retina-decision.md`
> (à jour) + `ETAT_DES_LIEUX.md` §8. Discipline CLAUDE.md §1 (diagnostiquer GRATUITEMENT d'abord, gater le
> cher) + §2 (pas de fausse solution, coordonnées DÉBRANCHÉES pour de vrai).

## 0. Pourquoi (le mur établi cette session)
🅑 = la créature planifie en **notant des états LATENTS via une tête apprise** (pas de coordonnées). Tout
est en place SAUF la **recherche** :
- Le WM `wm_command_hex_retina_eat_v2` est **food-aware** : son RÊVE (rollout open-loop) garde la bouffe
  (food_auc 0.84). Ce n'est PLUS le problème.
- Le planner actuel teste une **grille FIXE d'arcs (vx, ω)** (~117 candidats). Mesuré : en **un seul rêve**,
  AUCUN arc n'atteint la bouffe (tous finissent ~1.7-1.9 m, 0 % < 1 m). → un readout latent n'a quasi **rien
  à classer** (plage minuscule + bruitée → corr ~+0.34 = ~⅓ de la géométrie). Muscler le food-awareness
  (w_food 1.5) n'aide pas (pire que 0.5) : ce n'est pas un manque de signal dans le latent, c'est que **la
  grille ne contient pas de bon plan à trouver**.
- Le planner-coordonnées marche sur la même grille uniquement car `-min_dist` donne le classement
  géométrique EXACT — ce qu'on s'interdit en 🅑.

**Idée du fix** : remplacer la grille figée par une **RECHERCHE** qui optimise la séquence de commandes
DANS l'espace continu, en se laissant guider par le **score latent** (énergie food-future sur le rêve).
La recherche fabrique elle-même les bons plans que la grille ne contenait pas → le readout retrouve une
vraie marge de manœuvre. C'est l'étape « recherche guidée » prévue avant Mode-1.

## 1. Deux variantes (commencer par CEM)
- **CEM (Cross-Entropy Method)** — SANS gradient, robuste, recommandé en premier :
  1. échantillonner N séquences de commandes ~ N(μ, σ) (μ,σ par pas, ou par segment) ;
  2. les dérouler dans le WM (`rollout_open_loop`) → latents rêvés ;
  3. scorer chacune par le **readout latent** (énergie future food-aware ; cf §3) ;
  4. garder les K meilleures (élites), refit μ,σ dessus ; itérer 3-5 fois ; renvoyer μ (1er pas exécuté).
- **Gradient** (plus tard, plus tranchant mais fragile) : back-prop du score latent vers la séquence de
  commandes à travers le WM différentiable (`dream_latents` est déjà grad-enabled). Risque : minima locaux,
  pas-de-temps, clamp des commandes dans la plage propre.

## 2. Score (JEPA-pur, coordonnées DÉBRANCHÉES — assert)
- Lire UNIQUEMENT des readouts du latent rêvé : **énergie future** `out["predicted_next_obs"][...,-1]`
  (l'instrument qui marche marginalement sur eat_v2, car physique → généralise aux commandes cherchées),
  − pénalité de chute (`done`). PAS de `food_xz`, PAS de `-min_dist`, PAS de heading géométrique.
- Option (à tester) : une **tête de valeur ré-entraînée sur latents RÊVÉS sous commandes DIVERSES** (babbling
  + bruit), pour qu'elle généralise aux séquences que la CEM explore (la tête actuelle, apprise sur les vraies
  commandes foraging, NE généralise PAS → corr négative mesurée). Mais commencer par l'énergie (plus simple).
- Garde-fou : les commandes restent dans le **régime propre** (vx 0.55-0.75, |ω| borné) sinon le WM sort de
  sa distribution → rêve non fiable.

## 3. Plan GATÉ (cheap → cher)
1. **GATE GRATUIT (offline, AUCUN Godot)** — `diag_cem.py` : sur des frames de `retina_forage` (bouffe visible
   1.5-4 m), lancer la CEM guidée par le **score énergie SEUL** (coordonnées débranchées) et mesurer la
   **distance MIN atteinte dans le rêve** par la séquence trouvée.
   - **SUCCÈS** : la CEM-énergie atteint min_dist nettement < la meilleure grille (cible < 1.0 m, vs ~1.7 m
     grille) sur une majorité de frames → la recherche débloque des rêves qui ATTEIGNENT la bouffe, guidée
     par le latent seul → 🅑 enfin exploitable. → passer au closed-loop.
   - **KILL** : la CEM-énergie ne fait pas mieux que la grille (reste ~1.7 m) → soit le rêve ne peut PAS
     représenter l'atteinte de la bouffe (limite du WM/horizon), soit le score énergie ne guide pas →
     diagnostiquer (essayer la tête de valeur diverse-commandes, ou horizon, ou gradient) AVANT tout closed-loop.
     Comparaison de contrôle : refaire la CEM guidée par la GÉOMÉTRIE (`-min_dist`) — si ELLE atteint < 1 m mais
     pas la version énergie → c'est le SCORE le problème ; si même la géométrie reste ~1.7 m → c'est le RÊVE/horizon.
2. **CHER (closed-loop)** — si le gate passe : câbler la CEM dans `command_planner.py`
   (`SYLVAN_PLANNER_SEARCH=cem`, score latent, **assert aucune coordonnée**), servir `wm_command_hex_retina_eat_v2`,
   forager (`run_forage_retina.sh` adapté). **JALON 🅑** : survie ≥ ~baseline coordonnées (~990, éco de vie ~2745),
   avec les coordonnées DÉBRANCHÉES (assert au runtime). KILL : erre/sous-perf nette → revenir au diag, ne PAS
   ré-injecter les coordonnées pour « faire marcher ».

## 4. Fichiers à toucher
- 🔴 `python/sylvan/control/planning/command_planner.py` : ajouter un chemin de recherche CEM (à côté de la
  grille, sélection par env `SYLVAN_PLANNER_SEARCH=grid|cem`). Réutiliser `world_model.rollout_open_loop`.
  Le coût latent (énergie future − chute) en option `latent_cost` ; assert pas de food_xz quand actif.
- 🟡 `scripts/serve_planner_command.py` : exposer le mode recherche + le coût latent (flags), servir eat_v2.
- 🆕 `diag_cem.py` (racine) : le GATE gratuit (§3.1).
- 🟢 WM inchangé (`dream_latents` + `rollout_open_loop` déjà dispos ; food_head non nécessaire à l'inférence).

## 5. Risques / inconnues
- **R1 — coût temps réel** : CEM = 5 itér × ~100 séquences = ~500 rollouts WM par replan (vs ~117 grille).
  Sur CPU, par replan toutes les 10 steps : à mesurer. Mitigations : moins d'itér/échantillons, horizon plus
  court, replan moins fréquent, ou warm-start μ du replan précédent.
- **R2 — le rêve ne sait peut-être pas ATTEINDRE la bouffe** (horizon/compounding) : c'est précisément ce que
  le GATE §3.1 teste AVANT de payer le closed-loop. Si la géométrie-CEM elle-même plafonne à ~1.7 m → c'est le
  WM/horizon, pas la recherche → escalade (horizon plus long, re-feed rétine périodique dans le rollout…).
- **R3 — score énergie trop mou** : la tête de valeur diverse-commandes est le plan B du score (§2).
- **R4 — commandes hors-régime propre** → rêve non fiable : clamp dur dans la plage propre.

## 6. Ordre d'exécution
0 (gratuit) `diag_cem.py` énergie + contrôle géométrie → décide SCORE-vs-RÊVE.
1 (si gate) câbler CEM + coût latent dans le planner, assert coordonnées débranchées.
2 (cher) closed-loop foraging vs baseline. MPC brute-force grille gardé en fallback. Mode-1 plus tard.

## 7. RÉSULTAT DU GATE §3.1 (2026-06-19) — KILL, mais la cause n'est NI la recherche NI le score NI l'horizon
`diag_cem.py` (offline, 40 frames bouffe 1.5-4m, e0=0.4, eat_v2) :
| variante | min_dist médiane | <1m |
|---|---|---|
| GRILLE (réf) | 1.75 m | 0% |
| CEM-ÉNERGIE (score latent, coords débranchées) | 1.80 m | 0% |
| CEM-GÉOMÉTRIE (contrôle, score=-min_dist) | 1.75 m | 0% |

La CEM-géométrie (qui PEUT utiliser les coordonnées et optimise DIRECTEMENT la distance) plafonne aussi à 1.75 m
→ **ce n'est pas le score, pas la grille, pas la recherche : le RÊVE ne transporte pas le corps.** Probes gratuits
(refus de la conclusion-horizon de commodité, CLAUDE.md §1/§2) :
- Rêve droit 120 pas : **eat_v2 = 0.19 m vs live `wm_command_hex_v2` = 0.75 m (4×)**.
- **Family-wide** (jepa_v2 pré-eat 0.18, eat_v1 0.18, eat_v2 0.19) → pas causé par le retrain eat-aux.
- Motion réelle ~5 mm/pas (~0.60 m/120) **identique dans tous les datasets** → pas la data.
- **1 pas teacher-forced parfait pour tous** (≈4.9 mm ≈ réel) → la tête déplacement va bien.
- ⇒ effondrement = **dérive open-loop du latent** ; meta : retina `latent_loss=cosine`+VICReg(1,1,1) vs v2 `mse`.
  Le cosine ne contraint que la DIRECTION → la magnitude dérive → le déplacement décroît à ~⅓ sur le rollout.

**Le débloqueur de 🅑 n'est donc PAS la recherche guidée (saine mais affamée — rien à atteindre) mais réparer la
dérive open-loop du WM-rétine** : terme latent MSE (ancre magnitude) à côté du cosine/VICReg, ou supervision du
déplacement intégré sur le rollout (scheduled sampling), sans re-collapser le latent → puis re-passer `diag_cem.py`.
Niveau = retrain-nuit du WM → décision owner, pas lancé. NE PAS câbler le closed-loop (gate non passé).
