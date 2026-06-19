# Design — 2ᵉ pulsion (SOIF + EAU) : arbitrage homéostatique émergent

> **STATUT : ÉTAGE 1 IMPLÉMENTÉ & VALIDÉ (2026-06-18).** Arbitrage émergent 11/12 au diag gratuit
> (`diag_arbitration.sh`), WM inchangé, non-régression A→B OK. Détails+code : `memory/sylvan-second-drive-arbitration.md`.
> Le doc ci-dessous = le design tel que conçu (étage 2 = section "ÉTAGE 2" reste à faire si voulu).


> But north-star : passer d'« une seule pulsion (faim) » à « plusieurs pulsions en compétition »,
> et **faire ÉMERGER l'arbitrage** (« j'ai faim ET soif, qu'est-ce qui est le plus urgent ? »)
> au lieu de le coder par une préférence fixe. C'est le premier vrai pas vers l'ALife émergente
> au-dessus de la nav A→B (rendue solide le 2026-06-18, voir `memory/sylvan-ab-navigation-fix.md`).

## Principe directeur : l'arbitrage doit ÉMERGER de l'URGENCE, pas d'un poids fixe

Mauvais design (anti-émergence) : `coût = w_food·dist_food + w_water·dist_water` avec w fixes
→ c'est NOUS qui décidons la priorité. Pas d'émergence.

Bon design (ALife) : chaque ressource a une **courbe d'urgence non-linéaire** qui explose quand le
niveau approche de 0. La priorité **tombe alors toute seule** de l'état interne + la géométrie :
- soif critique + eau loin → le terme soif domine → il va boire (même si la bouffe est à côté) ;
- les deux moyens → il va au plus proche ;
- une ressource pleine → son terme s'éteint → il ignore cette ressource.
L'arbitrage est un *résultat*, pas une règle. Forme : `urgency(level) = (1 - level/max)²` (ou `exp(-k·level)`),
le carré rend la chute critique beaucoup plus chère qu'un déficit modéré.

## Architecture en 2 ÉTAGES (gate cher derrière pas-cher)

### ÉTAGE 1 — test d'arbitrage GRATUIT (le WM ne bouge PAS, zéro entraînement)

Insight : la position des ressources est reconstruite **géométriquement** dans le planner
(`food_xz_from_radar`), pas par le WM. Le `vision_fine` est déjà un input **planner-only**. Donc l'eau
+ la soif s'ajoutent comme inputs **planner-only**, et la décroissance énergie/soif est modélisée
**analytiquement** (drain ≈ linéaire par pas) dans l'intégration des candidats — exactement comme le
déplacement est intégré. **Le WM reste à 145 dims, le policy résiduel à 144, aucun checkpoint cassé.**

Ce que ça ajoute, étage par étage (réfs depuis la carte du code) :

| Étage | Fichier | Ajout |
|---|---|---|
| Monde/spawn | `godot/scripts/world/food_manager.gd` | dupliquer pour l'eau (positions, try_consume eau→restore soif, get_water_positions, nearest). Env `SYLVAN_WATER_COUNT/ANGLE_DEG/MIN_RADIUS/SPAWN_RADIUS`. ~60 L |
| Perception | `godot/scripts/agent/perception.gd` | `water_radar()` identique à `food_radar` (12 + 36 fin). ~25 L |
| Homéostasie | `godot/scripts/agent/homeostasis.gd` | état `thirst` parallèle (max, drain passif, restore_thirst), `is_critical()` → `energy<=0 OR thirst<=0 OR health<=0`. ~40 L |
| Glue serveur | `godot/scripts/main.gd` + `python/scripts/serve_planner_command.py` | Godot envoie `water_radar` (planner-only, comme `vision_fine`) + scalaire `thirst` ; le serveur les passe au planner. **Le WM obs reste `proprio+food_radar+energy` = 145.** ~15 L |
| Coût planner | `python/sylvan/control/planning/command_planner.py` | `water_xz_from_radar()` ; intégrer énergie ET soif analytiquement le long du candidat (drain/pas, +restore si la trajectoire imaginée atteint la ressource) ; score = `-Σ urgency(level_t)` + shaping de cap vers la ressource LA PLUS URGENTE. Env `SYLVAN_PLANNER_THIRST_W`, courbes d'urgence. ~50 L |

**Le WM, le policy, les dims 132/18/144/145 : INCHANGÉS.** C'est ça qui rend l'étage 1 gratuit.

### ÉTAGE 2 — internaliser dans le WM (CHER, gaté derrière l'étage 1)

Seulement SI l'étage 1 montre un arbitrage émergent correct ET qu'on veut que le WM *imagine* la
dynamique des ressources (events boire/manger, décroissance apprise plutôt qu'analytique) :
- obs WM 145 → **158** (`+ water_radar(12) + thirst(1)`), resync `config.py`, `constants.py`,
  `command_wm.py` (obs_head), `wm_dataset.py`, `symmetry.py`, `observation_builder.gd`.
- recollecter des épisodes AVEC l'eau (régime propre hexapode), réentraîner le WM.
- Coût justifié uniquement par un gain mesuré vs l'étage 1 analytique.

## Le DIAGNOSTIC GRATUIT d'arbitrage émergent (critères falsifiables AVANT tout)

Comme pour A→B : scénarios contrôlés, 1 bouffe + 1 eau à des positions/azimuts fixes, niveaux
internes forcés, on mesure QUELLE ressource il choisit. Réutilise le moteur de `diag_nav_ab.sh`.

Cas de test (chacun = une prédiction falsifiable) :
1. **Faim critique, soif pleine**, bouffe et eau équidistantes → doit aller à la BOUFFE.
2. **Soif critique, faim pleine**, idem → doit aller à l'EAU. (symétrie : pas de biais)
3. **Soif critique mais eau LOIN (5 m), bouffe PROCHE (2 m), faim moyenne** → doit quand même aller à
   l'EAU (l'urgence bat la proximité) = LE test d'arbitrage non-trivial.
4. **Les deux moyens, eau proche / bouffe loin** → va au plus proche (eau).
5. **Switch dynamique** : parti vers la bouffe, la soif devient critique en route → doit *re-router*
   vers l'eau (le receding-horizon re-décide). = preuve d'arbitrage continu, pas one-shot.

KILL/escalade : si l'arbitrage suit la proximité en ignorant l'urgence (cas 3 échoue), la courbe
d'urgence est trop molle → ajuster k (gratuit) AVANT de toucher au WM. Si même réglé l'urgence ne
peut pas battre la géométrie sans casser les cas 1-2 → re-penser la forme du coût, pas entraîner.

## Pourquoi SOIF+EAU en premier (vs danger, fatigue)

- **Réutilise 100 % de la machinerie bouffe/énergie** (spawn, radar, homéostasie, reconstruction) →
  risque d'intégration minimal, c'est le test d'arbitrage le plus PUR (aller à A vs B selon l'urgence).
- Progression ensuite : **danger+prédateur** (ressource NÉGATIVE → évitement + réactif, le WM doit
  modéliser une menace mobile) ; puis **fatigue+repos** (l'action « ne rien faire » → patience temporelle).
  Chacun ajoute une *nature* de décision nouvelle. Soif d'abord = fondation propre.

## Risques / pièges (de la carte du code)

1. Ordre de concaténation des obs : l'énergie DOIT rester en dernier (`out[..., -1]`). En étage 1 on
   n'y touche pas (soif/eau sont planner-only) → risque nul. En étage 2, attention au slicing.
2. Le policy résiduel ne voit PAS la soif (reste 144) — normal, il ne fait que l'équilibre/propulsion ;
   la décision vit dans le planner. Cohérent avec l'archi 3-couches.
3. Garder l'agnosticité §14 : la soif/l'eau vivent UNIQUEMENT dans le coût du planner (corps/WM/CPG
   restent agnostiques). L'étage 1 le respecte par construction (WM inchangé).
