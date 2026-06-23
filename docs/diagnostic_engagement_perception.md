# Diagnostic — engagement d'une cible non-perçue / perception sous auto-mouvement (2026-06-21)

> Résout/diagnostique le problème posé dans `PROMPT_engagement_descriptif.md` (l'entité latent-pure atteint
> une cible devant/côté mais n'**engage** pas une cible derrière). Toutes les mesures sont **gratuites**
> (offline, 0 entraînement) et **multi-méthode** (code + données). À lire avant de re-toucher au WM.

## TL;DR

- Ce **n'est PAS** la magnitude du latent, **NI** le champ de vision (la rétine voit à 360° et *lit* le derrière).
- **Cause prouvée** : l'**encodeur** du WM (277→128) **jette l'information de CÔTÉ (gauche/droite)** d'une cible
  **derrière** — parce qu'elle est rare (~9 % des frames) et que la perte est dominée par le devant.
- Conséquence : en latent-pur, l'entité **ne peut pas savoir de quel côté tourner** pour une cible non frontale.
- **Reframe** : ce n'est pas un bug « bouffe-derrière », c'est la **capacité générale manquante de CHERCHER**
  ce qu'on ne perçoit pas (l'étape 2 du north-star : faim→**chercher**→aller→survie).
- **Enseignement structurel** : une recherche *latent-pure* est forcément **non-dirigée** (le latent ne porte
  pas la direction d'une cible non perçue) → faible pour des cibles éparses/loin. Dirigée = lire la rétine
  (perception-pure, pas latent-pure) ou réparer le WM.

## Hypothèses RÉFUTÉES (ne pas re-tester)

### 1. « C'est l'effondrement de magnitude du latent en rollout » — RÉFUTÉE
Outil : `diag_dream_rotate.py`. Mesures (wm_rich_fidele_sym + value_head_food_dream, 40 frames/catégorie) :
- `value` max sur l'horizon : **derrière 0.004 vs devant 0.509** (morte derrière).
- collapse magnitude @119 : **devant 0.46 < derrière 0.56** → le devant collapse PLUS et **marche** quand même.
- renormaliser le latent rêvé **ne ressuscite pas** la value (0.000).
- corr(pose prédite A, ahead lu sur le latent) **−0.29** : la perception latente *anti-suit* la pose que le WM
  prédit lui-même → découplage perception/pose dans le rêve.
- reproduit le « fait 2 » : le coût choisit un virage du **mauvais côté** 82–88 % des frames derrière.
→ La magnitude n'est pas le verrou.

### 2. « La rétine a un champ de vision avant (angle mort derrière) » — RÉFUTÉE
- **Code** (`godot/scripts/agent/perception.gd`) : rétine = **36 rayons sur 360°** × [depth, R, G, B] = 144 dims,
  **pas d'angle mort**. `food_rel0` est un **oracle séparé** (recherche analytique 360°, plus proche bouffe).
  Bouffe par défaut = 10 (`food_manager.gd`, env `SYLVAN_FOOD_COUNT`).
- **Données** (`diag_retina_fov_probe.py`, sonde fraîche sur la rétine **brute**) : le bearing derrière est lisible
  à **87 % devant/derrière, 72 % côté, 34° d'erreur** (≈ aussi bon que devant 84/79/14). L'info **est** dans la
  perception.

## Cause PROUVÉE : l'encodeur jette le côté arrière

Outil : `diag_latent_bearing_probe.py` (sonde fraîche du bearing à chaque étage du pipeline, cible derrière) :

| étage | devant/derrière | **côté (g/d)** | \|Δθ\| médian |
|---|---|---|---|
| rétine brute (144) | 87 % | **72 %** | 34° |
| encoder(obs0) (128) | 81 % | **37 %** | 86° |
| latent RSSM t0 (128) | 63 % | **52 %** (=hasard) | 77° |
| OrientHead entraînée | 35 % | **13 %** | 120° |

Le **devant survit partout** (76–98 %). Donc : l'info de côté de la cible-derrière **meurt dans l'encodeur 277→128**
(le goulot la compresse), pas dans la dynamique ni le readout. Le « 84 % décodable » du problème initial était
**dominé par le devant**. Cause profonde : derrière = **rare** (9 % des frames) + perte de reconstruction dominée
par le cas fréquent (devant) → côté-arrière jeté. **Faiblesse générale** : re-frappera pour tout percept
rare/périphérique (prédateur derrière, eau quand on a faim).

→ Corollaire : amorcer un cap dead-reckoné **depuis le latent** est mort (gate `diag_heading_seed.py` : seed côté
13 %, plafond facing A_final +0.28). Le dead-reckon **seedé-rétine** marcherait (72 %) mais lit la rétine
(perception-pure, pas latent-pure).

## CHERCHER (perception active) — conçu, implémenté, gate FC=1 négatif

Idée : capacité générale, JEPA-pure, drive-agnostique, 0 retrain. Quand la value latente de la pulsion active est
PLATE (`engage < τ`, rien d'engageant perçu), explorer (scan du cap → errance) jusqu'à ce qu'une cible entre dans
le cône avant (latent fort) → handoff auto au mode-avant latent. Déclencheur lu dans le latent + approche dans le
latent = pur ; seul l'acte d'explorer = réflexe substrat (comme le CPG).

- **Gate offline PASSÉ** (`diag_search_trigger.py`) : `engage` (max proba-repas du meilleur futur) sépare net —
  front-proche 0.98 / front-mid 0.945 / side 0.681 / **derrière 0.17** → seuil **τ=0.5** (trou 0.33–0.73).
- **Implémenté** : `command_planner.plan_latent` renvoie `engage` ; `serve_planner_command._apply_search` (machine
  scan→errance, état entre replans, handoff `engage≥τ`) ; env `SYLVAN_SEARCH_ENABLE/TAU/VX/OMEGA/SCAN/WANDER/
  PATIENCE/LOG`, **OFF par défaut**.
- **Gate closed-loop FC=1 (cas dur, 1 bouffe éparse) = NÉGATIF** (`gate_search_fc1.sh`, search OFF vs ON) :
  search reste **plus loin** de la bouffe (moy **5.38 m** vs base 4.50) et **n'arrive JAMAIS à portée** (<1.2 m :
  **0 %** vs base 2 %). CHERCHER s'active bien (27 transitions) mais la recherche **non-dirigée éloigne**. Survie
  inutilisable (saturée à 1500 : drain 0.05 trop doux). Repas bruités.
- **Enseignement structurel** : recherche *latent-pure* = forcément **non-dirigée** → faible pour cibles
  éparses/loin (le latent ne "voit" une cible à approcher que dans ~2–3 m, portée du rêve ; au-delà, aveugle quelle
  que soit l'orientation). **Pureté maximale et efficacité sont en tension** ici (prouvé, pas théorique).
- **Gate closed-loop FC=6 (réaliste, drain 0.12 discriminant) = NÉGATIF aussi** (`gate_search.sh 6 0.12`) :
  survie base méd **1200**/moy 1106 → search méd **900**/moy 1028 ; repas base **12** → search **9** ; CHERCHER
  déclenché **27×** (pas dormant). → **Verdict des DEUX densités : la recherche latent-pure NON-DIRIGÉE est morte**
  (laisser `SYLVAN_SEARCH_ENABLE` OFF par défaut). Le hack « gain » n'a même pas livré le gain.
- **Pourquoi** (cf `diagnostic_perception_rotation_wm.md`) : le rêve du WM est trop faible → `engage` est souvent
  bas même quand la bouffe est proche → CHERCHER sur-déclenche → l'errance non-dirigée éloigne. La vraie cause
  remonte à la **fidélité du rollout du WM** (la clé de voûte), pas au mécanisme de recherche.

## Outils gratuits créés (réutilisables)

| Script | Mesure |
|---|---|
| `diag_dream_rotate.py` | magnitude/value/ahead/pose le long d'un rêve de virage (réfute magnitude) |
| `diag_retina_fov_probe.py` | bearing lisible depuis la rétine BRUTE (réfute FOV) |
| `diag_latent_bearing_probe.py` | bearing à chaque étage encoder→latent (localise la perte) |
| `diag_heading_seed.py` | gate du dead-reckon seedé-latent (le tue) |
| `diag_search_trigger.py` | séparation du déclencheur `engage` (fixe τ) |
| `diag_search_sanity.py` | sanity câblage `engage` (offline) |
| `gate_search_fc1.sh` / `gate_search.sh` | gate closed-loop search OFF vs ON (survie/repas/distance) |
