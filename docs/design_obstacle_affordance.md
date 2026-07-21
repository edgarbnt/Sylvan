# Design — Canal OBSTACLE / affordance physique : première conséquence NON-homéostatique, pré-inscrit 2026-07-17

## Mission
Prouver, **en vies**, que l'entité **CONTOURNE** un obstacle qui bloque son mouvement (au lieu de
foncer dedans), **sans aucun coût-obstacle codé-main dans la boucle de décision** — la réaction est
**APPRISE / ÉMERGENTE** du seul vécu. C'est le **PREMIER canal de conséquence NON-homéostatique** :
la conséquence porte sur le **mouvement**, pas sur les drives (faim/soif/santé). C'est la
généralisation de « apparence → conséquence apprise » (chantier baie-buisson **clos**, G0-G3 PASS) à
une **nouvelle nature de conséquence**. Et ça crée la **TOPOLOGIE** (détours, impasses, occlusions)
qui rendra *chercher + mémoire* enfin décisifs (aujourd'hui triviaux en arène ouverte).

## À lire d'abord
- `docs/research_appearance_consequence.md` **AXE 3** (la synthèse de recherche : « absorber
  l'obstacle DANS le WM » est LE PLUS MINCE en sources — une seule réf vague — la littérature
  préférant un **prédicteur d'affordance séparé** appris du signal commandé-vs-réel).
- `docs/design_attribution_credit.md` (le **patron** du chantier précédent : pré-inscription,
  gates cheaper-first, G0 gratuit qui gate le cher, négatif commité).
- `docs/design_perception_types.md` (le WM typé = la base perceptive vivante).
- `memory/MEMORY.md` + fin de `memory/sylvan-mode1-build.md`.

## Pourquoi ce canal, et pourquoi maintenant
1. **Premier canal non-homéostatique.** Jusqu'ici toute conséquence apprise était un drive
   (rouge→énergie, bleu→soif, vert→dégâts). Le blocage du mouvement est une conséquence d'une
   **autre nature** — la brique manquante pour une famille de conséquences ouverte et hétérogène
   (research AXE 1).
2. **Le coût paie par CANAL (un sens fixé au corps), pas par objet.** Le canal-mouvement payé UNE
   fois → toute la famille des affordances physiques (bloque / traverse / ralentit) arrive
   **découverte**, zéro code par objet. C'est le même contrat que « une tête par pulsion » (§3
   CLAUDE.md), étendu à un canal sensori-moteur.
3. **La topologie débloque le vrai mur de capacité.** `docs/roadmap_vers_monde_v3.md` (encart
   2026-07-17 tard) inverse l'ordre : obstacles AVANT chercher/mémoire, parce qu'en arène ouverte
   chercher/mémoire sont triviaux. Les détours/occlusions de l'obstacle sont ce qui les rend
   décisifs.

## Le principe (à garder honnête — §2, §3 CLAUDE.md)
- **MONDE / physique / sens = donnés légitimes** (§3). Un solide qui bloque le corps est une
  addition **PHYSIQUE** (couche CORPS/MONDE). La **RÉACTION** (contourner) est **APPRISE ou
  ÉMERGENTE**, JAMAIS un `if gris then évite` ni un coût-obstacle codé-main dans la décision.
- **Coût par CANAL, pas par objet.** Le canal-mouvement est écrit/fourni UNE fois ; l'apparence de
  l'obstacle et son effet (bloque) sont **découverts**.
- **Critère de pureté (le test officiel « ça survit à un changement de monde ? ») :** si l'apparence
  de l'obstacle change, l'entité **ré-apprend** qu'il bloque — elle ne présume rien de la couleur.

## État à la reprise (rien à re-découvrir)
- WM vivant = `wm_objcentric_kin_typed` (perception 100 % apprise au sens connaissance-du-monde).
- **Le corps cinématique glisse par TÉLÉPORTATION** (`sylvan_agent.gd` `_kinematic_step`,
  `PhysicsServer3D.body_set_state`) : il **NE respecte PAS les solides** aujourd'hui. Un obstacle
  bloquant exige une addition **physique du CORPS** (shapecast / collision avant le glide), jamais le
  cerveau.
- Le buisson (chantier précédent) a établi le patron d'**ingrédient-monde opt-in, défaut OFF,
  bit-identique** (`food_manager.gd`, flags `SYLVAN_FOOD_BUSH*`) : le reprendre pour l'obstacle.

## Les deux voies (le diag G0 en TRANCHE une)
- **Voie A — « absorber dans le WM ».** Re-collecter + fine-tuner le WM avec des interactions
  d'obstacle pour que la dynamique `(vx,ω)→déplacement` encode le blocage ; le planner MPC évite
  alors **par rollout**, zéro coût codé. **Séduisante** (pas de nouveau module, pureté maximale) mais
  **PEU ÉTAYÉE** (research AXE 3 : une seule réf vague) **et COÛTEUSE** (cycle WM) — et elle frôle le
  **SIGNAL D'ALERTE §3** (ne pas retrain le WM pour *une* ressource). À ne payer QUE si G0 prouve que
  le WM *peut* représenter le blocage.
- **Voie B — « prédicteur d'affordance séparé ».** Une fonction apprise `rétine → facteur de
  blocage`, entraînée du signal **commandé-vs-réel** (le WM prédit un déplacement, le réel ≈ 0 ⇒
  collision), **lue par le planner comme un coût** — dans la lignée **exacte** de la lunette
  saillance-danger déjà vivante et pure. **MIEUX ÉTAYÉE** (research AXE 3 : traversabilité
  recon-error / commandé-vs-réel + MPC).
- **Le signal commandé-vs-réalisé est le pont des deux voies** : déplacement PRÉDIT (WM) vs RÉALISÉ
  (proprio / odométrie), **déjà disponible, aucun nouveau capteur**.

## Contrainte de monde DÉCLARÉE (viabilité — mesurée AVANT de juger l'agent, §2)
- L'obstacle doit créer une **VRAIE place** : contourner **coûte un détour** mais reste **POSSIBLE et
  SURVIVABLE** (portée métabolique soutenable ≥ longueur du détour). Sinon on juge des vies
  **condamnées par l'arithmétique** (leçon plafond métabolique).
- L'apparence de l'obstacle (couleur/teinte, taille, position) = **propriété du monde déclarée**,
  jamais ajustée pour faire passer un gate (§2). Elle est **datée** dans ce doc au build.
- **Séparabilité rétine** : l'obstacle est perceptible et distinct du fond et des ressources
  (mesuré en G1, écart teinte inter > dispersion intra).

## Gates PRÉ-ENREGISTRÉS (falsifiables, ordre pas-cher-d'abord)

### G0 — DIAG GRATUIT DÉCISIF (0 run, 0 Godot, 0 train). **GATE LA VOIE ET TOUT LE BUILD.**
**Question :** le WM gelé conditionne-t-il son **déplacement prédit** sur la **PERCEPTION** (la
rétine), ou seulement sur la **commande** (vx, ω) ? Ce diag TRANCHE la voie et évite un cycle WM
deviné (la leçon anti-boucle §1).

- **G0a — architectural (inspection, gratuit).** Le chemin de prédiction du déplacement prend-il en
  entrée le **latent** (qui encode la rétine) ou **seulement la commande** ? *(Inspection faite :
  `MetricsPredictionHead` = `cat([latent(128), commande(2)])` → la rétine **peut** atteindre le
  déplacement, mais **seulement via le latent** ; et la CIBLE d'entraînement du déplacement est de la
  cinématique corporelle pure obstacle-free → le couplage rétine→déplacement est **probablement
  faible**, à MESURER.)* La porte architecturale est donc **ouverte** → G0b tranche.
- **G0b — comportemental (le WM répond-il DÉJÀ à un obstacle ?).** Sur `N` obs RÉELLES du corpus
  existant (aucun obstacle rendu — injection synthétique, façon `diag_credit_g0`), injecter un
  **obstacle synthétique DEVANT** à courte portée dans la rétine, re-encoder, et mesurer **DEUX**
  réponses (pour une **commande avant fixe** vx > 0, ω = 0) :
  1. `Δ_disp = |déplacement_prédit(obstacle) − déplacement_prédit(sans)|` (le couplage
     rétine→**déplacement**) ;
  2. `Δ_slot = |slot_prédit(obstacle) − slot_prédit(sans)|` (le couplage rétine→**latent**, borne de
     **REPRÉSENTABILITÉ** : le slot est le readout le plus retina-dérivé → un `Δ_slot` fort prouve que
     l'obstacle **atteint le latent**, donc qu'un fine-tune *pourrait* le lire).
  - **Contrôles (pré-enregistrés)** : obstacle injecté **sur le côté / derrière** (le blocage vers
    l'avant devrait **moins** réagir → réponse spatialement sensée) ; rayon **PLACEBO** lointain /
    non-coloré (Δ ≈ 0 attendu) ; **plancher-bruit MESURÉ** = dispersion de Δ sous injections nulles
    (pas réglé — §2).

**Verdict G0 (le SUCCÈS de G0 = TRANCHER + localiser le coût, pas valider une voie) :**
- **`Δ_disp` significatif** (`> 3·plancher` ET `>` réponse-côté) → le WM **répond déjà** au blocage →
  **voie A quasi-gratuite** (fine-tune léger de la tête déplacement).
- **`Δ_disp ≈ 0` mais `Δ_slot` fort** (l'obstacle est **dans le latent**, la tête déplacement
  l'IGNORE — issue **attendue**) → voie A **possible mais COÛTEUSE** (re-collecte obstacle +
  fine-tune tête, l'info est là) ; **voie B lit le MÊME latent/rétine bien moins cher** (jumeau de la
  lunette danger, pur) → **voie B recommandée** (research AXE 3 + §3).
- **`Δ_disp ≈ 0` ET `Δ_slot ≈ 0`** (l'obstacle n'atteint même pas le latent) → voie A exigerait un
  **retrain encodeur complet** → **voie B tranchée** sans ambiguïté.
- **ÉCHEC G0** = diag ambigu / contrôles incohérents (réponse-côté ≥ réponse-avant, placebo ≈ signal)
  → **re-concevoir la sonde AVANT tout build**, ne rien lancer.

### G1 — Godot léger (viabilité du monde + physique du corps)
- Ajouter un obstacle **COLORÉ bloquant**, opt-in `SYLVAN_OBSTACLE_*` **défaut OFF bit-identique**
  (patron buisson) ; **rendu rétine** ; **AUCUN drive**, aucune consommation.
- Le corps cinématique **RESPECTE les solides** (shapecast / `move_and_collide` avant le glide dans
  `_kinematic_step`) = **physique du CORPS**, pas du cerveau. `main.gd` jamais stagé.
- **MESURER** : (a) le corps **s'arrête** contre le solide → déplacement réalisé **<** commandé quand
  un obstacle est devant (le signal du canal existe) ; (b) **VIABILITÉ** : obstacle placé entre spawn
  et ressources → contourner reste **possible + survivable** (détour ≤ portée soutenable) ; (c)
  **séparabilité rétine** de l'obstacle vs fond/ressources.
- Si non-viable ou non-séparable → ajuster le **MONDE** (déclaré), **jamais le gate** (§2).

### G2 — offline (apprendre l'affordance ; **selon le verdict G0**)
- **Voie B** : entraîner le prédicteur d'affordance (`rétine → facteur de blocage`) sur le corpus G1,
  **labels bootstrappés du commandé-vs-réel** ; MESURER AUC / séparation sol-vs-obstacle vs labels de
  contact ; intégration planner comme **coût** (lu comme la lunette danger). WM **gelé**.
- **Voie A** : re-collecte + fine-tune WM avec obstacles ; MESURER que le déplacement prédit chute
  **au bon endroit** (open-loop) et que le reste (perception / position) **ne régresse pas**.
- **Non-régression (les deux voies)** : les drives (food / water / danger) restent corrects.

### G3 — juge closed-loop (payé si G0-2 passent)
- **2×24 vies seeds 1+2**, monde avec obstacle **entre spawn et ressources**.
- **PASS** : taux de **collision ↓** vs baseline **obstacle-aveugle** (l'entité contourne) **ET**
  forage ≥ config vivante − bruit (l'obstacle coûte un détour sans casser la survie). **Zéro
  coût-obstacle codé** dans la boucle de décision.
- **KILL précoce** : seed 1 **fonce** systématiquement dans l'obstacle (collision ≈ aveugle), OU le
  forage s'effondre.

## ⭐ VERDICT G0 (2026-07-17, `diagnostics/diag_obstacle_g0.py`, 0 run / 0 Godot / 0 train) : PASSÉ — voie B tranchée
Corpus PROPRE `critic_kin_typcorp` (256 frames à **front dégagé**), WM vivant
`wm_objcentric_kin_typed` (obs 277, 3 slots), commande avant fixe vx=0.65, obstacle synthétique
injecté dans la rétine. **Sanity null** (obs identique) : ratio 1.000, Δ = 0 exact → sonde
déterministe ✅.

- **(1) La tête déplacement IGNORE l'obstacle** (le décisif) : obstacle gris **DEVANT proche
  (~1.85 m)** → `d_fwd` **8.21 mm vs 8.00 mm de base, ratio 1.03** — elle ne **FREINE pas** (si quoi
  que ce soit elle accélère très légèrement = réponse anti-blocage, pur bruit de perturbation).
  `|Δdisp|` = 0.17 mm (~2 %), **sans sélectivité spatiale** : DEVANT 0.17 ≈ DERRIÈRE 0.16 ≈ CÔTÉ 0.10
  ≈ LOINTAIN 0.12 mm (tous dans la bande-bruit) → la tête n'a **aucun modèle d'obstacle**. Attendu :
  sa cible d'entraînement = cinématique corporelle pure, monde **sans** obstacle.
- **(2) MAIS l'obstacle ATTEINT le latent** (représentabilité) : `Δlatent(1−cos)` DEVANT **0.0118**
  vs null **0.00000** (le latent qui nourrit la tête déplacement bouge bien) ; `Δslot` du readout
  requêté (vert devant) **6.58 m** (la perception object-centric répond fortement). L'information de
  l'obstacle est **présente dans le latent** — la tête déplacement la laisse simplement tomber.

**➜ VERDICT : issue 2/3 pré-enregistrée (`Δ_disp ≈ 0` mais `Δ_latent`/`Δ_slot` fort) → VOIE B
(prédicteur d'affordance séparé).** La voie A (« absorber dans le WM ») n'est **pas**
architecturalement impossible (l'info est dans le latent) mais exigerait une re-collecte obstacle +
fine-tune du WM (cycle coûteux, SIGNAL D'ALERTE §3) pour un gain modeste ; la voie B lit le **même
latent/rétine** comme un coût planner — **jumeau exact de la lunette saillance-danger déjà vivante et
pure** (research AXE 3 : la voie la mieux étayée). Le diag a **tranché à coût nul** (0 run) = la
discipline anti-boucle §1 respectée. **PROCHAIN = G1** (obstacle bloquant Godot + le corps
cinématique respecte les solides + viabilité du monde mesurée).

## ⭐ VERDICT G1 (2026-07-18, `obstacle_manager.gd` + `_kinematic_step` respecte les solides + `diagnostics/diag_obstacle_g1.py`) : PASSÉ
Construit : **`godot/scripts/world/obstacle_manager.gd`** (mur SOLIDE + perceptible, opt-in
`SYLVAN_OBSTACLE_*` défaut OFF bit-identique, patron buisson/hazard ; cyan déclaré 2026-07-17,
cos≈0 avec le rouge-bouffe → ne fire aucun slot en collecte food-only) + **collision MANUELLE dans
`sylvan_agent._kinematic_step`** (le corps cinématique est GELÉ → aucune résolution physique → un
raycast sur une couche dédiée bit 2 arrête le glide ; jamais le sol bit 0 ni le corps bit 1 ;
bit-identique quand `SYLVAN_OBSTACLE_COUNT=0` : masque 0 → no-op). Hooks `main.gd` LOCAUX non stagés
(placement + `SYLVAN_OBSTACLE_AHEAD`/`SYLVAN_DRIVE_STRAIGHT` pour le test A/B).

- **(a) LE CORPS S'ARRÊTE** — A/B drive-straight SOLIDE vs PASSABLE (mur droit devant le spawn, profondeur
  du rayon-mur = odomètre) : **SOLIDE `min_devant 0.69 m`, 0 pénétration** (le corps ne descend JAMAIS sous
  la distance d'arrêt, épinglé 1404/1404 frames ; le déplacement cumulé Godot plafonne à 1.53 m au mur) ;
  **PASSABLE `min 0.35 m`, 41 pénétrations** (il TRAVERSE). ✅ La collision manuelle arrête le corps.
- **(b) VIABILITÉ géométrique** (le critère PRÉ-INSCRIT) : détour autour du mur étroit (demi-largeur
  0.35 m) = **0.7 m ≤ portée soutenable 4 m**, bouffe reachable autour (mur non-enclosant). ✅
  **Occlusion RÉELLE rapportée honnêtement (§2, pas cachée)** : bouffe perçue **21 %** — un mur SOLIDE
  occulte la ligne de vue frontale (tension fondamentale, cf. leçon hazard : rétine horizontale). Ce
  n'est PAS un death-trap (détour court, bouffe perceptible hors-axe) mais une **FEATURE de topologie** :
  elle rendra la MÉMOIRE décisive, et explique que l'agent NAÏF forage mal (7 repas / 12 vies = attendu,
  l'évitement n'existe pas encore). ⚠️ **Dette G3** : le juge devra DISSOCIER « évite le solide » de
  « peine à cause de l'occlusion » — la baseline `SYLVAN_OBSTACLE_SOLID=0` (obstacle rendu mais
  traversable) est déjà branchée exactement pour ça.
- **(c) SÉPARABILITÉ** : `cos(bouffe, obstacle) 0.39 ≪ intra 1.00` → cluster couleur distinct → voie B
  pourra l'apprendre. ✅

**➜ G1 PASSÉ** : monde-obstacle bâti, le corps respecte les solides (physique du CORPS, jamais le
cerveau), monde viable (détour court ; l'occlusion est une feature, pas un death-trap), obstacle
séparable. Harnais : `scripts/collect_obstacle_g1.sh`. **PROCHAIN = G2** : entraîner le prédicteur
d'affordance (voie B) sur le signal commandé-vs-réel (le WM prédit un déplacement, le réel ≈ 0 ⇒
blocage), lu par le planner comme un coût — jumeau de la lunette saillance-danger.

## ⭐ VERDICT G2 (2026-07-18, en cours) : infra + G2-0 PASS + prédicteur entraîné (sélectivité OOD = dette)
**Infra label** (commité) : champ `torso` (pose x,z,yaw) ajouté au payload Godot→serveur
(`policy_player.gd` whitelist) + loggé dans le BC_LOG (`serve_planner_command.py`, `wm.torso0`) + hook
local `main.gd` (non stagé) → **déplacement RÉALISÉ** disponible = le label commandé-vs-réel.

**G2-0 (GRATUIT, `diag_obstacle_g2.py`) : PASSÉ** — label auto-supervisé « commandé avant + réalisé ≈ 0
APRÈS avoir bougé » (exclut le reset) = **1230 bloqués / 4916 libres** ; profondeur devant méd bloqué
**0.70 m** vs libre **10.0 m** → **AUC(géométrie seule) = 1.000** (apprenable) ; obstacle cyan devant :
bloqués **100 %** vs libres 20 % (le label NE regarde PAS la couleur). → licencie l'entraînement.

**Prédicteur entraîné** (`train_obstacle_affordance.py`, jumeau `DangerSaliency` : MIL max-pool
`s(rgb)·g(dist)`, BCE + parcimonie, `obstacle_points()` déployable) :
- **(G-auc) AUC CV-par-segment = 1.000** ✅ — apprend « bloqué » du seul label moteur ; **ρ̂ = 0.63 m** appris.
- **(G-sel) sélectivité** : `s(cyan)=1.00` (obstacle) ✅, **`s(rouge)=0.00`** (bouffe) ✅ — **il a DÉCOUVERT,
  du SEUL label moteur, que la bouffe ne bloque PAS et l'obstacle si** (le résultat central : la
  discrimination appearance-agnostic MARCHE). **MAIS `s(bleu)=1.00`, `s(vert)=1.00`** ❌.
- **Diagnostic honnête (§2)** : bleu/vert sont **HORS-DISTRIBUTION** — le monde d'entraînement était
  food-only + obstacle (seules couleurs présentes : rouge + cyan) → la MLP d'apparence extrapole
  n'importe comment sur une teinte jamais vue. Ce n'est PAS un échec de la méthode ; c'est que le
  prédicteur ne peut être sélectif que sur les couleurs **vues**. AUC=1.0 reflète aussi un monde SIMPLE
  (un seul type de bloqueur). **DETTE** : pour un monde multi-drive, ré-entraîner sur un corpus
  CONTENANT eau(bleu)+danger(vert) comme NÉGATIFS (perçus, non-bloquants) → sélectivité robuste.
- **Deux voies pour la suite** (choix owner) : (i) **intégrer + G3 en monde food+obstacle** (couleurs
  bleu/vert ABSENTES → le prédicteur cyan-sélectif suffit, pas de faux positif) et noter la dette
  multi-drive ; (ii) **d'abord ré-collecter en monde riche** (food+eau+danger+obstacle) pour une
  sélectivité juste, puis G3. PROCHAIN quel que soit le choix : `_obstacle_lens` + `SYLVAN_WP_OBSTACLE`
  dans `waypoint_layer.py` (jumeau du swap `_lens`, marge de standoff DÉCLARÉE, pas la ρ̂ réfutée) → G3.

**INTÉGRATION FAITE (2026-07-18) + FINDING G3-monde (honnête).** `waypoint_layer.py` : `SYLVAN_WP_OBSTACLE`
charge la lentille apprise (opt-in, défaut OFF bit-identique) ; `_lens` fusionne les points-obstacles
dans la MÊME machinerie d'intrusion que le danger (validation food+obstacle, sans danger). Vérifié : la
lentille se charge (AUC 1.0, ρ̂ 0.63 m), l'étage waypoint tourne. **MAIS le smoke A/B est INCONCLUSIF** :
aware (3 repas / 12 frames près-du-mur) ≈ blind (3 repas / 0). Cause DIAGNOSTIQUÉE = le mur OCCULTE la
bouffe → le slot single-food se vide → l'agent **PERD sa cible et erre** AVANT d'atteindre le mur → il
ne l'ENGAGE jamais → le contournement n'est pas exerçable. **C'est un finding structurel** : contourner
un obstacle OCCLUANT exige une **persistance de cible (mémoire)** ; sans elle, l'agent erre. → G3 doit
soit (a) un monde NON-occluant (obstacle DÉCALÉ à côté du trajet, bouffe visible, protubérant dans le
chemin → contournement pur, testable sans mémoire), soit (b) activer la mémoire de slot (module
`memoire_spatiale`, partiel) — ce qui **révèle la dépendance à la mémoire = le chantier suivant**. Voie
B (perception+intégration) est bâtie et branchée ; la preuve-en-vies du contournement attend ce choix
de monde G3.

## ⭐ GEL DU CHANTIER (2026-07-18, owner) : voie B bâtie + branchée ; G3 DIFFÉRÉ (pas escamoté)
Le chantier est **GELÉ** à l'état « perception + intégration LIVRÉES, preuve-en-vies EN ATTENTE » :
- **BANKÉ** : G0 PASS (voie B tranchée à coût nul), G1 PASS (monde solide viable + le corps respecte
  les solides), G2 PASS offline (AUC CV 1.000, sélectivité cyan-vs-bouffe DÉCOUVERTE du seul label
  moteur), intégration `SYLVAN_WP_OBSTACLE` dans `waypoint_layer.py` (opt-in, défaut OFF
  bit-identique). WM intact, zéro coût-obstacle codé-main dans la décision.
- **G3 (preuve-en-vies du contournement) = DIFFÉRÉ sur finding structurel**, pas relâché (§2) : un
  mur occluant fait perdre la cible au slot single-food → l'agent erre AVANT d'engager le mur → le
  contournement n'est **pas exerçable sans persistance de cible (mémoire)**. Deux réouvertures
  pré-inscrites : **(a)** monde G3 NON-occluant (mur décalé, protubérant dans le chemin, bouffe
  visible → contournement pur testable sans mémoire) ; **(b)** mémoire de slot (module
  `memoire_spatiale`) — la voie qui révèle la vraie dépendance = le chantier mémoire.
- **Dettes déclarées à la réouverture** : sélectivité OOD (ré-entraîner la lentille sur un corpus
  riche avec eau/danger comme négatifs perçus non-bloquants) ; test « survit au changement
  d'apparence » (critère de pureté officiel) pas encore payé.
- **Pourquoi geler ICI** : l'interface de l'étage waypoint est désormais STABLE (lunette danger +
  lentille obstacle branchées, plus personne n'édite `waypoint_layer.py`) → c'est le socle gelé
  qu'exige le chantier suivant (**critique appris à l'étage waypoint**, pré-inscription
  `docs/design_critique_waypoint.md`) sans cible mouvante ni train-deploy mismatch.

## Ce qu'on ne touche JAMAIS
Le WM (gelé — **sauf** voie A explicitement gatée par G0+G2) ; le readout géométrique du slot ; le
transport ; les drives eux-mêmes ; le planner bas (coût de décision). Le corps qui **respecte les
solides** = **physique**, pas cerveau. Godot : `godot/scripts/main.gd` **jamais stagé**, `ui/`
jamais stagé ; obstacle **opt-in défaut OFF = bit-identique**. Carte
`tools/archi_hud/architecture.json` mise à jour **DANS LE COMMIT du build** (pas de la
pré-inscription).

## Critère de succès = le BUT
En vies : l'entité **contourne** un obstacle qu'elle a **appris** bloquant (pas codé), **forage
préservé**, et la réaction **survit à un changement d'apparence** de l'obstacle. Zéro code par objet :
le **canal** (mouvement) est fourni UNE fois ; l'apparence et l'effet (bloque) sont **découverts**.
Si PASS : premier canal de conséquence non-homéostatique appris ; la **topologie** existe → débloque
*chercher + mémoire spatiale*. Si échec : **négatif commité**, cause diagnostiquée sur trace (le G0
gratuit d'abord = négatif à coût nul).

## Sources
- `docs/research_appearance_consequence.md` **AXE 3** : Montesano et al., *Learning Object
  Affordances* (IEEE T-RO 2008, réseau bayésien action-features-effet — la couleur détectée
  non-pertinente, forme/contact retenus) ; **traversabilité par erreur de reconstruction** (81-85 %
  vs labels, zéro label — jumeau de la lunette saillance-danger) ; **traversabilité self-supervisée
  commandé-vs-réel + MPC** (labels bootstrappés du mouvement commandé-vs-réel = le signal exact du
  canal) ; **évitement model-based émergent** (rollouts dans un WM appris — piste unique et vague,
  côté voie A, **non prouvée**).
- Composition GVF « près d'un obstacle » (Horde, AXE 1) : P(pic capteur-contact) comme affordance
  apprise et composable.
</content>
</invoke>

---

# G3 DÉGELÉ (2026-07-21) — la situation manquante existe enfin

Le chantier avait été **gelé** parce que le G3 n'avait pas de monde où se juger : en arène ouverte,
l'obstacle unique était rencontré trop rarement. La forêt navigable le fournit — **45 arbres**,
réglage mesuré : immobile 5,4 %, vitesse pleine, ≥201 blocages en 3 épisodes. **L'entité se cogne
régulièrement tout en restant fonctionnelle** : c'est exactement la fenêtre où une perception
d'obstacle peut se voir.

Le prédicteur transfère sans ré-entraînement : `s(vert foncé) = 0,985` → **bloquant** (mesuré,
`diag_foret_g0.py`).

## Critères PRÉ-INSCRITS (avant lancement)

Témoin = `arbgrad_graded_s7` (45 arbres, prédicteur OFF) déjà collecté. Traité = même monde, même
seed, `SYLVAN_WAYPOINT=1` + `SYLVAN_WP_OBSTACLE`. Run court, 3 épisodes.

- **PASS** : les **blocages baissent d'au moins 30 %** ET l'immobilité ne monte pas (≤ 8 %).
  *Percevoir les arbres doit faire CONTOURNER, pas figer.*
- **ÉCHEC INFORMATIF** : blocages inchangés → le coût d'intrusion ne pèse pas assez dans le score.
- **KILL** : immobilité > 15 % ou consommations effondrées → le prédicteur **paralyse** (il voit des
  murs partout). ⚠️ Attendu comme plausible : `s(bleu) = 1,00`, donc **il prend l'eau pour un mur**.
  En multi-drive, il pourrait fuir ce qu'il doit boire — c'est le défaut connu, non corrigé.
- **NUL** : `guards.sanity()` échoue → verdict nul, pas négatif.

## RÉSULTAT DU G3 DÉGELÉ (2026-07-21) — **ÉCHEC**, et la cause est la PORTÉE apprise

| | blocages | immobile | conso | ticks |
|---|---|---|---|---|
| prédicteur OFF | ~200-400 | 5,4 % | 4 | 6221 |
| prédicteur **ON** | **~1400-1600** | 3,6 % | 3 | 5003 |

**6× plus de collisions** par tick (le critère demandait −30 %). L'entité n'est pas paralysée
(immobilité en baisse, 3,6 %), elle se cogne simplement bien davantage.

**Cause mesurée, lue dans le log** : `ρ̂ = 0,63 m`. La portée apprise du prédicteur est de **63 cm** —
il ne signale un obstacle qu'au contact. Elle a été apprise sur un monde à **un seul mur**, où réagir
tard suffisait. Dans une forêt espacée de 1,3 m, réagir à 63 cm arrive trop tard pour éviter, et le
détour de l'étage waypoint (couronne 8×2,5 m) part alors **dans les arbres voisins** : chaque
évitement engendre de nouvelles collisions.

⇒ **Le transfert est PARTIEL et ASYMÉTRIQUE** : la reconnaissance de **couleur** transfère
parfaitement (`s(vert) = 0,985`), la **portée** ne transfère pas. Je n'avais vérifié que la couleur —
vérification incomplète, et c'est ce qui a rendu le résultat surprenant.

**Ce que ça implique.** Le ré-entraînement du prédicteur n'est plus optionnel, et il ne concerne plus
seulement le faux positif sur l'eau (`s(bleu) = 1,00`) : il doit **réapprendre ρ̂ dans un monde
d'arbres**. C'est toujours une petite tête (minutes), mais elle a besoin d'un corpus forestier — qui
existe désormais (`arbgrad_graded_s7`, 45 arbres, ≥201 blocages étiquetés commandé-vs-réel).

**Négatif banké** : ne pas rebrancher ce checkpoint tel quel dans un monde dense.
