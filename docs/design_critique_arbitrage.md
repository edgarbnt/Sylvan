# Design — Critique d'ARBITRAGE DE CIBLE (multi-drive) à l'étage waypoint — pré-inscrit 2026-07-18

> Pré-inscription écrite AVANT tout diag/run/train (§1). Chantier ouvert à la clôture/gel du canal
> obstacle (voir `design_obstacle_affordance.md` §GEL) sur la direction owner « on passe au
> critique ». L'étage waypoint est désormais à INTERFACE GELÉE (lunette danger + lentille obstacle
> branchées) : ce chantier LIT cet étage, il ne réécrit ni les lentilles ni le scoring du sprint.

## Mission
Rendre APPRISE la dernière grande décision DESIGNÉE du chemin vivant : **QUEL besoin poursuivre
(bouffe vs eau vs rien) et QUAND basculer**. Aujourd'hui cet arbitrage est un coût designé
(`command_planner.py` `cost_mode="survival"` : rollout drain/refill analytiques + **continuation
ALTERNÉE codée-main** `_survival_extension` + foresight `survival_weight=300`) — déclaré en carte
« queue analytique = échafaudage 3ᵉ verrou, à remplacer par le critique appris », et déclaré dans
le code lui-même (`command_planner.py:103` : « la version pure = tête drive-dynamics APPRISE »).
Le critique renaît à l'étage HAUT, où sa condition de validité (peu d'options très dissemblables →
écarts de valeur larges) est satisfaite — le sprint-critique l'a PROUVÉ en vies (juge 45/8).
Objectif falsifiable : l'arbitrage appris **≥ l'arbitrage designé, jugé en vies**, zéro nouvelle
constante fittée.

## À lire d'abord
- `docs/design_critique_sprint.md` — LE gabarit (chantier PASS) + la doctrine de FORME : liens
  APPRIS × constantes MESURÉES du corps + socle analytique consistant + modification plafonnée ;
  ni remplacement, ni monolithe sur label net.
- `docs/design_critique_waypoint.md` — les 3 négatifs v1/v2/v3 (le déficit est la FORME) =
  CONTRAINTES, pas des options.
- `docs/design_purete_hjepa.md` §P2/P2-bis — **CLOS, NE PAS ROUVRIR** : forme pure jugée ÉCHEC
  (49/14), G-kill échec MATHÉMATIQUE (l'espérance risque-neutre ne refuse pas une traversée
  rentable-en-moyenne qui tue) → **W = PRÉFÉRENCE DU CORPS** (aversion câblée, §3) ; têtes
  douleur̂/P̂mort bankées et réutilisables.
- Mémoire `sylvan-second-drive-arbitration` — le MUR documenté : arbitrage MYOPE (« meurt de faim
  campé sur l'eau »), fix designé foresight = gain réel mais MODESTE (PLAFONNE) ; 3 verrous notés
  de la voie apprise (MC off-policy, WM aveugle au repas, eat-dynamics).
- Code : `python/sylvan/control/planning/command_planner.py:83-105` (coût survie designé),
  `:167` (`_survival_extension`, continuation alternée), `:932-941` (foresight déficit),
  `python/scripts/serve_planner_command.py:112-113` (`order_scores` sf/sw = l'instrumentation de
  l'ordre food-vs-water, loggée par replan).

## Le mécanisme DESIGNÉ visé (état des lieux exact, pour mesurer honnêtement)
1. À chaque replan multi-ressource, le planner score les candidats par PAS-VÉCUS SIMULÉS :
   rollout WM (drain analytique + refill au contact) puis **continuation ALTERNÉE codée-main**
   (l'entité est supposée alterner bouffe/eau indéfiniment) + pénalité foresight
   `survival_weight·deficit` (anti-myopie designée).
2. L'ordre food-vs-water qui en sort est loggé (`order_scores` sf/sw) ; la cible retenue descend
   à l'étage waypoint qui route VERS elle (le « vers où » est arbitré AVANT l'étage waypoint).
3. Constantes du modèle interne : `resource_drain=0.0016`, `resource_restore=0.5`,
   `survival_weight=300` — un MODÈLE designé du métabolisme dans l'imagination, distinct des
   drains RÉELS du corps (qui, eux, sont légitimes §3).

## Essayé → résultat (contraintes héritées, ne pas répéter)
- **Fente arithmétique du BAS** (`critique_appris`, carte) : erreur réseau 19-47× l'écart à
  trancher entre 33 candidats quasi ex-aequo → tout critique-cible vit à l'étage HAUT. Jamais de
  critique par-commande.
- **3 négatifs critique-waypoint** : notes MC par état = choix FLOTTANTS ; l'analytique gagne par
  CONSISTANCE → G-consist obligatoire, socle analytique conservé, modification graduelle/plafonnée.
- **P2/P2-bis** : l'aversion au risque n'est PAS apprenable en espérance → les préférences
  (aversion, W) restent au CORPS ; on n'apprend que des LIENS (P̂ d'issue), valués par des
  constantes MESURÉES.
- **Foresight designé PLAFONNE** (2026-06-26, époque hexapode) — fait fondateur à RE-MESURER sur
  le corps cinématique en G0, pas à présumer (§2).
- **Plafond épars MÉTABOLIQUE** (leçon critique-résidu) : en 1+1 épars une part des morts est
  ARITHMÉTIQUE (rien d'atteignable) → G0 doit SÉPARER mort-par-arbitrage de mort-métabolique,
  sinon on chiffre une place fantôme.
- **Leçon auto-confirmante** (corpus sprint g24, 0 % ε) : le vécu designé ne contient PAS les
  contrefactuels de bascule → toute voie apprise exigera une collecte ε-CIBLE (G1).

## Les voies (le G0 en TRANCHE une, comme au chantier obstacle)
- **Voie A — liens composés au niveau CIBLE** (réplique du sprint-pattern) :
  `score(cible|s) = coût-temps analytique (distance/odométrie) − P̂(obtenir|s,cible)·bénéfice_mesuré(drive,satiété) + risque appris (douleur̂/P̂mort en route)`,
  hystérésis pro-cible-courante CONSERVÉE (anti-flottement). Réutilise le contrat `sprint_inputs`
  (features candidates + drives + douleur̂) et les têtes bankées.
- **Voie B — tête drive/eat-dynamics APPRISE** (la voie nommée dans le code, 3ᵉ verrou) : la
  continuation alternée + drain/refill du MODÈLE INTERNE remplacés par une tête apprise du vécu
  (ce que le contact restaure, à quelle vitesse chaque drive descend EN SITUATION) ; l'algorithme
  de simulation (général) reste.
- **Voie C (étroite) — bascule apprise seule** : apprendre uniquement QUAND abandonner la
  poursuite courante pour l'autre besoin (la décision de switch), le reste designé intact.

## Gates PRÉ-ENREGISTRÉS (falsifiables, ordre pas-cher-d'abord)

### G0 — LA PLACE, chiffrée gratuitement (0 run, 0 Godot, 0 train). **GATE TOUT LE CHANTIER.**
Le sprint-critique n'a été licencié QUE parce que sa place était chiffrée (géométrie 34/11 vs
oracle 47/9). Ici pareil : pas de place mesurée → pas de chantier.
Sur les corpus vécus EXISTANTS (multi-drive 1+1 : `critic_kin_*`, runs judge/pure instrumentés ;
BC_LOG + `decisions.jsonl` + `godot.log`, machinerie de jointure `load_sprint_decisions`) :
1. **Reconstruire par replan** : cible choisie (`plan.target`/`order_scores`) + drives + issue de
   poursuite (obtenu / basculé / mort, fenêtres de poursuite v3).
2. **Chiffrer le déficit d'arbitrage** en séparant les classes (§2) :
   - (a) **morts-par-arbitrage** : mort d'un drive avec l'AUTRE drive haut ET une ressource du
     type manquant PERCEPTIBLE dans l'épisode ET métaboliquement atteignable au moment du dernier
     replan utile (distance×drain < réserve) — vs **morts MÉTABOLIQUES** (rien d'atteignable :
     hors-place, plafond connu) ;
   - (b) **bascules pathologiques** : flottement (switches répétés sans consommation) et bascules
     TARDIVES (l'autre drive sous seuil critique avant le switch) ;
   - (c) **campements** : séjour prolongé près d'une ressource pendant que l'autre drive plonge.
3. **Borne d'oracle-hindsight** : sur les morts-par-arbitrage, rejouer géométriquement (odométrie
   + drains mesurés) « et si la bascule avait eu lieu au meilleur replan ? » → nb de vies/repas
   RÉCUPÉRABLES = LA PLACE.
**VERDICT G0** (pré-enregistré) :
- Place > bruit d'instrument (±5 repas équivalents / 24 vies, la barre du sprint) → chantier
  LICENCIÉ, et la LOCALISATION du déficit TRANCHE la voie : jamais-basculé/campé → voie A ou C ;
  modèle interne faux (drains/refill simulés ≠ vécus, continuation alternée irréaliste) → voie B.
- Place ≤ bruit → **STOP chantier, négatif commité** (« l'arbitrage designé n'est PAS le goulot du
  monde courant ») — on ne bâtit pas un critique pour une place qui n'existe pas ; retour owner
  (candidats suivants : configurator, mémoire).
- Données insuffisantes pour reconstruire (champs manquants) → G0b : instrumentation ADDITIVE
  cheap (log enrichi) + petite collecte de re-mesure AVANT toute conclusion — jamais conclure sur
  un corpus qui ne porte pas la question.

### G1 — corpus ε-CIBLE (payé si G0 licencie)
Collecte avec ε-switch de CIBLE (contrefactuels de bascule absents du vécu designé), petit ε,
**seeds 3+4** (les seeds 1+2 restent la propriété du juge), log enrichi ADDITIF : par replan, les
features des DEUX cibles candidates + drives + cible retenue. Smoke bit-identité à ε=0 avant la
collecte pleine.

### G2 — train + gates offline (forme PINNÉE au verdict G0 ; budget dur 1 train + 1 re-train)
Gates du gabarit sprint, adaptés à l'arbitrage et pré-chiffrés AVANT le train dans ce doc (les
seuils exacts seront écrits au moment du pin de forme, jamais après coup) :
- **G-rank** : la forme apprise ordonne les poursuites empiriquement meilleures par bucket
  d'états (drives×distances), AUC > 0.70 CV-4 par vie ;
- **G-res** : choix de cible simulé ≥ designé + marge pré-fixée sur décisions tenues ;
- **G-consist** : taux de bascule simulé ≤ 1.2× le designé (LE tueur historique) ;
- **G-mono** : satiété fait décroître le bénéfice ; urgence du drive fait croître la priorité —
  gradients dans le bon sens sur les buckets peuplés.
Interdits reconduits : label U net monolithique ; constante ajustée pour passer un gate ;
distillation du designé (blanchiment).

### G3 — juge closed-loop (payé si G2)
2×24 vies **seeds 1+2** (jamais vus au train), monde multi-drive vivant, bras appris vs réf
designée MESURÉE en G0 (pas de re-run de la réf). **PASS pré-chiffré à l'issue de G0** (survie /
repas+boissons / morts-par-arbitrage poolés, seuils écrits dans ce doc avant le run). **KILL
précoce** : premier seed < réf − bruit. Échec → négatif commité, le designé RESTE (il est jugé).

## ⭐ VERDICT G0 (2026-07-18, `diagnostics/diag_arbitrage_g0.py`, gratuit 0 run/0 train) : PASSÉ — place LARGE, voie A
240 vies vécues (10 runs instrumentés monde v2 1+1 : g24×4, spx×2, judge×2, pure×2), selfcheck
découpage de vies 24/24 vs godot.log (la 1ʳᵉ détection ratait exactement le cas intéressant —
mort d'un drive pendant que l'autre est plein — corrigée sur la signature de reset MESURÉE
(70,70,100)).
- **LA PLACE** : morts-par-arbitrage (ressource du drive mortel VUE + métaboliquement ATTEIGNABLE
  au dernier replan utile, cible AUTRE choisie à ce moment) = **13.1/24 vies** (atteignabilité
  ligne-droite) et **16.1/24** (errance ×2) — **2.6-3.2× la barre pré-enregistrée (5.0)**. NB
  honnête : les deux facteurs d'atteignabilité ne bornent pas la place de façon monotone
  (resserrer déplace des « poursuites échouées » vers « bascule tardive ») ; les DEUX lectures
  concordent largement au-dessus de la barre — c'est le critère.
- **Contrôle dilemme (§2)** : à la mort, l'AUTRE drive = q1/méd/q3 **40/40/80**, **92-94 % > 40**
  → vrais ratés d'arbitrage (le pic à 40 = l'autre ressource VENAIT d'être consommée : signature
  du campement), PAS des dilemmes perdus d'avance.
- **Localisation** : jamais-basculé **73**, bascule-tardive **58-93**, **campements 116/240 vies
  (48 %)**, flottement **2022 bascules food↔water sans consommation** (~8.4/vie). Le déficit est
  LE CHOIX DE CIBLE ET LE MOMENT DE BASCULE, pas le modèle interne de continuation. **→ VOIE A
  (liens composés au niveau cible)**, avec la CONSISTANCE (hystérésis pro-cible-courante) en
  contrainte DURE (le flottement mesuré est déjà pathologique) ; voie C = fallback étroit ; voie B
  non indiquée par les données.
- **Découverte** : metabolique_vue = **0**, jamais_vue = **1** → dans ce monde, quasi TOUTES les
  morts de drive avaient leur ressource vue+atteignable à un moment de la vie ; la mort venait du
  CHOIX. (Le plafond « métabolique » documenté reste vrai en SOUTENABILITÉ long-terme : la borne
  hindsight par-décision est une borne SUPÉRIEURE — récupérer une décision peut ne faire que
  reporter la mort. C'est le juge G3, en vies, qui dira la part réellement convertible.)
- **Sanity croisée** : morts-danger 56/240 = 5.6/24 vies ≈ réf juge g24 (11/48 = 5.5) ✓ ;
  237/240 morts (3 tronquées) = monde dur, cohérent avec les réfs.
**➜ CHANTIER LICENCIÉ (per pré-enregistrement). PROCHAIN = G1** (collecte ε-CIBLE seeds 3+4 — le
flottement vécu n'est PAS un contrefactuel de bascule TENUE ; la machinerie ε-cible est à écrire),
puis PIN de la forme voie A + seuils G2/G3 chiffrés dans ce doc AVANT tout train.

## Ce qu'on ne touche JAMAIS
Le WM (gelé) ; slots/transport ; les drains/restores du CORPS réel (homeostasis — §3, conception) ;
W/marges/aversion (préférences du corps, closes P2/P2-bis) ; le sprint-critique vivant et les
lentilles de l'étage waypoint (interface gelée) ; les candidats géométriques ; les seeds du juge.
`main.gd` jamais stagé. Carte `architecture.json` mise à jour dans le même commit que chaque
verdict.

## Critère de succès = le BUT
En vies : survie multi-drive ≥ config vivante ET morts-par-arbitrage ↓ (les deux, poolés, seeds du
juge — jamais un proxy offline). Si PASS : la queue analytique designée (continuation alternée +
foresight) sort du chemin vivant, remplacée par la décision apprise — deuxième décision apprise
validée à l'étage waypoint, et la principale décision de SURVIE de l'entité devient apprise du
vécu. Si échec : négatif commité, cause diagnostiquée sur trace, le designé reste (il est jugé).
