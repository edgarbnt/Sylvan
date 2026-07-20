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

## ⭐ VERDICT G1 (2026-07-18, machinerie commit dfd4a75 + corpus arb3/arb4) : PASSÉ — corpus contrefactuel livré
**Machinerie ε-CIBLE** : `SYLVAN_TARGET_EXPLORE_EPS` (command_planner.py, branche `plan_multi_surv`)
flippe le choix de cible avec prob ε et le **TIENT K replans** (un flip non tenu = le flottement
pathologique de G0, pas un contrefactuel) ; RNG dédié seedé, décisions forcées **flaggées
`explore_target`** dans le BC_LOG (le trainer sépare politique et contrefactuel). Sonde de
placement préalable (gratuite) : 55,5 % des replans sont MULTI dans ce monde et **127/131 morts-
par-arbitrage ont les deux ressources visibles au dernier replan utile** → la branche multi est le
bon point d'injection. Bit-identité à ε=0 : **par construction** (garde `eps>0` = zéro instruction
nouvelle exécutée, aucun tirage, aucune clé de log) — déviation assumée de la lettre du gate (pas
d'A/B ε=0 payé) ; smoke fonctionnel à ε=0.6 : tenues 15-45 replans, 72 % des forcés contredisent
le choix designé.
**Corpus** (`collect_arb_corpus.sh`, base = CONFIG VIVANTE waypoint [lunette + sprint-critic
décontaminé] = parité train/déploiement ; ε=0.05, K=15 calibrés et déclarés ; seeds 3+4) :
- **48 vies (24+24, cross-check godot.log exact)**, 34.7k ticks/run, 3471 replans/run,
  847+983 décisions forcées.
- **128 bascules TENUES** (segments n≥5, méd 12-15 replans ≈ 2.4-3 m de poursuite) ;
  **désaccord avec le choix designé à l'initiation : 108/128 (84 %)** = vrais contrefactuels ;
- **contraste d'issue : 54/128 (42 %) des poursuites forcées PAIENT** (conso du type forcé
  pendant/juste après la tenue) — ni degeneré (0 %) ni trivial (100 %) → apprenable ;
- **viabilité sous ε** : 19+24 et 13+21 repas+boissons /24 vies (bande des réfs designées) ;
  structure de morts inchangée (danger 5/run ≈ réf) ; les morts-par-arbitrage PERSISTENT sous ε
  (13 et 11) — l'ε n'a pas « réglé » le déficit par accident, attendu.
**➜ G1 PASSÉ. PROCHAIN = PIN DE FORME voie A + seuils G2 chiffrés dans ce doc (owner), AVANT
tout train** (budget dur : 1 train + 1 re-train diagnostiqué).

## ⭐ PIN DE FORME (D1, tranché owner 2026-07-20) + seuils G2 — écrits AVANT tout train
**Forme au déploiement (voie A, réplique exacte de la doctrine sprint)** — à chaque replan MULTI,
pour chaque cible t ∈ {food, water} visible :
```
S(t) = dist(t) − 0.02 · max(0, P̂(obtenir|s,t)·bénéfice(t) − κ_data·douleur̂(t)·100 − P̂mort(t)·κ_data·100)
```
choix = argmin S(t), avec COMMITTMENT pro-cible-courante : le challenger doit battre l'incumbent
de **δ = 75 pas × 0.02 = 1.5 m d'équivalent-score** — l'ancre de bruit slot-jitter DÉJÀ calibrée
du planner (commentaire COMMITTMENT, command_planner.py), réutilisée telle quelle : zéro constante
nouvelle.
- **P̂(obtenir|s,t) = LA SEULE TÊTE ENTRAÎNÉE** : MLP jumeau SprintCritic (14-d), BCE sur `got`
  (le type POURSUIVI consommé dans la fenêtre de poursuite : jusqu'à bascule de cible /
  consommation / mort / cap 600 pas), corpus = replans MULTI des 12 runs instrumentés (10 runs G0
  + arb3/arb4, les lignes ε flaggées incluses), **CV-4 par VIE**. Entrées = contrat
  `sprint_inputs` INCHANGÉ appliqué au candidat DIRECT-vers-t : `candidate_features(t, t, greens)`
  (10-d) + [énergie/100, soif/100, santé/100, douleur̂(t)/100].
- **GELÉ** : douleur̂ = `waypoint_pain_decont` (le vivant) ; P̂mort = `sprint_critic/death_best.pt`
  (bankée P2-bis, AUC 0.839) ; greens offline = lunette `danger_saliency` (la vivante) sur le
  retina0 loggé — parité featurization train/déploiement.
- **MESURÉ (zéro constante fittée)** : bénéfice(t) = min(restore_mesuré, 100−drive(t))/drain_mesuré
  (satiété exacte, pas apprise) ; κ_data re-mesuré sur le corpus poolé ; 0.02 m/pas = corps calibré.
- **Interdits reconduits** : label U net ; distillation du designé (blanchiment) ; constante
  ajustée pour faire passer un gate ; toucher au scoring waypoint/sprint (interface gelée).

**Seuils G2 (pré-enregistrés, offline, gratuits — écrits ICI avant le train)** :
1. **G-rank** : AUC(P̂, got) > **0.70** en CV-4 par vie sur les décisions tenues (ε incluses).
2. **G-res** : précision du choix simulé (argmin S + committment) vs la cible empiriquement
   meilleure du bucket (état drives × écart de distance) ≥ **designé + 10 pts** sur décisions
   tenues.
3. **G-consist** : taux de bascule du choix simulé, rejoué sur les séquences réelles de replans
   multi, ≤ **1.2×** celui du designé (LE tueur historique v2/v3).
4. **G-mono** : P̂·bénéfice strictement DÉCROISSANT par bande de satiété du drive de la cible ET
   remise CROISSANTE par bande d'urgence (buckets peuplés).
Budget dur : **1 train + 1 re-train** sur hypothèse nouvelle diagnostiquée sur trace ; au-delà →
négatif commité + STOP. Le juge G3 (2×24 vies seeds 1+2, PASS chiffré vs réf vivante MESURÉE avant
le run) n'est payé QUE si G2 passe.

## ⭐ VERDICT G2 (2026-07-20, `train_arb_critic.py`, 1 train — budget re-train NON consommé) : 3/4 GATES ✅, G-mono ❌ DIAGNOSTIQUÉ CRITÈRE CONFONDU → correction owner en attente
Corpus : 288 vies / 11 688 décisions multi (842 ε ; 13 114 records waypoint-override ÉCARTÉS —
positions non fiables, honnêteté corpus), got=0.45, mesures κ=9.14 / drain 0.05 / restore 39.95
(cohérentes sprint). Featurization = parité stricte (candidate_features + lunette saillance +
douleur̂ decont importés des modules vivants).
- **G-rank ✅ AUC CV-4 par vie = 0.787** (plis 0.749/0.767/0.772/0.861, tous > barre 0.70).
  Nettement au-dessus du plafond ~0.68 du niveau sprint — le niveau CIBLE a le rapport
  signal/bruit que le diagnostic de la fente prédisait.
- **G-res ✅ 85.9 % vs designé 69.6 % (+16.3 ≥ +10)** — sans fuite : table de buckets sur plis
  d'entraînement, scores par le réseau du pli, jugé sur plis tenus (11 688 décisions).
- **G-consist ✅ 3.7 % ≤ 6.5 %** (1.2× designé 5.4 %) — le choix appris + committment δ=1.5 m
  bascule MOINS que le designé (le tueur historique est absent).
- **G-mono ❌ — mais le diagnostic gratuit sur trace montre un CRITÈRE CONFONDU, pas un modèle
  pervers** : (1) `bén = min(40, 100−drive)/drain` est PLAT (=799) sous drive 60 → les bandes
  [0,30) et [30,60) comparaient P̂ seul, pas la satiété ; (2) P̂ est CALIBRÉE par bande
  (P̂ vs got réel : 0.367/0.387, 0.513/0.486, 0.479/0.484, 0.341/0.326) et sa chute à drive<30
  est la VÉRITÉ vécue — **52 % de ces poursuites meurent avant d'obtenir** ; exiger P̂·bén ↓
  strict sur cette zone = exiger que la tête MENTE sur la mortalité du désespoir. (3) Là où bén
  varie réellement (drive>60) : **P̂·bén = 325 → 86, strictement décroissant ✓**. Le volet
  urgence↑remise est confondu par la même mortalité (la remise DOIT baisser quand mourir en
  route est probable — c'est précisément ce que la forme doit pricer).
- Sonde comportementale annexe (consignée, pas un gate) : le choix appris suit l'urgence plus
  DOUCEMENT que le designé (choix-food 0.32→0.36→0.42 par bande d'écart d'urgence vs designé
  0.12→0.35→0.74) — c'est le juge G3, en vies, qui dira si cette pondération douce (informée par
  P̂/mortalité) bat le suivi d'urgence raide du designé.
**PROPOSITION G-mono-v2 (correction à découvert, précédent sprint volet-blessés — décision
OWNER, juge G3 INCHANGÉ)** : remplacer les deux volets confondus par (a) P̂·bén strictement
décroissant sur la zone où bén VARIE (bandes [60,80) vs [80,100] — mesuré 325 > 86 ✓) ET
(b) CALIBRATION : |P̂ − taux réel d'obtention| ≤ 0.05 par bande de satiété (mesuré
0.020/0.027/0.005/0.015 ✓). Le volet urgence-remise est RETIRÉ comme confondu (la mortalité est
pricée par construction). Si l'owner tranche G-mono-v2 → gates 4/4 → **G3 licencié** (2×24 vies
seeds 1+2, PASS chiffré vs réf vivante mesurée AVANT le run). Sinon → négatif commité, designé
conservé. Ckpt bankée : `data/checkpoints/arb_critic/arb_best.pt` (gates_pass=False tant que
G-mono n'est pas tranché).

## ⭐ G-mono-v2 TRANCHÉ (owner 2026-07-20) → gates 4/4 → G3 LICENCIÉ + protocole pré-enregistré
**Décision owner** : G-mono-v2 accepté tel que proposé — (a) monotonie P̂·bén sur la zone où bén
varie (mesuré 325 > 86 ✓) + (b) calibration |P̂−réel| ≤ 0.05 par bande (mesuré 0.005-0.027 ✓) ;
volet urgence-remise retiré (confondu par la mortalité, pricée par construction). **G2 = 4/4.**
**Déploiement (commit de ce jour)** : `SYLVAN_ARB_CRITIC=<ckpt>` dans `command_planner.py` —
remplace le CHOIX de cible du replan multi par la forme pinnée (toutes constantes lues du ckpt :
κ, drain, restore, δ committment) ; le scoring des COMMANDES vers la cible choisie reste designé ;
sf/sw designés toujours loggés (diagnostic) + `arb_scores`/flag `arb` au BC (corpus honnête) ;
`arb_ok=False` pendant les legs waypoint (positions = overrides, exclues du corpus au train =
parité) ; santé passée au planner (parité features). Opt-in, défaut OFF bit-identique.
**Protocole G3 (pré-enregistré AVANT tout run)** : 2 bras × 2 seeds (1+2, propriété du juge) ×
24 vies, monde v2, config vivante waypoint (lunette + sprint décontaminé), ε OFF, harnais
`scripts/judge_arb_critic.sh <ref|arb> <seed>` (même collecte instrumentée BC que les corpus).
**La réf (bras designé) est RE-MESURÉE D'ABORD**, puis le PASS est chiffré à partir d'elle AVANT
de lancer le bras appris, par cette formule :
- **conso poolées** (repas+boissons) du bras arb ≥ **réf − 5** (bruit d'instrument) ;
- **morts-par-arbitrage poolées** (`diag_arbitrage_g0`, facteur 1.0, même parseur pour les 2
  bras) ≤ **réf_arb − 8** (une vraie baisse, au-delà du bruit) ;
- **morts totales** ≤ **réf + 2** ;
- **KILL précoce** : au seed 1, conso(arb) < conso_réf_s1 − 10 → stop avant le seed 2.
Échec → négatif commité, le designé reste (il est jugé), la tête reste bankée.

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
