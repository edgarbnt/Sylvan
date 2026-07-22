# Design — Monde « bosquets » : ressources fixes, épuisables, qui repoussent (2026-07-21)

## Mission
Sept capacités ont été réfutées parce que **rien dans le monde ne les demandait**. Ce chantier
ne touche pas au cerveau : il construit un MONDE, une bonne fois, puis le GÈLE. Prompt de
départ : `docs/prompt_construire_le_monde.md`.

## À lire d'abord
- `ETAT_DES_LIEUX.md` §DIRECTION COURANTE
- `diagnostics/diag_monde_bosquets_g0.py` — le gate gratuit et son `--selfcheck`

---

## 1. Le diagnostic qui commande le design (mesuré, pas supposé)

**La nourriture suit l'entité.** `food_manager.gd:348` `_respawn_near(agent_pos)` téléporte la
pastille consommée dans un anneau **2,0-4,5 m autour de l'agent**, la même frame. Le commentaire
du fichier l'assume (« PERPETUAL FIELD … survival is limited by falling, not by walking out of a
fixed patch »). Mesuré sur corpus : bouffe à **3,2-4,1 m** en médiane, **jamais au-delà de
10,17 m**.

C'est un diagnostic plus précis que « monde 9 m < rétine ». **Agrandir l'arène ne changerait
rien** : le respawn suivrait l'agent dans un monde de n'importe quelle taille.

**Budget métabolique** (mesuré, 5 corpus forêt, téléports filtrés) :

| grandeur | valeur |
|---|---|
| vitesse | 0,011 m/tick (moyenne), 0,018 (pic) |
| drain | 0,05 (faim) + 0,035 (soif) par tick, jauges 0-100 |
| restore | 39,95 par pastille |
| ⇒ une conso obligatoire toutes les | **470 ticks** |
| ⇒ trajet disponible entre deux repas | **5,2 m** |
| ⇒ excursion max réservoirs pleins | **22 m** (l'énergie borne) |
| épisode | 3000 ticks ≈ 33 m parcourus |

**Corrections de constantes documentées** (code vérifié) :
- **rétine = 10,0 m**, pas 12 (`perception.gd:14`, + 12 copies Python). Le « 12 » est
  `NUM_SECTORS`, l'ancien radar-oracle — une autre grandeur. L'argument « 12 ⩾ 9 » qui a clos le
  chantier mémoire reposait sur un nombre emprunté à un autre capteur.
- **`r = √u` n'existe nulle part** : tout placement radial est `randf_range(min,max)`, linéaire
  en r → densité en 1/r. Ce n'est pas un réglage à conserver, c'est un réglage à écrire.
- **espacement mini 1,3 m non appliqué** : `forest_solid.gd` `_min_gap = 0.0` par défaut.

## 2. Le design proposé

Quatre bosquets à **positions fixes**, 2 bouffe / 2 eau, côté 9 m, dans une forêt. Chaque
bosquet contient 4 portions ; une portion restaure **20** points ; une portion repousse toutes
les 600 ticks. Traverser 9 m coûte 818 ticks = **41 d'énergie / 29 de soif** : confortable
réservoir plein, quasi mortel à moitié vide.

Contrainte d'instrument : `restore` doit rester **> 5**, sinon `guards.CONSUME_JUMP = 5.0` rend
les consommations invisibles et annule tout verdict.

## 3. Gate G0 — PRÉ-INSCRIT (avant lancement)

`diagnostics/diag_monde_bosquets_g0.py`, 0 run / 0 Godot / 0 entraînement. On simule le monde
hors moteur et on y fait tourner une échelle de politiques ; **l'écart entre elles EST la mesure
de ce que le monde exige**.

- **PASS** : glouton < 1800 **ET** mémoire > 2600 **ET** écart ≥ 800
- **N'EXIGE RIEN** : écart < 300 → re-dimensionner AVANT de construire
- **TROP DUR** : mémoire < 1800 → monde non viable
- **SUSPECT** : aléatoire > 1000

## 4. RÉSULTAT — **NÉGATIF** : le monde exige l'ENGAGEMENT, pas la MÉMOIRE

24 graines, selfcheck passé.

| politique | survie méd | pleins | conso méd |
|---|---|---|---|
| aléatoire | 1400 | 0/24 | 0 |
| glouton (purement réactif) | 1400 | 0/24 | 0 |
| **`sticky` (engagement, AUCUNE carte)** | **3000** | **23/24** | **14** |
| mémoire complète | 3000 | 24/24 | 13 |

**`sticky` égale la mémoire complète.** Or `sticky` ne retient qu'**un seul point** et s'y tient
jusqu'à l'atteindre — c'est exactement ce que le planner vivant fait déjà via le slot. Le monde
proposé ne demande donc **aucune mémoire spatiale**.

⚠️ **Le « PASS » affiché est un FAUX POSITIF, et le critère était mal spécifié.** Il compare la
mémoire à `greedy`, un homme de paille : `greedy` re-décide à chaque tick, oscille entre deux
bosquets qui tirent en sens opposés et se fige à 1,50 m d'une ressource (le « flottement », déjà
mesuré en vie réelle : 2022 bascules). Battre `greedy` ne prouve rien. Conformément à la règle du
projet, la barre n'est **pas** déplacée après coup : elle est déclarée mal posée.
Second critère mal posé : `RANDOM_MAX = 1000` est **insatisfiable** — sans manger, on survit
exactement `INIT/DRAIN_E = 1400` ticks, donc le plancher est 1400 et l'alerte se déclenche
toujours.

## 5. Ce que le balayage révèle — **l'occlusion, pas l'espacement**

Écart mémoire − engagement, 16 graines :

| côté \ p_visible | 0,58 *(forêt actuelle)* | 0,30 | 0,12 |
|---|---|---|---|
| **9 m** | +0 | +0 | **+734** |
| 14 m | +0 | +0 | +0 |
| 20 m | +0 | +0 | +0 |

Deux enseignements :

1. **Écarter les bosquets ne crée aucun besoin de mémoire.** À 14 et 20 m, les deux politiques
   s'effondrent *ensemble* (2000 puis 1400 ticks) : le monde devient trop grand pour le
   métabolisme, et ça pénalise autant celui qui se souvient. C'est un plafond de **PORTÉE**, le
   même que celui déjà banké — pas un manque de mémoire.
2. **Seule l'occlusion sévère sépare les deux**, et il en faut **p_vis ≈ 0,12** : une ressource
   visible 12 % du temps quand elle est à portée, contre **0,58 mesuré dans la forêt à 40
   arbres**.

🚨 **Et c'est là qu'est le mur, mesuré.** On ne peut pas atteindre p_vis = 0,12 en ajoutant des
arbres : 45 arbres = fenêtre navigable (immobile 5,4 %), **54 arbres → immobile 85 %, 0 repas**.
La densité qui rendrait la mémoire payante est **au-delà de celle que le corps sait traverser**.

## 6. Direction qui en découle (non vérifiée — prochain gate gratuit)

Il faut de l'**occlusion par unité d'obstruction navigationnelle**. Beaucoup de troncs fins
coûtent cher en navigation et occultent peu ; **peu de masses opaques larges** (affleurements
rocheux, fourrés denses de 2-3 m) occultent beaucoup en n'occupant qu'un peu de sol.

**Ce n'était pas démontré — c'est maintenant RÉFUTÉ.**

### 6-bis. GATE GÉOMÉTRIQUE 2D (`diagnostics/diag_occlusion_geom_g0.py`) — **RÉFUTÉ**

Réplique 2D fidèle de `perception.gd` : 36 rayons à 10°, portée 10 m, un bosquet est vu ssi un
rayon l'atteint avant un obstacle. Modélise donc l'occlusion **et** le sous-échantillonnage
angulaire.

**Fait dominant, inattendu : sans AUCUN obstacle, p_vis = 0,64.** L'essentiel de l'invisibilité
ne vient pas des arbres mais du **sous-échantillonnage angulaire** — une cible de 0,35 m à 8 m
sous-tend 5°, soit une demi-inter-rayon. Les 45 arbres ne font passer p_vis que de **0,64 à
0,44**.

Coût de l'occlusion en couverture de sol :

| configuration | p_vis | sol couvert |
|---|---|---|
| 45 arbres r=0,35 m *(monde actuel)* | 0,476 | 5 % |
| 14 masses r=2,5 m | 0,270 | 72 % |
| **10 masses r=3,5 m** | **0,206** | **101 %** |
| 20 masses r=3,5 m | 0,102 | 202 % |

⇒ Atteindre la cible p_vis ≤ 0,20 exige de couvrir **~100 % du sol** de masse opaque ; le p_vis
= 0,12 qui faisait apparaître l'écart mémoire en exige **200 %** (disques largement
superposés). **Ce n'est pas un monde, c'est un mur.** L'hypothèse « peu de masses larges » est
réfutée.

⚠️ **Le volet NAVIGABILITÉ du gate est NUL** (déclaré avant lancement) : aucun des deux proxys
essayés ne sépare les deux ancres mesurées. La percolation en espace de configuration donne
83,7 % contre 80,6 % (il fallait ≥ 20 points d'écart) ; la simulation de la règle de collision
exacte (`_kin_collide` : rayon unique, arrêt net, plus la replanification à 10 ticks) donne
**1,3 % contre 1,4 %**, et reste sous 2,2 % jusqu'à **150 arbres**.
⇒ **Le figeage à 54 arbres n'est PAS géométrique.** L'espace reste connexe et traversable. La
cause vit dans la pile planner/perception, pas dans le monde. **Suspect n°1, déjà documenté et
mesuré** : `slot_vis_thr = 1e-3` (`command_planner.py:121`), dont la fuite sous occlusion vaut
0,0477 = **48× le seuil** → le planner ne sait JAMAIS qu'une ressource est cachée, continue de
viser à travers le tronc et s'y encastre. Correctif = un seuil (~0,15), zéro ré-entraînement.
C'est le prochain test à faire, et il est cheap.

### 6-ter. Conclusion de portée — ce qui est fixable par le monde, et ce qui ne l'est pas

Le facteur limitant de la mémoire n'est pas la topologie du monde, c'est **l'omniscience de la
rétine 360° à 10 m** dans une arène que le corps ne traverse qu'en partie. Aucun agencement
d'obstacles ne la défait à un coût navigable.

- **se souvenir** et **chercher** : **PAS fixables par le monde** ici. Ils demandent de réduire
  la couverture angulaire — donc le cône, donc un ré-entraînement du WM, décision que l'owner
  avait déclinée. Ce gate est un argument NOUVEAU en faveur du cône : on sait désormais que
  l'occlusion ne peut pas s'y substituer.
- **prédire** (repousse différée, non calculable analytiquement) et **éviter** (ratio
  dangereux/inoffensif constant par décile de distance) : **restent fixables par le monde**, et
  ne dépendent ni du champ de vision ni de la portée.

🚫 **Piège à refuser** : rendre les ressources plus PETITES ferait baisser p_vis par pur
artefact d'échantillonnage. Ce serait fabriquer de l'invisibilité au lieu de la structurer —
fausse solution au sens du §2.

---

# 9. LA SOLUTION — ALIASING PERCEPTUEL (mesuré, PASS)

**Reformulation qui débloque tout.** L'invisibilité qui compte n'est pas « je ne vois pas
l'objet » mais « je vois l'objet et son ÉTAT ne se lit pas ». Deux bosquets identiques à l'œil,
l'un plein l'autre vidé : la vision ne les sépare pas, seule la mémoire de ce qu'on a mangé et
quand le fait. C'est la condition FORMELLE du POMDP — dès deux états observationnellement
équivalents aux conséquences différentes, toute politique réactive déterministe est
strictement sous-optimale.

Et ça n'exige **aucun changement de capteur** : le buisson (1,5 m) sous-tend 21° à 8 m et est
échantillonné de façon fiable ; la baie (0,35 m) sous-tend 5°, une demi-inter-rayon, et ne
l'est pas. L'aliasing est une propriété que la rétine a DÉJÀ ; le monde n'a qu'à placer l'état
décisif à cette échelle.

**Mesure** (`--alias 1,5`, 20 graines, `regrow 2000`, 3 portions) :

| | stock lisible partout | **aliasing à 1,5 m** |
|---|---|---|
| `sticky` (engagement, sans carte) | **15/20** pleins, 11 repas | **8/20** pleins, 8 repas |
| `memory` | 20/20 pleins, 11 repas | **20/20** pleins, **14** repas |
| écart | +5 | **+12** |

L'aliasing **divise par deux** la survie de l'engagement et laisse la mémoire intacte. Sur les
repas ils étaient à égalité (11 = 11) ; ils passent du simple au double (8 contre 14). Robuste
sur 6 configurations de rareté.

**L'objection traitée** (soulevée par la recherche) : « la mémoire ne bat l'inspection sur place
que si s'approcher coûte trop cher ». Chiffré ici : vérifier un bosquet à 8 m impose de
s'approcher à 1,5 m, soit **591 ticks = 126 % du budget entre deux repas**. Le seuil de bascule
donné par la littérature est 40 %. **L'inspection n'est pas une alternative** — l'objection est
levée par l'arithmétique, pas écartée.

⚠️ **Limite honnête** : dans la simulation, `memory` connaît le taux de repousse. L'entité
réelle devrait l'apprendre — c'est un seul scalaire, apprenable du vécu, mais ce n'est pas
gratuit et il ne faut pas le compter comme acquis.

# 10. LE CÔNE — pourquoi il n'a rien changé, et ce qu'il coûterait vraiment

**Correction factuelle : le cône EXISTE dans le code.** `SYLVAN_OCCLUDE_FOV_DEG`
(`serve_planner_command.py:46,424`), défaut 360 ; `scripts/diag_nav_ab_memory.sh:25` le met à
**180** par défaut. Il n'a jamais été promu parce qu'il dégradait.

**Pourquoi il dégrade — l'implémentation est une perte sèche.** `occlude_retina` met les rayons
hors-cône à `depth=1, RGB=0` **en conservant l'espacement de 10°**. À ±45° il ne reste que
**9 rayons actifs sur 36** ; les 27 autres transportent « rien ». On perd 75 % de la couverture
sans gagner un iota d'acuité. Un VRAI cône redistribuerait les 36 rayons sur ±45°, soit
**2,5° d'écart, 4× plus fin** — ce qui attaquerait directement le sous-échantillonnage qui
plafonne p_vis à 0,64.

**Et un vrai cône est BON MARCHÉ.** `slot_head.py:41-43` :
`th = [k·2π/NRAY]` → `register_buffer("sin"/"cos")`. Ce sont des **buffers géométriques, pas des
poids appris** ; le décodage de position (`:138`, `:165`) est un soft-argmax sur des rayons
« d'angle CONNU ». Changer le champ de vision = changer une table d'angles, plus une ligne dans
`perception.gd`. **Zéro poids appris touché.** (Reste à mesurer : la dérive du latent, la rétine
passant de 63 % de rayons vides à presque aucun — c'est un check open-loop, pas un ré-entraînement.)

**🚨 MAIS le cône est INVIVABLE dans ce corps, et voici pourquoi.** Simulé : cône seul → **0/20**
épisodes pleins pour l'engagement ET pour la mémoire ; cône + aliasing → 0/20 aussi. Cause,
calculée sur constantes mesurées (`surv_turn_rate = 0,015 rad/tick`, vérifié correct par
l'audit) :

| balayage | ticks | % du budget inter-repas (471) |
|---|---|---|
| 90° | 105 | 22 % |
| 180° | 209 | 45 % |
| **360°** | **419** | **89 %** |

**Se retourner pour regarder consomme 89 % de ce qui sépare l'entité de la famine.** Elle ne
peut pas se payer de regarder. ⇒ **La rétine 360° n'est pas un luxe de design : elle COMPENSE un
corps qui tourne trop lentement pour son métabolisme.** C'est la contrainte cachée que personne
n'avait chiffrée.

**Condition pour que le cône devienne viable** : rendre le balayage payable, c'est-à-dire
`kin_turn` **×4 à ×6** (0,015 → 0,060-0,090 rad/tick), ce qui ramène le tour complet à 105-70
ticks = 22-15 % du budget. C'est **une variable d'environnement** (`SYLVAN_KIN_TURN`), pas un
ré-entraînement. À quoi il faut ajouter un comportement de BALAYAGE, que les politiques
actuelles n'ont pas.

⚠️ Limite de ce verdict : mes politiques simulées ne balaient jamais, donc elles subissent le
coût du cône sans jamais en tirer l'usage. Le cône est donc jugé ici dans sa version **la plus
défavorable**. Ce que le chiffre du balayage établit solidement, c'est le **prix d'entrée** ; il
ne réfute pas le cône, il en donne la condition.

## 7. Palette — mesurée, et un critère du prompt réfuté

Requêtes = canaux purs rouge/bleu/vert, masque dur par argmax (`slot_head.py:59,87-119`).

**Le critère « fuite après seuil = 0 » est insatisfiable** : le gris parfait donne
`cos = 1/√3 = 0,5774` sur les trois requêtes, au-dessus du seuil **0,55**. Aucune couleur RGB ne
le passe. Le modèle reproduit exactement les mesures du projet (brun foncé **0,2271** au
quatre-millième).

Avec l'ancienne palette, la bande de séparation est **[0,885 arbre ; 0,870 eau] = négative** :
aucun seuil ne peut séparer un arbre d'une ressource. Le verrou est **le seuil**, pas la palette.

**Règle proposée : la saturation porte le statut, la teinte porte l'identité.**
Ressource `cos ≥ 0,93` ; décor `cos ≤ 0,75` ; seuil par-type à **0,85** → bande libre **0,195**.

| objet | RGB | cos | canal |
|---|---|---|---|
| bouffe | 0,95 0,22 0,14 | 0,964 | R |
| eau | 0,12 0,34 0,97 | 0,937 | B |
| danger | 0,12 0,95 0,10 | 0,987 | V |
| tronc bloquant | 0,30 0,26 0,34 | 0,651 | — |
| buisson inoffensif | 0,40 0,34 0,16 | 0,729 | — |

Effet secondaire : **le brun redevient utilisable** (0,777 < 0,85). « Tronc brun interdit » était
une conséquence du seuil 0,55, pas une propriété de la couleur.
Limite honnête : sous le seuil le cône neutre est **étroit** — roche, sol et tronc ne sont qu'à
2,5-4,6° de teinte l'un de l'autre. Seule la paire à discriminer (bloquant/inoffensif) obtient
**23,6°**, mieux que l'ancienne paire (21,2°).
**Ne PAS toucher à la rétine** : le correctif est un seuil (`query_thr`, buffer `persistent=False`,
déjà par-type, déjà écrit depuis des mesures par `build_typed_slots.py`). Zéro poids appris
touché. Changer portée/rayons/canaux changerait `obs_dim` → ré-entraînement complet du WM.

## 8. Dettes d'instrument à traiter avant toute mesure en monde occulté
- `command_planner.py:121` `slot_vis_thr = 1e-3` : son propre commentaire mesure la fuite sous
  occlusion à 0,0477 = **48× le seuil** → la porte de visibilité ne se déclenche jamais sous un
  arbre. À porter à ~0,15.
- `diag_reach_curve.py` : bandes `(0,2,4,6,8)` et `SLACK = 3,0 × d/vitesse` supposent l'anneau
  actuel **et un trajet en ligne droite**. En monde à détours, la portée baisserait pour une
  raison de *monde* et se lirait comme une perte de capacité.
- `hazard_manager.gd:156` et `obstacle_manager.gd:120` placent le danger et le mur **sur la ligne
  droite spawn→bouffe**. C'est viser l'agent, ce que le prompt interdit, et c'est le chemin par
  défaut.

## Critère de succès = le BUT
Un écart **mémoire − engagement** ≥ 800 ticks dans un monde dont la **navigabilité reste celle
des 45 arbres**. Tant que cet écart est nul, construire ne sert à rien.


---

# 11. CALIBRATION (étape 1) — le monde est RÉGLÉ, en mono-pulsion

Harnais `scripts/baseline_bosquets.sh`, corps promu + `wm_objcentric_kin`, 5 vies, seed 1.

| config | pleins | repas méd | au plancher de famine | verdict |
|---|---|---|---|---|
| bi-pulsion, 2 bosquets | 0-1/5 | 0-1 | **3-5/5** | MUR |
| mono, 2 bosquets, buisson englobant | 1/5 | 0 | 3/5 | MUR |
| mono, 2 bosquets, baies en couronne | 4/5 | 4 | 1/5 | TROP FACILE |
| **mono, 4 bosquets × 2 baies, repousse 2500** | **3/5** | **2** | **0/5** | **CALIBRÉ** |

Le régime retenu : personne ne meurt aveugle (0/5 au plancher), et 2 vies sur 5 échouent
quand même. C'est la fenêtre où une meilleure décision peut se voir.

**Pourquoi 4 bosquets et pas 2** : avec deux bosquets, se souvenir lequel est vide est trivial
(« pas celui-là »). Avec quatre, il faut retenir **lesquels** on a vidés et **quand** — c'est la
forme what-where-when, et c'est là que la mémoire paie.

## Trois murs traversés, dans l'ordre

1. **Bi-pulsion = impossible par arithmétique.** Bouffe et eau en bosquets séparés à ~10 m : chaque
   bascule de pulsion coûte 909 ticks pour un budget inter-consommation de 471, soit **1,9×**.
   Mesuré : 5/5 vies mortes au plancher avec `repas=0 boissons=1`. **Négatif banké** — ne pas
   ré-essayer le multi-drive en bosquets ségrégués. Il faudrait des bosquets MIXTES (une clairière
   qui porte les deux), au prix de faire disparaître le choix bouffe-vs-eau.
2. **Le marqueur englobait les baies** (§9-bis) — cécité totale, 0 % de localisation.
3. **Trop facile** une fois la vue rendue — corrigé en fractionnant en 4 bosquets plus pauvres.

## Ce qui reste à faire avant de croire quoi que ce soit
- **n = 5, une seule graine.** Le verdict est directionnel, pas solide. L'A/B se fera poolé 2 graines.
- **La mémoire n'est pas branchée.** Rien n'a encore été mesuré sur ce que ce monde EXIGE ; on a
  seulement montré qu'il est vivable et non saturé.
- **Mono-pulsion assumé** : ce monde ne teste PAS l'arbitrage. C'est délibéré (isoler la mémoire),
  pas un oubli.

---

# 12. A/B MÉMOIRE (le juge) — **PARTIEL / effet non distinguable de zéro**

Pré-inscription : `docs/prereg_ab_memoire_bosquets.md`, écrite et commitée AVANT lancement.
Monde calibré (4 bosquets × 2 baies, repousse 2500, mono-pulsion), 2 graines × 20 vies par bras.
Activation vérifiée dans le log serveur : `MÉMOIRE SPATIALE active` côté ON, absente côté OFF.

| | OFF (n=40) | ON (n=40) | écart |
|---|---|---|---|
| repas **moyens** | 1,45 | 1,68 | **+0,23** |
| IC 95 % bootstrap (20k) | | | **[−0,05 ; +0,52]** — contient zéro |
| repas médians | 1,0 | 2,0 | +1,0 *(artefact, voir ci-dessous)* |
| épisodes pleins | 25 % | 20 % | **−5 pts** (confirmation ÉCHOUE) |
| vies au plancher de famine | 8 % | **0 %** | −8 pts |
| direction par graine | | | +0,50 / **−0,05** |

**Verdict pré-inscrit : PARTIEL** (gain poolé atteint, mais la direction s'inverse sur la graine 2).

⚠️ **Ma métrique primaire était mal spécifiée, et je le dis plutôt que d'en changer.** La médiane
d'un petit compte entier saute d'une unité entière : les deux distributions sont quasi identiques
et la médiane passe pourtant de 1 à 2. La moyenne est la statistique honnête ici, et elle donne
+0,23 avec un intervalle qui contient zéro. La barre n'est pas déplacée — elle est déclarée
mauvaise.

**Puissance** : taille d'effet 0,34 σ → il faudrait **~139 vies par bras** pour trancher à 80 %.
Contre 40 mesurées. Payer 3,5× plus pour confirmer un effet aussi mince n'est pas recommandé.

## Le seul effet propre : la mémoire supprime les catastrophes, pas la médiocrité

`off` contient **trois vies à zéro repas** ; `on` n'en contient **aucune** (minimum 1). Le plancher
de famine tombe de 8 % à 0 %. La mémoire ne relève pas la performance typique — elle rattrape les
vies où l'entité errait sans jamais rien trouver. Mécaniquement cohérent : elle sert quand on ne
voit rien, pas quand on voit.

## Cohérence avec le G0 simulé — ce n'est PAS une surprise

La simulation (§9) prédisait un écart mémoire ≈ 0 tant que la visibilité reste à p_vis ≈ 0,58, et
un écart réel seulement vers 0,12. Le monde à bosquets crée bien de l'aliasing, mais l'entité voit
encore assez pour s'en passer la plupart du temps. **Le réel confirme la simulation.** Et le gate
géométrique (§6-bis) a déjà montré que descendre à 0,12 exigerait ~200 % de couverture du sol.

⇒ **Conclusion de l'arc** : l'aliasing par épuisement est un vrai levier mais MODESTE dans ce
corps ; le facteur qui domine reste l'omniscience de la rétine 360° à 10 m. Cf §6-ter : mémoire et
recherche ne sont pas fixables par le monde seul ici.

---

# 13. A/B MÉMOIRE SOUS CÔNE — **PASS**, et c'est une INTERACTION

Pré-inscription + avenant cône + caveat d'interprétation : tous commités AVANT le run.
Activation vérifiée par bras dans le log serveur (cône des deux côtés, mémoire côté ON seulement).

| condition | repas moy | pleins | plancher |
|---|---|---|---|
| 360°, rotation 1,5, mémoire off | 1,45 | 25 % | 8 % |
| 360°, rotation 1,5, mémoire ON | 1,68 | 20 % | 0 % |
| cône 120°, rotation 6,0, mémoire off | 1,62 | 28 % | 0 % |
| **cône 120°, rotation 6,0, mémoire ON** | **3,80** | **98 %** | 0 % |

- effet mémoire à 360° : **+0,23** (rien)
- effet corps rapide + cône sans mémoire : **+0,18** (rien)
- **effet mémoire SOUS CÔNE : +2,17**, IC 95 % bootstrap **[+1,77 ; +2,58]**, 1,52 σ,
  direction identique sur les 2 graines (+2,20 / +2,15)

**Ni l'un ni l'autre séparément ; les deux ensemble.** C'est la signature d'une interaction, et elle
était PRÉDITE : le G0 simulé annonçait un écart nul tant que le hors-vue reste marginal. On l'a fait
passer de 6,2 % à 73,2 % en redistribuant les rayons, et la mémoire est devenue porteuse.

Le bras ON fait 3,80 repas pour un besoin métabolique de 3,75 : il mange exactement ce qu'il faut.

## Caveats — à lire avec le résultat, pas après

1. **Confusion cône × vitesse de rotation.** Les deux ont changé ensemble. Le bras OFF montre que la
   vitesse seule n'apporte rien (+0,18), donc elle ne porte pas l'effet — mais la cellule
   « cône + rotation lente + mémoire » n'a PAS été mesurée en vrai (seulement en simulation, où elle
   donnait 0/20). Le rôle exact de la vitesse reste non isolé.
2. **La survie sature de nouveau** (98 % pleins) : ne plus juger là-dessus, utiliser les repas.
3. **C'est de la mémoire PASSIVE, pas de la perception active** (caveat pré-inscrit, `ee5dd65`) :
   le coût du planner ne contient aucun terme d'information, l'entité ne tourne jamais POUR regarder.
   Elle retient ce qu'elle a vu en se déplaçant pour d'autres raisons. Ne pas revendiquer autre chose.
4. **Fidélité du latent du WM sous cône : NON VÉRIFIÉE.** Le slot transfère par construction (angles
   géométriques recalculés), mais l'encodeur a été entraîné sur des rétines 360° et voit désormais une
   distribution inédite. Dette à payer avant de construire dessus.
5. **Le corps a changé** (rotation ×4) : toutes les constantes de décision calibrées sur l'ancien
   corps sont désormais suspectes, `surv_turn_rate = 0,015` en premier.
