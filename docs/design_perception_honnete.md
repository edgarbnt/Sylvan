# Design — PERCEPTION HONNÊTE : qu'elle sache quand elle ne voit pas — pré-inscrit 2026-08-02

> Pré-inscription écrite AVANT tout diag/train (§1). Ouvre après le STOP du chantier mouvement-des-objets,
> qui redirige ici — comme sa réserve l'annonçait avant d'être lancé.

## Mission
L'entité **invente une position et ne sait pas qu'elle invente**. Lui donner la connaissance de sa
propre ignorance. Le but n'est PAS qu'elle voie mieux — c'est **qu'elle arrête de mentir** au reste
de l'architecture.

## Pourquoi c'est architectural, pas cosmétique
Une perception qui ment **casse tout ce qui est au-dessus d'elle** : le modèle du monde rêve sur de
la fiction, et le planificateur choisit sur ce rêve. `[MESURÉ: diag_critique_rang_g0.py]` Aucune
lecture du rêve ne corrèle avec le résultat réel (meilleur |ρ| = 0,209 ; hasard corrigé p = 0,334).

À l'inverse, une perception qui dit « je ne sais pas » rend la prédiction et la mémoire
**nécessaires** au lieu d'optionnelles. C'est la raison d'être de l'architecture.

⚠️ **Le but n'est pas « voir tout le temps ».** Une entité qui voit tout n'a besoin ni de mémoire,
ni de prédiction, ni de perception active — elle rendrait le modèle du monde inutile. Le capteur
limité (cône 120°, occlusion) est une propriété du CORPS, définie une fois, et elle reste.

## L'état mesuré
| | valeur |
|---|---|
| erreur de gisement quand un rayon touche VRAIMENT la cible | **10,5°** |
| erreur quand aucun ne la touche | **33°** |
| part des ticks sans aucun rayon sur la cible | **61 %** |
| jauge de visibilité SERVIE (taux de bon classement) | **56 %** — quasi pile ou face |
| tête apprise déjà mesurée (juillet) | **67 %** |
| coût du biais de visée en capture | **−14,8 pts** |

## QUATRE chemins indépendants mènent ici, tous mesurés le 2026-08-02
1. ablation de capture — le biais de visée coûte **−14,8 pts** ;
2. G0 critique-de-rang — le rêve ne porte aucun signal sur le résultat (p = 0,334) ;
3. estimation naïve du mouvement — **3,3× pire** que de supposer l'objet immobile ;
4. G0 mouvement — SNR max **0,59**, et **68°** d'erreur même avec des étiquettes parfaites.

## RÉFUTÉ AVANT DE COMMENCER — ne pas construire ça
**« Confiance apprise par consistance de transport »**, que j'allais proposer. Deux raisons
indépendantes de l'écarter :

1. `[MESURÉ: design_perception_pure_faim.md §5]` la consistance de transport seule **se verrouille
   sur les troncs** — résidu = `prey_speed × gap` exactement ; un arbre immobile est plus
   consistant qu'une proie qui fuit. La confiance dirait « je vois très bien cet arbre ».
2. [Learning from World Feedback (arXiv 2607.16591)](https://arxiv.org/html/2607.16591) : pénaliser
   l'incertitude du MODÈLE **augmente** les collisions (26 % → 34 %), car elle est anticorrélée au
   danger réel (r < 0,15) — elle pousse vers ce qui est *prévisible*, pas vers ce qui est *sûr*.
   Ce qui marche chez eux, ce sont des signaux de **retour du monde** (observables du capteur).

⇒ **Le signal doit venir du CAPTEUR, pas de ce que le modèle pense de lui-même.**

Leur règle de sélection, qu'on adopte : *pour tout signal candidat, mesurer sa corrélation au
résultat réel sur des données tenues à l'écart ; un signal faiblement corrélé est inutilisable,
quelle que soit sa précision sur la tâche nominale.* (C'est exactement ce qu'on a fait ce soir avec
les 7 lectures du rêve.)

## G0 — GATE GRATUIT : un signal du CAPTEUR sait-il dire « je n'ai rien » ?
Candidats, tous calculables directement depuis la rétine, **zéro oracle en entrée** :
nombre de rayons au-dessus du seuil · masse de saillance totale · **piqué de la distribution
d'attention** (une cible nette donne un pic, une invention un étalement) · entropie de l'attention ·
profondeur des rayons retenus · écart entre le 1er et le 2e pic.

**Cible à prédire** : « un rayon touche-t-il réellement la proie ? », vérité géométrique dérivée de
`food_rel0` — **ORACLE D'ÉVAL UNIQUEMENT**. Découpe **par épisode**.

| gate | barre | pourquoi ce chiffre |
|---|---|---|
| **G0-honnête** | taux de bon classement ≥ **80 %** | à 67 % (déjà mesuré en juillet) elle se trompe **une fois sur trois** : elle abandonnerait une vraie proie une fois sur trois, ce qui coûte plus que ça ne rapporte sur un budget de ~350 pas |
| **G0-contrôle** | la jauge SERVIE reste à ~56 % | sinon il n'y a rien à améliorer |
| 🛑 **STOP** | meilleur signal < **70 %** | le capteur ne porte pas l'information ; ne pas payer la suite |

⚠️ **Correction pour comparaisons multiples OBLIGATOIRE** : on teste ~6 signaux et on garde le
meilleur. Le max sous l'hypothèse nulle doit être estimé par permutation, comme ce soir — c'est la
faute exacte que j'ai commise sur le G0 critique-de-rang (seuil posé sur un max non corrigé).

## G1 — GATE D'USAGE : à quoi sert de le savoir ? (payé seulement si G0 passe)
**C'est la barre la plus importante, et elle porte sur le comportement, pas sur la lucidité.**

Savoir qu'on ne voit pas ne vaut que s'il existe un meilleur comportement à adopter. Trois options,
à départager **avant** de coder :

| option | ce que ça suppose | statut |
|---|---|---|
| **A. Perception ACTIVE** — tourner la tête pour aller voir | que le regard soit pilotable et qu'un balayage retrouve la cible | ⭐ candidate principale : c'est ce que « savoir qu'on ne voit pas » débloque naturellement, et c'est le seul usage qui AJOUTE de l'information |
| **B. Mémoire** — viser la dernière position connue | que la mémoire ait enfin un signal pour prendre le relais | G0 juillet = STOP, mais mesuré **sous le plafond levé depuis** → à rouvrir avec un G0 propre |
| **C. Renoncer** — ne pas poursuivre un fantôme | rien | ⚠️ risque : plus honnête et **moins de repas** |

**Gate gratuit préalable** : sur les corpus existants, parmi les ticks où elle invente, quelle part
aurait été récupérable par (A) un balayage du regard, (B) une position mémorisée récente ? Si les
deux sont sous 20 %, seule (C) reste et le chantier **n'améliorera pas la vie** — le dire alors.

## G2 — juge en vies
2 bras × 3 graines × 12 vies, contrôle de viabilité du monde passé d'abord, **et contrôle d'action**
(le bras a-t-il réellement agi ? mesurer le mécanisme, pas la bannière).

| gate | barre |
|---|---|
| **le BUT** | consommation par 1000 pas **VÉCUS** > le bras de référence (métrique non bimodale) |
| non-régression | survie moyenne ≥ référence |
| 🛑 **KILL** | consommation en baisse de plus de **15 %**, ou morts-danger **+3 ou plus** (magnitude, pas direction) |

## Réserve à dire d'avance
`[HYPOTHÈSE]` La barre G0 à 80 % a une chance réelle de ne pas être atteinte : le meilleur résultat
connu est 67 %, et l'information manquante vient de ce qu'**aucun rayon ne touche la cible** — il
n'y a alors rien à extraire, quelle que soit la tête.

Et même si G0 passe, `[MESURÉ: A/B perception du matin]` un composant peut avoir d'excellents
chiffres de perception et **zéro effet en vies**. C'est pourquoi G1 existe et pourquoi le juge final
porte sur le comportement.

## Ce que ce chantier ne fait pas
Il ne corrige ni le rayon de braquage (−23,3 pts), ni la cécité au mouvement des objets (dette n°1,
qui reste ouverte en aval de celui-ci), ni la survie bimodale.
