# Design — CRITIQUE DE RANG (classer, pas prédire) — pré-inscrit 2026-08-02

> Pré-inscription écrite AVANT tout diag/train (§1). Ouvre après la sonde de bifurcation qui a
> établi, pour la première fois du projet, qu'une marge existe à l'étage décision.

## Mission
Rendre APPRISE la valeur d'une décision : au lieu d'un coût écrit à la main (`-min_dist` + énergie
+ queue analytique), une tête qui **classe** les commandes candidates par conséquence future.
Objectif falsifiable : **capturer une part de la marge mesurée**, jugée en vies, sans nouvelle
constante fittée.

## Pourquoi maintenant (et pas en juillet)
`[MESURÉ: scripts/cf_fork_probe.sh, foret_v1, corps cinématique, 3 graines, rejeu déterministe]`
Critère pré-déclaré avant de lancer — marge confirmée si `max > ref` sur ≥2 graines sur 3.
**Résultat 3/3** :

| graine | déterminisme | ref (le planner seul) | max (meilleur candidat) | candidats à ZÉRO |
|---|---|---|---|---|
| 1 | ✅ A=B=1 | 1 repas | **3** | 9/21 |
| 2 | ✅ A=B=1 | 1 repas | **2** | 7/21 |
| 3 | ✅ A=B=0 | **0 repas** | **3** | 13/21 |

Une seule décision, tenue une période de replan, fait varier le résultat de **0 à 3 repas**. Sur la
graine 3 le choix du planner est parmi les PIRES.

`[INFÉRÉ]` Ce résultat renverse le verdict de juillet (« aucune décision unique ne compte »,
conséquence 1,9 %), et la raison est identifiable : le monde a depuis acquis une **irréversibilité**
que ce verdict n'avait pas — la nourriture est devenue une **proie qui fuit**
(`[MESURÉ: SYLVAN_FOOD_PREY_SPEED=0,023 ; autocorrélation de direction 0,986]`). Qui hésite la perd.
La mémoire projet nommait l'irréversibilité comme le levier unique ; elle est arrivée par la porte
de derrière et le verdict n'a jamais été rejugé.

## Essayé → résultat (NE PAS répéter)
| tentative | résultat |
|---|---|
| Critique par-commande, monde d'avant | **fente arithmétique** : erreur réseau 19-47× l'écart à trancher entre 33 candidats quasi ex-aequo |
| Critique de cible appris remplaçant l'ordre designé (G3, juillet) | critère visé atteint, échec **EXPORTÉ** (danger 5→13, conso 108→96) |
| Notes Monte-Carlo par état (3 négatifs waypoint) | choix **flottants** ; l'analytique gagne par CONSISTANCE |
| Arbitrage homéostatique (aujourd'hui) | 51 % des choix changés, **taux d'acquisition inchangé** (3,23→3,30, p=0,877) |

## L'HYPOTHÈSE CENTRALE — c'est l'OBJECTIF d'entraînement qui a échoué, pas l'idée
Le code du projet le note lui-même (`command_planner.py`, bloc `critic_mode`) : on demandait au
réseau de **RANGER** des options presque identiques alors qu'il avait été entraîné à **PRÉDIRE une
valeur moyenne** (MSE sur retours Monte-Carlo). *Deux tâches différentes.* La piste inscrite dans ce
même commentaire est « changer l'OBJECTIF (apprendre à CLASSER — préférences/TD), pas les données ».

Ce chantier teste **cette différence de forme**, et rien d'autre. Refaire une régression de valeur
en espérant mieux serait la boucle que le PRINCIPE N°1 interdit.

## G0 — GATE GRATUIT D'APPRENABILITÉ (0 entraînement, 0 Godot)
**La question** : le signal qui distingue une bonne décision d'une mauvaise est-il **présent dans le
rêve du WM** ? Si le rollout ne le porte pas, aucune tête posée dessus ne peut le trouver — et on
l'apprend pour zéro.

**Matériel déjà payé** : 3 forks × 21 candidats × conséquence RÉELLE mesurée = **63 triplets
(état, action, résultat) contrefactuels**. C'est rare et cher ; c'est exactement ce qui manquait
en juillet (données off-policy sans contrefactuel).

**Protocole** : pour chacun des 63, faire rouler le WM GELÉ sur la commande candidate et extraire
plusieurs lectures — le coût servi (`-min_dist` + énergie), `min_dist` seul, la distance finale,
l'énergie prédite, l'alignement, la distance à l'eau. Puis corréler chaque lecture au **résultat
réel** (repas dans la fenêtre).

| gate | critère |
|---|---|
| **G0-signal** | au moins une lecture du rêve atteint une corrélation de rang \|ρ\| ≥ **0,40** avec le résultat réel |
| **G0-marge** | la lecture la mieux classée désigne un candidat qui **bat `ref`** sur ≥ 2 forks sur 3 |
| **G0-contrôle** | le coût SERVI, lui, ne les bat PAS — sinon il n'y a rien à apprendre, il suffit de le lire |
| 🛑 **STOP** | aucune lecture ne dépasse \|ρ\| = 0,20 ⇒ **le rêve ne porte pas le signal**, le chantier s'arrête ici et le levier redevient le WM (mouvement des objets), pas le critique |

⚠️ **Puissance faible assumée** : 63 points, 3 forks. Un ρ mesuré sur si peu est bruité. Si G0 tombe
entre 0,20 et 0,40, la règle **pré-déclarée** est : collecter 3 forks de plus (graines 4-6, mêmes
paramètres) et re-juger UNE fois — pas de collecte à répétition jusqu'à obtenir le bon chiffre.

## G1 — forme, si G0 passe
Tête de **RANG** (perte par paires sur les préférences observées), pas de régression de valeur.
Socle designé CONSERVÉ et hystérésis d'incumbent INTACTE — la doctrine validée aujourd'hui : le
seul bras qui n'a pas exporté son échec est celui qui gardait le socle
`[MESURÉ: G2 arbitrage, faim 12→12, conso en hausse]`.

## G2 — juge en vies
2 bras × 3 graines × 12 vies, contrôle de viabilité du monde passé d'abord, **et contrôle d'action**
(le bras appris a-t-il réellement décidé ? mesurer son mécanisme, pas sa bannière — le piège du
2026-08-02).

| gate | critère |
|---|---|
| **le BUT** | consommation par 1000 pas VÉCUS > celle du bras designé (métrique non bimodale) |
| non-régression | survie moyenne ≥ designé |
| 🛑 **KILL** | morts-danger **+3 ou plus**, ou consommation en baisse (magnitude, pas direction — la faute de pré-inscription du 2026-08-02) |

## Ce que le chantier N'apporte PAS
`[MESURÉ: diag_portee_g0.py]` Il ne corrige ni la visée (−14,8 pts) ni le rayon de braquage
(−23,3 pts), et il ne dissout pas l'échafaudage sprint (dette n°1 = cécité du WM au mouvement des
objets). Ce sont des chantiers de SUBSTRAT, distincts et à traiter séparément.
