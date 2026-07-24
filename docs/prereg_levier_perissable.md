# Pré-enregistrement — Levier CONSÉQUENCE n°3 : baies PÉRISSABLES (relocalisantes)

**Date** : 2026-07-23. **Branche** : `feat/perception-consequence`.

## Contexte
Le chantier critique appris a été fermé faute de **conséquence** : dans le monde `bosquets_v2`,
aucune décision unique ne compte (juge contrefactuel déterministe `scripts/cf_fork_distribution.sh`
= **17 % (3/18)** de points où forcer le pire choix change les repas ; le reste est récupéré au
replan suivant). Cause-racine unique : le corps/monde est trop **RÉCUPÉRABLE**. Owner : ne pas
abandonner le critique, mais **modifier le monde** pour rendre les décisions conséquentes (leviers
1 danger, 2 coût-virage, 3 périssable). On teste **UN levier à la fois**, chacun gaté par le même
balayage de conséquence + un contrôle de survie (empilement interdit, §4).

## Mécanisme testé (levier 3)
Baie PÉRISSABLE **relocalisante** (`SYLVAN_FOOD_PERISH`, preset `bosquets_v3_perish`, T=800 ticks) :
une baie vivante non mangée depuis T ticks **SAUTE sur un autre bosquet** (elle ne disparaît PAS —
ce monde n'a que 2 baies, disparaître = famine, mesuré 0 repas). Le compte de baies vivantes reste
2 (densité constante → survie préservée, §2), mais la baie que l'agent visait n'est plus là où il
allait → **un choix trop lent (hésitation, virage inutile) perd son trajet**. Naissances staggerées.
GRATUIT côté WM (règle de monde, perception inchangée). OFF par défaut = bit-identique.

## Contrôle de survie (PAYÉ avant le gate)
Smoke 5 vies, `wm_objcentric_kin` + corps promu : survie médiane **3000** (pleine), repas médian
**2** = parité avec `bosquets_v2`. La relocalisation ne famine PAS. ✅

## Critère de conséquence (falsifiable, AVANT de lancer)
Instrument identique au baseline : 6 graines × 3 ticks (600/1200/1800), pire choix (0.75,−0.6) tenu
240 ticks, repas comparés dans [t, t+800] vs référence. Rejeu déterministe (seed + mono-thread +
serveur frais/run).

- **SUCCÈS** : taux **≥ 33 % (≥ 6/18)** — le périssable double au moins la conséquence du baseline
  (17 %). → le levier crée de la conséquence apprenable → on garde et on gate le critique dessus.
- **KILL** : taux **≤ 22 % (≤ 4/18)** — dans le bruit du baseline → le périssable n'ajoute pas de
  conséquence aux décisions réelles du planner → banké NÉGATIF, on passe au levier 2 (coût-virage).
- **MARGINAL** (23–32 %) : signal faible → noté, à combiner éventuellement avec le levier 2.

## Résultat — SUCCÈS (2026-07-23)

Balayage `PRESET=bosquets_v3_perish bash scripts/cf_fork_distribution.sh` :

| seed | 600 | 1200 | 1800 |
|------|-----|------|------|
| 1 | **OUI** | non | non |
| 3 | non | non | non |
| 5 | non | **OUI** | **OUI** |
| 6 | non | non | non |
| 8 | **OUI** | **OUI** | non |
| 9 | non | **OUI** | non |

**TAUX DE CONSÉQUENCE = 6/18 = 33 %** (baseline `bosquets_v2` = 3/18 = 17 %). Le levier périssable
**DOUBLE** la conséquence, atteint pile le seuil de succès pré-enregistré (≥ 33 %), SANS effondrer la
survie (smoke : médiane 3000, repas 2 = parité v2). Là où le monde était récupérable, forcer le pire
choix 240 ticks fait maintenant relocaliser la baie visée → le repas est perdu (seeds 1/5/8/9).

**Décision** : levier 3 VALIDÉ comme source de conséquence. `bosquets_v3_perish` devient le monde
candidat pour re-gater le critique appris (dont l'échec venait de l'absence de conséquence, pas de
l'appris). Prochaine étape : construire la tête critique et l'A/B pleine-politique à ce régime
conséquent. Les leviers 2 (coût-virage) et 1 (danger) restent à tester/combiner ensuite (§4, un à la
fois). Note honnête : 33 % = seuil, pas une marge confortable ; 3/6 graines portent toute la
conséquence (5, 8, +1,9) — à confirmer si on veut plus de robustesse (plus de graines/ticks).

---

# Suite (2026-07-24) — Y a-t-il une MARGE pour un critique ?

Le levier a rendu les décisions conséquentes (33 %). Question suivante, posée AVANT d'entraîner
quoi que ce soit : un critique appris aurait-il quelque chose à gagner ?

## Gate n°1 (gratuit, POOLÉ) — la cible est-elle plus que de la géométrie ?
`diagnostics/diag_critic_beyond_geometry.py` sur le corpus `critic_bosq_a` (20 vies, WM vivant,
held-out PAR ÉPISODE). AUC(dist seule) 0,709 → AUC(GEO=dist+cap) **0,800** → AUC(GEO+énergie)
0,774, **delta −0,026 = KILL** selon le pré-enregistrement.

**MAIS ce verdict n'est PAS retenu**, pour deux raisons dites honnêtement :
1. **Sous-puissant** : seulement **25 vrais repas** (les 44 comptés la veille incluaient les 19
   resets d'épisode — chiffre corrigé). L'effectif utile est 25 événements, pas 53 000 ticks.
2. **Question POOLÉE** = très exactement l'erreur des 3 échecs historiques. Le planner classe
   117 candidats DANS LE MÊME ÉTAT ; seule la variance INTRA-état décide.

Bug de mesure corrigé en route : `torso0` = `(x, z, YAW)` ; inclure le yaw dans une norme faisait
passer chaque enroulement à ±π (saut de 2π ≈ 6,28) pour un téléport → 94 fausses frontières
d'épisode, labels et split corrompus. Frontières désormais sur (x,z) seuls → 19 frontières = 20
épisodes exactement, insensible au seuil de 0,3 à 2,0 m.

## Gate n°2 (le vrai, INTRA-état) — le coût analytique choisit-il déjà le mieux ?
`scripts/cf_fork_probe.sh` (paramétré `PRESET=` + `HOLD=`) : à un fork conséquent, on rejoue les
**21 candidats** en déterministe et on compare au repas que le planner obtient de lui-même (ref).
`max(candidats) > ref` ⇒ marge capturable par un critique.

| fork | ref (planner) | distribution des 21 candidats | marge |
|------|---------------|-------------------------------|-------|
| seed 5, k=1800 | **2** | 20 → 0 repas ; (0,75 ; +0,0) → 2 | **0** (planner déjà optimal) |
| seed 8, k=1200 | **0** | 5 → 0 repas ; **16 → 1 repas** | **+1** (planner battu par 16/21) |

**VERDICT : la marge EXISTE, et elle est grosse là où elle existe.** Au fork seed 8 le planner
analytique est battu par **76 %** des commandes fixes : il replanifie toutes les 60 ticks,
**hésite**, et n'atteint jamais la baie — alors que presque n'importe quel engagement TENU y
arrive. C'est la pathologie que le monde périssable punit (hésiter ⇒ la baie se relocalise) et
précisément ce qu'un critique valorisant l'engagement pourrait corriger. La marge est
fork-DÉPENDANTE (nulle au fork seed 5) — conforme à la leçon « un fork ne suffit pas ».

Contrôles : déterminisme vérifié à chaque fork (A=B), serveur frais par run, mono-thread.

### 3ᵉ fork + contrôle : la marge est-elle juste de l'ENGAGEMENT ?

| fork | ref (planner) | max des 21 candidats | marge |
|------|---------------|----------------------|-------|
| seed 5, k=1800 | 2 | 2 (1 seul candidat) | **0** |
| seed 8, k=1200 | 0 | **1** (16 candidats sur 21) | **+1** |
| seed 9, k=1200 | 1 | 1 (6 candidats sur 21) | **0** |

Marge à **1 fork sur 3** conséquents. Réelle mais pas systématique.

Au fork 8, ce qui gagne est « n'importe quel engagement TENU » contre un planner qui hésite →
HYPOTHÈSE CONCURRENTE bien moins chère : allonger l'engagement (replan) capturerait la marge sans
rien apprendre. **RÉFUTÉ** (A/B `bosquets_v3_perish`, 8 vies, seed 1, mémoire ON) :

| replan | survie méd. | repas (8 vies) | vies pleines |
|--------|-------------|----------------|--------------|
| **60** | **2900** | **11** | **4/8** |
| 120 | 2600 | 8 | 2/8 |
| 240 | 2000 | 4 | 1/8 |

Dégradation MONOTONE. L'engagement aveugle n'est pas la réponse : ce qui gagnait au fork 8 ne se
généralise pas. Il faut savoir **QUAND** s'engager et **VERS QUOI** — un jugement dépendant de
l'état, c'est-à-dire précisément un critique. Le négatif du commitment (déjà obtenu dans l'ancien
monde) TIENT dans le monde périssable.

## Bilan et décision
1. Le levier périssable rend les décisions conséquentes (33 % vs 17 %), survie intacte.
2. Il existe une marge de choix (+1 repas) à ~1 fork conséquent sur 3.
3. Cette marge n'est PAS capturable par un réglage grossier d'engagement (réfuté ci-dessus).
⇒ Le prochain pas justifié est bien la tête-valeur apprise, jugée INTRA-état (juge = ce probe des
21 candidats, pas un R² poolé), puis A/B pleine-politique. Réserve honnête : la marge est modeste
et fork-dépendante (1/3), et le gate poolé ne montre pas de signal au-delà de la géométrie —
un critique qui la capture devra être jugé sur le BUT (repas/survie), pas sur une AUC.
