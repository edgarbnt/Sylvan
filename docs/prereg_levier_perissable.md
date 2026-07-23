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
