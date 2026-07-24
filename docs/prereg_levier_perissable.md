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

---

# La tête est bâtie — et le JUGE INTRA-ÉTAT la REFUSE (2026-07-24)

## Ce qui a été construit
`python/sylvan/critic_corpus.py` (corpus/cible/token en UN seul endroit, pour que entraîneur, juge
et diagnostic ne divergent pas) + `python/scripts/train_meal_critic.py` (tête V(token) → P(repas
dans K ticks), WM GELÉ, split par ÉPISODE, stats calées sur le train seul, |sin| pour imposer la
symétrie miroir) + `diagnostics/diag_critic_intra_state.py` (LE juge).

Entraînement : AUC POOLÉE held-out **0,786** (convergée, plateau). Ce chiffre ne juge rien.

## Juge intra-état, fork seed 8 k=1200 (celui QUI A de la marge)
Vérité-terrain sauvegardée : `data/forks/s8_k1200_outcomes.txt` (16/21 candidats → 1 repas).

|  | top-1 choisi | repas obtenus | AUC intra-état |
|--|--------------|---------------|----------------|
| CRITIQUE | (0,75 ; +0,6) | **0** | 0,562 |
| ANALYTIQUE `-min_dist` | (0,75 ; +0,6) | **0** | 0,587 |

**ÉCHEC des deux**, au niveau du hasard. Deux causes MESURÉES, pas supposées :

1. **Le critique est `-min_dist` déguisé** : corrélation de RANG entre V et −min_dist = **+0,930**.
   Il a ré-appris la géométrie — cohérent avec le gate poolé (aucun signal au-delà de la géométrie).
2. **Le rêve est MYOPE** : horizon 80 ticks = **0,88 m** parcourus, pour une baie à **7,61 m**. Les
   21 candidats finissent étalés sur **0,62 m** → états quasi indiscernables. Aucune fonction de cet
   état ne peut les classer, critique ou pas.

## Balayage d'horizon (gratuit) — la myopie est-elle LA cause ?
| horizon | parcouru | écart min-max des candidats | AUC intra-état de −min_dist |
|---------|----------|------------------------------|------------------------------|
| 80 (actuel) | 0,88 m | 0,62 m | 0,587 |
| 200 | 2,20 m | 2,51 m | 0,613 |
| 400 | 4,40 m | 4,95 m | 0,637 |
| **700** | **7,70 m** | **5,72 m** | **0,663** |
| 1000 | 11,00 m | 3,52 m | 0,637 |

L'optimum tombe là où le rêve ATTEINT enfin la baie (7,70 m parcourus vs 7,61 m de distance).
Allonger l'horizon aide donc réellement, mais modestement (0,587 → 0,663) : la myopie est UNE cause,
pas toute la cause (les issues mesurées sont en partie dispersées — 16/21 à 1 repas avec des trous
isolés, ce qui plafonne ce que TOUT classement peut atteindre).

## Décision honnête
Le critique tel que conçu (token à 5 dims, noté sur un rollout de 0,88 m) **n'a rien à apporter** :
il ne voit que ce que `-min_dist` voit, appliqué à des états que le rollout ne sépare pas. Négatif
BANKÉ. Deux pistes distinctes en sortent, à gater séparément (§4) :
  (a) **HORIZON** — designed, cheap, testable en closed-loop tout de suite (80 → ~700).
  (b) **ENTRÉE PLUS RICHE** — le token jette la scène (autres bosquets, buissons) que le latent
      porte ; un critique sur le LATENT verrait ce que `-min_dist` ne voit pas. C'est la seule
      forme qui puisse battre la géométrie, puisque sur la géométrie seule il n'y a rien à gagner.

---

# Suite (2026-07-24) — critique-LATENT, horizon, et MATURITÉ VISIBLE

## 1. Critique-LATENT : ÉCHEC, la donnée est la contrainte
`python/scripts/train_latent_critic.py` (entrée = latents RÊVÉS open-loop sous les commandes vécues,
donc train = déploiement ; WM gelé ; split par épisode). AUC POOLÉE held-out **0,623**, avec
SURAPPRENTISSAGE franc (la loss descend pendant que l'AUC se dégrade après l'époque 40) : 128
dimensions d'entrée pour **25 repas** vécus.

Juge intra-état (fork seed 8 k=1200) :

| critique | top-1 | AUC intra-état | verdict |
|----------|-------|----------------|---------|
| token (géométrie) | 0 repas | 0,562 | pas de gain (analytique 0,587) |
| latent (scène) | 1 repas | **0,325** | ÉCHEC, classement ANTI-corrélé |
| analytique `-min_dist` | 0 repas | 0,587 | — |

⚠️ **Bug de verdict corrigé** : le script concluait « LE CRITIQUE GAGNE » sur le seul top-1, alors
que le pré-enregistrement exigeait top-1 ET meilleur classement. Or **76 % des candidats atteignent
le max ici** → un choix ALÉATOIRE « gagne » le top-1 3 fois sur 4. Le top-1 seul ne prouve rien ; le
juge affiche désormais le taux de base et exige les deux conditions. Sans ce garde-fou j'aurais
relayé de la chance comme un succès.

⇒ La contrainte n'est ni la cible, ni le monde, ni la forme : c'est le **volume de vécu**. 25 repas
ne peuvent entraîner aucun critique riche. Un corpus puissant demande ~150-200 vies (~200+ repas).

## 2. Horizon : le gain OFFLINE ne TRANSFÈRE PAS (négatif)
A/B closed-loop `bosquets_v3_perish`, 8 vies, seed 1 :

| horizon | imagination | survie méd. | repas (8 vies) | vies pleines |
|---------|-------------|-------------|----------------|--------------|
| **80** | 0,88 m | **2900** | **11** | **4/8** |
| 300 | 3,30 m | 2000 (plancher) | 2 | 0/8 |

L'offline prédisait l'INVERSE (AUC 0,587 → 0,637 en allant de 80 à 400). Cause probable : sur 300
pas le rêve open-loop DÉRIVE, donc le planner optimise une fantaisie. H=700 TUÉ avant la fin (9× le
coût, même logique de transfert déjà réfutée). **Piste horizon fermée** ; leçon re-confirmée : un
gain de classement offline ne prédit pas le comportement en vie.

## 3. MATURITÉ VISIBLE — bâtie et PROUVÉE neutre pour la perception de position
`SYLVAN_FOOD_RIPE_CUE` (preset `bosquets_v4_ripe`) : la LUMINOSITÉ du buisson-marqueur encode l'âge
de sa baie (x1,0 fraîche → x0,2 imminente ; bosquet vide = éteint).

**Pourquoi le buisson et PAS la baie** : le slot pondère ses rayons par une saillance
`max(RGB) − min(RGB)`. Teinter la BAIE ferait mécaniquement préférer la plus fraîche AU SLOT — la
perception arbitrerait à la place du critique, et `-min_dist` en profiterait aussi. Raccourci câblé,
refusé (§2/§3).

**Preuve d'invariance** (mesurée, pas supposée) : le buisson est à cos **0,402** du rouge et
**0,453** du bleu, sous le seuil 0,55 → ses rayons sont exclus EN DUR des deux slots ; et l'affinité
étant un COSINUS, elle est invariante par changement d'échelle. Vérifié sur **2000 observations
réelles** (10 708 rayons de buisson touchés) : luminosité ×0,8 / ×0,5 / ×0,2 → écart max du slot
bouffe = **0,00000000 m**.
⇒ l'indice est PROUVABLEMENT invisible à `-min_dist` et présent dans la rétine, donc exploitable
UNIQUEMENT par un critique qui lit la scène. C'est le premier signal du monde que la géométrie ne
peut pas voir.

⚠️ Première vérification MAL CONÇUE et écartée : comparer deux runs closed-loop (cue OFF vs ON) à
graine égale ne teste rien, car l'indice change la rétine → le latent → le plan → les trajectoires
divergent (divergence ATTENDUE). L'invariance devait être testée à OBSERVATION ÉGALE ; c'est ce qui
est fait ci-dessus.

## Prochain pas
Collecter UN gros corpus sur `bosquets_v4_ripe` (~150-200 vies) : il sert les DEUX besoins d'un coup
— le volume qui manquait, et le signal que seule la scène porte. Puis ré-entraîner le critique-latent
et le repasser au juge intra-état. Ne PAS ajouter d'autre levier avant ce verdict (§4).

---

# CRITIQUE TD (valeur TERMINALE + bootstrap) — il BAT `-min_dist` (2026-07-24)

## La réécriture, et pourquoi les deux versions précédentes ne POUVAIENT pas marcher
Recherche : **TD-MPC** (Hansen et al., ICML 2022) traite exactement notre problème — un rollout
COURT dans un WM appris qui doit décider en fonction d'un avenir LOINTAIN. Son objectif :

    max_a  Σ_{i=0}^{H-1} γ^i R(z_{t+i}, a_{t+i})  +  γ^H · Q(z_{t+H}, a_{t+H})

soit un rêve court + une valeur **TERMINALE**, « qui fait entrer l'information de long horizon dans
un plan de court horizon ». Deux erreurs corrigées :
1. **agrégat `mean` → TERMINAL** : moyenner V sur le rêve le traite comme une récompense par pas et
   re-note ce qui est DÉJÀ dans l'horizon, au lieu de résumer ce qui vient APRÈS ;
2. **cible Monte-Carlo à fenêtre fixe → BOOTSTRAP TD** : « repas dans K=200 ticks » est
   structurellement aveugle au-delà de K ; `V(z_t) ← r_t + γ·V(z_{t+1})` (réseau-cible retardé)
   propage la valeur depuis arbitrairement loin.
Et surtout **PAS** un rollout plus long : mesuré (H=300 → survie au plancher) et prédit par la
littérature (l'erreur du modèle se compose, le planner optimise une fantaisie).

Alignement TD propre : on rêve sous les commandes RÉELLEMENT exécutées, donc le rêve suit la
trajectoire vécue et la récompense du tick t+d s'aligne sur la profondeur d — le bootstrap tourne
sur la DISTRIBUTION DE DÉPLOIEMENT. WM gelé ; récompense = 1 au tick d'un repas.

## Juge intra-état — 2 forks sur 2

| fork | taux de base | CRITIQUE TD | ANALYTIQUE `-min_dist` |
|------|--------------|-------------|------------------------|
| seed 8, k=1200 | 76 % | top-1 **1 repas**, AUC **0,688** | top-1 0 repas, AUC 0,587 |
| seed 5, k=1800 | **5 %** | top-1 **2 repas**, AUC **1,000** | top-1 0 repas, AUC 0,950 |

Le second fork est le test FORT : un seul candidat sur 21 atteint le max, donc le hasard ne gagne le
top-1 que 5 % du temps — et le critique TD désigne exactement ce candidat, avec un classement
parfait. Rappel des formes précédentes au fork 1 : token 0,562, latent 0,325 (sous le hasard).

| seed 9, k=1200 | 29 % | top-1 **1 repas**, AUC **0,778** | top-1 0 repas, AUC **0,467** |

**3 forks sur 3.** Le critique TD désigne un candidat qui atteint le MAX aux trois ; l'analytique
désigne un candidat à 0 repas aux trois, et au fork 3 son classement tombe SOUS le hasard (0,467).
AUC moyenne : TD **0,822** vs analytique 0,668.

## Ce que ça ne prouve pas encore
3 forks, pas une distribution. Le juge final reste un **A/B pleine-politique** (repas/survie sur
plusieurs vies) avec `SYLVAN_PLANNER_COST=critic`. À noter aussi : V est SOUS-PROPAGÉE (moyenne
held-out 0,015 là où le taux de repas implique ~0,16) — plus d'itérations TD laisseraient peut-être
encore du signal. Le gros corpus `bosquets_v4_ripe` (maturité visible) est en cours et n'a PAS servi
ici : ce gain vient de la FORME (terminale + TD), pas du volume.

## ⚠️ A/B PLEINE-POLITIQUE — le gain offline NE TRANSFÈRE PAS (négatif, il prime)

`bosquets_v3_perish`, 8 vies, seed 1, replan 60, mémoire ON :

| bras | survie méd. | repas (8 vies) | vies pleines |
|------|-------------|----------------|--------------|
| **ANALYTIQUE `-min_dist`** | **2900** | **11** | **4/8** |
| CRITIQUE TD SEUL (remplace) | 2000 (plancher) | **0** | 0/8 |
| ANALYTIQUE + V terminale (w=20) | 2615 | 6 | 0/8 |

Le critique DÉGRADE, en remplacement comme en complément. **Le verdict closed-loop PRIME sur les
3 forks offline** : la conclusion « le critique bat -min_dist » est RÉFUTÉE.

### Pourquoi — deux causes, dont une erreur de méthode que je dois nommer
1. **ÉCHANTILLON BIAISÉ (ma faute).** Les 3 forks jugés ont été CHOISIS parce qu'ils sont
   conséquents — c'est-à-dire précisément ceux où l'analytique se plante. Ils représentent 17-33 %
   des décisions. Mieux classer LÀ ne dit rien des 70-80 % de décisions ORDINAIRES, où `-min_dist`
   fait le travail essentiel : fournir une pente DENSE et toujours valide vers la nourriture.
2. **V est SOUS-PROPAGÉE et donc quasi plate** (moyenne held-out 0,015 là où le taux de repas
   implique ~0,16). Dans un état ordinaire elle ne distingue rien → son argmax est du bruit → l'agent
   erre et meurt de faim (0 repas, 8/8 au plancher).

### LEÇON DE MÉTHODE (3ᵉ fois dans la même session)
Un gain de CLASSEMENT OFFLINE, mesuré à des points de décision choisis, NE PRÉDIT PAS le
comportement en vie. Ça a raté pour l'horizon (AUC↗ offline, effondrement closed-loop) puis deux
fois pour le critique. ⇒ **Le juge d'un critique est l'A/B pleine-politique, point.** L'intra-état
reste utile comme filtre PAS-CHER (il a bien tué les formes token et latent), mais il ne promeut
rien : il ne peut que DISQUALIFIER.

### Ce qui reste actionnable (ne PAS empiler avant de l'avoir testé)
- **Réparer la sous-propagation de V** : c'est un défaut CONCRET et mesuré (0,015 vs ~0,16 attendu),
  pas une conjecture — plus d'itérations TD / meilleure propagation. Une valeur plate ne peut rien
  porter, quelle que soit sa forme.
- **Le gros corpus `bosquets_v4_ripe`** (~150 vies, maturité visible) : 25 repas ne pouvaient rien
  entraîner ; il apporte le volume ET un signal non-géométrique.
- Balayer le poids du mélange plutôt que le deviner (w=20 était un choix arbitraire de ma part).

---

# RÉPARATION DE V + GROS CORPUS → et le VRAI verrou, structurel (2026-07-24)

## 1. La sous-propagation était réelle et grossière — réparée
Cause : la boucle faisait **UN pas de gradient par époque**, soit **400 pas au total**, alors qu'en
TD 1-pas la valeur ne remonte que d'UN tick par mise à jour et que γ=0,999 demande ~1000 remontées.
Deux ordres de grandeur manquants. Corrigé par des **retours n-PAS** (récurrence arrière sur toute la
fenêtre du rêve : 80 ticks remontent en UNE mise à jour) et **8000 pas de gradient**.
Effet : écart-type de V ×3 (0,072 → 0,224) = la valeur est devenue discriminante.

## 2. Mon ancre était fausse (corrigé)
`E[V] = taux/(1−γ)` suppose un processus INFINI. Or les épisodes se terminent (mort ou plafond) :
près de la fin l'avenir est tronqué. Vraie ancre Monte-Carlo bornée à l'épisode = **0,228** et non
0,361 (surestimation 1,6×). Détail parlant : la **médiane du vrai retour est 0** — la plupart des
états n'ont réellement aucun repas futur dans l'horizon actualisé.

## 3. Gros corpus : 149 vies, 137 repas (5,5× plus)
`bosquets_v4_ripe` (maturité visible), 3 tranches gzippées.

## 4. ⛔ LE VERROU EST STRUCTUREL : le LATENT NE PORTE PAS L'OBJET
Malgré la réparation ET le gros corpus, corr(V, vrai retour actualisé) held-out = **+0,047**
(contre −0,125 avant) : toujours rien.

Sonde GRATUITE et décisive — peut-on lire la position de la bouffe DEPUIS le latent rêvé ?
(régression ridge, held-out, 6000 latents à profondeur 80)

| cible lue depuis le latent | R² held-out |
|----------------------------|-------------|
| slot x (droite) | **−0,011** |
| slot z (avant) | **+0,042** |
| **distance à la bouffe** | **−0,884** (pire que prédire la moyenne) |

**L'information n'est pas dans le latent.** C'est la confirmation directe d'un acquis déjà écrit du
projet (« le pur-latent-valeur `plan_latent` était lossy, perdait l'objet » — la raison d'être même
du slot), auquel je n'avais pas donné assez de poids. ⇒ **tout critique LATENT est condamné**, ce qui
explique d'un coup le −0,325 (latent MC) et le +0,047 (latent TD).

## 5. La conséquence, dure, à assumer
Avec CE WM, le critique n'a aucune entrée qui soit à la fois FIABLE et PLUS RICHE que la géométrie :
- **slot** (transporté, fiable) = exactement ce que `-min_dist` calcule → rien à gagner (corr. de
  rang +0,93 mesurée) ;
- **latent** (riche en principe) = ne porte pas l'objet → inutilisable.
Dans le rêve, la SEULE information d'objet fiable est le slot TRANSPORTÉ, et c'est de la géométrie.

**Piste qui reste ouverte** (non testée, à gater) : donner au critique le slot transporté ET un
readout APPRIS de la rétine RÉELLE à t=0 (elle, non rêvée, porte prouvablement l'indice de maturité —
invariance mesurée à 0,00000000 m sur les slots). La partie dépendante du candidat reste géométrique,
mais la VALEUR de la baie visée serait modulée par une information que `-min_dist` ne voit pas.
