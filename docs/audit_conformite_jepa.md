# Audit de conformité JEPA / LeCun — ce qui est faux, et ce qui SEMBLE pur sans l'être

**Date** : 2026-07-24. Demandé par l'owner après l'échec répété du critique appris.
**Méthode** : confronter l'archi RÉELLEMENT SERVIE aux prescriptions de LeCun (*A Path Towards
Autonomous Machine Intelligence*) et à la littérature JEPA récente — **sur mesures**, pas sur
impressions. Chaque anomalie ci-dessous est vérifiée sur le checkpoint vivant.

---

## A1. ⚠️ CORRIGÉ LE 2026-07-24 — voir l'encadré en fin de section

## A1 (énoncé initial, TROP FORT). Le latent est EFFONDRÉ et NE PORTE PAS L'OBJET (mesuré)

| mesure | valeur |
|--------|--------|
| eff_rank du latent (128 dims) | **34,2 = 27 % de la capacité** |
| R² pour lire slot x depuis le latent rêvé | −0,011 |
| R² pour lire slot z | +0,042 |
| R² pour lire la DISTANCE à la bouffe | **−0,884** (pire que la moyenne) |

Le critère JEPA est qu'une représentation soit **maximalement informative tout en restant
prédictible** ; l'effondrement est défini comme l'encodeur qui minimise la perte de prédiction **en
perdant l'information discriminante** — VICReg existe exactement pour l'empêcher. Objection possible
(« JEPA jette délibérément l'imprévisible ») : **rejetée**, la position de la nourriture est la chose
la plus prédictible et la plus centrale de la tâche. C'est un vrai effondrement partiel.

### ⚠️ CORRECTION MESURÉE (2026-07-24) — l'énoncé ci-dessus était TROP FORT

La sonde initiale visait la **position précise du slot**, et seulement à la profondeur 80. En
décomposant proprement (sondes linéaire ET MLP, à plusieurs profondeurs) :

| lu dans le latent, à la profondeur 0 | R² held-out |
|--------------------------------------|-------------|
| rétine entière | **+0,798** |
| rétine SANS les rayons bouffe | +0,737 |
| **rayons bouffe seulement** | **+0,572** (MLP +0,595) |
| **« y a-t-il de la bouffe en vue »** | **+0,593** (MLP +0,630) |
| position PRÉCISE du slot | +0,046 |

Et la dégradation le long du rêve (« bouffe en vue ») :
profondeur 0 → **0,556** ; 20 → 0,304 ; 40 → 0,254 ; **79 → 0,160**.

**Énoncé CORRIGÉ de A1** : le latent porte la **PRÉSENCE et l'APPARENCE** de l'objet (R² ≈ 0,6 à la
profondeur 0 — donc potentiellement l'indice de maturité), mais **PAS ses coordonnées précises**
(R² ≈ 0,05) ; et cette information **se dégrade d'un facteur 3,5 le long du rollout open-loop**
(0,556 → 0,160). Les rayons bouffe pèsent 35 % de la variance de la rétine : l'objet n'est pas un
détail négligeable que JEPA jetterait légitimement.

**Conséquence RÉVISÉE, et elle est actionnable** : les critiques latents n'ont pas échoué parce que
l'information est absente, mais parce qu'ils la lisaient **au pire endroit** — le latent TERMINAL
(profondeur 80), là où il n'en reste que 0,16. C'est aussi le mécanisme de l'effondrement à H=300.

⇒ La forme correcte n'exige NI ré-entraînement du substrat NI lecture de la rétine brute :
**V(latent à t=0, slot terminal)** — l'état FIABLE du world-model (perception à t0, qui porte la
scène et l'apparence) combiné à ce que le candidat OBTIENT géométriquement. C'est la forme Q(s,a) de
TD-MPC, les deux entrées sont l'état du WM (donc architecturalement pur), et le latent y apporte ce
que `-min_dist` ne voit pas.

---

## A2. ⚠️ CE QU'ON APPELLE « PERCEPTION OBJECT-CENTRIC PURE » EST CODÉ-MAIN (mesuré)

C'est l'anomalie la plus importante, parce qu'elle porte sur un module que la doc du projet classe
`pur`. Sur le checkpoint **RÉELLEMENT SERVI** (`wm_objcentric_kin`) :

```
color_queries = [[1.0, 0.0, 0.0],      <- rouge primaire EXACT, codé à la main
                 [0.0, 0.0, 1.0]]      <- bleu primaire EXACT
```
(le WM « typé » qui a des requêtes MESURÉES — [[0.876,0.349,0.333], …] — existe mais **n'est PAS
promu** ; ce n'est pas lui qu'on sert.)

Et avec `slot_resources=2`, dans `slot_head._attend` :
```python
scores = [self.score[k](r) ...]      # scoreur APPRIS calculé…
a_list = [softmax(s) for s in scores]
if self.color_queries is not None:
    ...
    a_list = []                       # …puis INTÉGRALEMENT ÉCRASÉ
    logit = log(sal * aff * prox) - 4.0 * dist
```
**2498 paramètres appris sont calculés puis jetés.** Le readout effectif est 100 % géométrique :
seuil 0,55 codé, prior de proximité codé, prior de distance −4/m codé, masque couleur dur codé, et
transport `slot_calib=(1,−1,−1)` fixé « c'est une géométrie, pas une quantité à fitter ».

**Nuance d'équité** : ce choix est DOCUMENTÉ et MESURÉ (7 variantes apprises, chacune trouvant un
optimum pathologique). Ce n'est pas une négligence. **Ce qui ne survit pas à l'audit, c'est le LABEL
« pur »**, pas la décision d'ingénierie.

---

## A3. On a codé-main le CRITIQUE (TC), pas seulement le coût intrinsèque (IC)

LeCun est explicite : le module de coût = **IC immuable** (faim, douleur — non entraînable) **+
critique ENTRAÎNÉ** qui prédit l'IC futur. Notre coût servi est
`-min_dist + heading·align + energy - done_penalty`.

Or `-min_dist` **n'est pas un coût intrinsèque** : ce n'est ni la faim ni la douleur, c'est une
heuristique *« rapproche-toi »* qui tient lieu de **critique**. Nous avons donc codé-main exactement
la pièce que LeCun dit devoir être apprise — et c'est précisément celle qu'on n'arrive pas à
remplacer depuis 4 tentatives.

**Effet pervers mesuré** : `-min_dist` est un très bon heuristique sur ce monde, donc tout critique
appris doit le battre **sur son propre terrain, la géométrie**. Corrélation de rang mesurée entre
notre meilleur critique-token et `-min_dist` : **+0,93** — il ne pouvait que le ré-apprendre.

---

## A4. WM DÉTERMINISTE dans un monde devenu STOCHASTIQUE ⛔ (mesuré)

- Aucune variable latente d'incertitude dans le WM (aucun `sample`/`reparam`/`logvar`) : le rollout
  produit **une seule** trajectoire.
- `transport_slot(slot, disp_real)` déplace le slot **par la seule ego-motion** : il suppose donc
  l'objet **IMMOBILE dans le monde**.

Mais le levier périssable qu'on vient d'ajouter fait **SAUTER la baie sur un autre bosquet**. Le rêve
est donc **structurellement incapable de représenter l'événement même qui crée la conséquence**.

La littérature nomme exactement ce défaut : *« un world-model JEPA déterministe ne peut pas
représenter un branchement stochastique : il prédit une ESPÉRANCE latente là où la planification a
besoin de futurs ÉNUMÉRABLES »* (MoP-JEPA). LeCun prescrit des **variables latentes** pour porter les
futurs multiples ; nous n'en avons aucune.

⇒ Piste sérieuse : on a rendu le monde conséquent par un mécanisme que le modèle ne peut pas voir.

---

## A5. Aucune abstraction TEMPORELLE (pas de H-JEPA)

La réponse de LeCun au long horizon est la **hiérarchie** : des prédicteurs à plusieurs échelles de
temps, le niveau haut planifiant en sous-buts. Nous planifions à **une seule échelle**, avec
**0,88 m** d'imagination (80 ticks) pour des cibles à **7,6 m** — mesuré. Et allonger le rollout est
réfuté (H=300 → survie au plancher) : c'est bien la hiérarchie qui manque, pas la longueur.

---

## LA CHAÎNE CAUSALE UNIFIÉE (ce que l'audit explique)

```
A1  latent effondré (27 %), ne porte pas l'objet
      ↓  (compensation historique)
A2  slot codé-main, 100 % géométrique  ← étiqueté « pur » à tort
      ↓
    la DÉCISION devient de la pure géométrie
      ↓
A3  -min_dist codé-main est ~optimal sur de la pure géométrie
      ↓
    un critique appris n'a AUCUN avantage informationnel  → il échoue (×4)
      ↓
A4  + le monde est devenu stochastique, mais le rêve suppose l'objet immobile
    → même la conséquence ajoutée est invisible au planner
```

**Les 4 échecs du critique ne sont pas 4 problèmes : c'est UN seul, en bout de chaîne.** Le critique
est la dernière pièce d'une chaîne dont les deux premiers maillons (représentation, perception) ne
sont pas JEPA. On demandait à un module appris d'améliorer un pipeline où **rien de ce qui décide
n'est appris**.

---

## Ce que l'audit invalide dans nos propres croyances

1. « Le slot est de la perception apprise, object-centric et pure » → **faux** sur la config servie :
   requêtes RGB codées, scoreur appris jeté, readout géométrique zéro-paramètre.
2. « Le WM est un substrat riche sur lequel empiler des têtes » → **faux** : son latent ne porte pas
   l'objet, donc aucune tête ne peut lire la scène.
3. « Le coût codé-main n'est qu'un échafaudage secondaire » → **faux** : c'est le critique de LeCun
   écrit à la main, c'est-à-dire la pièce centrale.
4. « Le monde périssable donne du travail au critique » → **partiellement faux** : il crée bien de la
   conséquence (33 % mesurés), mais par un mécanisme (relocalisation) que le WM déterministe à
   transport ego-only **ne peut pas prédire**.

## Ordre de traitement suggéré (du plus causal au plus dérivé)
1. **A1** — rendre le latent informatif (pression information-content type VICReg, jugée par la sonde
   « peut-on lire le slot depuis le latent ? » qui est gratuite et déjà écrite).
2. **A4** — donner au WM de quoi représenter l'incertitude, OU choisir un mécanisme de conséquence
   que le modèle peut voir. (Le moins cher : rendre la conséquence PRÉDICTIBLE plutôt que aléatoire.)
3. **A3/A2** — ne re-tenter un critique appris qu'une fois A1 réglé ; avant, il n'a rien à lire.
4. **A5** — la hiérarchie, chantier de fond, après.

**Ne rien empiler avant A1** : c'est le maillon dont tout le reste dépend.

---

# RÉPARATION D'A1 — exécutée, et ce qu'elle a vraiment montré (2026-07-24)

## Ce qui a été fait
Forme **Q(s, a) = V(latent à t=0, slot TRANSPORTÉ à l'horizon)** — l'état FIABLE du WM (le latent à
t0, non dégradé) combiné à ce que le candidat obtient géométriquement. Architecturalement pur : on ne
lit NI la rétine brute (ce qui donnerait au coût sa propre perception), NI le latent terminal
(dégradé). Cible = le **vrai retour actualisé, calculé exactement** (épisodes complets → aucun
bootstrap, donc aucune erreur de propagation — le défaut mesuré de la version TD). Corpus : 149 vies,
137 repas. `python/scripts/train_q_critic.py`.

## Résultat, avec CONTRÔLE D'ABLATION (posé d'avance)
| prédicteur du vrai retour | R² held-out |
|---------------------------|-------------|
| **token géométrique SEUL** | **+0,179** |
| latent + token | +0,149 |
| **apport du latent** | **−0,030** |

Le latent **n'apporte rien**, il nuit légèrement. Corrélation ~+0,32 dans les deux cas.

## Et pourtant l'indice de maturité EST bien dans le latent (mesuré)
| | R² held-out |
|--|-------------|
| lire la MATURITÉ (luminosité du buisson) depuis le latent | lin **+0,476** / MLP **+0,650** |
(indice servi et variable : buisson visible 80 % des ticks, luminosité moy 0,443 sd 0,276)

⇒ **Le changement de monde a fonctionné** : l'indice est rendu, il varie, il traverse la perception et
il est lisible dans la représentation. **Mais il ne prédit pas l'issue.**

## CONCLUSION HONNÊTE, et elle clôt la chaîne
Dans ce monde, **les repas futurs sont prédits par la GÉOMÉTRIE, point**. Ce n'est pas que le latent
soit vide (il porte la scène à 0,80, la présence à 0,59, la maturité à 0,65) : c'est que tout ce
qu'il porte en plus est **REDONDANT avec la distance** ou **sans effet sur l'issue**.

Cela révise A3 : `-min_dist` n'est pas une béquille qu'un critique devrait battre — c'est, dans ce
monde, **proche du meilleur prédicteur disponible**. Les 5 échecs du critique ne sont donc pas 5
erreurs d'ingénierie : ils mesurent une propriété du MONDE.

Ce qui rendrait un critique utile n'est donc PAS une meilleure tête ni une meilleure entrée, mais un
monde où **quelque chose de perceptible et de non-géométrique change l'issue**. Le levier périssable
crée bien de la conséquence (33 %), mais son mécanisme (relocalisation aléatoire) est
(a) imprévisible par construction et (b) invisible au rêve (A4, transport ego-only). C'est cohérent
avec tout ce qui précède : **A4 est maintenant le maillon critique, pas A1.**

---

# A1, TROISIÈME MESURE — l'encodeur ne perçoit PAS l'apparence de la NOURRITURE (2026-07-24)

Contexte : le monde v7 donne aux proies des TYPES dont la valeur est arbitraire — la seule condition
mesurée où un critique est NÉCESSAIRE. Précondition évidente avant d'entraîner : le critique doit
PERCEVOIR le type. Mesuré, et c'est non.

| où l'on lit le TYPE de la proie visée | teinte | luminosité |
|---------------------------------------|--------|------------|
| RÉTINE brute | **82,9 %** | **67,0 %** |
| sortie de l'ENCODEUR | 29,5 % | 35,6 % |
| LATENT | 27,3 % | 31,2 % |
| (majorité = hasard à battre) | 44,2 % | 32,3 % |

**L'information est dans l'observation et l'ENCODEUR la détruit** — dès la première couche, pas le
prédicteur.

Trois explications testées et TOUTES RÉFUTÉES :
1. « c'est le canal » (teinte vs luminosité) → non, les deux sont détruits ;
2. « l'objet est trop petit » → non, mesuré : baie 3,44 rayons, buisson 3,17 — quasi identiques ;
3. « ma sonde demande trop » (attribut de la plus proche, par argmin) → non : même la LUMINOSITÉ
   MOYENNE des rayons bouffe est illisible (**R² −0,659**), alors que la MÊME mesure sur le BUISSON
   donne **+0,650**.

**CAUSE RETENUE** : pendant l'entraînement du WM, la couleur de la NOURRITURE était CONSTANTE, tandis
que l'environnement/les buissons variaient. L'encodeur n'a donc alloué aucune capacité à représenter
l'apparence de la nourriture — c'est une constante pour lui. C'est le cas « hors-distribution » qui
avait été listé comme le seul justifiant un ré-entraînement.

## Conséquence : DEUX voies, et pour la première fois un retrain est FONDÉ
- **(a) CHEAP — porter le signal par le BUISSON** : mesuré à 0,650, donc ça marche tout de suite.
  Le buisson annonce le type de SA baie. Limite : incompatible avec la proie MOBILE (une baie qui
  vagabonde quitte son buisson), donc il faudrait choisir entre les deux leviers.
- **(b) FONDÉ — ré-entraîner le WM avec une couleur de nourriture VARIABLE dans la collecte.**
  Ce n'est PAS câbler une ressource dans le WM (§3) : on donne au substrat perceptif l'exposition à
  une variation qu'il n'a jamais vue, et ce qu'il apprendrait est GÉNÉRAL (discriminer des apparences),
  pas « la bouffe vaut tant ». C'est aussi exactement ce qu'exige le critère JEPA : la représentation
  doit porter l'information disponible et pertinente. Coût : une recollecte + un entraînement WM.

C'est le premier ré-entraînement du substrat que la mesure JUSTIFIE dans tout ce chantier — les
précédents auraient été des raccourcis.
