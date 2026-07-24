# Audit de conformité JEPA / LeCun — ce qui est faux, et ce qui SEMBLE pur sans l'être

**Date** : 2026-07-24. Demandé par l'owner après l'échec répété du critique appris.
**Méthode** : confronter l'archi RÉELLEMENT SERVIE aux prescriptions de LeCun (*A Path Towards
Autonomous Machine Intelligence*) et à la littérature JEPA récente — **sur mesures**, pas sur
impressions. Chaque anomalie ci-dessous est vérifiée sur le checkpoint vivant.

---

## A1. Le latent est EFFONDRÉ et NE PORTE PAS L'OBJET ⛔ (mesuré)

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

**Conséquence** : tout module lisant le latent est condamné. Cela explique d'un coup le critique
latent MC (corr −0,325) et le critique latent TD (+0,047).

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
