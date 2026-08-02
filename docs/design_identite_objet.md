# Design — IDENTITÉ D'OBJET : « est-ce la même qu'avant ? » — pré-inscrit 2026-08-02

> Pré-inscription écrite AVANT tout diag/train (§1). Ouvre après le verdict de la question
> reformulée, qui a renversé le cadrage de la journée.

## Mission
Le repère de l'entité répond à « où est le truc rouge le plus proche ? », **à neuf à chaque
instant**. Il n'a aucune notion de continuité : si la proie poursuivie passe derrière un arbre
pendant qu'une autre apparaît, le repère bascule **silencieusement**. Lui donner la liaison
temporelle : *ce que je vois maintenant est-il la continuation de ce que je voyais avant ?*

## D'où vient cette piste (et pourquoi elle n'est pas une intuition de plus)
`[MESURÉ: diag_perception_honnete_g0.py --question une-proie]` **85,0 %** des ticks, la position lue
par le slot est **réellement occupée par une proie**. ⇒ **L'entité n'hallucine pas.** Le « elle
invente 61 % du temps » valait pour la question *« voit-elle LA plus proche ? »* — la mauvaise
question, posée avec un oracle qui ne connaît qu'une seule proie.

Elle regarde donc une VRAIE proie, simplement **pas la même d'un instant à l'autre**.

Trois observations indépendantes que ça expliquerait :
1. `[observé par l'owner à l'écran]` « il change énormément d'avis, plein d'aller-retours » — c'est
   la signature exacte d'une cible qui se réinitialise sans que l'entité le sache ;
2. `[MESURÉ]` la poursuite échoue alors que la proie est à 3 m — si la cible se réinitialise en
   route, l'effort ne s'accumule jamais ;
3. `[MESURÉ: G0 mémoire, juillet]` la mémoire spatiale n'a **jamais** rien rapporté — on ne mémorise
   pas un objet qu'on est incapable de reconnaître comme étant le même.

⚠️ **PISTE, PAS CONCLUSION.** Elle explique bien trois faits, ce qui est encourageant et ne prouve
rien. Les gates ci-dessous existent pour la tuer si elle est fausse.

## Ce que dit la littérature
- **Suivi à slots** (SAVi / SAVi++, SlotContrast, TSA) : le défi est nommé — *chaque slot doit
  maintenir une identité cohérente entre images malgré les changements d'apparence*.
- **Conflit d'objectifs** ([Dual-State Slot Attention](https://arxiv.org/html/2606.12601v1)) : *un
  slot à vecteur unique ne peut pas être à la fois sensible aux changements transitoires (pour la
  position) et invariant à eux (pour l'identité)*. ⇒ chez nous, le slot porte une POSITION qui doit
  changer chaque tick ; **il ne peut pas porter l'identité aussi**. Il faut deux choses séparées.
- **Occlusion** (TSA) : *désactiver le slot absent par une porte, en PRÉSERVANT son état pour une
  ré-acquisition cohérente*. C'est exactement le besoin d'une forêt.
- **Signal sans étiquette** (ré-ID auto-supervisée) : *deux détections dans la MÊME image sont à coup
  sûr des objets DIFFÉRENTS* (négatifs gratuits) ; le même objet entre images voisines est un
  positif.

⭐ **Ce dernier point contourne le piège documenté du projet** : la cohérence de transport seule
**verrouille sur les troncs** (un arbre immobile est plus prévisible qu'une proie qui fuit). Une
perte contrastive ne récompense pas la PRÉVISIBILITÉ mais la DISCRIMINATION entre objets
simultanés — le tronc n'y gagne rien.

## G0 — GATES GRATUITS, dans cet ordre (le plus tuant d'abord)

### G0-1 — EST-CE QUE ÇA COÛTE ? (usage avant faisabilité)
Sur les corpus existants : détecter les **bascules de cible** (la position lue saute de plus de
2 m en un tick sans que l'agent ait bougé d'autant), puis comparer le taux de réussite des
approches AVEC et SANS bascule.

| gate | barre |
|---|---|
| **fréquence** | ≥ **20 %** des approches sous 3 m contiennent une bascule |
| **coût** | les approches avec bascule réussissent **≥ 15 points** de moins |
| 🛑 **STOP** | bascules < 10 %, **ou** écart de réussite < 5 points ⇒ ça ne coûte rien, ne pas payer la suite |

### G0-2 — EST-CE FAISABLE PAR LA GÉOMÉTRIE SEULE ? (payé si G0-1 passe)
Si oui, le chantier est **cheap et sans apprentissage** : il suffit de suivre. Mesurer, pour chaque
éclipse (suite de ticks où la cible n'est plus vue) : sa durée, le déplacement de la proie pendant
l'éclipse, et la distance à la proie **la plus proche AUTRE**.

| gate | barre |
|---|---|
| **PASS géométrique** | dans ≥ **80 %** des éclipses, déplacement pendant l'éclipse < ⅓ de la distance à l'autre proie ⇒ ré-identification par simple continuité, **zéro apprentissage** |
| **sinon** | l'apparence est nécessaire → G0-3 |

### G0-3 — Y A-T-IL DE QUOI APPRENDRE ? (payé si G0-2 échoue)
Le signal contrastif exige que deux proies simultanées soient **distinguables**. Le monde sert
4 teintes ; deux proies de teintes différentes le sont trivialement, deux de la même teinte non.

| gate | barre |
|---|---|
| **PASS** | ≥ **50 %** des paires simultanées ont des teintes distinctes |
| 🛑 **STOP** | < 25 % ⇒ l'apparence ne discrimine pas, et la position ayant échoué en G0-2, il n'y a rien pour ré-identifier |

## Forme visée (si les gates passent)
**Ne PAS toucher au slot existant.** Il porte la position, c'est son métier, et il est mesuré bon
(85 % des lectures sont sur une vraie proie). On AJOUTE une liaison temporelle séparée — conforme au
conflit d'objectifs identifié par la littérature : position sensible au temps, identité invariante.

Entraînement **auto-supervisé** : négatifs = deux détections du même tick ; positifs = ticks voisins.
Zéro étiquette, zéro oracle. `food_rel0` reste un oracle d'ÉVAL.

## Critère de succès (le BUT)
`[MESURÉ comme référence: A/B sprint]` Pas une précision de suivi — **une vie meilleure** :
consommation par 1000 pas **VÉCUS** > le bras de référence, et survie moyenne ≥ référence.

🛑 **KILL** : consommation par temps vécu en baisse de plus de **15 %**, ou morts-danger **+3 ou
plus** (magnitude, pas direction).

## Réserves dites d'avance
- `[HYPOTHÈSE]` G0-1 peut très bien échouer : il est possible que les bascules soient fréquentes
  mais **sans coût**, parce qu'une autre proie fait tout aussi bien l'affaire. Dans ce monde, la
  théorie du régime optimal a déjà montré que **prendre n'importe quelle proie est optimal**
  (`[MESURÉ]` prendre tout 278 pts/1000 pas contre 125 pour le meilleur type seul). Si changer de
  cible ne coûte rien, le chantier meurt en G0-1 — et ce serait cohérent avec le reste de la journée.
- **Correction pour comparaisons multiples** partout où plusieurs signaux sont testés, par
  permutation.
- Un composant peut avoir d'excellents chiffres et **zéro effet en vies** (`[MESURÉ]` A/B perception
  du matin). D'où un critère de succès comportemental, et non de suivi.

---

## ⭐⭐ VERDICT G0 (2026-08-02, `diagnostics/diag_identite_objet_g0.py`) : **PASSÉ, et il survit à ses contrôles**

331 approches sous 3 m, 6 corpus.

### G0-1 ✅ — la bascule COÛTE
| | valeur |
|---|---|
| approches contenant une bascule | **59,8 %** (barre 20 %) |
| réussite SANS bascule | **66,2 %** (n=133) |
| réussite AVEC bascule | **43,4 %** (n=198) |
| écart | **+22,7 points** (barre 15) |

**La réserve du design est LEVÉE** : je craignais que les bascules soient fréquentes mais sans coût,
puisque la théorie du régime optimal montre que prendre n'importe quelle proie est optimal ici. Elles
coûtent.

### Contrôles adverses — les deux tiennent
**Causalité inverse ?** Une approche qui échoue dure plus longtemps, donc contient mécaniquement plus
de bascules (durée médiane 41 ticks sans, 86 avec). L'écart **survit à la stratification par durée** :

| durée | sans bascule | avec bascule | écart |
|---|---|---|---|
| 20-50 ticks | 81,2 % | 31,1 % | **+50,1** |
| 50-120 | 70,7 % | 44,9 % | **+25,8** |
| 120+ | 63,6 % | 54,9 % | +8,7 |

**Précédence temporelle ?** Une bascule dans les **20 PREMIERS** ticks → réussite **35,7 %** contre
**57,1 %** sans. La bascule PRÉCÈDE l'échec.

⇒ Ce n'est pas un artefact de durée. La bascule est bien **en cause**.

### G0-2 ✅ — et il n'y a probablement RIEN À APPRENDRE
2 260 segments sans bascule, durée médiane 12 ticks, q90 = 89 ticks. Une proie parcourt
**2,05 m** en 89 ticks, contre un seuil d'ambiguïté de **2,90 m** (⅓ de l'espacement minimum entre
bosquets). ⇒ **la simple continuité géométrique suffit à ré-identifier** : suivre, pas apprendre.

### G0-3 — 47,4 % de teintes distinctes (barre 50 %), **sans objet** puisque G0-2 passe.

## Conséquence pour G1
Le chantier devient **peu cher** : une persistance de cible (garder l'objet suivi tant qu'un candidat
reste dans un rayon de continuité, plutôt que de recalculer à neuf), **sans réseau**. La forme
apprise ne devient nécessaire que si la persistance géométrique échoue en vies.

⚠️ `[INFÉRÉ]` « zéro apprentissage suffit » vient d'un calcul de marge, pas d'une implémentation.
G1 doit l'implémenter et le juger EN VIES, avec contrôle d'action et KILL à magnitude.
