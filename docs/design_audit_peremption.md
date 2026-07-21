# Audit de péremption — les constantes designées survivent-elles au changement de corps ?

## Mission
Le corps a changé (pivot pattes → cinématique, 2026-07-06). Les constantes designées du planner ont
été calibrées **avant**. `far_align` a prouvé qu'une telle constante peut devenir **activement
nuisible** sans que personne le remarque. Ce chantier balaie les autres — **en supprimant ce qui est
périmé**, pas en remplaçant du designé par de l'appris (mauvais bilan : critique-arbitrage échoué
faute de matière ; P2-bis : l'aversion est une préférence du corps).

## À lire d'abord
1. `diagnostics/diag_reach_curve.py` — l'instrument de jugement (Phase 0)
2. `diagnostics/diag_body_model_audit.py` — la mesure du modèle du corps (Phase 1)
3. `diagnostics/guards.py` — les gardes à appeler avant tout verdict

## La coupe qui structure tout
Les constantes ne sont pas de même nature :
- **(a) MODÈLE DU CORPS** (`nominal_speed`, `surv_turn_rate`, `resource_drain`, `restore`) : elles
  encodent un **fait physique mesurable**. Il y a une bonne réponse → verdict **gratuit**, et une
  valeur fausse est un **bug**, pas un arbitrage.
- **(b) PRÉFÉRENCES DE SCORE** (`align_gain`, `align_mode`, `urgency_weight`, `surv_margin_weight`,
  `surv_horizon`) : aucune valeur « vraie » → seul un **A/B au but** tranche.

## Carte d'exécution (vérifiée, pas supposée)
`heading_weight` et `far_align`/`align_gain` sont dans des **branches complémentaires** de `plan()`,
**jamais actives ensemble** : mono-drive (`water is None`) → `heading_weight` actif (L560/L609),
`far_align` inerte ; multi-drive `COST=survival` (**vivant**) → `surv_mode` retourne à **L1063**,
donc `heading_weight` **inerte**, seul `far_align` (L977) agit. **Auditer chaque réglage dans sa
branche**, sinon on mesure du bruit sur du code mort.

## Résultats acquis

**Phase 0 — instrument** (2026-07-21). `diag_reach_curve.py` : opportunité = tick avec cible planner
à distance *d* **devant** ; atteinte = consommée avant l'échéance `slack × d / vitesse` (échéance
**proportionnelle** à *d* → équitable entre bandes) ; vitesse **mesurée** par corpus ; poolage
multi-seed ; `guards.sanity()` par corpus ; `--selfcheck`. Validé en reproduisant le verdict
`far_align`. **Chiffres canoniques far_align** (poolé 2 seeds, n ≥ 823/bande), l'effet **croît avec
la distance** : [0,2) 88,1→94,4 (+6,4) · [2,4) 80,7→89,7 (+9,0) · [4,6) 64,7→78,0 (+13,3) ·
[6,8) 27,2→47,2 (+20,0). Ils **remplacent** le « 47→70 » (calcul inline non reproductible).

**Phase 1 — modèle du corps** (2026-07-21, 3 corpus, zéro run) :

| constante | déclaré | mesuré | verdict |
|---|---|---|---|
| `nominal_speed` | 0.02 m/pas | **0.0100** | 🚨 **PÉRIMÉ ×2** |
| `surv_turn_rate` | 0.015 rad/pas | 0.0150 | ✅ correct |
| `resource_drain` (+`_T`) | 0.0005 / 0.00035 | idem | ✅ correct |
| `resource_restore` | 0.4 | 0.3995 absorbé | ✅ correct |

`SYLVAN_PLANNER_SPEED` et `SYLVAN_PLANNER_TURN_RATE` ne sont overridés par **aucun** harnais.
Conséquence de `nominal_speed` (`t = d / nominal_speed`, `coût = t × drain`) : pour l'opportunité
médiane réelle (3,15 m) le planner croit que le trajet coûte **7,88** points de jauge au lieu de
**15,76** → il sous-estime d'un facteur 2 le prix métabolique du déplacement ; et le coût de virage
n'étant pas mis à l'échelle, il croit **tourner 2× plus cher** qu'en vrai.

**Négatif informatif banké** : `surv_turn_rate` avait été annoncé « suspect n°1 » sur une estimation
analytique (~0,019). La mesure la **réfute**. L'estimation n'avait été avancée qu'assortie de
l'obligation de la mesurer — ne pas la ré-ouvrir.

## Phase 2 — item n°1 : `nominal_speed` (PRÉ-INSCRIT, avant lancement)

**Hypothèse.** Le planner sous-estimant d'un facteur 2 le prix du déplacement, il accepte des
trajets qu'il ne peut pas payer → contribue aux morts « ressource vue mais inatteignable », le motif
qui a fait clore **mémoire** et **arbitrage** en l'imputant à un plafond de substrat.

**Contre-hypothèse à garder en tête (§2).** Le coût est **comparatif** entre candidats : un biais
partagé peut laisser l'argmax largement inchangé. Un résultat nul est donc **attendu comme
plausible**, et ne serait pas une déception — il localiserait le problème ailleurs.

**Protocole.** Bras témoin = corpus existants `arbgrad_graded_s{1,2}_r40_fa0` (`nominal_speed=0.02`,
jamais overridé ; harnais non dérivé depuis, et le seul diff du planner depuis leur collecte est un
`print` → chemin de décision bit-identique). Bras traité = même harnais, `PSPEED=0.010`, **mêmes
seeds 1 et 2**, tout le reste identique. Monde = arène ouverte, `FA=0` (condition propre),
multi-drive `COST=survival` (la branche où `nominal_speed` agit).

**Instrument.** `diag_reach_curve.py --a <témoin×2> --b <traité×2>`, conditionné devant.

**Validité (avant tout verdict).** `guards.sanity()` doit passer sur les **4** corpus. Un corpus
dégénéré (immobile > 60 %) rend le verdict **NUL**, pas négatif.

**Décision — pré-enregistrée.** C'est une constante de **catégorie (a)** : la valeur mesurée EST la
vérité, donc le test est une **non-régression**, pas un concours.
- **ADOPTER 0.010** si aucune bande (n ≥ 500) ne régresse de ≥ **5 points**. *(Inclut le cas « aucun
  effet » : la valeur juste est adoptée par défaut, et on note que la constante n'était pas
  porteuse.)*
- **KILL / investiguer** si une bande (n ≥ 500) régresse de ≥ **5 points** → alors le modèle faux
  compensait quelque chose, et ce quelque chose doit être nommé avant tout autre changement.
- **SOUS-PUISSANT** si n < 500 sur une bande → le dire, ne pas trancher sur cette bande.

**Budget.** Phase 2 ≤ 3 facteurs × 2 bras × 2 seeds = 12 runs. **Règle d'arrêt** : si les deux
premiers facteurs sortent sans effet, on arrête le balayage (les poids de score ne sont pas le
levier) et on passe à la mémoire.

### RÉSULTAT (2026-07-21) — **KILL**, et l'hypothèse de la Phase 1 est **RÉFUTÉE**

Poolé 2 seeds, conditionné devant, `guards.sanity()` OK sur les 4 corpus. Durée : ~15 min/run.

| bande | v0.02 (témoin) | v0.010 (mesuré) | Δ | n |
|---|---|---|---|---|
| [0,2) | 94,4 % | 93,2 % | −1,2 | 2011/1996 |
| [2,4) | 89,7 % | 89,5 % | −0,2 | 3431/3473 |
| [4,6) | 78,0 % | 75,0 % | −2,9 | 1698/1790 |
| **[6,8)** | **47,2 %** | **32,5 %** | **−14,7** | 864/1012 |

Direction **cohérente sur les 2 seeds** en [6,8) (−4,7 et −23,7), mais chaque seed prise seule y est
**sous-puissante** (n = 421 et 443 < 500) : l'effet n'existe qu'en poolé, et il est porté surtout par
la seed 2. Survie **inchangée** (195 vs 188 consommations ; 2153 vs 2167 ticks/vie) — cohérent avec
un instrument aveugle, et raison de plus de ne pas juger là-dessus.

**Verdict pré-inscrit appliqué : KILL** — une bande (n ≥ 500) régresse de ≥ 5 points. Donc, selon la
règle écrite d'avance, **le modèle faux compensait quelque chose, à nommer avant tout autre
changement**.

**Ce qu'il compensait — ⚠️ MA PREMIÈRE ATTRIBUTION ÉTAIT FAUSSE.** J'avais accusé
`deficit = relu(d_fin / vitesse × drain − niveau)` (`command_planner.py:1084`). Ce terme est dans la
branche **`else` de `surv_mode`** — donc **inerte** en config vivante (`surv_mode` retourne à L1063).
J'ai encore nommé un mécanisme sans tracer le chemin d'exécution. Le code réellement exécuté est
`_survival_extension`, sondé **gratuitement** par `diagnostics/diag_survival_tail.py`.

**Le vrai mécanisme : une FALAISE suivie d'un PLATEAU PLAT.** Dans la queue, `lived = min(t_die,
travel)` **sature** dès que le trajet dépasse le temps-avant-mort, et `margin` reste à **0** quand le
candidat meurt en route. Donc au-delà de la distance jugée atteignable, le score est une
**constante** — mesuré à énergie 0,30 : chute de −2400 puis **exactement plat** (0 gradient) à toute
distance ultérieure. Un score plat = **aucune préférence entre candidats** = l'entité cesse de
s'approcher. C'est la pathologie « knife-edge » déjà documentée pour `min_dist`
(`command_planner.py:62`).

Et le second terme ne sauve rien : `Δtime` est **exactement 0** dans toute la zone utile — **toute**
la préférence de la queue passe par `margin_w × margin`.

**Pourquoi la vitesse fausse le cachait.** La falaise se situe à la distance max jugée atteignable,
qui vaut mécaniquement `niveau / drain × vitesse` :

| énergie | falaise à spd=0.020 (déclaré) | falaise à spd=0.010 (vrai) |
|---|---|---|
| 0,20 | 7,9 m | 3,9 m |
| 0,30 | 11,9 m | 5,9 m |
| 0,50 | 16,0 m | 9,9 m |

Les ressources apparaissent entre **2 et 8 m**. Avec la vitesse fausse la falaise est **hors du
monde** → jamais rencontrée. Avec la vraie, elle tombe **en plein dedans** dès que l'énergie est
moyenne-basse → les cibles à 6-8 m entrent dans la zone plate. C'est exactement l'effondrement
observé sur `[6,8)`, bandes proches intactes.

⇒ **`nominal_speed = 0.02` n'est pas un modèle du corps : c'est le réglage qui poussait une falaise
hors du monde.** DÉCLARÉ comme tel (§2 : ne pas le laisser passer pour une mesure).

**Conséquence pour l'hypothèse de la Phase 1 : RÉFUTÉE.** J'avais avancé que la constante périmée
pouvait contribuer aux morts « ressource vue mais inatteignable ». C'est **l'inverse** : elle *aidait*
la portée lointaine. La contre-hypothèse pré-inscrite (« un biais partagé peut laisser l'argmax
inchangé ») était trop douce — l'effet n'est ni nul ni dans le sens prévu.

**Ce que ça vaut méthodologiquement.** Insérer la vérité mesurée aurait **dégradé** l'entité de 15
points de portée lointaine. Sans le critère KILL écrit d'avance, j'aurais adopté 0.010 comme
« évidemment correct » et perdu cela en silence. C'est l'argument le plus net de la session pour la
pré-inscription — et une limite réelle de la doctrine « purifier = mettre la vraie valeur ».

**Décision.** Garder `nominal_speed = 0.02` pour l'instant, **reclassé « compensation déclarée »** et
non « modèle du corps ». Ne PAS re-tester cette constante **seule** : le négatif est banké. Elle ne
pourra être corrigée qu'**en même temps que la falaise** — c'est la version exigeante de la doctrine :
*on ne corrige pas une constante sans corriger ce qu'elle compensait.*

---

## Phase 3 — supprimer la FALAISE (PRÉ-INSCRIT, avant lancement)

**Le défaut, en une phrase.** Quand la queue juge une cible inatteignable, elle renvoie une
**constante** au lieu d'un signal gradué — elle perd l'information « à quel point c'est loin », donc
le planner n'a plus de raison de s'en approcher.

**Le fix, minimal et sans nouvelle constante.** Aujourd'hui `margin = 0` quand le candidat meurt en
route (plancher artificiel). On **retire le plancher** et on laisse la marge devenir **négative** :

```
margin = − (travel − t_die) × drain        # le manque, en unités de jauge
```

C'est **continu** à la frontière (quand `travel → t_die`, la marge → 0 des deux côtés) et
**monotone décroissant** au-delà → le gradient est restauré partout. Zéro paramètre ajouté, zéro
oracle : on **enlève** un clamp, on n'empile rien. Derrière un flag `SYLVAN_PLANNER_TAIL_GRADED`,
défaut **OFF** = bit-identique.

**Hypothèse.** La falaise est le vrai défaut ; la vitesse fausse ne faisait que la pousser hors du
monde. Donc **falaise supprimée + vraie vitesse** doit valoir au moins la ligne de base — et rendrait
en prime le modèle du corps honnête.

**Contre-hypothèse gardée (§2).** Le gradient restauré pourrait faire **poursuivre des cibles
réellement inatteignables** (l'entité meurt en chemin au lieu de se rabattre sur l'autre ressource).
Ce serait un négatif informatif : la falaise serait alors une **prudence utile**, pas un bug.

**Protocole.** Témoin = les mêmes corpus `arbgrad_graded_s{1,2}_r40_fa0` (falaise, spd=0.02).
Traité = `TAIL_GRADED=1` **et** `PSPEED=0.010`, mêmes seeds. 2 runs.

**Décision — pré-enregistrée.**
- **ADOPTER** (falaise gradée + vraie vitesse) si aucune bande (n ≥ 500) ne régresse de ≥ 5 points.
  Double gain : gradient sain **et** modèle du corps honnête.
- **NÉGATIF INFORMATIF** si une bande régresse de ≥ 5 points → la falaise est une prudence utile ;
  on la garde, on l'écrit, et l'audit des constantes s'arrête là.
- **SOUS-PUISSANT** si n < 500 sur une bande.
- Validité : `guards.sanity()` sur les 4 corpus, sinon verdict **NUL**.
- Non-régression exigée **avant** le run : avec le flag ON et une cible **atteignable**, le score
  doit être **bit-identique** à l'actuel (le fix ne touche que la zone morte).

**Non-régression VÉRIFIÉE avant lancement** (2026-07-21) : bit-identique sur toutes les cibles
atteignables (énergie 0,3/0,5/0,9 × distances 1-5 m), et gradient restauré dans la zone morte à
**−10 points/m — exactement la même pente que dans la zone vivante** (score continu en pente).

⚠️ **Limite assumée** : la **falaise subsiste** (−2410 à la frontière) ; seul le **plateau plat** est
corrigé. C'est délibéré — « y aller me tue » est un signal juste ; c'était l'**absence de préférence
entre cibles lointaines** qui était le bug. Ne pas revendiquer plus que ça.

### RÉSULTAT (2026-07-21) — **NÉGATIF**, et la limite assumée était en fait la cause principale

| bande | témoin (falaise + 0.02) | gradué + vraie vitesse | Δ |
|---|---|---|---|
| [0,2) | 94,4 % | 94,3 % | −0,1 |
| [2,4) | 89,7 % | 88,5 % | −1,2 |
| **[4,6)** | **78,0 %** | **71,2 %** | **−6,8** |
| **[6,8)** | **47,2 %** | **35,5 %** | **−11,7** |

Deux bandes régressent de ≥ 5 pts (n ≥ 845) → **négatif informatif** selon la règle écrite d'avance.
Et surtout : comparé à la Phase 2 (vraie vitesse **seule** : 75,0 / 32,5), le gradient donne
**−3,8 / +3,0** — un **lavage**. Il n'apporte rien.

**Pourquoi il ne pouvait rien apporter** (sondé après coup, gratuitement). À énergie 0,30, vraie
vitesse : cible à 3 m → score 3030 ; à 7 m → **600**. Le gradient de −10 pts/m **ordonne les cibles
lointaines entre elles**, mais l'écart avec une cible proche reste de **~2430**. Une cible lointaine
ne redevient **jamais** compétitive. **Le plateau était réel mais SECONDAIRE ; c'est la FALAISE qui
décide — et je l'avais délibérément laissée.** J'ai corrigé le symptôme mesurable en laissant la
cause, tout en l'écrivant comme une « limite assumée ».

**Décision.** Le flag `SYLVAN_PLANNER_TAIL_GRADED` est **RETIRÉ du code** : il n'achète rien, et
garder un bouton de plus sur une fonction qui en compte déjà quinze contredit la doctrine de
l'audit (« chercher ce qui peut être RETIRÉ »). Le constat reste écrit ici et dans la sonde
`diagnostics/diag_survival_tail.py`, qui est **conservée** — c'est elle qui a localisé la falaise.

## CONCLUSION DE L'AUDIT (2026-07-21)

**Les constantes ne sont pas le levier.** Trois runs payés, deux négatifs, et le vrai résultat est
structurel : la queue analytique ne peut pas être réparée par ses paramètres, parce que sa **forme**
suppose un monde statique (un trajet atomique, sans replanification, sans respawn) et déclare
« mortelle » une cible que l'entité atteint en réalité 47 % du temps.

**Et derrière la queue, la cause-racine est l'HORIZON D'IMAGINATION.** Le WM déroule 80 pas
= **0,8 m**, dans un monde où les ressources sont à **2-8 m**. La phase 2 doit donc estimer ~97 % de
l'avenir — d'où sa nécessité, ses quinze boutons, et son modèle du monde codé à la main. C'est aussi
la raison déjà consignée de l'échec du critique appris (`command_planner.py:745-760`) : avec un rêve
de 0,8 m, les 33 candidats sont quasi ex-æquo (marge relative 0,003-0,005), et on demande à une tête
entraînée à *prédire une valeur* de *classer* des options indiscernables.

⇒ **Ni « régler les constantes » ni « remplacer la queue par de l'appris » ne marchent tant que
l'imagination est trop courte.** Le vrai choix structurel est l'horizon : rollout plus long (≈5× le
coût, fidélité open-loop dégradée) ou **abstraction temporelle** (WM qui saute dans le temps —
direction H-JEPA, déjà amorcée par l'étage waypoint).

### ⚠️ CETTE CONCLUSION EST RÉVISÉE — test gratuit de divergence (2026-07-21)

J'ai testé ma propre conclusion avant de la faire payer (§2 : « une conclusion qui arrange est
suspecte »). `diagnostics/diag_candidate_divergence.py` déroule les 117 candidats **analytiquement**
(le corps cinématique obéit exactement à (vx, ω) ; constantes mesurées en Phase 1) sur **300 états
réels** — borne supérieure, aucun WM, aucun run.

| horizon | trajet | divergence géométrique | ρ(score, −distance), ordre fixé |
|---|---|---|---|
| **80 (actuel)** | 0,80 m | **12,8 %** | **0,734** |
| 120 | 1,20 m | 25,8 % | 0,838 |
| **240** | 2,40 m | 73,1 % | **0,931** |
| 480 | 4,80 m | 135,8 % | 0,779 |

**Ce que ça réfute.** « Les candidats sont quasi ex-æquo, l'horizon est le mur » est **trop fort**.
À l'horizon actuel les candidats divergent déjà de **12,8 %** de la distance à la cible, et le coût
transmet cette géométrie à **ρ = 0,73** (1 % d'ex-æquo seulement). Ce n'est pas détruit.

**Ce que ça confirme, modestement.** Allonger l'horizon **améliore** la fidélité du classement
(0,73 → 0,93 à 240 pas ≈ 2,4 m), puis la **dégrade** au-delà (0,78 à 480) — les arcs longs courbent
et les hypothèses de la queue lâchent. Il y a donc un gain réel mais **borné**, autour de 2-3 m.

**Ce que ça révèle, et qui est neuf.** Le signal de valeur est **99,6 % constant** :
valeur médiane **3085**, étendue entre candidats **12,5** → part informative **0,40 %**. La cause est
`time.clamp(max=cap)` : tout candidat qui survit sature à 3000, et seul `margin_w × margin` varie
(cohérent avec `Δtime = 0` mesuré plus tôt).
- Pour un **argmax**, cet offset est sans importance — d'où un planner qui fonctionne.
- Pour une **tête entraînée en MSE** sur cette valeur, il est **fatal** : il faut résoudre 0,40 % de
  la cible pour classer, ce qui est sous le bruit d'apprentissage.

⇒ **L'échec du critique appris s'explique par le CADRAGE DE LA CIBLE, pas par l'horizon.** Il n'a
jamais été testé équitablement : on lui a demandé de prédire un nombre dont 99,6 % est un offset
constant. C'est corrigeable en centrant/normalisant la cible — **sans toucher au WM ni à l'horizon**.

**Négatifs bankés — ne pas répéter** : `nominal_speed` seule (KILL) ; `nominal_speed` + gradient de
zone morte (négatif) ; `surv_turn_rate` (déjà correct, réfuté gratuitement).

## Honnêteté sur le gain attendu
Un audit qui ne fait que retirer ne rendra pas l'entité intelligente. Sa valeur est **(i)** une ligne
de base à laquelle on peut se fier et **(ii)** éventuellement de la performance gratuite, comme
`far_align`. C'est un **prérequis**, pas le but — d'où le budget plafonné et la règle d'arrêt.

---

# GATE FORÊT (PRÉ-INSCRIT, 2026-07-21, avant lancement)

**Question.** Le monde forestier rend-il la politique actuelle — *aller vers la ressource urgente la
plus proche et visible* — **insuffisante**, sans pour autant la rendre **impossible** ?

⚠️ **Particularité de ce gate** : contrairement à tous les précédents, ici une **dégradation est le
signal RECHERCHÉ**. Le risque n'est donc pas de se tromper de signe, c'est de confondre *« plus
exigeant »* avec *« cassé »*. D'où une garde explicite sur la capacité de base.

**Protocole.** Témoin = corpus existants `arbgrad_graded_s{1,2}_r40_fa0` (aucun arbre). Traité = même
harnais, mêmes seeds, `SYLVAN_FOREST_COUNT=40` (anneau 2,5-11 m), tout le reste identique.
Instrument = `diagnostics/diag_reach_curve.py`, poolé 2 seeds, conditionné devant.

**Mesures rapportées, dont une de calibration.**
1. Courbe d'atteinte par bande.
2. Survie et consommations — sortent-elles du **plafond** ?
3. **Fraction de ticks où la bouffe n'est plus visible** dans la rétine = le hors-vue enfin créé.
4. **Occupation de rétine mesurée** (% de rayons sur un arbre) : c'est la VRAIE valeur du bouton
   densité, le `count` n'en est qu'un proxy. Le G0 borne l'utile à ~30 %.

**Décision — pré-enregistrée.**
- **SUCCÈS (le monde exige plus)** : les bandes lointaines `[4,6)` et `[6,8)` baissent de **≥ 5 pts**,
  **ET** la bande proche `[0,2)` reste **≥ 85 %** (l'entité sait encore fermer quand c'est possible),
  **ET** du hors-vue apparaît (> 5 % des ticks).
- **TROP DUR** : `[0,2)` tombe **sous 85 %** → la forêt casse la capacité de base, pas la stratégie.
  Réduire la densité et re-mesurer. *Ce n'est pas un échec du monde, c'est un mauvais réglage.*
- **SANS EFFET** : rien ne bouge de ≥ 5 pts → augmenter la densité.
- **NUL** : `guards.sanity()` échoue sur un corpus (entité immobile) → verdict nul, pas négatif.

## RÉSULTAT DU GATE FORÊT (2026-07-21) — **SUCCÈS**, sur les 3 conditions pré-inscrites

Poolé 2 seeds, `guards.sanity()` OK, forêt vérifiée active (40 arbres placés, ~3200 blocages).

| bande | sans forêt | FORÊT | Δ |
|---|---|---|---|
| **[0,2)** | 94,4 % | **92,9 %** | −1,5 *(dans le bruit)* |
| [2,4) | 89,7 % | 81,5 % | **−8,2** |
| [4,6) | 78,0 % | 68,5 % | **−9,4** |
| [6,8) | 47,2 % | 30,7 % | **−16,5** |

1. **Bandes lointaines ≥ 5 pts** ✅ (−9,4 et −16,5)
2. **Capacité de base préservée** ✅ ([0,2) = 92,9 % ≥ 85 %)
3. **Hors-vue créé** ✅ : 27,7 % → **41,8 %** des ticks (**+14,1 pts**)

**Le déficit CROÎT avec la distance** (−1,5 / −8,2 / −9,4 / −16,5) : c'est la signature attendue de
l'occlusion + détour — le proche reste atteignable, le lointain devient un problème de *stratégie*,
pas de *motricité*. Consommations 195 → 156 (−20 %).

⚠️ **Calibration à corriger** : l'occupation de rétine mesurée est **43,7 %**, au-dessus des ~30 %
que le G0 recommandait (à 60 % l'erreur du slot atteint 1,43 m). La densité est donc au **bord haut**
de l'utile. Baisser à ~28 arbres devrait préserver davantage la bande [2,4) tout en gardant l'effet
lointain — à mesurer, pas à supposer.

⚠️ **Ce que ce gate ne dit PAS.** Il établit que le monde **exige** davantage, pas que l'entité
**saura** y répondre. Il crée seulement la place pour que mémoire et détour comptent — la place qui
manquait depuis le début (G0 mémoire : `never_seen = 0`, tout était visible).
