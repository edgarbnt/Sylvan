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

## Honnêteté sur le gain attendu
Un audit qui ne fait que retirer ne rendra pas l'entité intelligente. Sa valeur est **(i)** une ligne
de base à laquelle on peut se fier et **(ii)** éventuellement de la performance gratuite, comme
`far_align`. C'est un **prérequis**, pas le but — d'où le budget plafonné et la règle d'arrêt.
