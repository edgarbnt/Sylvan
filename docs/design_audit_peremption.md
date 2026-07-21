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

## Honnêteté sur le gain attendu
Un audit qui ne fait que retirer ne rendra pas l'entité intelligente. Sa valeur est **(i)** une ligne
de base à laquelle on peut se fier et **(ii)** éventuellement de la performance gratuite, comme
`far_align`. C'est un **prérequis**, pas le but — d'où le budget plafonné et la règle d'arrêt.
