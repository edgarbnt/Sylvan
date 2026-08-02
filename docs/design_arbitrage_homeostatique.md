# Design — ARBITRAGE HOMÉOSTATIQUE (coût intrinsèque non-séparable) — pré-inscrit 2026-08-02

> Pré-inscription écrite AVANT tout diag/run/train (§1). Ouvre la branche `feat/arbitrage-homeostatique`
> après la clôture du chantier perception-de-la-faim et la réparation du monde `foret_v1`.

## Mission
L'entité meurt de soif **le ventre plein**. Rendre l'arbitrage entre pulsions correct **par la
FORME du coût intrinsèque**, pas par une tête apprise de plus : une seule fonction de besoin
NON-SÉPARABLE sur toutes les jauges, dont l'arbitrage se déduit au lieu d'être arbitré.

⚠️ **Ce chantier ne commence PAS par de l'appris, et c'est délibéré.** Dans le cadre LeCun le
coût intrinsèque est **immuable et non-entraînable** par définition ; seul le *critique* qui le
PRÉDIT est appris. Le chantier arbitrage précédent a fait l'inverse (tête apprise remplaçant
l'ordre designé) et a échoué en vies. On corrige d'abord l'étage qui est censé être designé.

## À lire d'abord
- `docs/design_critique_arbitrage.md` §VERDICT G3 + §INVESTIGATION POST-JUGE — **le négatif à ne
  pas répéter** et ses 3 sondes qui localisent le vrai mur.
- `python/sylvan/control/planning/command_planner.py:83-105` (coût survie designé), `:167`
  (`_survival_extension`, continuation ALTERNÉE codée-main), `:932-941` (foresight déficit).
- `diagnostics/diag_viabilite_monde.py` — le contrôle qui doit passer AVANT tout jugement.

## Bug mesuré (2026-08-02, corpus `eau_fix` + `fx_*`, 36 vies/bras)
- **55 % des morts sont des morts de SOIF avec de l'énergie en stock** (84 d'énergie restante,
  soif à 0). Identique avant et après la réparation du monde ⇒ ce n'est PAS un artefact du monde.
- **0,00 boisson / 1000 pas en médiane** pour **1,17 nécessaires** — alors que le budget de trajet
  est désormais VIABLE (27 m exigés sur 47 parcourables, 58 %).
- L'eau est à **4,01 m** en médiane et vue **37 %** des ticks : elle est atteignable (41 % des
  approches closent sous 1 m, contre 7 % pour la nourriture qui, elle, fuit).
⇒ Elle **peut** boire et **choisit** de ne pas le faire. C'est un défaut d'ARBITRAGE, désormais
attribuable puisque le monde ne l'empêche plus.

### Cause structurelle identifiée
Le coût servi est **séparable** (`survival_weight·deficit` + continuation alternée). Un coût
séparable ne peut PAS produire la propriété qui manque. La théorie homéostatique
([Keramati & Gutkin, eLife 2014](https://elifesciences.org/articles/04811)) définit

```
D(H) = ( Σᵢ |h*ᵢ − hᵢ|ⁿ )^(1/m)        avec n > m > 1
récompense d'un résultat K  =  D(H) − D(H+K)
```

et prouve que `n > m > 1` entraîne **quatre** propriétés, dont la 4ᵉ est exactement la manquante :
**effet INHIBITEUR des besoins non pertinents** — quand la soif est loin de sa cible, la valeur de
la nourriture s'effondre *automatiquement*, sans poids ni règle. Bonus démontrés par la même
forme : **aversion au risque** (concavité ∂²r/∂k² < 0 — ce que P2-bis avait dû câbler faute de
pouvoir l'apprendre) et **nécessité de l'escompte** γ<1 pour la stabilité physiologique.

## Essayé → résultat (NE PAS répéter)
| tentative | résultat |
|---|---|
| Tête apprise remplaçant l'ordre de cible (`SYLVAN_ARB_CRITIC`, G3 2026-07-20) | **ÉCHEC en vies** : critère visé atteint (morts-par-arbitrage 34→25) mais échec EXPORTÉ (danger 5→13, conso 108→96). Remplacement sans ancre d'aversion. |
| Amendement « correction plafonnée » de cette tête | **Mort-né avant paiement** : zone d'action 6 % des décisions, et P̂ n'y ajoute rien (13,3 % vs 14,2 % = pile ou face). |
| Notes MC par état (3 négatifs critique-waypoint) | Choix **flottants** ; l'analytique gagne par CONSISTANCE ⇒ contrainte de consistance obligatoire. |
| Foresight designé `survival_weight=300` | Gain réel mais **PLAFONNE** (2026-06-26). |
| P2/P2-bis — apprendre l'aversion en espérance | **ÉCHEC MATHÉMATIQUE** : une tarification risque-neutre ne refuse pas une option rentable-en-moyenne qui tue. |

## Le mur pré-inscrit, à dire d'avance
L'investigation post-juge du chantier précédent a montré que **le CHOIX n'était pas le goulot** :
basculer vers la ressource nécessiteuse ne paie que **37,7 %** (elle est LOIN), et basculer TÔT
exige de viser une ressource **hors du champ de vision** → dépendance à une mémoire spatiale.
**Si G2 échoue, la conclusion n'est pas « la forme homéostatique ne marche pas »** mais « le choix
n'était toujours pas le goulot », et le levier redevient portée/mémoire. Ces mesures datent d'un
AUTRE monde (1 bouffe + 1 eau, kin) : elles sont à **re-mesurer en G0**, pas à présumer (§2).

## Étapes — cheaper-first

### G0 — GRATUIT (0 run, 0 train), sur les corpus déjà payés
Rejouer hors-ligne, à chaque décision de replan des corpus `fx_*`/`eau_fix` :
1. l'ordre designé servi (`order_scores` sf/sw) ;
2. l'ordre qu'aurait donné `D(H)` non-séparable.

**PASS si** : (a) les deux ordres **diffèrent** sur ≥ 15 % des décisions ; (b) sur les états
« campés » (poursuit une jauge ≥ 60 alors que l'autre < 40), l'ordre homéostatique désigne la
jauge nécessiteuse dans ≥ 70 % des cas ; (c) le contrôle de viabilité du monde reste ✅.
**STOP si** les ordres coïncident (< 15 %) : la forme ne changerait rien, inutile de payer la suite.
Re-mesurer aussi les deux murs hérités (taux de succès des poursuites lointaines, part des
bascules exigeant une cible hors-vue) — s'ils dominent, le chantier est **re-scopé avant** G1.

### G1 — implémentation DESIGNÉE, opt-in, zéro constante fittée
`D(H)` dans le coût du planner, `SYLVAN_PLANNER_COST=homeostatic`, défaut inchangé.
**Forme PINNÉE au verdict G0** : `n=4, m=3` (n > m > 1, valeurs de la littérature), jauges
normalisées par leur plage, cible = jauge pleine. ⚠️ Ces exposants sont une **préférence du corps**
définie une fois (§3), **PAS** des constantes à ajuster jusqu'à ce qu'un gate passe. Un balayage
de sensibilité est autorisé comme MESURE rapportée, jamais comme sélection.
**Vérification obligatoire** : OFF ⇒ trajectoires bit-identiques.

### G2 — juge closed-loop (payé seulement si G0 et G1 passent)
2 bras × 3 seeds × 12 vies, monde `foret_v1` réparé, contrôle de viabilité passé d'abord.

## Critères de succès / KILL — pré-enregistrés
| gate | critère |
|---|---|
| **G-soif** (le BUT) | morts-de-soif-avec-énergie-en-stock **55 % → ≤ 30 %** |
| **G-boire** | boissons **0,00 → ≥ 1,17 / 1000 pas** (le nécessaire mesuré) |
| **G-non-régression** | consommation totale (repas+boissons) **≥** celle du bras designé, et survie moyenne **≥** 828 |
| 🛑 **KILL** | consommation totale en baisse, **ou** morts-danger en hausse (la signature exacte de l'échec exporté du G3 précédent) |

Le KILL est écrit ainsi **délibérément** : l'échec du chantier précédent n'était pas un raté du
critère visé, c'était un **transfert du problème ailleurs**. On le surveille par construction.

## Après (et seulement après)
Une fois la forme du coût intrinsèque validée, l'étage appris devient celui que LeCun rend
entraînable : un **critique qui PRÉDIT `D` futur** depuis le latent. Et la littérature
([Dulberg et al. 2022](https://arxiv.org/abs/2204.06608)) recommande de le faire **modulaire —
une tête par jauge, combinées au niveau de la valeur** (meilleure exploration, meilleure
efficacité en données, robustesse hors-distribution), ce qui est exactement le PRINCIPE N°3.
