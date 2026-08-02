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

### ⭐ VERDICT G0 (2026-08-02, `diagnostics/diag_arbitrage_homeo_g0.py`, 84 vies / 79 434 ticks) : PASSÉ, MAIS LA FORME EST DÉGÉNÉRÉE — G1 SIMPLIFIÉ
- **G0-a ✅** : désaccord **51,0 %** des 2 415 décisions jugées (barre 15 %).
- **G0-b ✅** : dans les 200 ticks précédant une mort de soif, la règle dit **BOIRE 83,2 %** du
  temps alors qu'elle ne poursuivait l'eau que **53,5 %** (barre 70 %).
- 🚨 **MAIS LA NORME n/m N'ACHÈTE RIEN, et c'est démontré** : le désaccord vaut 51,0 % pour
  **tous** les exposants testés (2;1,5 · 3;2 · 4;3 · 6;4) **et** tous les escomptes (0,999 ·
  0,997 · 0,99), à **tous** les apports (140 · 84 · 40 · 20). Raison analytique : `D` est
  symétrique et Schur-convexe pour `n>1`, donc comparer `(de−R, dt)` à `(de, dt−R)` revient
  toujours à comparer `de` et `dt`. **Pour deux jauges SYMÉTRIQUES à apport ÉGAL, la forme
  homéostatique se réduit EXACTEMENT à « la jauge la plus démunie d'abord ».**
  ⇒ La non-séparabilité achète l'aversion au risque et la dose-réponse non linéaire, **pas**
  l'arbitrage entre deux pulsions symétriques. Poser la norme `n/m` ici serait **habiller un
  résultat trivial d'une théorie qui ne travaille pas** (§2) → **G1 est simplifié**, et les
  exposants sortent du chantier.

🩹 **Deux erreurs de mesure attrapées dans ce G0 même, à ne pas refaire** : (1) la 1ʳᵉ version de
G0-b était **TAUTOLOGIQUE** — « campée » était défini comme *poursuivre la jauge haute alors que
l'autre est basse*, donc la nécessiteuse avait par construction le plus gros déficit et la règle
la désignait 100 % du temps ; les 100 % étaient dans ma définition, pas dans les données. (2) la
1ʳᵉ version comparait les gains **NUS**, sans coût de trajet, ce qui ne pouvait par construction
rien exercer de la forme. Le contrôle qui a sauvé le verdict est le **balayage de sensibilité** :
une ligne PLATE prouve qu'un test ne mesure pas ce qu'il prétend.

### G1 — implémentation DESIGNÉE, opt-in, zéro constante fittée
**FORME PINNÉE AU VERDICT G0 (simplifiée)** : priorité à la jauge la plus démunie, arbitrée contre
le coût de trajet. **Pas de norme `n/m`** — G0 a prouvé qu'elle est inerte ici. Les exposants et
l'escompte ne sont plus des paramètres de ce chantier.

### G1-bis — ancienne rédaction, PÉRIMÉE (conservée pour mémoire)
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

## ⭐⭐ VERDICT G2 (2026-08-02, runs `g2_{ref,homeo}_s{1,2,3}`, 36 vies/bras) : **PARTIEL** — direction bonne, barre visée NON atteinte, aucun échec exporté
Drapeau vérifié actif dans les 3 graines du bras homéo, absent dans les 3 graines de référence.

| | REF designé | HOMEO |
|---|---|---|
| survie moyenne | 828 | **1206** |
| survie médiane | 350 | 380 |
| consommation (repas+boissons) | 115 | **176** |
| vies pleines | 6 | **10** |
| morts de soif | 17 | **12** |
| morts de faim | 12 | 12 |
| morts au danger | 1 | 2 |
| boissons — médiane par vie /1000 pas | 0,00 | **0,99** |
| vies sans AUCUNE boisson | 67 % | **47 %** |
| **G-soif** (le BUT) — morts-de-soif-avec-énergie-en-stock | 53,3 % | **42,3 %** |

- **G-soif ❌** : 53,3 → 42,3 %, une vraie amélioration de 11 points, mais la barre pré-inscrite
  était **≤ 30 %**. Elle n'est pas atteinte. Pas de re-négociation du seuil (§2).
- **G-boire ⚠️ barre AMBIGUË DE MA PART** : j'avais écrit « 0,00 → ≥ 1,17 » en mélangeant une
  observation MÉDIANE-par-vie (0,00) et un seuil POOLÉ (1,17). En médiane par vie : 0,00 → **0,99**
  (sous la barre). En poolé : 1,67 → 1,72 (la réf passait déjà). **Faute de pré-inscription**, à ne
  pas refaire : fixer la STATISTIQUE en même temps que le seuil.
- **Non-régression ✅✅** : consommation +53 %, survie moyenne +46 %, vies pleines 6→10.
- **Aucun signal significatif** : permutation 20 000 tirages — survie p=0,151, repas p=0,146,
  boissons p=0,296. À 36 vies/bras, rien n'est établi.
- 🛑 **KILL — déclenché à la LETTRE, et c'est un défaut de MA pré-inscription** : « morts-danger
  en hausse » est vrai (1 → 2), mais sur 36 vies c'est du bruit, et la consommation MONTE au lieu
  de baisser. J'avais écrit ce KILL sans magnitude. Je le rapporte tel quel plutôt que de le
  réinterpréter après coup ; la leçon est qu'un KILL doit porter un seuil, pas une direction.

**Ce qui distingue nettement ce résultat de l'échec de juillet** : là-bas le critère visé passait
mais l'échec était EXPORTÉ (danger 5→13, consommation 108→96). Ici c'est l'inverse — le critère
visé échoue mais **rien n'est exporté** : morts de faim identiques (12→12), consommation en
HAUSSE. La forme « socle designé + hystérésis conservée » n'a pas reproduit la pathologie du
remplacement. C'est l'acquis principal.

**⚠️ Ne PAS étendre l'échantillon maintenant.** Ajouter des graines après avoir vu une tendance
favorable fabrique de la significativité (p-hacking). Si on veut trancher, il faut **pré-déclarer**
l'extension et sa règle d'arrêt AVANT de la lancer.

## Après (et seulement après)
Une fois la forme du coût intrinsèque validée, l'étage appris devient celui que LeCun rend
entraînable : un **critique qui PRÉDIT `D` futur** depuis le latent. Et la littérature
([Dulberg et al. 2022](https://arxiv.org/abs/2204.06608)) recommande de le faire **modulaire —
une tête par jauge, combinées au niveau de la valeur** (meilleure exploration, meilleure
efficacité en données, robustesse hors-distribution), ce qui est exactement le PRINCIPE N°3.
