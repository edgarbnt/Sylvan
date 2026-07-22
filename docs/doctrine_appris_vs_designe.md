# Doctrine — appris contre designé (décision owner, 2026-07-22)

## La décision

**Le but du projet est une thèse d'architecture, pas une performance.** Un coût codé à la main qui
bat une tête apprise n'est pas un succès : c'est exactement l'objet qu'on cherche à faire
disparaître. L'owner tranche explicitement :

> « Je préfère du code appris, comme LeCun et ses chercheurs le veulent, un peu moins performant,
> plutôt que d'avoir du déterministe — c'est la dernière chose dont j'ai envie, c'est l'inverse du
> projet. Je préfère passer des semaines à faire marcher un critique pour que mon entité soit
> intelligente, plutôt que de regarder des formules tourner. »

**Conséquence sur les critères de promotion** : un composant appris légèrement moins performant que
son équivalent designé est **promouvable**. La barre n'est plus « ≥ baseline » mais « ne s'effondre
pas ». Ce qui reste refusé : un composant appris **activement nuisible**.

⚠️ Ceci AMENDE la lecture usuelle de `CLAUDE.md` §4 (« JAMAIS échanger robustesse contre pureté »).
La robustesse reste exigée — pas la parité de performance.

## Pourquoi l'ancien palmarès de l'appris ne vaut pas comme argument

Trois échecs sont bankés : critique de valeur (R² **−0,567**), critique d'arbitrage (échec exporté
au danger), prédicteur d'obstacle (**6×** plus de collisions). J'en avais tiré « remplacer le
designé par de l'appris a un mauvais rendement ». **Cette généralisation était fausse, et la mesure
du 2026-07-22 la réfute.**

Les trois ont échoué dans un monde qui **n'exigeait rien** :
- le critique de valeur, parce que survivre ≈ géométrie et que la règle innée est *exacte* — dans un
  monde dégénéré, rien ne bat l'exact ;
- le prédicteur d'obstacle, parce que « tout bloque de près » était la solution *optimale* du monde
  d'entraînement.

Le contre-exemple est arrivé le jour même : la **mémoire spatiale** — composant 100 % appris
(EgomotionHead + encodeur de slot) — rapporte **+2,17 repas** (IC [+1,77 ; +2,58], 1,52 σ) là où le
designé plafonnait. Le palmarès n'était pas une propriété de l'appris, **c'était une propriété du
monde**. Le monde a changé.

## Le prérequis DUR avant de rouvrir le critique appris

Le critique avait échoué en partie parce que le **signal de valeur est constant à 99,6 %** (médiane
3085, étendue 12,5) : tout candidat qui survit sature au plafond, donc une tête entraînée en MSE doit
résoudre 0,40 % de sa cible pour classer — sous le bruit d'apprentissage.

🚨 **Dans la config validée le 2026-07-22, 98 % des épisodes sont PLEINS.** Le signal de valeur est
donc **de nouveau quasi-constant**. Relancer le critique appris tel quel retomberait exactement dans
le même mur.

⇒ **Il faut d'abord RESSERRER LE MONDE jusqu'à ce que la valeur discrimine** — que survivre redevienne
incertain. C'est un réglage de monde, pas un chantier d'apprentissage, et il conditionne tout le
reste de la direction. Critère opérationnel : la survie ne doit ni saturer (≥ 80 % pleins) ni
s'effondrer ; viser une dispersion réelle des durées de vie.

## Ordre des chantiers qui en découle

1. **Payer la dette du 2026-07-22** — fidélité open-loop du WM sous cône (l'encodeur a été entraîné
   sur des rétines 360°), isoler la cellule manquante (cône + rotation lente + mémoire), re-baser
   `diag_reach_curve`.
2. **Nettoyer les constantes périmées** — le corps a changé (rotation ×4), toutes les constantes de
   décision calibrées sur l'ancien corps sont suspectes.
3. **Geler le monde mécaniquement** (`WorldPreset` importé, écrit dans chaque corpus, asserté par
   `guards`) — sinon « geler » reste une promesse.
4. **Resserrer le monde** pour que la valeur discrimine (prérequis ci-dessus).
5. **Alors seulement** : attaquer le designé avec de l'appris — critique de valeur, puis perception
   active (terme d'information, la capacité qui n'existe pas du tout).

Supprimer le code mort reste prioritaire et orthogonal (`residu_ppo`/`corps_cpg` inatteignables,
signaux de récompense lisant des clés jamais produites).
