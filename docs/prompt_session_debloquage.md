# Prompt de démarrage — session « audit de péremption, puis démonstration d'un choix complexe »

> Copier-coller le bloc ci-dessous comme premier message de la nouvelle session.

---

Salut. On reprend Sylvan. Lis d'abord, dans cet ordre : `ETAT_DES_LIEUX.md` (handoff courant,
2026-07-21), `memory/sylvan-guards.md`, puis `memory/sylvan-foraging-economy.md`.

**Contexte en une phrase.** La session précédente a produit **8 auto-corrections de mesure** — à
chaque fois une constante ou une étiquette CRUE au lieu d'être MESURÉE. Bilan : le chantier arbitrage
a été clos (sa « place » était un artefact de mesure), la mémoire a été réhabilitée (**×2** en
food-only), et surtout un échafaudage (`far_align`) s'est révélé **activement nuisible** en arène
ouverte. Des garde-fous automatiques existent maintenant (`diagnostics/guards.py`) — **utilise-les
systématiquement**.

## Le constat qui fixe la priorité de cette session

`far_align` avait été réglé pour le corps à **pattes**, et **jamais revérifié après le pivot vers le
corps cinématique**. Le retirer en arène ouverte améliore l'atteinte à **toutes** les distances :
gain de performance ET de pureté, en supprimant du code.

**Chiffres canoniques** (instrument persisté `diagnostics/diag_reach_curve.py`, poolé 2 seeds,
n ≥ 823 par bande) — l'effet **croît avec la distance**, cohérent pour un échafaudage « far-target » :

| bande | FA=1 | FA=0 | Δ |
|---|---|---|---|
| [0,2) m | 88,1 % | 94,4 % | **+6,4** |
| [2,4) m | 80,7 % | 89,7 % | **+9,0** |
| [4,6) m | 64,7 % | 78,0 % | **+13,3** |
| [6,8) m | 27,2 % | 47,2 % | **+20,0** |

⚠️ Ces chiffres **remplacent** le « 47 → 70 » banké le 2026-07-21 : celui-ci venait d'un calcul
python **inline non persisté**, donc non reproductible. Direction et classement des bandes sont
confirmés ; les niveaux absolus diffèrent car la définition diffère. **Ne plus citer le 47 → 70.**

**Or TOUS les autres réglages de décision sont dans le même cas** — ajustés à la main sur un corps qui
n'existe plus, jamais revalidés depuis :

**Le suspect n°1, trouvé en vérifiant ce prompt** (`command_planner.py:121`) :

```
surv_turn_rate: float = 0.015   # rad/pas de virage imaginé phase-2 (hexapode ~25-50°/s ...)
```

**Le planner modélise encore le corps à PATTES.** C'est sa vitesse de virage *imaginée* — donc tout
le coût de survie raisonne sur un corps qui n'existe plus. Ordre de grandeur : `kin_turn=1.5` avec
l'échelle de temps mesurée (0,8 → 0,0100 m/tick) donnerait plutôt **≈ 0,019 rad/pas** (~25 % de
plus). ⚠️ **Ce calcul est le mien — MESURE-le sur corpus** (`guards.measured_constants`, virage réel
rad/tick) avant d'en faire quoi que ce soit. C'est exactement le pattern `far_align` : une constante
calibrée sur l'ancien corps, jamais revue.

## ⚠️ Carte d'exécution — à connaître AVANT d'auditer quoi que ce soit

J'ai d'abord annoncé un « suspect n°2 » (le retrait de `heading_weight` qui n'aurait jamais atterri,
18 harnais/34 à 2.0). **C'était FAUX** — j'avais compté des `grep` au lieu de lire les branches.
Après avoir tracé `plan()` : le retrait **a bien atterri** dans les harnais single-drive vivants
(tous à `0.0`), et surtout les deux réglages vivent dans des **branches COMPLÉMENTAIRES, jamais
actives ensemble** :

| config | `heading_weight` | `far_align` / `align_gain` / `align_mode` |
|---|---|---|
| mono-drive (`water is None`) | **ACTIF** (L560 slot, L609 coords) | inerte |
| multi-drive `COST=survival` (**vivant**) | **INERTE** (`surv_mode` retourne à L1063) | **ACTIF** (L977) |
| multi-drive `COST=designed` | ACTIF (L1091) | inerte |

⇒ **Poser `SYLVAN_PLANNER_HEADING_W` dans un harnais multi-drive survival est sans effet ; auditer
`far_align` en mono-drive est sans objet.** Auditer chaque réglage **dans sa branche**, sinon on
mesure du bruit sur du code mort. `heading_weight` n'est donc **pas** prioritaire.

## Le tableau à balayer

| constante | défaut code | env | rôle / branche |
|---|---|---|---|
| **`surv_turn_rate`** | **0.015** | `SYLVAN_PLANNER_TURN_RATE` | **virage imaginé — calibré HEXAPODE** ⇦ commencer ici |
| `align_gain` | 60.0 | `SYLVAN_PLANNER_ALIGN_GAIN` | poids de far-align — **multi-drive survival seulement** |
| `align_mode` | `"mean"` | `SYLVAN_PLANNER_ALIGN_MODE` | `mean` (spirale) vs `end` (tourne tôt puis commit) — **jamais A/B testé** |
| `urgency_weight` | 6.0 | `SYLVAN_PLANNER_URGENCY_W` | poids de l'inconfort futur |
| `surv_margin_weight` | 200.0 | `SYLVAN_PLANNER_SURV_MARGIN_W` | tie-break de marge |
| `surv_horizon` | 3000.0 | `SYLVAN_PLANNER_SURV_H` | cap de la simulation de survie |
| `resource_drain` / `restore` | 0.0016 / 0.5 (code) ; 0.0005 / 0.4 (harnais) | `SYLVAN_PLANNER_DRAIN` / `_RESTORE` | **modèle interne du métabolisme**, posé à la main — le vrai drain mesuré est 0,05/tick |
| `heading_weight` | 2.0 | `SYLVAN_PLANNER_HEADING_W` | mono-drive seulement ; retrait déjà atterri → **basse priorité** |

On a **prouvé** qu'au moins l'une d'elles était devenue nuisible. Rien ne dit qu'elle est la seule.
Ne PAS auditer : `surv_discount` (négatif banké 2026-07-04, ne pas activer) et `commit_delta`
(à 0.0 = OFF, appartient au chantier arbitrage CLOS).

⚠️ **Nuance vérifiée dans le code** : `far_align` vaut **`False` par défaut** dans
`command_planner.py` — c'est **chaque harnais qui l'allume**. L'échafaudage vit donc dans les
*scripts de mesure*, pas dans la boucle. Inventaire fait (2026-07-21) — 4 harnais l'allument :
`ab_obstacle_memory_multi.sh`, `collect_arb_graded.sh`, `collect_critic_corpus_kin.sh`,
`collect_reachprobe.sh`. Ce dernier — **l'instrument de jugement** — l'avait **en dur, sans
override** : il ne pouvait littéralement pas exprimer la condition propre. **Déjà corrigé** :
`FA`/`AG`/`HW`/`UW`/`AMODE` sont paramétrables, défauts inchangés (campagnes passées reproductibles),
et le tag de corpus les porte. Les 3 autres restent à vérifier avant de re-mesurer.

## Ordre de marche (ne pas sauter d'étape)

**0. FAIT (2026-07-21) — l'instrument existe.** `diagnostics/diag_reach_curve.py` : courbe
atteinte-vs-distance, conditionnée **devant**, échéance **proportionnelle à la distance**
(`slack × d / vitesse`), vitesse **mesurée** par corpus, poolage multi-seed (`--a`/`--b`), gardes
`guards.sanity()` par corpus, `--selfcheck`. Validé en reproduisant le verdict `far_align` sur les
2 seeds, et la garde se déclenche bien sur le corpus dégénéré connu (monde-mur FA=0, immobile 79 %).

**1. AUDIT DE PÉREMPTION + ligne de base propre — LA priorité.**
Balayer ces constantes une par une, chacune jugée sur la **courbe atteinte-vs-distance** (jamais sur
la survie), exactement comme on l'a fait pour `far_align`. Les harnais existent
(`scripts/collect_arb_graded.sh`, `scripts/collect_reachprobe.sh`), les gardes aussi.
Objectif double : **supprimer/recaler les béquilles périmées** ET obtenir enfin une **référence saine**
— tous les chiffres de la semaine passée sont teintés par `far_align` et par une constante de vitesse
fausse d'un facteur 2.
⚠️ `far_align` est **dépendant du monde** : nuisible en arène ouverte, **PORTEUR en monde-mur** (sans
lui l'entité est immobile 79 % des ticks). Auditer chaque constante **dans les deux mondes**.

**2. Consolider la mémoire** (2-3 seeds, multi-drive, monde-mur) : passer le **+23 %** de « suggestif »
à solide. Juge = **courbe d'atteinte** (n en milliers), pas les consommations (n en dizaines).

**3. Dégeler le G3 obstacle** avec la mémoire branchée : la démonstration visée — **atteindre une
ressource vue puis cachée derrière un mur, via un détour, mieux qu'un agent sans mémoire**. C'est le
plus petit exemple concret de « décider », et il sera bien plus convaincant sur une base assainie.

**4. Si ça passe** → promouvoir la mémoire dans la config vivante, et seulement ensuite enrichir le
monde (topologie, cône).

**Anomalie gratuite à investiguer en chemin** : en monde-mur, l'entité est **immobile 49 % des ticks
même avec `far_align`**. Elle percute massivement l'obstacle. Personne n'a regardé pourquoi — gisement
potentiellement gros, coût nul.

## Ce sur quoi je veux que tu insistes

- **Mesure avant de croire.** Toute constante utilisée dans un jugement doit être vérifiée sur le
  corpus (`guards.check_constants`). Toute anomalie (métrique à 0, entité immobile) = on **CREUSE**,
  on ne rapporte pas.
- **Juge sur des métriques qui voient** : courbe d'atteinte, ratio d'errance, budget/cycle. La survie
  est un instrument **aveugle** ici (budget par cycle ≈ 0 → dérive nulle → dominée par la variance).
- **Magnitude vs bruit** : pas de « PASS » si l'effet est dans le bruit ; pas de « réfuté » sur
  quelques dizaines d'événements. Dis **« sous-puissant »** quand c'est le cas.
- **Pré-inscris** les critères avant de lancer, et ne les déplace jamais après.
- **Supprimer du code périmé est un gain double** (pureté + performance) : c'est la leçon
  `far_align`. Chercher activement ce qui peut être RETIRÉ, pas seulement ajouté.

## Ce qu'il ne faut PAS faire

- Rouvrir le **critique d'arbitrage** (place réelle 2/24 contre une barre de 5).
- Remplacer du **designé par de l'appris « pour la pureté »** : mauvais bilan (critique-arbitrage
  échoué faute de matière à apprendre ; P2-bis : l'aversion est une préférence du corps). Le gisement
  est dans la **suppression de béquilles périmées**, pas dans l'apprentissage à tout prix.
- Espérer que **la vitesse** règle quelque chose : elle **masque** (courbe normalisée pire, qualité de
  décision identique).
- Toucher au **restore** : plafonné à 100 (+50 % nominal = +8 % absorbé).
- Retirer `far_align` **en monde-mur** (il y est porteur).

Commence par me proposer ton plan d'audit pour l'étape 1 — quelles constantes, dans quel ordre, avec
quels critères pré-enregistrés — **sans rien lancer**.
