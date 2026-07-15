# État des lieux — le critique (2026-07-15)

> Handoff court et honnête. Détail et sources : `docs/recherche_critique_argmax.md`.
> Carte vivante : `tools/archi_hud/architecture.json` (module `critique_appris`).

## Mission

Que la note des 33 plans imaginés soit, à terme, **inné (câblé, minimal) + correction apprise du vécu**
(forme LeCun `C = IC + TC`), et non plus une formule entièrement codée-main. Locomotion = prérequis donné.

## Le critique en une phrase

Le planner imagine 33 mouvements `(vx, ω)`, déroule chacun dans le world-model, **note** chacun, exécute
le meilleur. Le critique **est le donneur de notes** — rien d'autre.

## Ce qui est ÉTABLI (mesuré, pas supposé)

1. **Le critique ne peut PAS remplacer la formule** — démontré, pas supposé. Les 33 plans finissent quasi
   au même endroit (rêve 1,6 m contre ressources 2-8 m) → l'écart entre le meilleur et le 2ᵉ est **1e-5**,
   alors que l'erreur d'un réseau est **2e-4** (20-50× trop grosse). La formule y arrive car son erreur est
   **exactement zéro**. *(sonde `diag_critic_aggregation.py`)*
2. **Le cadrage « pureté = supprimer la formule » était faux.** LeCun : `note = coût inné (immuable) +
   critique (appris par-dessus)`. Le coût survie **est** un coût inné à ses yeux. Le `CLAUDE.md` PRINCIPE N°3
   le disait déjà. → la cible est `inné + correction`, pas `critique seul`.
3. **`inné + correction apprise` : bâti, et REFUSÉ par son gate.** La correction (le critique apprend
   l'*erreur* de l'inné) ne prédit la survie que **+0.023** de R² mieux que l'inné seul sur des vies jamais
   vues (gate ≥ +0.10 ; un pli sur 4 **négatif**). Code présent (`--labels residual`,
   `SYLVAN_PLANNER_COST=residual`), **NON promu**. *(`train_survival_critic.py`)*
4. **La cause n'est pas le cerveau, c'est le VÉCU.** 57 vies, **une seule politique déterministe** → l'entité
   revit la même vie (les spawns varient, la réaction non). Rien à apprendre d'un vécu qui se répète.

## État courant qui TOURNE

| donneur de notes | forage (repas+eau, 12 vies) | statut |
|---|---|---|
| **formule innée seule** (`SYLVAN_PLANNER_COST=survival`) | **34,5** | ✅ le vivant |
| valeur parfaite (oracle, sonde) | 36,5 | plafond de la fente |
| critique appris **à la place** | 16 | ☠️ prouvé impossible |
| **inné + correction** (`=residual`) | non mesuré | 🔨 bâti, gate refusé, non promu |

## Exploration en espace-commande : TESTÉE, mauvais levier (2026-07-15)

Hypothèse : l'entité revit la même vie → lui faire vivre des vies **variées** donnerait à la correction un
contraste à apprendre. Le bouton `SYLVAN_CMD_EXPLORE_STD` était un no-op (bruit tiré par replan, re-corrigé
aussitôt). **Fix implémenté** : `SYLVAN_CMD_EXPLORE_PERSIST=K` — tenir le biais K replans (K=5 ≈ 1 m de
déviation engagée). **A/B court (8 vies × 2, `diag_life_diversity.py`)** :

| | exploration OFF | persistante (std 0.3, K=5) |
|---|---|---|
| ω saturé aux bornes ±0.6 | 44 % | 22 % → le bruit **atteint** bien la commande |
| **dispersion de survie** (le juge : diversité des issues) | 1120 | **742 (×0.66)** |

**Verdict = négatif propre.** L'exploration atteint la commande (le no-op est corrigé) mais **COMPRIME** les
issues au lieu de les diversifier : tout le monde meurt un peu plus tôt, sans structure. **Le monde est
simple et la politique quasi-optimale → perturber au hasard ne peut que dégrader.** Le bruit de commande
n'est pas le bon levier. *(errance non concluante : métrique polluée par les refills intra-vie.)*

## Le vrai fork (décision d'owner)

La marge du critique est **structurellement petite ICI** : l'inné forage déjà 34,5 contre un plafond de
36,5 (et l'écart restant est surtout métabolique, `diag_metabolic_ceiling.py`). Trois voies :

1. **Accepter l'inné comme point de fonctionnement.** Honnête : dans ce monde plat sans danger, la survie ≈
   géométrie, que l'inné capture déjà → un critique appris n'a presque rien à ajouter, par construction.
2. **Exploration au niveau du PLAN** (dernier levier d'exploration) : choisir un candidat non-argmax et le
   **tenir** — l'entité fait des CHOIX différents (aller à l'eau quand la bouffe primait), pas juste un ω
   bruité. Test gratuit : re-collecter + `diag_life_diversity` (dispersion de survie ↑ ?) avant tout gate.
3. **Enrichir le MONDE** (obstacles, ressources qui s'épuisent, danger) : alors l'issue dépend de plus que
   la géométrie → le résidu devient GRAND et apprenable → le critique reprend son sens. « Enrichir le monde
   avant le cerveau. »

**Gate déjà écrit** (`train_survival_critic --labels residual`, 2 min, critère +0.10) : à rejouer dès qu'un
corpus **réellement varié** existe (voie 2 ou 3). C'est lui qui dira si `inné + correction` prend vie.

## Direction choisie (2026-07-15, owner) : ENRICHIR LE MONDE — zone nocive (danger)

Décision owner : Sylvan doit être un système où l'expérience compte → voie 3. Premier élément = **zone
nocive** (une région qui abîme la santé). Choisi car la **place est prouvable** : le coût inné n'a aucun
terme de danger → l'entité fonce dedans en aveugle, coût inévitable par construction. Pas de piège collision.

**Bâti** : `godot/scripts/world/hazard_manager.gd` (disque sur le trajet spawn→bouffe, opt-in
`SYLVAN_HAZARD_COUNT`, défaut OFF = zéro régression). Branché dans `main.gd` par 4 lignes **NON stagées**
(chantier HUD owner — hooks locaux à intégrer côté owner ; toute la logique est dans le manager, stageable).
Gate gratuit : `diagnostics/diag_hazard_gate.py` (critères pré-enregistrés : aveuglement ≥50 %, coût réel).

**Méthode anti-boucle** (ce qui rend ce chantier différent) : le cher (WM-retrain pour percevoir le danger,
puis composant qui apprend à l'éviter) est **gaté derrière la preuve gratuite que la place existe**. On ne
paie l'étape N+1 que si l'entité aveugle SOUFFRE mesurablement du danger. Le baseline aveugle (santé perdue /
morts par danger) = le chiffre que l'entité percevante+décidante devra battre ensuite.

**Gate PASSÉ et CONSÉQUENT (2026-07-15)** : `diag_hazard_gate.py`, 12 vies OFF vs ON, plusieurs niveaux de
dégâts. Résultat : la santé est du **slack** (rien ne la lit avant 0 : ni planner, ni récompense, ni corps →
vérifié) → un danger sous-létal ne change rien (la faim tue avant). À **dégâts 0.5** (défaut verrouillé), la
zone devient **létale** : traverser vide la barre (100 dégâts) → **7/12 vies aveugles TUÉES par le danger**
(vs 0 sans danger ; morts de soif 8→0). Éviter = retour au régime normal → monde survivable *si contourné*.

- **BASELINE AVEUGLE = 7/12 (58 %) tuées par un danger invisible.** C'est le chiffre à faire tomber vers 0
  par une entité qui PERÇOIT et CONTOURNE. Payoff net, non ambigu (≠ marge floue du critique).
- **Config verrouillée** : `SYLVAN_HAZARD_COUNT=1` (r=1.3, dégâts=0.5, frac=0.55). Défaut OFF = zéro régression.

**ÉTAPE 1 FAITE + VÉRIFIÉE (2026-07-15) : le danger est PERCEPTIBLE.** `hazard_manager.gd` est devenu un
`Node3D` qui pose un cylindre **violet** `(0.6,0.12,0.85)` (mesh + Area3D couche-8 `retina_color`, calqué
sur food_manager) au centre de chaque zone. Vérif gratuite (collecte hazard-ON + parse `wm.retina0`) : la
rétine renvoie exactement le violet, distinct du rouge(bouffe)/bleu(eau) ; danger VU dans **79 % des frames**.
Zéro ligne changée dans la rétine (elle encode déjà RGB). main.gd : +1 ligne `add_child(hazard_manager)` (local, non stagé).

**PROCHAIN PAS — ÉTAPE 2 (le cher légitime, PRINCIPE N°3) : apprendre à VOIR le danger.** Aujourd'hui la
rétine capte le violet mais le WM/slot ne requête que rouge+bleu → l'entité voit sans comprendre. Il faut
**re-collecter le monde (hazard-ON) + ré-entraîner le WM** pour qu'il perçoive la position du danger (3ᵉ slot,
requête-couleur violet, comme le slot-2 eau). Gaté derrière l'étape 1 (faite). Puis :
3. Mesurer : une entité qui voit le danger le contourne-t-elle → morts-par-danger ↓ vers 0 (baseline 7/12), forage préservé ?
4. Alors le **critique-correction déjà codé** (`IC + λ·TC`, `SYLVAN_PLANNER_COST=residual`) a enfin un résidu
   structuré à apprendre (« près du danger → valeur basse ») → re-gate `--labels residual`.

## Critère de succès = le BUT

Forage (repas + boissons sur 12 vies), jamais la survie médiane (plafond épars = **métabolique**,
`diag_metabolic_ceiling.py`). Référence à battre : **34,5**. Plafond de la fente : **36,5**.
