# ÉTAT DES LIEUX — handoff courant (2026-07-21)

> Constat au PRÉSENT. L'historique long vit dans `memory/` ; l'état des modules dans
> `tools/archi_hud/architecture.json` (source de vérité). Ce fichier dit **où on en est** et
> **quoi faire ensuite**, rien d'autre.

## Mission
Une entité ALife qui **décide elle-même** (faim → chercher → approcher → survivre) par planification
dans un WM appris. Le but courant : qu'elle **fasse de vrais choix** et que son intelligence soit
**visible et mesurable**.

## À lire d'abord
1. `CLAUDE.md` (les 4 principes — ils ont tous servi cette semaine)
2. `memory/sylvan-guards.md` + `diagnostics/guards.py` — **les garde-fous de mesure, à appeler avant
   tout verdict** (nés de 8 auto-corrections en une journée)
3. `memory/sylvan-foraging-economy.md` — l'économie de survie, la courbe atteinte-vs-distance, et
   toutes les corrections de constantes

---

## 1. Ce que l'entité EST aujourd'hui (mesuré, pas supposé)

- **Perception : apprise et load-bearing.** Types découverts par conséquence, saillance-danger,
  affordance-obstacle. C'est la vraie réussite du projet.
- **Corps : cinématique**, vitesse **0,0100 m/tick** (MESURÉE ; la constante 0,02 qui traînait dans
  les diags était fausse d'un facteur 2).
- **Motricité : bonne.** Trajets quasi droits (ratio d'errance **1,08**), ferme une cible proche
  devant à **90-100 %**.
- **Décision : designée**, sauf UNE décision apprise validée en vies (le sprint-critique).
- **Survie : modeste** en multi-drive épars.

## 2. Le flow de la semaine (pourquoi on en est là)

Chaîne réelle, avec les verdicts :

1. **Audit** des formes d'intelligence → perception forte, décision designée.
2. **Obstacle** → G0-G2 PASS, **G3 gelé** (mur occluant fait perdre la cible → besoin de mémoire).
3. **Critique-arbitrage** → G0 « place 13-16 morts/24 » → G1 ε-cible → pin D1 → G2 3/4 → **G3 ÉCHEC**
   (morts exportées au danger).
4. **Mémoire (monde éparse)** → **G0 STOP** (la ressource est vue, ou hors de portée).
5. **Sondes** : mono-vs-multi (le trou de près = arbitrage, pas moteur), commitment (**négatif**),
   cône-vs-360 (**réfuté**), métrique juste (**l'artefact-derrière gonflait le déficit**).
6. **Cône de vision** pré-inscrit → **G0 partagé** → retrain WM **non licencié** → redirection obstacle.
7. **Mémoire (monde-mur)** → G0 place réelle → **A/B PASS**.
8. **kin_speed** → **la vitesse MASQUE** (courbe normalisée pire, décision identique).
9. **Tâche d'arbitrage DÉGÉNÉRÉE** découverte (drains identiques → écart binaire 0 ou 40) → drains
   asymétriques → **zone grise 0 % → 42 %** ; **fix drain par-jauge** dans le planner (bit-identique
   en symétrique).
10. **AUTOPSIE** → la « place d'arbitrage » était un **ARTEFACT DE MESURE** (2,0 cas réels/24 vies
    contre une barre de 5) → **chantier arbitrage CLOS, pour la bonne raison**.
11. **Budget métabolique ≈ 0** → la survie est une **loterie à dérive nulle** → instrument aveugle.
12. **far_align** → l'échafaudage **handicape** en arène ouverte (2 seeds), mais est **PORTEUR** en
    monde-mur (sans lui : immobile 79 %).
13. **Verdicts mémoire CORRIGÉS** (respawns comptés comme repas) : food-only **+107 %**,
    multi-drive **+23 %**.
14. **Garde-fous implémentés** (`diagnostics/guards.py`).

## 3. Les 8 auto-corrections — et la règle qui en sort

Toutes de la même forme : **une valeur crue au lieu d'être mesurée**.
portée théorique · étiquette de mort non décomposée · constante de vitesse (×2) · restore nominal
vs absorbé · respawns comptés comme repas · échafaudage jamais re-testé · critère trop faible
(direction sans magnitude) · anomalie prise pour un résultat.

> **RÈGLE : aucune constante ni étiquette ne fonde un verdict sans avoir été MESURÉE sur le corpus.**
> Appliquée par `diagnostics/guards.py` — `sanity()` + `check_constants()` avant tout verdict,
> `consumptions()` au lieu de tout compteur ad-hoc.

## 4. État des chantiers

| chantier | état | à savoir |
|---|---|---|
| **Arbitrage (critique de cible)** | **CLOS** | Pas de déficit réel (2/24 vs barre 5). Ne PAS rouvrir. |
| **Mémoire spatiale** | **FORT, non promu** | food-only **×2**, multi-drive **+23 %** (sous-puissant). `MultiSlotMemory` existe et marche. |
| **Obstacle (affordance)** | **GELÉ au G2, déblocable** | G3 attendait la mémoire — condition désormais satisfaite. |
| **Vision en cône** | pré-inscrit, **non licencié** | G0 partagé : crée du hors-vue (15,6 %) mais payoff non mesurable offline. Retrain WM non payé. |
| **Mode-1 (RL)** | parqué | plafond BC. |
| **Curiosité / configurator** | manquants | demandent un monde plus riche. |

## 5. Ce qui bloque VRAIMENT

1. **L'instrument.** Le budget par cycle est ≈ 0 → la survie est dominée par la variance. Tous les
   verdicts rendus sur la survie/les consommations sont **sous-puissants**. Les métriques qui VOIENT :
   courbe atteinte-vs-distance (n en milliers), ratio d'errance, budget par cycle, efficacité de cycle.
2. **Un biais dans toutes les mesures de la semaine** : `far_align` était allumé partout. En arène
   ouverte il **handicapait** (atteinte 4-6 m : 47 % → 70 % sans lui). Les chiffres antérieurs sont
   donc **pessimistes**.
3. **Le monde ne réclame pas d'intelligence.** Arène ouverte + 360° + 2 ressources : aller au plus
   proche suffit. C'est la raison de fond pour laquelle chaque brique ajoutée paraît « modeste ».

## 6. Ce qu'il faut creuser (priorisé, cheaper-first)

1. **AUDIT DE PÉREMPTION + ligne de base propre** (gratuit→1 run). **Tout le reste en dépend.**
   `far_align` avait été calibré pour le corps à **pattes** et jamais revu après le pivot cinématique
   → il handicape. **Toutes les autres constantes de décision sont dans le même cas.** Suspect n°1
   trouvé le 2026-07-21 : `surv_turn_rate = 0.015` (`command_planner.py:121`), commenté
   « hexapode ~25-50°/s » → **le coût de survie imagine encore le virage d'un corps à pattes**.
   Suspect n°2 : le retrait de `heading_weight` **décidé le 2026-06-25 n'a jamais atterri** — défaut
   du code toujours `2.0`, **18 harnais sur 34** le codent en dur à 2.0 (6 seulement à 0.0), alors que
   `CLAUDE.md` et la carte affirmaient « coût redevenu un pur `-min_dist` ». Carte corrigée.
   Balayer chacune (cf. le tableau dans `docs/prompt_session_debloquage.md`), jugée sur la **courbe
   d'atteinte**, **dans les deux mondes** (l'effet de FA est dépendant du monde).
   ⚠️ L'instrument était contaminé : `collect_reachprobe.sh` avait `FAR_ALIGN=1` **en dur, sans
   override** — corrigé (paramétrable, défauts inchangés). 3 autres harnais allument encore FA
   par défaut (`ab_obstacle_memory_multi`, `collect_arb_graded`, `collect_critic_corpus_kin`).
2. **Consolider la mémoire** (2-3 seeds) : c'est la SEULE brique qui *ajoute* une capacité mesurée.
   Objectif : passer le multi-drive de « suggestif » à solide → promotion.
3. **Dégeler l'obstacle G3** avec mémoire ON : première démonstration d'un **choix complexe**
   (atteindre une cible mémorisée derrière un mur).
4. **Anomalie non expliquée** : en monde-mur, l'entité est **immobile 49 % des ticks même avec
   far_align**. Elle percute énormément. Gratuit à investiguer, potentiellement gros.
5. **Puis seulement** : enrichir le monde (topologie, cône) pour que la planification et la curiosité
   aient un sens.

## 7. Ce qu'il ne faut PAS refaire (négatifs bankés)

- Rouvrir le **critique d'arbitrage** (place réelle sous la barre).
- Croire que **la vitesse** résout : elle masque (courbe normalisée pire, décision identique).
- Régler le **restore** : plafonné à 100, +50 % nominal = +8 % absorbé.
- Juger un changement de substrat sur **la survie brute**.
- Retirer `far_align` **en monde-mur** (il y est porteur).

## Critère de succès (le BUT)

L'entité fait un **choix complexe démontrable** : atteindre une ressource **vue puis cachée**, via un
détour, mieux qu'un agent sans mémoire — mesuré sur la **courbe d'atteinte**, pas sur la survie.
