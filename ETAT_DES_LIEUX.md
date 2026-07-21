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

## 4-bis. ASSAINISSEMENT DE L'INVENTAIRE (2026-07-21) — le code était en MEILLEUR état que les registres

Déclenché par un constat simple : à chaque fois qu'on regardait de près, quelque chose n'était pas ce
qu'il prétendait. Ce n'était pas le code qui s'écroulait — c'était la **comptabilité** qui était
optimiste. Trois corrections, toutes mesurées :

**1. La « baseline vivante » multi-drive ne tournait pas la config vivante.** Le corps cinématique est
promu depuis le 2026-07-07, mais `baseline_multidrive_slot.sh` servait le corps ET le WM **hexapodes**.
Le record cité (méd 2735-2820) venait donc d'une config **supersédée**. Re-mesuré, seed 1, 10 ép :

| | documentée (hexapode) | **réellement promue (cinématique)** |
|---|---|---|
| survie médiane | 2295-2525 | **3000 = PLAFOND** |
| épisodes pleins | 4/10 | **9/10** |
| repas médians | 2-3 | **6,5-8** |
| atteinte [0,2) | 87,7 % | **97,2 %** (+9,5) |
| atteinte [2,4) | 63,0 % | **93,8 %** (+30,8) |

⇒ **L'entité est nettement meilleure que ne le disait sa propre documentation.** Et la survie est
désormais **saturée** (9/10 au plafond) : **ne plus juger là-dessus**, utiliser la courbe d'atteinte.

**2. Deux modules comptés « purs » sont du code mort.** `residu_ppo` et `corps_cpg` : depuis la
promotion, `sylvan_agent.gd:831` fait `if kinematic_mode: … return` — le code dit lui-même
« ne sont jamais atteints ». Requalifiés `partiel` dans la carte. L'inventaire réel passe de
**7 purs / 7 partiels** à **5 purs / 9 partiels**.

**3. Le corps promu n'a atterri que dans 10 harnais sur 77**, dont **10 cités dans `CLAUDE.md` comme
vivants** tournent encore l'hexapode (`run_forage_wmslot`, `diag_nav_ab_wmslot`, `run_forage_purslot`,
`diag_nav_ab_purslot`, `run_hesitation_probe`, `run_forage_hex`, `run_forage_retina`, …). Dette
restante, à traiter avant de citer un de leurs chiffres.

**Correctif d'hygiène au passage** : `baseline_multidrive_slot.sh` faisait un `pkill -9` **global** qui
tuait les serveurs des autres runs. Retiré (il ne tue plus que le sien) ; port paramétrable.

## 4-ter. CRITIQUE APPRIS À CIBLE CENTRÉE — RE-TESTÉ, RÉFUTÉ (2026-07-21)

Le test de divergence avait montré que le signal de valeur est **99,6 % constant** (médiane 3085,
étendue 12,5) → une tête entraînée en MSE doit résoudre 0,40 % de sa cible pour classer. D'où
l'hypothèse : « le critique n'a jamais eu sa chance, il faut centrer la cible ».

**C'était déjà fait.** `train_survival_critic.py --labels residual` apprend `(vécu − inné)`, avec un
argument plus fin que le mien : l'inné est **exact** pour classer (écart d'action 1e-5 ≪ erreur d'un
réseau) ; ce que le critique peut seul apporter, c'est ce que l'inné **ignore**. Checkpoint
`survival_critic_resid_kin`, non promu, gate historique faible (R² moyen +0,13, un pli à +0,01).

**Rejoué sur données fraîches du corps promu** (34 vies, 9864 instants, CV 4 plis, critère écrit
avant) :

| | R² sur vies jamais vues |
|---|---|
| inné seul | **+0,437** |
| inné + correction apprise | **−0,129** |
| **gain** | **−0,567** (barre ≥ +0,10) — plis −0,219 / −0,406 / −1,111 / −0,531 |

**La correction apprise DÉGRADE l'inné, sur les 4 plis.** ⇒ voie « critique appris » **CLOSE pour ce
monde**, sur critère pré-enregistré.

**Mesure utile au passage** : l'inné est **optimiste ×2,07** (survie réelle méd 1470 pas contre 3036
prédits) — à rapprocher du `nominal_speed` 2× trop grand, dont l'A/B a montré qu'il est **porteur**.
L'optimisme est load-bearing, pas un simple bug.

**Et la raison était déjà écrite par le projet** (`docs/etat_critique.md`), désormais confirmée sur
données fraîches : *« en monde plat sans danger, la survie ≈ géométrie, que l'inné capture déjà → un
critique appris n'a presque rien à ajouter, par construction »*. La pré-inscription n'autorise le
rejeu que sur un corpus **réellement varié** ; il n'en existait aucun sur disque (nettoyés).
⚠️ **Correction le jour même** : j'avais écrit que `collect_sprint_corpus_v2.sh` « ne sert pas le corps
promu ». **Faux** — j'avais grepé le *wrapper* au lieu du fichier *délégué* (`collect_critic_corpus_kin.sh`,
qui utilise déjà `SYLVAN_KINEMATIC=1`). Aucune dette : le corpus varié est collectable immédiatement,
et il est en cours de collecte. **Règle : ne rien affirmer sur un harnais sans suivre ses délégations.**

## 5. Ce qui bloque VRAIMENT

1. **L'instrument.** Le budget par cycle est ≈ 0 → la survie est dominée par la variance. Tous les
   verdicts rendus sur la survie/les consommations sont **sous-puissants**. Les métriques qui VOIENT :
   courbe atteinte-vs-distance (n en milliers), ratio d'errance, budget par cycle, efficacité de cycle.
2. **Un biais dans toutes les mesures de la semaine** : `far_align` était allumé partout. En arène
   ouverte il **handicapait à TOUTES les distances**, et l'effet **croît avec la distance** — mesuré
   avec l'instrument persisté `diagnostics/diag_reach_curve.py`, poolé 2 seeds, n ≥ 823/bande :
   [0,2) 88,1→94,4 (**+6,4**) · [2,4) 80,7→89,7 (**+9,0**) · [4,6) 64,7→78,0 (**+13,3**) ·
   [6,8) 27,2→47,2 (**+20,0**). Les chiffres antérieurs sont donc **pessimistes**.
   ⚠️ Ces valeurs **remplacent** le « 47 → 70 » banké le même jour (calcul inline non persisté, donc
   non reproductible) : direction et classement confirmés, niveaux absolus différents car la
   définition diffère. **Ne plus citer le 47 → 70.**
3. **Le monde ne réclame pas d'intelligence.** Arène ouverte + 360° + 2 ressources : aller au plus
   proche suffit. C'est la raison de fond pour laquelle chaque brique ajoutée paraît « modeste ».

## 6. Ce qu'il faut creuser (priorisé, cheaper-first)

0. **FAIT — Phases 0 et 1 de l'audit (2026-07-21, zéro run).**
   **Phase 0** : l'instrument de jugement existe enfin (`diagnostics/diag_reach_curve.py`), validé en
   reproduisant le verdict `far_align` sur 2 seeds.
   **Phase 1** : le modèle du corps interne du planner a été **mesuré** sur 3 corpus
   (`diagnostics/diag_body_model_audit.py`). **Un vrai bug trouvé, un faux suspect réfuté :**
   - 🚨 **`nominal_speed` déclaré 0.02, mesuré 0.0100** — périmé ×2, et `SYLVAN_PLANNER_SPEED`
     n'est overridé par **aucun** harnais. Le planner croit qu'un trajet médian (3,15 m) coûte
     **7,88** points de jauge au lieu de **15,76** → il sous-estime d'un facteur 2 le prix
     métabolique du déplacement, et croit tourner 2× plus cher qu'en vrai.
     *Hypothèse à tester, pas conclusion* : cela pourrait contribuer aux morts « vue mais
     inatteignable » — le motif qui a fait clore mémoire et arbitrage en l'imputant au substrat.
   - ✅ **`surv_turn_rate = 0.015` est CORRECT** (mesuré 0,0150 rad/pas) : mon estimation
     analytique « ≈0,019 » était fausse. Elle n'avait été avancée qu'assortie de l'obligation de
     la mesurer — c'est ce qui a évité de « corriger » une constante juste.
   - ✅ drain (0.0005 / 0.00035 par jauge) et restore absorbé (0.3995) : corrects.

1. **AUDIT DE PÉREMPTION — Phase 2, item n°1 : FAIT → KILL, hypothèse RÉFUTÉE.**
   Corriger `nominal_speed` à sa valeur mesurée (0.02 → 0.010) **DÉGRADE** la portée lointaine :
   [6,8) m **47,2 → 32,5 %** (−14,7 pts, n=864/1012, direction cohérente 2 seeds) ; bandes proches
   dans le bruit ; survie inchangée. Critères pré-inscrits **avant** lancement (commit `8dd0216`).
   **Mécanisme — ⚠️ ma 1ʳᵉ attribution était FAUSSE** (j'avais accusé `deficit = relu(...)`,
   `command_planner.py:1084`, qui est **inerte** en config vivante). Tracé et **mesuré** ensuite
   (`diagnostics/diag_survival_tail.py`, gratuit) : le vrai code est `_survival_extension`, et son
   défaut est une **FALAISE suivie d'un PLATEAU PLAT**. `lived = min(t_die, travel)` **sature** et
   `margin` reste **0** quand le candidat meurt en route → au-delà de la distance atteignable le
   score est une **constante** (chute −2400 puis *exactement* plat). Score plat = **aucune préférence
   entre candidats** = l'entité cesse de s'approcher. Et `Δtime` est **exactement 0** : toute la
   préférence passe par `margin_w × margin`.
   **Pourquoi la vitesse fausse le cachait** : la falaise est à `niveau/drain × vitesse` — à énergie
   0,30 elle passe de **11,9 m à 5,9 m**. Les ressources apparaissent entre **2 et 8 m** : avec la
   vitesse fausse la falaise est **hors du monde**, avec la vraie elle tombe **dedans**.
   ⇒ `nominal_speed=0.02` n'est pas un modèle du corps mais **le réglage qui poussait une falaise
   hors du monde** — conservé, déclaré, négatif banké, ne pas re-tester seul.
   **Leçon de méthode** : insérer la vérité mesurée aurait coûté 15 points de portée en silence.
   Limite réelle de « purifier = mettre la vraie valeur » : **on ne corrige pas une constante sans
   corriger ce qu'elle compensait**.

2. **Phase 3 : NÉGATIF — et l'audit des constantes est CLOS.** Supprimer le plateau plat (marge =
   manque, zéro paramètre ajouté) **ne rachète pas** la vraie vitesse : [4,6) 78,0 → **71,2** (−6,8),
   [6,8) 47,2 → **35,5** (−11,7). Comparé à la vraie vitesse *seule* (75,0 / 32,5), le gradient donne
   −3,8 / +3,0 = **un lavage**.
   **Pourquoi** : à énergie 0,30, une cible à 3 m score 3030, à 7 m **600**. Le gradient (−10 pts/m)
   ordonne les cibles lointaines entre elles, mais l'écart avec une cible proche reste **~2430** —
   une cible lointaine ne redevient jamais compétitive. **Le plateau était réel mais SECONDAIRE ;
   c'est la FALAISE qui décide, et je l'avais délibérément laissée.** Flag **retiré du code** (il
   n'achète rien ; garder un bouton de plus contredit la doctrine). Sonde `diag_survival_tail.py`
   **conservée** — c'est elle qui a localisé la falaise.

3. **CONCLUSION DE L'AUDIT : les constantes ne sont pas le levier.** La queue analytique n'est pas
   réparable par ses paramètres : sa **forme** suppose un monde statique (trajet atomique, sans
   replanification ni respawn) et déclare « mortelle » une cible atteinte en vrai 47 % du temps.
   **Cause-racine derrière elle : l'HORIZON D'IMAGINATION.** Le WM déroule 80 pas = **0,8 m** dans un
   monde où les ressources sont à **2-8 m** → la queue codée main doit estimer ~97 % de l'avenir,
   d'où ses quinze boutons. C'est aussi la raison déjà consignée de l'échec du critique appris
   (`command_planner.py:745-760`) : rêve de 0,8 m → 33 candidats quasi ex-æquo (marge 0,003-0,005).
   ⇒ **Ni régler les constantes, ni remplacer la queue par de l'appris ne marchent tant que
   l'imagination est trop courte.** Vrai choix structurel : rollout plus long (~5× le coût, fidélité
   dégradée) ou **abstraction temporelle** (WM qui saute dans le temps — H-JEPA, amorcé par l'étage
   waypoint). Détail : `docs/design_audit_peremption.md`.
   `far_align` avait été calibré pour le corps à **pattes** et jamais revu après le pivot cinématique
   → il handicape. **Toutes les autres constantes de décision sont dans le même cas.** Suspect n°1
   trouvé le 2026-07-21 : `surv_turn_rate = 0.015` (`command_planner.py:121`), commenté
   « hexapode ~25-50°/s » → **le coût de survie imagine encore le virage d'un corps à pattes**.
   **Auto-correction n°9 (même jour)** : j'avais annoncé un « suspect n°2 » — le retrait de
   `heading_weight` qui n'aurait jamais atterri (18 harnais/34 à 2.0). **FAUX, corrigé après avoir
   tracé le chemin d'exécution** : le retrait a bien atterri dans les harnais single-drive vivants
   (tous à `0.0`), et surtout `heading_weight` est **INERTE en multi-drive `COST=survival`** (la
   config vivante : `surv_mode` retourne à L1063 avant la ligne 1091). J'avais compté des grep au
   lieu de lire les branches — exactement l'erreur que l'audit combat.
   ⇒ **`heading_weight` et `far_align` sont dans des branches COMPLÉMENTAIRES** : mono-drive → `hw`
   actif, `far_align` inerte ; multi-drive survival → l'inverse. Auditer chacun **dans sa branche**.
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
