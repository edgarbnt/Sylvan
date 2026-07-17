# Design — Gate-capacité : « survit à un changement de monde », prouvé en vies (embryon jour/nuit)

## Mission
Le chantier perception-types a atteint la PARITÉ (juge 42/10, WM typé promu) mais **pas la
VALEUR**. La valeur = la capacité que le codé-main NE PEUT PAS avoir : quand l'apparence du monde
change, l'entité s'adapte au lieu de devenir aveugle. Ce chantier la construit et la PROUVE en
vies : **swap d'apparence en cours de vie** + **re-mesure périodique de la perception** (l'embryon
du cycle jour/nuit) → l'entité re-perçoit et re-mange, là où les requêtes figées s'effondrent.

## À lire d'abord
- `docs/design_perception_types.md` (WM typé promu = la base) + `scripts/build_typed_slots.py`
  (la MESURE cluster+lien, réutilisée telle quelle en live).
- Mémoire 2026-07-08 (cycle naïf auto-confirmant) : pourquoi l'embryon PERCEPTION est bien moins
  exposé à ce mur que le jour/nuit DÉCISION.

## Pourquoi l'embryon, pas le jour/nuit complet (décision tranchée owner 2026-07-17)
1. **Verdict propre** : une variable à la fois (§4). Si on bâtit tout le cycle et que le swap
   échoue, on ne saura pas si c'est la re-mesure perception qui ne boucle pas ou le ré-entraînement
   des têtes de décision qui déstabilise. L'embryon ne touche QUE la perception.
2. **Mur orthogonal** : le cycle décision naïf est auto-confirmant (ε-manager + gates requis,
   2026-07-08) — ce mur vit dans les têtes de DÉCISION. La perception se voit passivement (les
   rayons rendent la bouffe qu'on la poursuive ou non) et les conséquences arrivent en vivant →
   l'embryon perception y est peu exposé. Ne pas payer ce mur pour un gate qui n'en a pas besoin.
3. **Zéro besoin démontré côté têtes** : le monde des têtes de décision n'a pas changé ; les
   ré-entraîner déstabiliserait une config fraîchement promue. La perception, elle, a un besoin
   démontré (le swap). Et l'embryon EST le squelette du jour/nuit — l'étendre plus tard = additif.

## Les deux pièces à construire
1. **Godot — swap d'apparence** (`SYLVAN_FOOD_SWAP_TICK=T`, `SYLVAN_FOOD_SWAP_HUE=Δ`) : à T pas dans
   CHAQUE vie, faire tourner la teinte de base `_albedo` de Δ et ré-appliquer à tous les items
   (avant T = apparence apprise, après = nouvelle). Opt-in, absent = OFF bit-identique. **Δ =
   propriété du MONDE déclarée** (jamais ajustée pour faciliter la récupération).
2. **Serveur — re-mesure périodique** (l'embryon, `SYLVAN_REMEASURE_EVERY=N`) : bufferiser par tick
   (rgbn du rayon le plus proche, distance, Δdrive), et toutes les N pas re-lancer cluster+lien
   (cœur de `build_typed_slots`) → mettre à jour `slot_encoder.color_queries` + `query_thr` en live.
   **N = période circadienne, constante du CORPS déclarée**. MESURE (zéro gradient). OFF = requêtes
   statiques (= comportement promu actuel, bit-identique).

## Bootstrap (honnête — comment on re-perçoit une bouffe qu'on ne perçoit plus)
Après le swap l'entité est aveugle à la bouffe. Elle récupère par deux voies vécues :
- **clustering** : la nouvelle couleur est RENDUE passivement → le groupe food se déplace en ≤ N pas ;
- **liaison** : `try_consume` mange par DISTANCE, pas par perception → même aveugle, l'entité mange
  ce qu'elle TOUCHE par hasard → l'énergie remonte → événement vécu → la re-mesure re-lie la
  nouvelle couleur à l'énergie.
Donc le critère est un **TEMPS-DE-RÉCUPÉRATION**, pas un instantané.

## Gates PRÉ-ENREGISTRÉS (cheaper-first ; budget : 2 pièces + pré-gate gratuit + 1 juge 2-bras)
0. **G-pré-swap (GRATUIT, offline — gate le travail Godot/serveur)** : sur `critic_kin_typcorp`,
   appliquer une rotation de teinte Δ à TOUS les rayons food, re-lancer la mesure → le prototype
   food récupère la couleur swappée (**cos ≥ 0.95** au food-swappé vrai) ET la liaison reste
   food→énergie. Échec → la mesure ne suit pas un swap → STOP avant tout Godot.
1. **G-swap-control (closed-loop — prouve que le défi est réel)** : bras CONTRÔLE = WM typé
   STATIQUE (pas de re-mesure) + swap Δ → forage **s'effondre** après le swap : taux de repas
   post-récupération ≤ **0.3×** le taux pré-swap. Si le contrôle ne s'effondre pas → Δ trop petit
   (l'augmenter — propriété monde déclarée), JAMAIS relâcher le gate.
2. **⭐ G-capacité (LE BUT, closed-loop)** : bras APPRIS = WM typé + re-mesure périodique + swap Δ
   → **RÉCUPÈRE** : taux de repas dans la fenêtre tardive (après swap+grâce) ≥ **0.6×** le taux
   pré-swap, ET **≫ contrôle** (taux appris − taux contrôle > bruit ±). Fenêtres pré-enregistrées :
   swap à T = 700 pas/vie ; grâce de récupération 200 pas ; fenêtre tardive [900, fin]. 2×24 vies
   seeds 1+2 poolés, chaque bras. KILL : appris ne récupère pas plus que contrôle → négatif.
Interdits : re-mesure = MESURE (pas de gradient) ; N (période) et Δ (magnitude) DÉCLARÉS, jamais
ajustés pour passer un gate ; pas d'oracle (le swap-hue du monde n'est ni entrée ni label — juste
la physique du monde qui change).

## ⭐ VERDICT G-pré-swap (2026-07-17, offline sur typcorp, 0 run) : **PASSÉ + contrainte découverte**
La mesure SUIT un swap vers une couleur LIBRE : food re-teinté magenta (0.83) → re-clusté
**cos 1.0000**, re-lié **food→énergie** (jaune 0.15 idem). **Contrainte de monde établie** : un swap
vers vert (=danger) ou cyan (≈eau) → food devient INSÉPARABLE du type existant → aucun learner ne
peut (cousin de G-sep) → la cible du swap Godot doit être une teinte LIBRE (magenta), **déclarée,
jamais l'inverse**. Leçon technique : swap = poser une teinte CIBLE propre (HSV, garder S/V), pas
une grande rotation `hue_matrix` (clip → couleur distordue — 1ᵉ faux négatif du pré-gate).
→ Godot/serveur LICENCIÉS. Reste à construire : swap Godot (teinte cible libre) + re-mesure
périodique serveur + juge 2-bras (contrôle statique s'effondre / appris récupère).

## Critère de succès = le BUT
Le temps-de-récupération après swap, mesuré en REPAS : bras appris ≫ bras contrôle. C'est la preuve
DIRECTE de « survit à un changement de monde » — celle que le monde figé ne pouvait pas donner
(Mur C). Si PASS : la re-mesure périodique (embryon jour/nuit) est promue ; le jour/nuit v1 (têtes
de décision) hérite d'un squelette validé. Si échec : négatif commité, la perception statique reste
le vivant (parité déjà acquise), l'embryon retourne au tiroir avec la cause diagnostiquée.

## ⭐ VERDICT (2026-07-17, juge 4×24 vies + diagnostic offline GRATUIT) : ÉCHEC DIAGNOSTIQUÉ — le monde ne peut pas poser ce test
Juge 2 bras × 2 seeds (contrôle STATIQUE / appris RE-MESURE, seeds 1+2, swap magenta T=700). Les
DEUX gates ratent : contrôle ratio tardif/pré = **0.50** (effondrement exigé ≤0.3 — ne s'effondre
PAS) ; appris **0.48 ≈ contrôle** (récupération exigée ≥0.6 ; Δtardif 0.00003 < bruit → ZÉRO
bénéfice). `scripts/judge_gate_capacite.sh` + `scripts.judge_gate_capacite` (segmentation des vies
via godot.log = vérité-terrain ; l'heuristique par saut de drive FUSIONNAIT des vies).

Diagnostic OFFLINE gratuit (0 run neuf, rejeu des corpus déjà collectés,
`diagnostics/diag_gate_capacite_offline.py`) — 3 causes, la 1ʳᵉ **décisive** :
1. **CAUSE-RACINE — le swap n'aveugle JAMAIS l'œil.** Le slot lit par COSINUS en RGB-normalisé, pas
   par teinte. Le magenta (teinte 0.83, « libre » en HUE) partage le canal ROUGE avec la requête
   food → **cos 0.90 > seuil 0.808 → 94 % des rayons magenta restent VISIBLES**. Le contrôle
   continue de forager (approche dirigée mesurée 85 %→62 %, baisse modeste, pas d'effondrement).
   `G-swap-control` était **structurellement inatteignable**. Piège de fond : aveugler le rouge
   exigerait une couleur à faible-R = vert(danger)/bleu(eau), OCCUPÉES → food inséparable (G-sep).
   **Dans un monde à 3 teintes primaires, AUCUNE couleur n'est à la fois invisible-au-rouge ET
   séparable → l'espace couleur est trop encombré pour poser un swap propre.** (G-pré-swap avait
   « passé » car il testait la MESURE/clustering — qui retrouve le magenta comme point distinct —
   PAS la cécité de la requête statique. « Libre en teinte » ≠ « loin en cosinus-RGB » : le trou
   qui a laissé passer un swap trop faible.)
2. **Le mécanisme n'isole pas food en live.** Food = ~8 % des rayons touchés, dominé par le vert
   (50 %, confond Mur A). K-means à K-découvert isole rarement un cluster food propre ; quand
   « energy » se lie, il choisit le vert (gclrn1, 1 fois — faux) ou un mélange rouge-dominant
   (gclrn2) — **jamais magenta**. Dépendant de la graine (1 vs 43 liaisons). Mauvaise liaison =
   risque de régression de la perception.
3. **Confond de protocole — swap par-vie + buffer inter-vies.** Le collecteur `policy_server`
   n'envoie jamais de reset TCP entre les vies → le buffer glissant (6000) mêle toujours rouge
   (pré-swap) et magenta (post-swap) de vies différentes → jamais de « changement de monde » stable
   auquel s'adapter.

**Conclusion honnête (§2)** : la capacité « survit à un changement de monde » reste NON prouvée —
pas parce que le mécanisme est impossible, mais parce que **CE MONDE ne peut pas poser le test**
(viabilité du test mesurée avant de juger l'agent — leçon métabolique appliquée à l'apparence).
Interdit de relâcher le gate. La perception STATIQUE reste la config vivante (parité 42/10 acquise).
Pièces construites conservées, défaut OFF **bit-identique**, re-mesure **NON promue** :
`SYLVAN_FOOD_SWAP_TICK/HUE` (`food_manager.gd`), `PeriodicRemeasure`
(`python/sylvan/control/remeasure.py` + `SYLVAN_REMEASURE_EVERY` dans `serve_planner_command.py`),
juge 2-bras.

## ⭐ PIVOT (owner, 2026-07-17) : de « swap d'apparence » vers « apparence→conséquence découverte »
Le swap-vers-couleur-proche est intestable dans un espace à 3 teintes (cause-racine ci-dessus). La
BONNE capacité, plus générale : l'entité se construit elle-même un dictionnaire {apparence →
interaction découverte du vécu}, OUVERT, sur PLUSIEURS canaux de conséquence — drives intéroceptifs
(déjà là) ET affordances physiques (obstacle bloque / herbe traverse), ces dernières apprises de
l'**ERREUR DE PRÉDICTION du WM** (déplacement commandé ≠ réalisé). Idée-clé : **absorber la dynamique
d'obstacle DANS le WM** → le planner l'évite par rollout imaginé, **zéro coût codé** (principiel §3 :
dynamique agnostique-drive, PAS un raccourci `--w-ressource`). Le coût d'ajout scale par **CANAL**
(sens — peu nombreux, fixés au corps à la conception), PAS par type d'objet (ouvert, découvert) : un
canal-mouvement payé une fois → toute la famille d'affordances physiques arrive découverte. 1ʳᵉ
instance testable = désambiguïsation **baie-buisson** (attribuer la conséquence au bon indice parmi
co-occurrents ; exige une DÉCORRÉLATION dans le monde = identifiabilité — buisson parfois sans baie,
baie parfois seule = ce qui la rend apprenable ET la teste). Deep research lancée (GVF/Horde, cue
competition, affordance par prédiction, découverte d'objets continue, motivation intrinsèque) puis
STOPPÉE (coût) et sauvegardée `.claude/workflows/deep-research.js` pour reprise. Prochain chantier à
pré-inscrire sur cette base.
