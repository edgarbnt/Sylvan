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

## Critère de succès = le BUT
Le temps-de-récupération après swap, mesuré en REPAS : bras appris ≫ bras contrôle. C'est la preuve
DIRECTE de « survit à un changement de monde » — celle que le monde figé ne pouvait pas donner
(Mur C). Si PASS : la re-mesure périodique (embryon jour/nuit) est promue ; le jour/nuit v1 (têtes
de décision) hérite d'un squelette validé. Si échec : négatif commité, la perception statique reste
le vivant (parité déjà acquise), l'embryon retourne au tiroir avec la cause diagnostiquée.
