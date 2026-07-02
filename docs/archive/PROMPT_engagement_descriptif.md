# Problème ouvert — engagement d'une cible non-frontale (description neutre)

## Contexte (architecture)

Projet d'ALife dans un world-model. Une entité hexapode se déplace dans un monde (Godot, physique). Pile à 3 couches :

- **Locomotion** : un CPG codé à la main + un résidu PPO borné. Exécute une commande continue `(vx, ω)` (vitesse avant, vitesse de rotation).
- **World-model (WM)** : réseau récurrent entraîné en auto-supervision (style JEPA). Il encode l'observation (proprioception + une **rétine** égocentrique de perception apprise + énergie) en un **latent**, et prédit la dynamique : à partir d'un latent et d'une commande, il prédit le latent suivant. Il peut donc « rêver » une trajectoire en boucle ouverte (open-loop) : à partir d'une observation réelle et d'une séquence de commandes, il génère une suite de latents imaginés. Il prédit aussi un déplacement corps-relatif par pas `(d_avant, d_latéral, d_yaw)`.
- **Planification** : un planner MPC (recherche, pas un réseau). À chaque replanification (toutes les ~10 étapes), il « rêve » ~117 séquences de commandes candidates (grille fixe d'arcs sur un horizon de ~300 pas), score chaque candidat par un **coût**, exécute la première commande du meilleur candidat, puis replanifie depuis une nouvelle observation réelle.

Deux modes de coût existent :
- **Mode coordonnées** : le coût lit une position de la cible fournie par un capteur privilégié (radar oracle), et récompense « se rapprocher » + « pointer vers la cible ». Ce mode fonctionne pour des cibles dans toutes les directions.
- **Mode latent-pur (dit « 🅑 »)** : aucune coordonnée. Le coût ne lit que ce que des **têtes apprises** extraient du latent rêvé. Une **value head** lit « va atteindre une cible bientôt » (proximité). La cible (nourriture) n'existe que dans la perception apprise (rétine→latent). C'est ce mode qui est étudié ici.

## Le problème

En mode latent-pur, l'entité **engage et atteint** une cible située **devant ou sur le côté** (mesuré : quand elle arrive à moins de 1,2 m d'une cible, elle conclut l'approche dans 100 % des cas). Mais elle **n'engage pas** une cible située **derrière** elle : elle ne déclenche pas le demi-tour, erre, et ne s'approche jamais. Sur une évaluation de 6 épisodes, les épisodes où la cible démarre derrière donnent 0 approche ; ceux où elle démarre devant/côté donnent des approches et des repas.

## Faits établis par diagnostics (mesures)

1. **Perception** : la cible est « visible » (capteur de référence) à 360°, y compris derrière (100 % des frames). La rétine est active à tous les azimuts. Le bearing (angle) égocentrique de la cible est décodable depuis le latent gelé : ~84 % de précision devant/derrière, erreur angulaire médiane ~25°.

2. **Le coût latent n'a pas de gradient d'engagement pour une cible derrière.** Mesuré sur des frames où la cible est derrière (azimut > 90°, distance 1,5–3 m) : parmi les 117 candidats rêvés, le **meilleur possible** ne rapproche l'entité qu'à ~2 m de la cible (aucun candidat ne l'atteint dans un seul rêve de 300 pas) ; et le candidat **choisi** par le coût augmente l'azimut (95° → ~130°, c.-à-d. tourne du mauvais côté) au lieu de le réduire.

3. **Origine localisée : le rêve open-loop ne représente pas l'acquisition-par-virage.** Le long du candidat qui, géométriquement, fait pivoter l'entité pour faire face à la cible (bearing 95° → ~19°, donc cible amenée devant), deux lectures indépendantes du latent rêvé restent « cible absente / derrière » : la value de proximité reste à 0,00 et la lecture d'orientation `ahead` reste négative (≈ −0,5), alors que la géométrie indique la cible passée devant. Ce comportement tient à tous les horizons, y compris courts (10–40 pas). Autrement dit : quand le WM imagine un virage, la perception de la cible **disparaît** du latent rêvé.

4. **La couverture des données n'est pas le facteur évident.** Les données existantes (collectées par le mode coordonnées) contiennent déjà des virages et des transitions « cible derrière → cible devant » (|ω| moyen ~0,36 ; ~5 croisements derrière↔devant par épisode). Une nouvelle collecte par « babbling » (commandes aléatoires) produit **moins** de virages utiles (|ω| ~0,13 ; l'entité avance et dépasse les cibles sans assez tourner).

5. **Teacher-forced vs open-loop, le long d'un vrai virage** : mesure non concluante. La corrélation entre la lecture d'orientation du latent et le vrai cos(bearing) est faible dans les deux cas (~+0,18 teacher-forced, ~+0,28 open-loop). L'isolement précis (représentation vs fidélité de déroulé) n'a pas donné de signal net.

## Solutions essayées (et mesures)

1. **Value head entraînée sur les latents rêvés multi-pas** (au lieu de teacher-forced 1-pas). Réalisée pour un problème connexe (la conclusion de l'approche, « close », pour des cibles devant). Résultat mesuré : a corrigé ce problème-là (0 → 3 repas / 4 épisodes ; rang du candidat au close 0,65 → 0,08). **N'a pas** d'effet sur l'engagement d'une cible derrière (le problème décrit ici).

2. **Tête d'orientation apprise (`OrientHead`)** : un readout du latent prédisant le bearing égocentrique de la cible (cos, sin), entraîné sur latents rêvés (devant/derrière ~80 %, erreur ~19°). Intégrée au coût du planner comme terme « récompenser le rêve qui finit orienté vers la cible », pondéré et atténué quand la value de proximité est haute. Résultat mesuré : avec ce terme actif (poids 3 ou 6), l'entité **n'engage toujours pas** le demi-tour pour une cible derrière (azimut final ~126°, réduit dans 1 cas sur 15). Diagnostic associé (fait 3) : la tête lit un latent rêvé qui ne contient pas l'acquisition, donc elle n'a rien à exploiter.

3. **Collecte de données de virages (« babbling » tournant)** : un collecteur produisant des commandes par morceaux avec rotation, avec des cibles à 360°, rétine et position de cible loggées. Test de couverture avant tout ré-entraînement : la donnée produite contient **moins** de virages/transitions utiles que la donnée existante (fait 4). Le ré-entraînement du WM sur cette base n'a pas été effectué.

## État

- Le mode latent-pur perçoit, planifie et atteint des cibles **devant/côté** (fonctionnel).
- L'engagement d'une cible **derrière** ne fonctionne pas, et le diagnostic le rattache à l'incapacité du **rêve open-loop du WM à représenter l'acquisition d'une cible pendant un virage imaginé**.
- Le mode coordonnées (capteur privilégié, non latent-pur) engage les cibles dans toutes les directions et reste disponible comme référence.
