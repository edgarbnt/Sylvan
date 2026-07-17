# Design — Monde INCRÉMENTAL et réouverture de la perception des types (2026-07-17)

## Mission
Objectif COURANT tranché owner : **rien dans le code de l'ENTITÉ ne doit encoder une connaissance
du MONDE écrite à la main.** Critère officiel : « est-ce que ça survit à un changement de monde ? »
(le simulateur Godot a le droit d'ÊTRE le monde ; c'est le cerveau de l'entité qui ne doit rien
présumer de son apparence). Ce doc corrige l'ordre des chantiers vis-à-vis de CET objectif et pose
le plan de réouverture de la reconnaissance des types d'objets.

## À lire d'abord
- `docs/design_purete_hjepa.md` §P5 (danger dissous, PASS 41/9) et §P6 (types d'objets, négatif
  instructif : 3 causes cartographiées).
- `python/sylvan/models/slot_head.py:44` (`color_queries` — le dernier monde codé-main).
- `docs/roadmap_vers_monde_v3.md` (à relire à la lumière de ce doc : l'ordre y est révisé).

## Le fait : ce qui reste codé-main dans l'entité (inventaire honnête)
Deux choses, et elles TOMBENT ENSEMBLE :
1. **Requêtes-couleur des slots** (`slot_head.color_queries` : food=rouge, water=bleu, danger=vert)
   — l'apparence de chaque type, écrite à la main.
2. **Le lien slot→drive** (`food_idx→énergie`, `water_idx→soif`, `hazard_idx→santé`) — quelle
   apparence soulage quel drive.
Apprendre « cet aspect a soulagé ce drive » donne les DEUX d'un coup (la requête ET le lien).
Déjà dissous : la perception DANGER de l'étage décisionnel (saillance apprise du vécu, P5) ; les
marges/W sont du CORPS (réglé P2-bis, pas du monde). Donc les 2 restes ci-dessus = la dernière
connaissance-du-monde codée-main dans la boucle.

## Essayé → résultat (la critique de l'ordre roadmap, correction assumée)
**« Jour/nuit v1 avant le bump apparences » était une recommandation de loyauté à la roadmap, pas
à l'objectif — RÉFUTÉE ici.**
- Jour/nuit = re-consolider la nuit des têtes DÉJÀ apprises (sprint, douleur̂, P̂mort, saillance).
  Il ne touche AUCUN des deux restes codés-main. Il sert un AUTRE but (auto-amélioration en vivant,
  le cycle de vie), orthogonal à la pureté. Pour l'objectif courant, son utilité est ≈ nulle.
- La reconnaissance des types (via l'ingrédient « apparences variées ») est le chemin DIRECT :
  c'est elle, et elle seule, qui retire le dernier monde codé-main.
→ Pour CET objectif : **reconnaissance des types AVANT jour/nuit.** Jour/nuit reste valable, mais
sur sa propre piste, quand on voudra le cycle de vie — et il sera même MEILLEUR après (toute la
perception sera consolidable, plus aucune lunette figée à ne pas toucher).

## Il n'y a pas de « monde v3 » : une PILE d'ingrédients découplés
La roadmap se contredit (§6 dit « un ingrédient à la fois », mais empaquette tout en un « v3 »
final). Chaque chantier réclame un ingrédient DIFFÉRENT (ou aucun) :

| Chantier | Ingrédient de monde requis |
|---|---|
| Jour/nuit v1 (têtes) | AUCUN (tourne dans le v2 actuel) |
| **Reconnaissance des types** | **apparences variées** (mêmes 3 types) |
| Chercher + mémoire | grande arène + ressources hors-vue |
| Jour/nuit v2, menace | mondes plus grands / menace mobile |

**« v3 simplifié » = UN ingrédient** : mêmes 3 types (eau/bouffe/danger), mais des objets VARIÉS
au lieu de sphères/zones de couleur unie. Pas de nouveau drive, pas de topologie, pas d'épuisement.
Le reste de la pile (corps, planner, drives, arbitrage) ne voit aucune différence. **« v4 » = tous
les autres ingrédients, plus tard, chacun sur son chantier.** C'est ce que la règle §6 demande déjà.

## L'ingrédient « apparences » est le SEUL qui touche le WM gelé (caveat dur, à mesurer AVANT)
Les autres ingrédients (topologie, épuisement, arène) sont de la LOGIQUE de monde : ils ne changent
pas ce que voit la rétine. Les apparences variées, SI : un objet texturé renvoie des couleurs qui
varient d'un rayon à l'autre et d'une instance à l'autre, alors que le WM a été appris sur des
sphères de couleur plate. Rien ne garantit d'avance que son latent gelé encaisse.
→ **Gate gratuit AVANT tout le reste** (discipline G-place appliquée à l'ingrédient lui-même) :
WM gelé en open-loop sur rétines à apparence variée → le latent / le slot tiennent-ils ? Deux
issues informatives : le WM généralise (la rétine est du RGB brut par conception → cheap, on
avance) ; ou il décroche (on l'apprend maintenant, sur un ingrédient isolé et pas cher, au lieu du
« v3 complet » où ce serait mêlé à cinq autres changements et impossible à diagnostiquer).
C'est le bon test de généralité du WM — à faire tôt, seul.

### ⭐ VERDICT DIAG WM (2026-07-17, `diagnostics/diag_wm_appearance_robustness.py`, 0 run) : **CHEAP**
Perturbations synthétiques des rétines réelles (jitter de teinte, bruit de texture par-rayon,
désaturation), 3000 ticks, WM gelé `wm_objcentric_kin_haz`. Étalon = dérive latente NATURELLE
tick-à-tick (0.044). Au niveau modéré réaliste (teinte 20° / σ 0.05 / désat 0.4), la dérive du
LATENT = **0.1-0.2× la dérive naturelle** (0.34× même au plus fort) → le latent bouge MOINS qu'un
pas de vie normal. **Le substrat gelé est ~invariant à l'apparence : pas de recollecte WM requise.**
Lecture : l'apparence vit dans le SLOT (readout), pas dans le latent de dynamique — cohérent avec
l'archi. L'objection « le bump est le mouvement le plus cher » TOMBE : le bump = retrain léger de
la requête seule, WM intouché.
⚠️ Limite honnête (mesurée) : le slot montre 0.000 m de dérive et 100 % de visibilité gardée à
TOUTES les magnitudes — non pas parce que la requête est robuste, mais parce que les perturbations
sont restées AU-DESSUS du seuil cosinus 0.55 (couleurs saturées du monde-jouet). Donc le diag
dé-risque le SUBSTRAT (établi), pas la requête face à un rendu réel désaturé/texturé (sous-testé
ici) — mais la requête est la pièce qu'on remplace de toute façon ; son comportement réel se
mesure au check open-loop du vrai bump.

## Prochain pas — plan de réouverture (PRÉ-ENREGISTRÉ dans `docs/design_perception_types.md`, rien lancé)
> Le chantier de réouverture a maintenant son doc dédié avec gates falsifiables :
> `docs/design_perception_types.md` (décision owner 2026-07-17 : pousser le curseur de
> pureté-du-monde au max AVANT tout gros build). Le résumé ci-dessous en est l'esquisse.
1. **Bump apparences** (opt-in Godot, défaut OFF = bit-identique) : les 3 types rendus avec de la
   variété intra-type (teinte/texture/forme), même `retina_color` moyen.
2. **Diag WM gratuit** (ci-dessus) : décide cheap-vs-recollecte.
3. **Reconnaissance en 2 étapes** (fondé sur la recherche, cf sources) :
   - **regrouper** les apparences vécues (non-supervisé) → « combien de sortes de choses ? » ; la
     marge de chaque groupe se MESURE sur l'écart réel entre groupes (résout le VERROU §P6 : les
     requêtes main étaient des séparateurs idéalisés plus écartés que le monde ; ici la séparation
     émerge des données au lieu d'être imposée) ;
   - **lier** chaque groupe à un drive par la conséquence AVEC BLOCAGE (Rescorla-Wagner : écarter
     l'indice déjà expliqué par une autre conséquence → le vert, qui prédit déjà les dégâts,
     n'explique pas le repas ; résout la CONTAMINATION §P6 « la bouffe a mesuré vert » sans
     bricolage ad hoc — c'est un principe d'apprentissage causal de base).
4. **Gate-CAPACITÉ (impossible en v2, c'est tout l'intérêt du bump)** : swap d'apparence en cours
   de vie → l'entité re-regroupe et continue de manger. C'est la seule preuve de « survit à un
   changement de monde » ; en v2 (une apparence figée par type) un succès ne prouvait que la
   parité avec la règle main, jamais la capacité.

## Critère de succès = le BUT
Le Gate-CAPACITÉ ci-dessus (swap d'apparence, mesuré en vies : mange-t-elle encore ?), poolé,
contre la réf vivante. Zéro connaissance du monde codée-main dans l'entité = requêtes ET lien
slot→drive appris du vécu. Discipline inchangée (CLAUDE.md) : pré-enregistrement, diag gratuit
avant tout train, budget 1+1, juge poolé, négatif commité, carte à jour dans le même commit.

## Sources (recherche 2026-07-17)
- Montesano, Lopes, Santos-Victor, *Learning Object Affordances* (IEEE T-RO 2008) : lier
  actions/effets à des TRAITS visuels, pas à des objets prédéfinis — le cadre exact de cet objectif.
- Rescorla-Wagner / blocage (cue competition) : « ce qui compte est l'INFORMATION que l'indice
  apporte, pas la proximité temporelle » → un indice déjà expliqué n'est pas ré-appris.
- *Unsupervised Object Discovery: A Comprehensive Survey* (arXiv 2024) : regrouper d'abord les
  apparences (prototypes), rattacher le sens ensuite — les 2 étapes séparées de §Prochain-pas.
