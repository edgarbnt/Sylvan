# Design — Attribution de crédit : la conséquence au BON indice (cas baie-buisson), pré-inscrit 2026-07-17

## Mission
Prouver, en vies, que l'entité attribue une conséquence à la **bonne apparence** quand plusieurs
co-occurrent au moment de l'interaction, ET qu'elle reconnaît un objet **neutre** (sans conséquence).
Cas minimal : la **baie** (nourrit) est TOUJOURS dans un **buisson** (neutre) pendant l'apprentissage
→ l'entité doit lier `baie→énergie` et `buisson→neutre`, puis **reconnaître une baie SEULE** (par
terre, sans buisson) et **ignorer un buisson seul**. C'est la « purification » poussée d'un cran :
non pas « associer une apparence à une conséquence » (déjà acquis, WM typé — K couleurs → K
conséquences découvertes), mais le faire **précisément** en présence de distracteurs et de neutres —
les deux ingrédients qui manquent pour un monde réaliste et encombré.

## À lire d'abord
- `docs/design_perception_types.md` (le WM typé = la base ; `scripts/build_typed_slots.py` Étape B =
  la contingence-contact à ÉTENDRE, WM gelé, invariant à l'apparence → zéro retrain).
- `docs/design_gate_capacite.md` (pourquoi le swap d'apparence a échoué → pivot vers CE cas ; la
  discipline pré-inscription).
- Mémoire 2026-07-17 (pivot owner « apparence→conséquence découverte », critère hard-code).

## Pourquoi ce cas (le plus petit qui exerce les 2 ingrédients manquants)
Le WM typé lie déjà 3 couleurs à 3 conséquences, découvertes, **sans code par couleur**. Ce qui
manque pour un monde ouvert :
1. **Attribution au bon indice** quand baie ET buisson sont présents au repas (crédit partagé).
2. **Classe NEUTRE** : un objet peut ne rien faire (le buisson) — le cas majoritaire dans un vrai
   monde. L'argmax actuel force TOUJOURS une conséquence → un neutre co-occurrent hérite d'un faux lien.
Le buisson est le plus petit monde qui teste les deux. Continuité directe : WM GELÉ, pas de retrain,
pas de physique du corps, on ne touche que l'étage de LIAISON (offline, jour/nuit).

## Le mécanisme (contingence ΔP + neutre + multi-rayon au contact)
Aujourd'hui : rayon le plus proche + **argmax de conséquence par groupe**. Deux limites au buisson :
- La rétine voit UNE couleur par direction → baie et buisson sont sur des **rayons voisins distincts**
  (pas le même). Il faut considérer **toutes les couleurs présentes dans la portée-contact**, pas
  seulement la plus proche.
- L'argmax choisit toujours une conséquence. Le buisson, co-occurrent avec les repas, hérite d'un
  `P(énergie|buisson)` gonflé par la co-occurrence.

Passage à une **contingence ΔP** (Rescorla-Wagner / compétition d'indices) : l'indice `c` mérite le
lien vers la conséquence `y` si sa présence **ÉLÈVE** `P(y)` **au-dessus du niveau de base**,
_sachant les autres indices présents_. Le buisson, toujours accompagné de la baie, n'élève rien seul
→ `ΔP(énergie|buisson) ≈ 0` → **neutre**. La baie élève franchement → `baie→énergie`.

**Classe neutre** : si aucune conséquence n'a un ΔP au-dessus d'un **plancher de significativité
MESURÉ** (issu du taux de base + son bruit d'échantillonnage), le groupe se lie à RIEN. Le plancher
est **mesuré, pas réglé** (§2 : pas un bouton pour faire passer le gate).

**Sortie** : `baie→requête-couleur du slot food` ; `buisson→aucune requête` (neutre). WM gelé ; seul
l'étage de liaison change (extension de `build_typed_slots`, tout hors-ligne).

## Contrainte de monde DÉCLARÉE (identifiabilité — mesurée avant de juger, §2)
Pour être apprenable ET testable, il faut une **décorrélation** :
- des **buissons parfois sans baie** (→ `P(repas|buisson)` diluée pendant l'apprentissage),
- des **baies parfois seules** (→ c'est le TEST de transfert).
Si baie et buisson sont TOUJOURS ensemble → **indécidable** (aucun mécanisme ne tranche). Les
distributions (fraction de buissons-sans-baie, fraction de baies-seules) sont des **propriétés du
monde déclarées, jamais ajustées** pour faciliter. Viabilité aussi conditionnée à : la rétine
**sépare** baie et buisson (teintes distinctes ; écart inter > dispersion intra — mesuré en G1).

## Gates PRÉ-ENREGISTRÉS (falsifiables, ordre pas-cher-d'abord)
**G0 — GRATUIT, synthétique (0 run, 0 Godot). GATE LE TRAVAIL GODOT.**
Sur le corpus EXISTANT, injecter une couleur-buisson synthétique : rayons buisson **au contact des
repas** (co-occurrence baie+buisson) ET rayons buisson **hors repas** (buissons seuls). Rejouer la
contingence ΔP. Doit :
- (a) lier **baie→énergie** (la vraie couleur food) ;
- (b) **buisson→NEUTRE** : `ΔP(y|buisson) < plancher mesuré` pour TOUTE conséquence `y`, MALGRÉ la
  co-occurrence 100 % au repas ;
- (c) **transfert** : une trame « baie seule » synthétique → slot food visible/positionné correct ;
- (d) **contraste décisif** : l'ancienne vue NAÏVE (argmax de co-occurrence) lie **buisson→énergie**
  (le piège), la contingence ΔP le RENVERSE (comme le vert 0.73→bloqué de P6).
Échec → le mécanisme est fautif, on le corrige AVANT de rendre quoi que ce soit dans Godot.

**G1 — Godot léger (viabilité du monde).** Ajouter un objet **buisson** perceptible NEUTRE (rendu
rétine, aucun drive, aucune physique) + co-location contrôlée avec la baie + baies-parfois-seules.
Collecter un petit corpus. MESURER : (a) la co-occurrence rendue au repas est bien celle voulue ;
(b) la décorrélation EXISTE (buissons sans baie, baies seules) ; (c) la rétine **sépare** baie/buisson
(écart teinte inter > dispersion intra). Si non-séparable ou non-décorrélé → ajuster le **MONDE**
(déclaré), JAMAIS le gate.

**G2 — mesure offline sur corpus RÉEL.** Rejouer la contingence ΔP sur le corpus buisson réel →
`baie→énergie`, `buisson→neutre`, et `eau→soif`/`danger→dégâts` **toujours corrects** (pas de
régression) ; sur les trames **baie-seule**, erreur de position/visibilité du slot food ≤ 0.5 m méd.
Émettre le WM typé avec la **requête-baie apprise** (buisson = pas de requête). WM gelé.

**G3 — juge closed-loop (payé si G0-2 passent).** 2×24 vies seeds 1+2, monde avec buissons. 
- **PASS-parité** : forage des baies (y compris **baies seules**) ≥ config vivante − bruit, ET le
  slot food **ne se verrouille jamais sur la couleur-buisson** (pas de poursuite de buissons).
- **KILL précoce** : seed 1 sous le seuil de repas, OU le buisson détourne le slot food.

## ⭐ VERDICT G0 (2026-07-17, `diag_credit_g0.py`, 0 run / 0 Godot / 0 train) : PASSÉ — mécanisme validé
Corpus PROPRE poolé (typcorp + 6 runs, 267 137 ticks-contact) + buisson synthétique injecté (baie
DANS un buisson **92 %**, buissons vides **1 %** = décorrélation ; **83 % des repas en buisson**).
K=3 retrouvé (rouge/vert/bleu). **Plancher-bruit MESURÉ (pas réglé)** = p95 `|coeff|` d'un placebo
**MÊME-STRUCTURE** (même fréquence + même collinéarité 92 % avec la baie — un indice quasi-collinéaire
a une variance de coeff GONFLÉE ; un aléatoire indépendant la sous-estime, leçon de la 1ʳᵉ passe) =
**0.0015**. Contingence PARTIELLE (Rescorla-Wagner = régression de la conséquence sur le vecteur des
indices présents) :
- **(a) baie→énergie** coeff **+0.0117** (8× le plancher) ✅ ;
- **(b) buisson NEUTRE** coeff **−0.0006** (dans la bande-nulle) **MALGRÉ 83 % de repas en buisson** ✅
  — le distracteur neutre est BLOQUÉ ;
- **(c) CONTRASTE** : le naïf `P(énergie|indice)` NE SÉPARE PAS (buisson 0.0091 ≈ baie 0.0112 ; pire,
  son argmax-par-cluster lie le buisson→**DÉGÂTS** 0.33 par co-occurrence) ; la partielle RENVERSE
  (buisson −0.0006 ≪ baie +0.0117) ✅ ;
- **(d) non-régression** : eau→soif (+0.0136), danger→dégâts (+0.31), et **vert PAS→énergie**
  (+0.0001 : Mur A dissous) ✅.
Mécanisme (contingence partielle + plancher mesuré par placebos) VALIDÉ hors-ligne → **G1 (objet
buisson Godot) LICENCIÉ**. Résidu noté (hors G0) : le rouge garde un coeff dégâts +0.14 (baies au
cœur du danger = confond Mur-A résiduel sur les DÉGÂTS, traité ailleurs par la lunette saillance —
n'affecte pas la liaison énergie/neutre). Refinement banké : **le plancher-neutre se MESURE par
placebos même-structure**, pas par un indice aléatoire indépendant.

## ⭐ VERDICT G1 (2026-07-17, `food_manager.gd` buisson + `diag_credit_g1.py`, corpus rendu) : PASSÉ
Buisson NEUTRE rendu dans Godot (opt-in `SYLVAN_FOOD_BUSH*`, défaut OFF bit-identique, PERCEPTIBLE
layer 8, SANS drive/consommation/physique — jamais dans `_positions`), co-localisé à la baie
(offset ±0.30 m, petit r=0.20 → co-perçu sans occulter) + buissons DISPERSÉS + baies parfois seules.
**Propriétés du MONDE déclarées** : teinte **0.45** (teal, dans le trou vert-bleu ; choisie par
mesure — voir ci-dessous), co-loc `_bush_p=0.9`, `_bush_alone=1`. Corpus 16 vies (kin base WM,
food-only WC=0 COST=mindist pour une nav propre). Mesures : **(a) co-occurrence au repas 0.67**
(21 repas ≥20) ✅ ; **(b) décorrélation** baies-seules 199 / buissons-seuls 8553 ✅ ; **(c)
séparabilité** cos(baie,buisson) 0.54 ≪ intra 0.99/1.00 ✅. → **monde viable, G2 LICENCIÉ.**

**Découverte mesurée (importante) : PAS de « zone morte » de couleur** — sous les requêtes du WM
vivant (typé, marges 0.81/0.86/0.92), TOUTE teinte saturée fire au moins un slot (les 3 requêtes +
marges couvrent tout le cercle des teintes ; l'espace couleur est encombré). Donc pendant la
COLLECTE avec le WM pré-buisson, un buisson coloré détourne un slot → l'agent se gare. Contourné en
G1 par une collecte **food-only** (WM single-food, requête rouge ≠ teal). **La neutralité réelle du
buisson sous le WM complet viendra de G2** (re-mesure des requêtes+marges AVEC le cluster buisson →
la marge du voisin se resserre pour l'exclure — c'est tout le point du chantier). Note : co-occ
~0.6 (pas 0.9) = géométrie (petit buisson parfois occulté au repas) ET **favorable à
l'identifiabilité** (plus de décorrélation).

## ⭐ VERDICT G2 (2026-07-17, `build_typed_slots_credit.py`, 0 run / 0 train) : PASSÉ — WM crédit-typé émis
Poolé : G1 `critic_kin_g1` (baie + buisson réel) + typcorp + DEFAULT_RUNS (eau/danger réels), 280 406
ticks, reliefs E=148 T=199 dgr=16256. K=4 découvert (baie/danger/eau/buisson), marges mesurées AVEC
le buisson.

**RAFFINEMENT (négatif informatif banké)** : la contingence ΔP sur la **PRÉSENCE** multi-indice (celle
validée en G0 synthétique) **RÉINTRODUIT le Mur A sur données réelles** — le rouge, PRÉSENT quand on se
fait mordre au cœur du danger, hérite d'un coeff DÉGÂTS (+0.13) > son coeff énergie rare (+0.011) →
argmax = DÉGÂTS (faux). Fix = le mécanisme VIVANT : contingence au **PLUS PROCHE** au contact (on est
le plus proche de ce qu'on CONSOMME) + test de **SIGNIFICATIVITÉ** ΔP > 3·SE pour le neutre. Résultat :
- **baie→ÉNERGIE** (ΔP +0.0135 ; et ΔP DÉGÂTS = **−0.2155** : rouge-proche ⇒ on mange, PAS on se fait
  mordre → le Mur A est renversé, les dégâts vont au vert) ✅ ;
- **eau→SOIF** (ΔP +0.0140), **danger→DÉGÂTS** (ΔP +0.3048) ✅ (non-régression) ;
- **BUISSON→NEUTRE** (aucun lift significatif : énergie −0.0013, soif −0.0019, dégâts −0.2314 — malgré
  8996 ticks-plus-proche) ✅ ;
- **(c) transfert baie-seule : erreur 0.061 m méd** (n=39) ≪ 0.5 m ✅ — le slot food localise une baie
  NUE quasi parfaitement.

**Marges émises [food 0.761, water 0.976, danger 0.936]** : water/danger RESSERRÉES (le cluster teal
buisson est leur voisin) → elles **excluent le buisson** = le fix « espace couleur encombré » (le
buisson ne fire plus water/danger sous le WM complet — testé en G3). Émis :
`data/checkpoints/wm_objcentric_kin_typed_credit/wm_best.pt` (WM gelé ; seuls color_queries+meta ;
buisson = cluster NEUTRE, pas de slot). PROCHAIN = **G3** (juge closed-loop, monde complet + buisson).

## Ce qu'on ne touche JAMAIS
WM (gelé, invariant à l'apparence), readout géométrique du slot, transport, corps/physique (pas
d'obstacle ici), les drives eux-mêmes. On ne change QUE l'étage de LIAISON (argmax → ΔP + neutre) +
une lecture multi-rayon au contact. Godot : `main.gd` jamais stagé ; objet buisson opt-in défaut OFF
= bit-identique. Carte `architecture.json` mise à jour DANS LE COMMIT du build (pas de la pré-inscription).

## Critère de succès = le BUT
Le **transfert** : baie apprise TOUJOURS-en-buisson, reconnue **SEULE**, buisson ignoré — mesuré en
vies, poolé, contre la config vivante. Zéro code par objet : le mécanisme (ΔP) et les canaux (sens)
sont écrits/fournis UNE fois ; la couleur baie/buisson et leur sens sont **découverts**. Si PASS :
l'attribution de crédit précise + la classe neutre sont acquises → l'axe « couleurs illimitées →
conséquences, sans code par objet » est débloqué ; le cas obstacle (nouveau CANAL) devient la suite.
Si échec : négatif commité, cause diagnostiquée sur trace (G0 gratuit d'abord = négatif à coût nul).

## Sources
- Rescorla-Wagner / blocage & compétition d'indices : « l'INFORMATION que l'indice apporte, pas la
  contiguïté » (déjà mobilisé pour le Mur A du WM typé ; ici étendu au distracteur NEUTRE via ΔP).
- Montesano et al., *Learning Object Affordances* (IEEE T-RO 2008) : lier traits visuels ↔ effets,
  sans concept d'objet pré-défini — le cadre.
