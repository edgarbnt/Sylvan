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
