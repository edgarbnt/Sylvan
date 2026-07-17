# Design — Reconnaissance des TYPES d'objets apprise de la conséquence (pré-enregistré 2026-07-17)

## Mission
Dissoudre la **dernière connaissance du monde codée-main** dans l'entité : les requêtes-couleur des
slots (`slot_head.color_queries` : food=rouge, water=bleu, danger=vert) ET le lien slot→drive.
Forme cible : « un type se reconnaît par l'apparence, et son SENS (nourrit / abreuve / blesse)
s'apprend de la conséquence vécue » — sans qu'aucune couleur ni aucun lien ne soit écrit. Sortie :
plus rien dans le cerveau de l'entité ne présume l'apparence du monde (critère owner « survit à un
changement de monde »). C'est le curseur de pureté-du-monde poussé à son maximum atteignable.

## À lire d'abord
- `docs/design_purete_hjepa.md` §P6 (les 3 murs mesurés du 1ᵉʳ essai) et §P5 (danger déjà dissous).
- `docs/design_monde_incremental.md` (le monde = ingrédients découplés ; verdict diag WM = CHEAP).
- `python/sylvan/models/slot_head.py:54` (`color_queries` + seuil 0.55 = la seule pièce à remplacer ;
  le readout est géométrique zéro-paramètre, on n'y touche pas).

## Limites MESURÉES qui fondent le chantier (ne pas re-deviner)
1. **Le WM gelé encaisse l'apparence** (diag_wm_appearance_robustness, 2026-07-17) : dérive du
   latent 0.1-0.2× la dérive naturelle tick-à-tick sous jitter modéré → **pas de recollecte WM.**
   L'apparence vit dans le SLOT (la pièce à remplacer), pas dans le substrat.
2. **Les 3 murs de P6** (la raison pour laquelle le 1ᵉʳ essai a échoué, à NE PAS répéter) :
   - **Mur A** — la bouffe « mesure vert » : 65 % des repas ont un rayon vert plus proche que la
     boule (bouffe au cœur du danger), 49/71 avec dégâts co-occurrents ;
   - **Mur B** — verrou structurel : les requêtes main sont des **séparateurs idéalisés** plus
     écartés que le monde rendu (cos(bleu-vrai, vert-vrai)=0.61 > seuil 0.55) → même une requête
     parfaite fuit ; le seuil GLOBAL 0.55 est une propriété d'appareil, pas d'apparence ;
   - **Mur C** — indécidabilité : en monde à apparence FIGÉE (1 couleur/type), un succès ne prouve
     que la parité avec la règle main, jamais l'adaptation → la capacité visée est intestable.
3. **Ce qui MARCHE déjà** (acquis P6, à réutiliser) : la MESURE retrouve les couleurs vraies (eau
   au millième, danger exact — 4ᵉ confirmation « une géométrie se mesure, ne se fitte pas »). Le
   fit par gradient, lui, ne l'identifie pas (jauge). Donc : **mesurer, jamais fitter la requête.**

## Essayé → résultat (P6, pour ne pas répéter)
- Fit gradient MIL de la requête → direction non-identifiée (jauge sur rayons monochromes). RÉFUTÉ.
- Fit cône-positif → init-dominée, AUC bonnes mais requête mensongère. RÉFUTÉ.
- Mesure médiane du rayon le plus proche → couleurs vraies retrouvées, MAIS Mur A (contamination
  vert) + Mur B (verrou séparateur). Négatif DÉFINITIF de la forme « mesure + seuil global ».
Leçon : le 1ᵉʳ essai n'avait ni le monde décidable (Mur C), ni le blocage (Mur A), ni la marge
par-type (Mur B). Ce chantier ajoute exactement ces trois pièces.

## Conception — chaque résultat de recherche répare un mur nommé
La recherche (2026-07-17) donne le CADRE et deux mécanismes, chacun ciblant un mur mesuré :

| Mur P6 | Principe de recherche | Pièce de conception |
|---|---|---|
| Cadre général | **Montesano, affordances** (IEEE T-RO 2008) : lier des TRAITS visuels à des EFFETS, sans concept d'objet pré-défini | la thèse du chantier : type = groupe d'apparence, sens = effet-drive appris |
| **Mur B** (séparateurs idéalisés) | **Découverte d'objets non-sup.** (survey 2024) : regrouper les apparences en prototypes D'ABORD, rattacher le sens ENSUITE | Étape A : la marge de chaque groupe se MESURE de l'écart réel entre groupes (plus de seuil global 0.55 imposé) |
| **Mur A** (bouffe mesure vert) | **Rescorla-Wagner, blocage** : ce qui compte est l'INFORMATION qu'un indice apporte, pas la co-occurrence temporelle | Étape B : le vert, déjà expliqué par les dégâts, est BLOQUÉ comme prédicteur du repas |
| **Mur C** (indécidabilité) | (pas de la recherche : le monde-jouet) | Ingrédient « apparences variées » = ce qui rend la capacité testable |

### Ingrédient de monde — le « bump apparences » (Godot, opt-in défaut OFF, bit-identique)
Les 3 types (eau/bouffe/danger) rendus avec **variété intra-type** : couleur tirée d'une
DISTRIBUTION par type (teinte/saturation) + texture (variation par-rayon) + éventuellement forme.
Aucun nouveau drive, aucune topologie, aucun épuisement (§ un ingrédient à la fois). Règle
d'honnêteté (§2) : les distributions d'apparence sont une **propriété déclarée du monde**, JAMAIS
ajustées pour rendre la reconnaissance facile — elles doivent chevaucher assez pour être réalistes
(des vrais objets ne sont pas infiniment séparés) tout en restant séparables EN PRINCIPE (sinon la
tâche est impossible, ce qui est un problème de VIABILITÉ du monde, mesuré avant de juger l'agent —
leçon du plafond métabolique appliquée à l'apparence).

### Étape A — REGROUPER les apparences (non-supervisé, mesure)
Sur les rétines vécues, regrouper les rayons colorés par ressemblance d'apparence → prototypes.
Chaque groupe = (centre = apparence médiane, MARGE = moitié de l'écart au groupe voisin). Le
centre remplace la requête `[K,3]` ; la marge remplace le seuil global 0.55 → **Mur B levé** (la
séparation ÉMERGE des données au lieu d'être imposée). Le nombre de groupes K est DÉCOUVERT (pas
injecté — « 3 types » serait de la connaissance-monde) ; un groupe qui ne se lie à aucun drive
= objet NEUTRE (comportement correct pour un monde réaliste où la plupart des choses sont neutres).
Continuité P6 : c'est la mesure P6-bis (qui retrouvait les couleurs vraies) généralisée d'une
médiane à une DISTRIBUTION. Mesure, pas gradient.

### Étape B — LIER chaque groupe à un drive par la conséquence, AVEC BLOCAGE
Pour chaque groupe : P(soulagement du drive | ce groupe était le plus proche avant l'événement),
MAIS en écartant l'indice déjà expliqué par une autre conséquence (Rescorla-Wagner). Le groupe
vert co-occurre avec les repas (bouffe engouffrée) mais **prédit déjà les dégâts** → il n'apporte
aucune information nouvelle sur « pourquoi j'ai mangé » → BLOQUÉ, non lié à l'énergie → **Mur A
levé**. Signal de blocage = les événements-dégâts VÉCUS (chute de santé), déjà dans le corpus.
Dépendance élégante et DÉJÀ satisfaite : le danger est reconnu (saillance P5) et son OUTCOME
(dégâts) est ce qui bloque la fausse liaison food↔vert. P5 débloque P6-rouvert.
Sortie : le lien slot→drive, appris (tombe avec les requêtes, comme prévu).

## Prochain pas — cheaper-first (gate AVANT de toucher Godot)
**G-pré (GRATUIT, synthétique — gate le travail Godot)** : rejouer les 2 étapes sur le corpus
EXISTANT à apparences SYNTHÉTIQUEMENT variées (même perturbateur que diag_wm : jitter+texture+
désat). Doit : (a) regrouper en 3 prototypes proches des couleurs vraies ; (b) lier food→énergie
et water→soif ; (c) le blocage met la liaison vert→énergie à ≈0 malgré la co-occurrence. Échec →
la machinerie de reconnaissance est fautive, on la corrige AVANT de rendre quoi que ce soit dans
Godot (zéro coût). Réussite → le bump Godot est licencié.

## ⭐⭐ VERDICT G-pré (2026-07-17, `diag_pretypes_recognition.py`, 0 train) : **MACHINERIE VALIDÉE**
Corpus réel perturbé (teinte 20°+texture 0.05+désat 0.4, combiné DUR), 150 361 ticks-objet.
- **Étape A (regrouper)** : K=3 découvert NET (silhouette 0.903 à K=3, pic franc vs 0.71/0.67/0.43).
  Décomposition du cos aux prototypes : **cos au centre de la classe PERTURBÉE = 0.9999-1.0000 pour
  les 3 types → le clustering est PARFAIT.** Le cos au vrai-PROPRE (rouge 0.967 / bleu 0.973 /
  vert 0.945) est plus bas UNIQUEMENT parce que comparer un centre désaturé à un oracle propre
  sous-compte le décalage de la perturbation — pas une erreur de clustering. Le strict G-pré-A
  (0.95-au-propre) reste ❌ AU DOSSIER (vert 0.945), mais la décomposition PROUVE que c'est un
  artefact de comparand ; le vrai gate (0.98) sera sur données RENDUES (comparand apparié).
- **Étape B (lier)** : contingence à PORTÉE-CONTACT (refinement, owner-approuvé) — **rouge→énergie
  (0.015) ✓, bleu→soif (0.014) ✓** (G-pré-B ✅).
- **⭐ Étape C (le cœur — ce que P6 ne savait pas faire)** : **vert→DÉGÂTS (0.347), PAS énergie
  (0.002)** ; séparation nette (P(en|rouge) 0.015 = 7.5× P(en|vert)). Contraste décisif : la vue
  NAÏVE (l'approche P6) donne au repas énergie **vert 0.73** / rouge 0.26 → le vert dominait ; la
  contingence forward le RENVERSE. **Le blocage Rescorla-Wagner résout le Mur A** (G-pré-C ✅).
**Conclusion** : les 3 étapes marchent en substance — 3 types recouvrés à l'identique (cos 0.9999
au centre perturbé), liaison correcte, blocage opérant (le confond fatal de P6 est dissous). Le
seul ❌ strict (A à 0.945-au-propre) est un artefact de comparand diagnostiqué, déféré au vrai gate
rendu (0.98). **Bump Godot LICENCIÉ** (décision owner). Refinement banké : contingence
portée-contact + décomposition cos-perturbé = la bonne métrique du pré-gate synthétique.

## Gates PRÉ-ENREGISTRÉS (falsifiables, ordre cheaper-first ; budget 1 train + 1 re-train diagnostiqué)
0. **G-pré** (ci-dessus, gratuit) : 3 prototypes + liaisons correctes + blocage à ≈0 sur corpus
   synthétique. Échec → corriger la reco, pas Godot.
1. **G-sep (viabilité du monde, sur corpus rendu)** : écart d'apparence entre types > dispersion
   intra-type (oracle couleur-rendue). Échec → ajuster les DISTRIBUTIONS du monde, JAMAIS les gates
   (leçon métabolique : ne pas juger une tâche impossible).
2. **G-cluster** : K découvert = nb de types vrais (oracle éval) ; cos(prototype, moyenne rendue
   vraie) ≥ 0.98 par type.
3. **G-bind** : chaque groupe lié au bon drive (oracle éval) ; poids de liaison vert→énergie ≈ 0
   (blocage vérifié) ; food→énergie et water→soif dominants.
4. **G-slot** : erreur de position du slot (requêtes apprises) vs positions oracles ≤ 0.5 m sur
   les ressources visibles, en monde à apparences variées.
5. **Juge closed-loop** (payé si 0-4), 2×24 vies seeds 1+2, monde v2 + bump apparences :
   - **PASS-parité** : reco apprise en monde VARIÉ ≥ config vivante actuelle en monde FIGÉ − bruit
     (l'apparence variée ne coûte rien) ;
   - **PASS-valeur** : reco apprise ≫ requêtes-main en monde VARIÉ (la règle main casse, l'apprise
     tient — la démonstration que ça sert) ;
   - KILL précoce seed 1 < 14 repas.
6. **⭐ GATE-CAPACITÉ (le BUT, impossible en v2)** : **swap d'apparence en cours de vie** → l'entité
   re-regroupe et continue de manger, là où les requêtes-main s'effondreraient. C'est la seule
   preuve directe de « survit à un changement de monde » ; tout le chantier existe pour ce gate.

## Ce qu'on ne touche JAMAIS
WM (gelé, diag-prouvé invariant à l'apparence), readout géométrique du slot, transport (géométrie
de l'espace), W / marges / hystérésis (préférences du CORPS, réglées P2-bis), drives eux-mêmes
(définis à la conception). On ne change QUE la source de la requête (mesurée) et du lien (appris).
Godot : `main.gd` jamais stagé ; bump opt-in défaut OFF = bit-identique.

## Critère de succès = le BUT
Le **Gate-CAPACITÉ** (swap d'apparence, mesuré en vies : mange-t-elle encore ?), poolé, contre la
réf vivante. Zéro connaissance du monde codée-main dans l'entité = requêtes ET lien slot→drive
appris/mesurés du vécu. Discipline inchangée (CLAUDE.md) : pré-enregistrement (ce doc), diag
gratuit avant tout train (G-pré), budget 1+1, juge poolé, négatif commité, carte à jour dans le
même commit que le build.

## Sources (recherche 2026-07-17)
- Montesano, Lopes, Santos-Victor, *Learning Object Affordances* (IEEE T-RO 2008) : lier
  actions/effets à des TRAITS visuels, pas à des objets pré-définis — le cadre de la mission.
- Rescorla-Wagner / blocage (cue competition, Frontiers in Psychology 2014) : « l'INFORMATION que
  l'indice apporte, pas la contiguïté » → Étape B, Mur A.
- *Unsupervised Object Discovery: A Comprehensive Survey* (arXiv 2024) : regrouper (prototypes)
  puis rattacher le sens — Étape A, Mur B.
