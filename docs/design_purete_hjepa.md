# Design — Purification de l'étage haut (post-critique-sprint, 2026-07-16)

## Mission
Rendre l'étage waypoint (et son socle bas) pur au sens du recadrage LeCun 2026-07-14 :
**pureté ≠ zéro formule** — l'IC analytique immuable est licite ; ce qui est impur, c'est la
CONNAISSANCE-DU-MONDE codée main (létalité du vert, géométrie des piliers, hints de cap) là où
elle devrait être apprise du vécu.

## ⭐ CRITÈRE OFFICIEL (owner, 2026-07-17) : « EST-CE QUE ÇA SURVIT À UN CHANGEMENT DE MONDE ? »
Le but à terme est un monde ressemblant au VRAI monde — plus de boules de couleur ni de formes
fluo comme béquilles. Le monde VA changer → toute variable clé-APPARENCE ou clé-GÉOMÉTRIE-du-monde
est un **échafaudage du monde-jouet**, à dissoudre AVANT tout enrichissement du monde :
- `green_points` (règle « danger = vert ») et les REQUÊTES-COULEUR des slots WM (« bouffe = rouge,
  eau = bleu ») — les plus graves : la perception entière est clé-apparence ;
- `green_margin`/`tangent_margin` (géométrie piliers) ;
- ⚠️ les entrées dg1/dg2 de douleur̂/P̂mort passent par la lunette verte : LABELS purs (vécus),
  FEATURES contaminées — les têtes devront être re-apprises sur percept brut.
SURVIVENT au critère : W-aversion, κ/drain/restore, machine à états, odométrie (corps/préférences,
zéro monde dedans) ; le WM lui-même (rétine RGB brute, latent général).
Forme cible : **perception ancrée dans la CONSÉQUENCE vécue** — « dangereux » = ce qui a précédé
mes dégâts (saillance-douleur sur rétine brute ; le corpus percept→dégâts existe déjà) ;
« nourrissant » = ce dont la consommation a soulagé le drive (recette §3, dont les requêtes-couleur
sont le bootstrap monde-jouet, pas la forme finale). Chantier dédié à licencier AVANT le monde v3.

## À lire d'abord
- `docs/design_critique_sprint.md` (le juge PASS 45/8 = la référence vivante et la méthode).
- `python/sylvan/control/planning/command_planner.py:602` (surv_mode = les DEUX visibles →
  le chemin designed + heading_weight vit sur tous les replans mono-ressource).
- `python/sylvan/control/waypoint_layer.py:172` (route_cost : W=25, green_margin=1.0).

## Inventaire (trié par nature, hygiène 2026-07-16)
- ❌ monde-codé-main À APPRENDRE : W=25 + green_margin=1.0 (létalité du vert) ; tangent_margin=1.4
  (bord létal 0.39 m = géométrie pilier connue d'avance) ; hints de cap du bas
  (HEADING_W=2.0 actif en mono-visible ; FAR_ALIGN=1 + ALIGN_GAIN=60 actifs en surv_mode).
- ⚠️ prior structurel (trou LeCun §6/§8.1) : le proposeur (anneau + tangents) — chantier futur.
- ✅ constantes de CONCEPTION (à déclarer, pas à apprendre — catégorie drives) : reach=1.2,
  timeout=180, hysteresis=0.15, patience=2, recheck=1 ; odométrie k_fwd/k_yaw (corps calibré).

## P1 — Débrancher les hints de cap du bas (pré-enregistré AVANT le run)
**Hypothèse falsifiable** : l'étage waypoint gère la topologie lointaine → les hints
(HEADING_W, FAR_ALIGN/ALIGN_GAIN) sont devenus REDONDANTS sur la config vivante.
- **Bras OFF** : config du juge (monde v2, waypoint + sprint-critic) avec `HW=0 FAR_ALIGN=0`,
  2×24 vies seeds 1+2. Réfs = bras juge hints-ON : **45 repas / 8 morts-danger** poolés.
- **PASS (hints retirés)** : repas poolés ≥ **40** (45 − bruit ±5) ET morts-danger ≤ **10** (8+2)
  → les hints sortent des DÉFAUTS du harnais ; le bras OFF devient la nouvelle référence vivante.
- **KILL précoce** : seed 1 < 14 repas (même seuil que le juge). Échec → hints conservés,
  négatif commité (ils ne sont PAS redondants — le bas en a encore besoin).
- ⚠️ URGENCY_W laissé tel quel (isole l'effet cap ; son chemin propre sera audité à part).

## ⭐ VERDICT P1 (2026-07-16) : **KILL — les hints sont PORTEURS, pas redondants**
Bras OFF seed 1 : **3 repas / 24 vies** (réf hints-ON : 19), 20 morts de FAIM, KILL précoce
déclenché (<14) → seed 2 non payé, hints CONSERVÉS. Diagnostic : l'étage waypoint décide OÙ aller,
mais le bas a besoin du shaping d'alignement pour TOURNER vers sa cible — sans `heading_weight`,
le mur A→B de 2026-06-18 revient (gradient de virage ≈ 0 dans `-min_dist` : l'entité voit la
bouffe et meurt devant). La note « hw=0 ≥ hw=2 » (2026-06-25) ne valait que pour l'ancienne
config sans danger/waypoint. RECLASSEMENT : les hints passent d'« impureté à retirer » à
**échafaudage PORTEUR daté** — leur remplacement propre exigera soit un bas qui apprend à tourner
(hors scope), soit un étage haut qui émet un cap (candidat lointain). Décomposition HW-seul vs
FAR_ALIGN-seul = sonde optionnelle future (licence owner), pas payée aujourd'hui.

## P2 — Absorber la tarification du vert (OUVERT 2026-07-16, gates pré-enregistrés ICI avant tout run)
**Forme pure** : `score(c) = longueur(c) + 0.02·max(0, κ_data·douleur̂(c)·100 − P̂(s,c)·bénéfice(drive))`
— W=25 et green_margin SORTENT du chemin décisionnel (la létalité devient 100 % apprise ; ils ne
survivent que dans le PROPOSEUR tangent, scope P3). Sans drives/cible-ressource : bénéfice=0
(pénalité = risque appris seul). ZÉRO entraînement : mêmes têtes que le juge PASS (ckpt re-taggé
`composed_pure_v1`, mêmes poids). ⚠️ Cousin du remplacement tué 2× — différences : max(0,·) plancher,
têtes validées closed-loop, G-consist obligatoire, KILL strict.
Gates OFFLINE (gratuits — le quantum de ranking P̂·ben−κ·pain̂ est INCHANGÉ, donc G-rank 0.683
owner-jugé et G-mono ✓ portent) :
  1. **G-res-pure** : choix simulé (forme pure) vs action empiriquement meilleure du bucket ≥
     analytique (72 %) — au minimum PARITÉ (le remplacement ne doit pas perdre ce que la remise a) ;
  2. **G-consist-pure** : bascule ≤ 1.2× analytique (le tueur historique du remplacement) ;
  3. **G-safe (nouveau)** : taux de traversée simulé sur bloqués BLESSÉS-PROFONDS (h<30 ET
     intr>médiane) ≤ forme-remise + 10 pts (un remplacement plus doux que W=25 ne doit pas ouvrir
     les vannes là où ça tue).
Juge closed-loop (si 1-3) : 2×24 vies seeds 1+2 vs réf vivante 45/8 — **PASS = repas poolés ≥ 40
ET morts-danger ≤ 10** ; KILL précoce seed 1 < 14. Échec offline OU juge → forme-remise conservée
(elle est jugée), négatif commité, W reste l'ancre déclarée.

## ⭐ VERDICT P2 (2026-07-16) : **ÉCHEC AU JUGE (morts) — négatif diagnostiqué, remise conservée**
Gates offline 3/3 passés (G-res-pure 75 %≥72, G-consist-pure 6.9 %≤7.8 — le remplacement ne
flotte PAS, G-safe parité) MAIS juge closed-loop : s1 17/6, s2 32/8 → **POOLÉ 49 repas / 14
morts-danger** vs gate ≥40 ET ≤10 : repas ✓ (bat même le plafond oracle 47 !) mais **morts ✗
(+6 vs remise 8, dégâts ×1.5)**. DIAGNOSTIC (structurel, pas un bug) : `κ·douleur̂` linéaire
prix une traversée profonde ~5 m là où `W·intr` montait à 25 m → sans l'ancre, l'entité troque
des vies contre des repas. **W=25 encode une PRIME DE RISQUE NON-LINÉAIRE** (mourir ≠ perdre
κ·dégâts — même thème que le plancher-mort du label) que les têtes actuelles (E[dégâts]) ne
portent pas. Per pré-enregistrement : forme-REMISE conservée (le vivant jugé 45/8), W = ANCRE
DÉCLARÉE ET DATÉE — sa purification exige une hypothèse NOUVELLE : une tête P(mort|s,c) (ou une
tarification convexe apprise de la queue des morts), licence owner requise. Ckpt sprint_pure.pt
bankée (judge_fail 49/14). Leçon : la frontière actuelle de pureté s'arrête à la remise-capée —
et on sait désormais EXACTEMENT ce que W contient.

## P2-bis — TÊTE P(mort|s,c) (OUVERT sur licence owner, gates pré-enregistrés AVANT le train)
Hypothèse (issue du diagnostic P2) : la prime de risque non-linéaire qu'encode W=25 est APPRENABLE —
    score(c) = longueur + 0.02·max(0, κ·douleur̂(c)·100 + **P̂mort(s,c)·κ·100** − P̂repas·bénéfice)
Le terme mort tarife la vie restante perdue (D_mort = κ_data·100 ≈ 920 pas — l'ancre déjà mesurée,
zéro constante nouvelle). P̂mort = MLP 14-d (contrat `sprint_inputs` inchangé), BCE sur
**died_danger** (poursuite finissant en mort-danger : santé→0 à la fin de vie).
CORPUS ÉLARGI (mesuré, gratuit) : 10 runs instrumentés (g24×4 + spx×2 + judge×2 + pure×2) =
**12 306 décisions, 173 positifs** (plis CV [56,27,45,45], 88 en classe cross) — apprenable.
⚠️ Les runs judge/pure ont des `costs` loggés NON-analytiques (leur forme de scoring) → ils servent
au TRAIN de la tête (features/intr/drives explicites) mais les replays de gates restent sur les
6 runs à coûts analytiques. Sans drives au déploiement : terme mort omis (documenté).
Gates PRÉ-ENREGISTRÉS :
  1. **G-death** : AUC(P̂mort, died_danger) > **0.80** CV-4 par vie ; ET monotonie santé —
     P̂mort moyen STRICTEMENT décroissant par bande h [0,30)/[30,60)/[60,100] sur traversées profondes ;
  2. **G-kill-decisions (le cœur)** : sur les décisions died_danger TENUES de classe cross, la
     forme v2 refuse la traversée ≥ **+30 pts** plus souvent que pure-v1 (elle doit dire NON aux
     traversées qui ont réellement tué) ;
  3. **G-res-v2** ≥ analytique (72 %) et **G-consist-v2** ≤ 1.2× (6 runs analytiques) ;
  4. **Juge** : 2×24 vies seeds 1+2, **repas poolés ≥ 40 ET morts-danger ≤ 10** ; KILL seed 1 < 14.
Budget dur : 1 train + 1 re-train diagnostiqué. Échec → remise conservée, W reste l'ancre, négatif
commité (et la piste ε-conditionné-blessé devient le préalable).

## ⭐⭐ VERDICT P2-bis (2026-07-17) : G-kill ÉCHOUÉ → W RECLASSÉ PRÉFÉRENCE DU CORPS (clôture)
Tête P(mort|s,c) : **G-death PASSÉ** (AUC CV-4 0.839, monotonie santé nette 0.140/0.052/0.009 —
le gradient de risque de mort EST appris ; label died_danger corrigé au passage : santé finale
par épisode, 207 positifs — h_at(end−1) débordait sur la vie suivante). G-res-v2 ✅ (76≥72),
G-consist-v2 ✅ (7.0 ≤ 7.8). MAIS **G-kill-decisions ❌ : refus 0 %→3 % (gate +30) sur les 102
décisions-tueuses tenues**. Cause MATHÉMATIQUE, pas un défaut de tête : la tarification
RISQUE-NEUTRE (prime = P̂mort·κ·100 ≈ 130 pas à P̂mort=0.15) ne dépasse jamais le bénéfice-repas
(~500 pas) — en espérance, traverser reste rentable même à 15 % de mort. Survivre exige de
l'AVERSION (utilité non-linéaire) ; un multiplicateur d'aversion fitté serait un raccourci
interdit (§2), et aucun re-train ne change l'algèbre → STOP per pré-enregistrement, budget rendu.
**CLÔTURE DE LA QUESTION DE PURETÉ (proposée) : W = PRÉFÉRENCE DU CORPS**, pas de la
connaissance-du-monde — l'attitude face au risque est une propriété de conception (catégorie
drives, §3 ; l'IC de LeCun câble les attitudes). La décomposition finale honnête :
`W·intr ≈ létalité (APPRISE : douleur̂, P̂mort — les deux têtes existent et sont bonnes)
× aversion (CORPS : constante de conception déclarée)`. La forme-remise (45/8) reste le vivant ;
les têtes douleur̂/P̂mort restent bankées pour tout futur usage (la mort est désormais PRÉDITE,
même si la préférence reste câblée). Acquis : on sait séparer ce que W contient.

## P4 — Reclassement (fait avec P1)
Les constantes de la machine à états sont déclarées CONSTANTES DE CONCEPTION en carte (comme les
drains des drives) — elles sortent du décompte de dette d'échafaudage.

## P5 — CHANTIER « PERCEPTION PAR LA CONSÉQUENCE » : le DANGER d'abord (ouvert 2026-07-17,
## branche feat/perception-consequence — gates pré-enregistrés ICI avant tout diag/train)
**Mission** : dissoudre la clé-apparence `green_points` (waypoint_layer.py:95, « danger = vert »)
en apprenant « dangereux = ce qui a précédé mes dégâts » sur la RÉTINE BRUTE (36 rayons ×
(d,r,g,b) = 144-d), labels = dégâts VÉCUS (chute de santé par tick, BC `obs.health`). Le volet
« nourrissant » (requêtes-couleur des slots WM) = chantier SÉPARÉ ultérieur.

**Forme (tranchée avant train — factorisation QUOI × OÙ, une géométrie de conception, pas à fitter)** :
    P(dégâts au tick t | rétine_t) = σ( b + Σ_{rayons k touchants} s(rgb_k) · g(d_k) )
- `s(rgb)` ∈ (0,1) = SAILLANCE D'APPARENCE (MLP 3→16→1 sigmoïde — la couleur seule, jamais la
  distance : l'apparence du danger ne dépend pas d'où il est) ;
- `g(d) = σ((ρ − 10·d)/τ)` = PORTÉE-MORSURE apprise (ρ, τ appris ; ρ̂ = distance où g=0.5) ;
- lecture déployée : rayon flaggé ⇔ d<0.999 ET s(rgb_k)>0.5 → mêmes points ego que green_points ;
  `green_margin` → ρ̂ (la marge devient MESURÉE du vécu, plus la géométrie pilier connue d'avance) ;
  `tangent_margin` → ρ̂ + 0.4 (le +0.4 = dégagement de CONCEPTION, relation structurelle conservée).
- Opt-in `SYLVAN_WP_SALIENCY=ckpt` (défaut OFF = bit-identique). W=25, hystérésis, machine à
  états : INTOUCHÉS (P2-bis : préférence du corps).

**Interdits** : positions hazard des logs = ORACLE D'ÉVALUATION SEULEMENT (jamais entrée/label) ;
PAS de distillation de green_points (labels = dégâts vécus uniquement — la règle-couleur ne sert
qu'à l'ÉVAL, le monde-jouet rend cet oracle licite : le vert EST la vérité rendue, cf
diag_hazard_slot) ; pas de constante ajustée pour passer un gate.

**G0 (diag gratuit AVANT tout train — `diag_saliency_corpus.py`, corpus = 10 runs instrumentés)** :
  1. ≥150 ONSETS-dégâts (premier tick de morsure après ≥20 ticks sains) avec rétine au tick ;
  2. visibilité : ≥90 % des ticks-dégâts ont ≥1 rayon vert-règle touchant (sinon label = bruit) ;
  3. contraste : ≥500 ticks proche-sans-dégât (≥1 rayon touchant <2 m, zéro dégât ±20 ticks) dont
     ≥100 avec rayon rouge/bleu proche (le confond « bouffe au cœur » doit être testable) ;
  4. diversité : onsets répartis sur ≥2 des 3 secteurs angulaires (avant/flanc/arrière).
  Échec → collecte ε seeds 3+4 d'abord (jamais les seeds 1+2 du juge) ; re-G0 ; échec encore → STOP.

**Gates OFFLINE pré-enregistrés (cheaper-first ; CV-4 par VIE ; échec → 1 seul re-train
diagnostiqué par tête, puis STOP négatif commité)** :
  1. **G-dmg** : AUC(P̂(dégâts|rétine), tick-dégâts) > **0.90** CV-4 par vie ;
  2. **G-loc (parité de lunette, le comportement-préservant)** : sur ticks tenus, rappel des
     rayons verts-règle ≥ **0.95** ET flag des rayons touchants NON-verts ≤ **2 %** (rouge proche
     pendant repas engouffré = LE test du confond) ;
  3. **G-ρ** : ρ̂ ∈ [médiane, q95 + 0.3 m] des distances min au point saillant aux ONSETS tenus
     (la portée apprise couvre la morsure vécue sans sur-couvrir) ;
  4. **G-feat (dé-contamination dg1/dg2)** : parité des ensembles de points saillance vs verts-règle
     sur les ticks de décision (Hausdorff ≤ 0.05 m ET même cardinal sur ≥99 %) ⇒ dg reconstruits
     à ±0.05 m ; puis RE-TRAIN des têtes sur feats recomputées lunette-saillance (repère vrai
     reconstruit via les candidats-anneau, positions déterministes → cible vraie) : AUC pain′ ≥
     **0.874** (0.894−0.02), AUC P̂mort′ ≥ **0.819** (0.839−0.02), plis P̂′ à ±0.02 des plis vivants ;
  5. **G-consist** : replay offline des poursuites — bascule du choix (lunette saillance + têtes
     ré-entraînées) ≤ **1.2×** l'analytique-vert ;
  6. smoke bit-identique : flag absent ⇒ decide() identique (selfcheck), garde hazard_manager ≥8.
**Juge closed-loop (payé SEULEMENT si 1-6)** : 2×24 vies seeds 1+2, monde v2, config vivante
(waypoint + sprint-critic dé-contaminé + lunette saillance) : **PASS = repas poolés ≥ 40 ET
morts-danger ≤ 10** (réf vivante 45/8, bruit ±5) ; KILL précoce seed 1 < 14 repas. Échec →
négatif commité, green_points conservé (lunette déclarée-datée), W/marges restent l'ancre.

### ⭐ NÉGATIF n°1 Phase A (2026-07-17) — forme SOMME : 1/4 gates, cause diagnostiquée sur trace
Train 1 : G-dmg **0.997** ✓ (la morsure EST prédite) mais G-loc ✗ (flag non-vert 48 %),
G-ρ ✗ (ρ̂=1.46 ≈ init), G-feat ✗ (0 %). Sonde census GRATUITE (s par couleur × distance) :
s(bleu)=0.010, s(vert)=1.000, **s(ROUGE)=0.601 PLAT à toute distance (100 % flaggé)** — pas une
dérive d'init : un ÉQUILIBRE. La forme SOMME donne au rouge un crédit PARTIEL de dégâts (repas
engouffrés : rouge près pendant la morsure), équilibré par les repas hors-zone → s(rouge) se pose
juste au-dessus du seuil. Et médiane **7 rayons verts aux ticks-dégâts vs 3 en approche** → la
somme sépare par le NOMBRE de rayons (taille angulaire = proxy de proximité) → g n'a jamais
besoin d'apprendre la portée (ρ̂ reste à l'init 1.5). Les gates ont fait leur travail (AUC 0.997
mentait, G-ρ/G-loc l'ont attrapé).
**RE-TRAIN pré-enregistré (le seul du budget)** : (1) **MAX-POOLING (MIL)** — physique du vécu :
la morsure a UNE source, pas une somme de rayons → dissout le crédit-partagé (rouge) ET le
comptage (ρ̂) d'un seul geste de FORME ; (2) **prior de parcimonie** λ=0.01·mean(s(touchants)) —
« rien n'est dangereux sans preuve vécue » : défaut sûr pour toute apparence jamais contrainte
(constante de conception, pas fittée — protège au changement de monde, le but du chantier).
Mêmes gates, aucun seuil déplacé.

### ⭐⭐⭐ VERDICT P5-MIXTE (2026-07-17) : **JUGE PASS 41/9 — LA CLÉ-APPARENCE DANGER EST DISSOUTE**
Smoke 3 vies ✓ (bannières, 0 crash, forage présent) → juge 2×24 vies seeds 1+2, bras lunette
APPRISE + marges du CORPS : s1 **23 repas/3 morts-danger** (vivant 19/5 — battu sur les 2 axes),
s2 **18/6** (vivant 26/3 — dessous), **POOLÉ 41/9 vs gate ≥40 ET ≤10 ✓✓** (réf 45/8 = parité
dans le bruit ±5/24-vies, gates poolés per pré-enregistrement). **PROMOTION** : la config vivante
devient `SYLVAN_WP_SALIENCY=data/checkpoints/danger_saliency/saliency_best.pt` +
`SYLVAN_WP_SPRINT_CRITIC=data/checkpoints/sprint_critic_decont/sprint_best.pt` (la paire JUGÉE :
têtes dé-contaminées, pain_ckpt→decont) — la règle « danger = vert » sort du chemin vivant
(green_points = secours déclaré, le défaut sans flag). Décomposition finale du volet danger :
**apparence = APPRISE (saillance, se ré-apprend du vécu si le monde change) · létalité = APPRISE
(douleur̂ 0.894, P̂mort 0.839) · standoff + aversion = CORPS (1.0/1.4 et W=25, constantes de
conception déclarées)**. Caveats honnêtes : s2 sous le vivant au-delà du bruit par-seed (18 vs 26 ;
le gate est poolé, consigné) ; multi-seed >1+2 = dette héritée ; monde-jouet (3 apparences pures).
Harnais du bras vivant : `scripts/judge_saliency_p5.sh`.

### ⭐⭐ VERDICT PHASE D (2026-07-17) : JUGE ÉCHOUÉ **29/14** (bras marges-ρ̂) — LA LUNETTE EST INNOCENTÉE,
### LA MARGE EST UNE PRÉFÉRENCE DU CORPS (négatif per pré-enregistrement)
2×24 vies seeds 1+2, bannières du bras vérifiées (ρ̂=0.63/tangent 1.03 + têtes décont) :
s1 **14 repas/5 morts-danger** (vivant 19/5 — KILL frôlé à =14), s2 **15/9** (vivant 26/3) →
POOLÉ **29/14** vs gate ≥40 ET ≤10 : ÉCHEC sur les DEUX axes. Per pré-enregistrement :
**green_points + marges main RESTENT le vivant**, lunette déclarée-datée, négatif commité.
Décomposition (sonde gratuite steps/vies) : temps vécu −10 % (morts-danger ×1.75) ET **taux de
forage en vie −25 %** (0.41-0.49 vs 0.54-0.70 repas/1000 pas) — les marges ρ̂ ne font pas que
tuer plus : elles dégradent les ROUTES (tangentes qui rasent le nuage à 1.03 m ; pénalité max
W·ρ divisée par ~1.6 → traversées longues non payantes remplacent les contournements).
**DIAGNOSTIC (pré-écrit au G0, confirmé et affûté)** : ρ̂=0.63 mesure « où ça mord » ; ce que
green_margin=1.0 encode n'est PAS cela — c'est une DISTANCE DE SÉCURITÉ (standoff) au-delà du
vécu = **préférence du corps (aversion)**, même clôture que W=25 (P2-bis : survivre exige
l'aversion, pas l'espérance vécue — ici la version SPATIALE du même théorème). La tête est
INNOCENTÉE : offline 4/4, lunette ≡ verte sur 100 % des 14 003 décisions.
**VOIE RESTANTE (hypothèse nouvelle, licence owner requise)** : bras « lunette APPRISE + marges
du CORPS » (green_margin=1.0/tangent=1.4 reclassées constantes de conception, comme W) —
dissoudrait la clé-APPARENCE (le but du chantier) en gardant le standoff déclaré ; vu la parité
de lunette 100 %, parité juge attendue. Décomposition finale candidate : apparence = APPRISE
(saillance), létalité = APPRISE (douleur̂/P̂mort), standoff + aversion = CORPS (déclarés).

### P5-MIXTE — bras « lunette APPRISE + marges du CORPS » (licencié owner 2026-07-17,
### pré-enregistré AVANT le run)
Hypothèse issue du diagnostic D : l'échec vivait dans les MARGES, pas la lunette → bras =
`SYLVAN_WP_SALIENCY` (lunette seule) avec **green_margin=1.0 / tangent_margin=1.4 CONSERVÉS et
reclassés CONSTANTES DE CONCEPTION** (standoff = préférence du corps, catégorie drives — comme
W=25). Le code saillance ne touche plus aux marges (le bras réfuté reste reproductible via les
overrides env, pas de bouton dédié). Attendu : parité stricte (lunette ≡ verte sur 100 % des
décisions mesurées) — mais on ne promeut JAMAIS sur un attendu.
Gates : 1) smoke 3 vies seed 3 (bannières, décisions loggées, zéro crash, comportement de forage
présent) ; 2) juge INCHANGÉ : 2×24 vies seeds 1+2, **repas poolés ≥ 40 ET morts-danger ≤ 10**
(réf 45/8), KILL seed 1 < 14. Si PASS → PROMOTION : la lunette saillance devient le défaut de la
config vivante (la clé-apparence danger est dissoute) ; échec → clôture définitive du volet danger
(green_points conservé), pas de 3ᵉ bras.

### ⭐ VERDICT PHASE C (2026-07-17) : DÉ-CONTAMINATION EXACTE — PARITÉ **4/4**
Filtre lunette (`decontaminate_heads.py`) : **0 décision écartée sur les 13 runs** (lunettes
identiques sur 100 % des décisions, wpx compris) → re-trains à corpus identique + mêmes seeds =
les têtes ′ reproduisent les vivantes À L'IDENTIQUE : pain′ **0.894** (= vivant), P̂′ **0.683**
(= vivant, G-mono/G-consist aux mêmes valeurs), P̂mort′ **0.839** ✓ mono ✓. Lecture : le savoir
des têtes ne portait AUCUNE connaissance spécifique à la règle-couleur au-delà de ce que la
lunette apprise fournit — la dé-contamination est une substitution de SOURCE, pas de valeurs.
**G-consist lunette+marges ρ̂** (replay exact `(ρ̂−dg)⁺` sur feats loggées, candidats du log =
limite déclarée) : bascule **5.1 %** vs analytique-vert 6.5 % ✓ — les marges vécues ne font pas
flotter les choix. Notes internes des trainers = celles du vivant (anti-myopie B2 réel=0
historique ; G-rank 0.683 = plafond de bruit du label, owner-jugé). Ckpts :
`waypoint_pain_decont`, `sprint_critic_decont` (pain_ckpt→decont), `sprint_death_decont`.
→ Phase D licenciée per pré-enregistrement (offline 1-6 tous passés).

### ⭐⭐ VERDICT PHASE A (2026-07-17) : RE-TRAIN MIL **4/4 GATES** — LA LUNETTE DANGER EST APPRISE
Re-train diagnostiqué (max-pool + parcimonie, budget 1/1 utilisé) : G-dmg **0.997** ✓ ;
G-loc rappel **1.000** / flag non-vert **0.00 %** ✓ (le crédit-partagé du rouge est dissous par
le max — s ne flagge plus que l'apparence qui précède la morsure) ; G-ρ **ρ̂=0.63 m** (τ=0.05)
∈ [0.39, 0.89] ✓ — la portée-morsure est MESURÉE du vécu (green_margin 1.0 → 0.63 : la géométrie
pilier connue d'avance sort, la marge vécue entre) ; G-feat lunettes identiques **100.0 %** des
14 003 décisions ✓ → les dg loggés = dg-saillance PAR IDENTITÉ (la phase C est exacte, zéro
reconstruction de repère). Ckpt `data/checkpoints/danger_saliency/saliency_best.pt`
(gates_pass=True). Branchement : `SYLVAN_WP_SALIENCY` (waypoint_layer `_lens`, OFF bit-identique,
selfcheck intégration ✓ : ρ̂=1.0 ⇒ décision identique, ρ̂ court suivi par l'intrusion).

### ⭐ VERDICT G0 (2026-07-17, `diag_saliency_corpus.py`, 0 train) : **PASSÉ 4/4**
183 onsets (≥150 ✓) ; visibilité verte **100 %** de 23 091 ticks-dégâts (✓ — le label vécu est
propre, zéro morsure aveugle) ; contraste 146 194 ticks proche-sans-dégât dont **96 115 rouge/bleu
proche** (✓ — le confond « bouffe au cœur » est massivement testable) ; secteurs avant 149 /
flanc 33 / arrière 1 (✓ ≥2 peuplés ; l'arrière rare = attendu, le corps avance — la lunette est
apparence-seule, insensible au bearing). **Mesure ρ̂ vécue** : onsets méd 0.39 / q90 0.57 /
q95 0.59 m ; tous ticks-dégâts méd 0.30 / q95 0.50 m → bande G-ρ effective = **[0.39, 0.89] m**.
⚠️ CONSIGNÉ AVANT LE TRAIN : la marge-main 1.0 est AU-DELÀ de la morsure vécue — elle porte un
headroom de sécurité que les données de dégâts ne justifient pas seules. Si le juge échoue sur
les MORTS avec un ρ̂ ~0.5-0.7, le diagnostic pré-écrit est « la marge contient une préférence de
sécurité du corps au-delà du vécu » (même famille que W=25, P2-bis) — pas un défaut de la tête.

## P6 — VOLET « NOURRISSANT » : requêtes de slot APPRISES du soulagement vécu (ouvert 2026-07-17
## soir, licencié owner — gates pré-enregistrés ICI avant tout diag/train)
**Mission** : dissoudre la DERNIÈRE clé-apparence de la boucle décisionnelle — les requêtes-couleur
des slots WM (« bouffe=rouge, eau=bleu, danger=vert », `slot_head.py color_queries`). La perception
ressource entière passe par ces 9 nombres écrits à la main. Forme cible : « nourrissant = ce qui a
soulagé MON drive » — les requêtes deviennent apprises des CONSÉQUENCES vécues.
**Le geste est minuscule (anatomie mesurée)** : dans le chemin vivant K>1, le readout du slot est
GÉOMÉTRIQUE zéro-paramètre (décision 2026-07-04) — la seule connaissance-du-monde est le buffer
`color_queries` [K,3]. Le seuil 0.55, le softmax masqué, le prior distance, la saillance-saturation
= appareil perceptif (constantes de conception, déclarées). Remplacer les requêtes par BUILD
(gabarit build_hazard_slot) ⇒ WM GELÉ, zéro retrain, tous les consommateurs préservés par parité.

**Forme (tranchée avant train)** : par drive d ∈ {énergie→food, soif→water, santé→danger} :
    P(conséquence_d | rétine) = σ( b_d + max_{rayons k touchants} σ(w_d·rgbn_k + c_d) · g_d(dist_k) )
— le gabarit P5 (MIL max-pool : la conséquence a UNE source ; prior parcimonie λ=0.01) avec
l'apparence LINÉAIRE en couleur NORMALISÉE : w_d·rgbn = cosinus × ||w_d|| = exactement la forme de
l'affinité du slot → **q̂_d = w_d/‖w_d‖ EST la requête**. Labels VÉCUS uniquement :
- food : soulagement énergie (drv[t]−drv[t−1] > +5), percept = rétine à **t−1** (l'objet est au
  contact juste avant, consommé/respawné à t) ; water : idem soif ;
- danger : ticks-dégâts (labels P5 réutilisés, même corpus).
**Interdits** : pas de distillation des requêtes main (les canaux purs ne servent qu'à l'ÉVAL,
licite monde-jouet) ; positions oracles = éval seulement ; WM/readout/transport/W/marges/hystérésis
intouchés. Résidus inventoriés HORS SCOPE (déclarés, pas cachés) : `_retina_food_pos` (sonde
SYLVAN_MULTI_FOOD_SLOT=0, défaut OFF), eau hors-vue→EMA du replan (frontière CHERCHER), tokens
color-gatés Mode-1 (branche mode1), `color_masses` (outil build-time, inutilisé à requêtes
drive-indexées).

**G0 (diag gratuit AVANT tout train — `diag_relief_corpus.py`, 10 runs)** :
  1. ≥100 soulagements ÉNERGIE et ≥100 SOIF avec rétine à t−1 et ≥1 rayon touchant < 1.5 m ;
  2. confond miroir testable : ≥30 repas ENGOUFFRÉS (rayon vert proche pendant le soulagement
     énergie — la requête-faim ne doit pas absorber le vert, symétrique du G-loc P5) ;
  3. contraste : ≥500 ticks avec rayon coloré proche SANS soulagement ±10 ticks du drive concerné.
  Échec → collecte ε seeds 3+4 ; re-G0 ; échec encore → STOP.
**Gates OFFLINE pré-enregistrés (budget : 1 train + 1 re-train diagnostiqué par requête)** :
  1. **G-q** : cos(q̂_food, rouge pur) ≥ 0.98 ET cos(q̂_water, bleu) ≥ 0.98 ET cos(q̂_danger, vert)
     ≥ 0.98 (oracle d'éval) ET affinité croisée post-seuil 0.55 = 0 (zéro fuite entre slots) ;
  2. **G-slot-parité** : slot head avec q̂ vs requêtes main sur ≥20k ticks BC : masque de
     visibilité identique ≥ 99.9 % ET |Δposition| ≤ 0.05 m sur ≥ 99.9 % des ticks visibles ;
  3. smoke 3 vies seed 3 (bannières, 0 crash, forage présent).
**Juge closed-loop (payé si 1-3)** : 2×24 vies seeds 1+2, config vivante (saillance + sprint
decont) avec `WM_CKPT=wm_objcentric_kin_lq` (requêtes apprises) : **PASS = repas poolés ≥ 36
(41−5) ET morts-danger ≤ 11 (9+2)** — réf vivante = bras saillance jugé 41/9 (2026-07-17) ;
KILL précoce seed 1 < 14. PASS → promotion : **plus aucune variable clé-apparence dans la boucle
décisionnelle** (sortie du chantier 1 de la roadmap). Échec → requêtes main conservées, négatif
commité.

### ⭐ NÉGATIF n°1 P6 (2026-07-17) — direction de requête NON-IDENTIFIÉE (jauge), cause sur trace
Train 1 : les têtes PRÉDISENT (AUC danger 0.997, food 0.867, water 0.758 ; ρ̂ 0.61-0.63 cohérents
P5) mais q̂_food/water ≈ **−gris** (tous canaux négatifs), q̂_danger blanchâtre → G-q ✗,
G-slot-parité ✗ (masque 47.75 %). Cause MATHÉMATIQUE : sur rayons monochromes (rgbn one-hot),
le comportement ne dépend que des scalaires w_i + c → w est libre le long de 1⃗ (jauge : w+α·1⃗,
c−α ⇒ comportement identique) ; le prior de parcimonie pousse cette jauge en bloc vers le négatif.
La requête normalisée est donc non-identifiée alors que la prédiction est bonne — un « résultat
qui a l'air bon et ment », attrapé par G-q comme prévu.
**RE-TRAIN pré-enregistré (le seul du budget, par requête)** : **w contraint au CÔNE POSITIF
(softplus)** — parité de déploiement, pas un fit : l'affinité du slot vit dans le cône positif
par construction (rgbn ≥ 0, requêtes main = gabarits non-négatifs, seuil cosinus 0.55) ; une
requête à composantes négatives n'a AUCUN sens dans la forme déployée. La non-négativité casse
la jauge du bon côté (canal OFF → w_i = 0 au bord actif ; canal ON → toute la masse). Mêmes
gates, aucun seuil déplacé.

### ⭐ VERDICT G0 P6 (2026-07-17, `diag_relief_corpus.py`, 0 train) : **PASSÉ 3/3**
Soulagements avec percept proche : énergie **201** / soif **277** (≥100/≥100 ✓) ; repas engouffrés
**160** (≥30 ✓) ; contraste **124 883** (≥500 ✓). Census (oracle d'éval) : énergie → rouge 94 %,
**vert 80 %** (le confond miroir est MASSIF : on mange le plus souvent dans le danger), bleu 4 % ;
soif → bleu 100 %, vert 10 %, rouge 2 %. Le MIL max-pool + les 124k contrastes (vert proche sans
soulagement = ultra-fréquent) doivent écraser s_food(vert) — exactement le mécanisme qui a écrasé
s(rouge) côté danger en P5.

## Critère de succès = le BUT
Chaque purification est jugée closed-loop contre la référence vivante (jamais un proxy offline
seul), au plancher de bruit ±5 repas/24-total, morts comprises. Un retrait qui coûte du forage ou
des morts n'est PAS une purification — c'est une régression déguisée en vertu.
