# Design — Purification de l'étage haut (post-critique-sprint, 2026-07-16)

## Mission
Rendre l'étage waypoint (et son socle bas) pur au sens du recadrage LeCun 2026-07-14 :
**pureté ≠ zéro formule** — l'IC analytique immuable est licite ; ce qui est impur, c'est la
CONNAISSANCE-DU-MONDE codée main (létalité du vert, géométrie des piliers, hints de cap) là où
elle devrait être apprise du vécu.

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

## P4 — Reclassement (fait avec P1)
Les constantes de la machine à états sont déclarées CONSTANTES DE CONCEPTION en carte (comme les
drains des drives) — elles sortent du décompte de dette d'échafaudage.

## Critère de succès = le BUT
Chaque purification est jugée closed-loop contre la référence vivante (jamais un proxy offline
seul), au plancher de bruit ±5 repas/24-total, morts comprises. Un retrait qui coûte du forage ou
des morts n'est PAS une purification — c'est une régression déguisée en vertu.
