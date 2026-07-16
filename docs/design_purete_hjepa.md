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

## P2 — Absorber la tarification du vert (pré-enregistrement complet AVANT son train)
`score = longueur + 0.02·max(0, κ_data·douleur̂(c)·100 − P̂·bénéfice)` — W et green_margin sortent
du chemin décisionnel (la létalité devient 100 % apprise). ⚠️ Cousin du remplacement tué 2× (v2/v3)
— différences : forme plafonnée par le bas (max(0,·)), deux têtes VALIDÉES par un juge, G-consist
obligatoire, KILL précoce strict. Gates détaillés à écrire À L'OUVERTURE du chantier (pas ici) ;
juge inchangé (2×24 vies, ≥ réf vivante − bruit ET morts ≤ réf+2).

## P4 — Reclassement (fait avec P1)
Les constantes de la machine à états sont déclarées CONSTANTES DE CONCEPTION en carte (comme les
drains des drives) — elles sortent du décompte de dette d'échafaudage.

## Critère de succès = le BUT
Chaque purification est jugée closed-loop contre la référence vivante (jamais un proxy offline
seul), au plancher de bruit ±5 repas/24-total, morts comprises. Un retrait qui coûte du forage ou
des morts n'est PAS une purification — c'est une régression déguisée en vertu.
