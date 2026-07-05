# Audit de pureté LeCun — Sylvan (2026-07-06)

> **Mission.** Inventorier la boucle vivante de Sylvan, la confronter à l'architecture
> LeCun (« A Path Towards Autonomous Machine Intelligence », 6 modules : configurateur,
> perception, world-model JEPA, coût = intrinsèque câblé + **critique appris**, acteur
> Mode-1/Mode-2, mémoire court-terme), et **classer les écarts** par
> `(sévérité × gain perf) / coût du fix`. Pas de nouvelle feature : on grappille par le haut.
> Réf. secondaire citable : blog Meta AI « Yann LeCun on a vision… » ; §13/§14 de `docs/BLUEPRINT.md`.

## Méthode

- **Gratuit d'abord (CLAUDE.md §1).** Vérif orphelins (0), disque OK, arbre propre. Verdict
  critique repêché depuis les **logs serveur** (source de vérité `[planner-cmd]`), pas les
  en-têtes de sonde (buggés). Deux sweeps de code parallèles (rétine/WM/mémoire ;
  coût/planner/drives/Mode-1), chaque écart vérifié **dans le code, pas dans les commentaires**.
- **Distinguer CORPS et impureté (CLAUDE.md §3).** CPG/moteur, drives câblés, et le coût
  intrinsèque homéostatique (« mourir = mal ») = le CORPS, **voulu**. Impuretés = oracles dans
  la boucle vive, poids-décision tunés tenant lieu de valeur apprise, échafaudages flaggés.
- **Ne pas masquer (CLAUDE.md §2).** Un écart qui arrange la carte est signalé, pas enterré.

## Verdict du fil critique (l'item ouvert de la session précédente)

Re-juge v2 du **critique appris** (`SYLVAN_PLANNER_COST=critic`), lu depuis
`/tmp/hesit_srv_*cr*.log` (`[planner-cmd] CRITIQUE APPRIS actif`) — les en-têtes de sonde
affichaient « coût designed » à tort (bug d'étiquette `$COST`). Mapping réel :

| bras | coût | survie méd 1+1 | survie méd 5+5 | avortées 1+1 / 5+5 |
|---|---|---|---|---|
| `cra_surv` | survie analytique | 1600 | 2255 | 79% / 64% |
| `crb` | **critique appris** | 1600 | **2850** | 95% / 76% |
| `crc` | designed (`-min_dist`) | 1595 | 2265 | 92% / 75% |

**Lecture honnête (n=8, seed unique).** Critique = **+585 vs designed en 5+5** (prometteur),
mais **plat en 1+1** (1600 vs 1595) et **hésitation inchangée**. Gates offline PASS (AUC .995,
non-sat .66, swap .95). → **Prometteur mais NON confirmé** : câblé comme mode optionnel,
**pas promu défaut**. Le multi-seed décisif n'a jamais été payé.

## Cartographie LeCun (6 modules)

| Module LeCun | Pièce Sylvan | État réel | Écart au référentiel |
|---|---|---|---|
| Perception | slot-head 2-slots requête-couleur | **géométrique pur (K=2)** | scoreur appris **court-circuité** → « appris » surévalué (E7) ; oracle-eau résiduel (E1) |
| World-Model (JEPA) | `SimpleRSSM` + slot object-centric | Dreamer-like **déterministe** | pas de latentes stochastiques → pas d'incertitude/curiosité (E5, flaggé) ; perte encore recon-radar en train (E8) |
| Coût intrinsèque | drain/refill homéostatique | **voulu (CORPS)** | — |
| Coût — valeur | survie analytique (LIVE) vs critique appris | **échafaudage LIVE**, critique existant mais optionnel | E2 (scaffolding tuné) / E3 (fix dispo) |
| Acteur Mode-2 | planner MPC commande-space | agnostique, sain | recherche plate (pas de sous-buts) — hors scope |
| Acteur Mode-1 | `DriveSymmetricPolicy` BC du planner | **distillé de Mode-2 (LeCun-correct)**, en jachère pré-RL | honnête (`partiel`) |
| Mémoire court-terme | `MultiSlotMemory` 3-états | learned egomotion + géom | valeur closed-loop non démontrée (E6, flaggé) |
| Configurateur | — | **absent** | manquant (peu urgent : 2 drives) |

## Table des écarts (classés)

| # | Écart | file:line | Classe LeCun | Carte ? | Sévérité | Coût fix | Gain perf |
|---|---|---|---|---|---|---|---|
| **E1** | **Oracle eau jamais-vue → radar-EMA** (`vision_water`→`_water_ema`) ; food dégrade correct à `None` | `command_planner.py:485-493` + `serve_planner_command.py:326,427` | **oracle** (le code l'appelle « l'oracle radar ») | **NON — nouveau** | **HAUTE** | faible-moyen | neutre/-léger |
| **E2** | Coût survie analytique LIVE = drain/refill codé + alternance + `surv_margin_weight=200` | `command_planner.py:604-668`, `_survival_extension:143-223` | échafaudage (proxy tuné pour la valeur) | oui (auto-flaggé) | MOY-HAUTE | — | — |
| **E3** | **Critique appris** existe, oracle-free, gates offline PASS, câblé mais **pas défaut** | `command_planner.py:528,568-602` + `scripts/train_survival_critic.py` | **critique appris (pur-aligné)** = LE fix de E2 | **carte dit `manquant` (STALE)** | — | **faible** (déjà bâti) | **+585 (5+5, à confirmer)** |
| **E4** | `cout_planner=pur` (prétend `-min_dist` géométrique) alors que le défaut multi-drive LIVE = survie analytique tunée | carte `cout_planner` | honnêteté | **carte MENT** | MOY | ~libre | — |
| **E5** | WM déterministe, zéro incertitude → bloque curiosité | `command_wm.py:69,181-217` | WM JEPA incomplet | oui (`partiel`) | ÉLEVÉ (retrain WM) | moyen |
| **E6** | Valeur closed-loop de la mémoire non démontrée (éclipses surtout brèves, ~17% queue) | `slot_memory.py` | mémoire court-terme | oui (`partiel`) | moyen | faible-modeste |
| **E7** | « Perception apprise » = géométrie fixe à K=2 (scoreur bypassé ; prior rouge/bleu codé ; seuils tunés 0.55/0.95/40/4.0) | `slot_head.py:85-92` + `train_slot_head.py:124-127` | honnêteté (pur par construction, pas « émergent ») | **NON — nouveau** | faible-moy | ~libre (carte) | — |
| **E8** | WM reconstruit encore le radar (oracle) en **entraînement** (obs live = proprio+rétine+énergie, pas radar) | `command_wm.py:41,279` | JEPA génératif (train-only) | non | faible | — | — |
| **H1** | Carte `world_model=wm_objcentric_s1` alors que défaut multi-drive LIVE = **s2** | carte `world_model` | honnêteté | **carte STALE** | faible | libre | — |
| — | `surv_discount` dormant (« NE PAS ACTIVER », télescope 1/(1−γ)) | `command_planner.py` | négatif documenté | oui | — | — | — |

**Dettes connues intégrées (ne pas re-découvrir) :** multi-seed de la promo s2 jamais payé ;
drive-dynamics analytique = 3ᵉ verrou (E2) ; eau jamais-vue→EMA (E1, ~2%) ; Mode-1 distillé
d'un planner à ~1900 sous-Gate ; Gate-S jamais passé ; `surv_discount` mort.

## Top 3 (ratio sévérité×perf/coût, pattern gagnant = pureté ET perf ensemble)

### #1 — Promouvoir le critique appris (gate multi-seed décisif) — E3 retire E2
**Pourquoi n°1 :** c'est LE module central manquant de LeCun (« trainable critic ») que
§14 du BLUEPRINT désigne explicitement comme le remplaçant du proxy codé-main. Il est **déjà
bâti et gate-offline PASS** → coût quasi nul, il ne reste que la mesure closed-loop jamais
payée. Promouvoir = retirer l'échafaudage LIVE E2 **et** potentiellement +585. Pureté+perf.
**Gate falsifiable (pré-enregistré) :**
- Closed-loop 3 seeds × {critic, survival, designed} sur 5+5 et 1+1 (harnais parallèle
  `run_hesitation_probe.sh` `PORT/WORLDS/PARALLEL`, ~20 min/4 bras).
- **SUCCÈS (promouvoir critic)** : médiane-sur-seeds critic ≥ designed dans les DEUX mondes,
  ET 5+5 critic ≥ designed+200 sur ≥2/3 seeds, ET 1+1 ≥ designed−200 (pas de régression).
- **KILL** : critic < survie−200 en médiane-sur-seeds dans un monde → reste optionnel,
  carte note le négatif. **PARTIEL** sinon → garder optionnel, documenter (ne PAS bouger les poteaux).

### #2 — Fermer/mettre en quarantaine l'oracle eau jamais-vue — E1
**Pourquoi n°2 :** **seul oracle authentique dans la boucle vive par défaut**, **non flaggé**,
et il **contredit** la mémoire (« l'eau a quitté l'oracle EMA » — vrai pour le cas VU, faux
pour jamais-vu). §2 impose de le surfacer, pas de le cacher.
**Gate falsifiable :**
- **Gratuit d'abord** : compter dans les buffers `hesit_probe_*` la fréquence réelle de
  déclenchement du fallback jamais-vu-eau (parse des logs de décision). Si <~1% et début-d'épisode → enjeu faible.
- Closed-loop A/B : fallback EMA **ON** (actuel) vs **OFF** (jamais-vu → `None`, ou dead-reckon
  mémoire). **SUCCÈS pureté** : survie OFF ≥ ON−200 (pas de régression réelle) → retirer l'oracle.
  Si OFF ≪ ON → l'oracle est porteur → **le dire** (§2), router via la mémoire, owner tranche.

### #3 — Resync carte + doc (libre, mandaté §CARTE + §2) — E4/H1/E3/E7
**Pourquoi n°3 :** gratuit, obligatoire, et supprime des **mensonges actifs** de la carte.
À corriger dans un commit : `cout_planner` pur→échafaudage (survie analytique LIVE) ;
`critique_appris` manquant→partiel (bâti+câblé, promo en attente) ; `world_model` s1→s2 ;
note d'honnêteté slot-head (géométrique à K=2, pas « émergent »). Validé par
`validate_architecture.py`. **Aucun run.**

## Ce qui est PUR / VOULU (ne pas toucher)
CPG + moteur + bornes d'action = CORPS. Drives câblés + coût intrinsèque homéostatique
(« mourir = mal ») = voulu (LeCun : « intrinsic costs, such as not wasting energy »).
Mode-1 = distillé de Mode-2 (BC du planner) = LeCun-correct, honnêtement `partiel`.
Transport slot = géométrie fixe équivariante par construction (dé-apprise volontairement).
