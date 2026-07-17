# Prompt — chantier « Gate-capacité » : prouver « survit à un changement de monde » (embryon jour/nuit)

> Copier-coller comme premier message d'une session fraîche. Écrit le 2026-07-17, à la clôture de
> la session P6-reopen (WM typé PROMU : perception 100 % apprise, juge PASS-parité 42/10 en monde
> varié ; branche `feat/perception-consequence`). La pré-inscription ET le pré-gate gratuit sont
> DÉJÀ FAITS — cette session CONSTRUIT.

---

Sylvan — chantier d'IMPLÉMENTATION : **Gate-capacité** = prouver en vies la capacité que la
perception codée-main ne peut PAS avoir — quand l'apparence du monde change en cours de vie,
l'entité s'adapte au lieu de devenir aveugle. Même branche `feat/perception-consequence` (ou
brancher `feat/gate-capacite` depuis elle). Venv `env_pytorch_3.12`, CPU, `PYTHONPATH=python`,
racine. Re-checker : orphelins (`pgrep -xc godot`), git log, disque (≥2 G).
⚠️ `SYLVAN_RUN_DIR` est RELATIF au projet Godot → purger `godot/data/replay_buffer/critic_tmp_*`
après chaque collecte. 🚫 **Ne JAMAIS publier d'artifact** (compte partagé — bloqué par
`disableArtifact`+hook, cf CLAUDE.md) : tout point/synthèse = texte ou fichier local.

LIRE D'ABORD : 1) **`docs/design_gate_capacite.md`** (LE doc : mission, pourquoi l'embryon et pas
le jour/nuit complet, les 2 pièces, gates pré-enregistrés, verdict G-pré-swap) ; 2)
`docs/design_perception_types.md` (la base : WM typé promu, la mesure) ; 3) `memory/MEMORY.md` +
fin de `memory/sylvan-mode1-build.md` (2026-07-17).

## État à la reprise (rien à re-découvrir)
- **Config vivante PROMUE** : monde v2 VARIÉ (`SYLVAN_HAZARD_COUNT=1 ENGULF_P=0.5 HEALTH_REGEN=0.05
  SYLVAN_FOOD_APPEARANCE_VAR=0.15 SYLVAN_WATER_APPEARANCE_VAR=0.15`), WM TYPÉ
  `data/checkpoints/wm_objcentric_kin_typed/wm_best.pt` (perception 100 % apprise : requêtes
  mesurées + marges par-type `query_thr` + lien slot→drive découvert) + `SYLVAN_WAYPOINT=1` +
  `SYLVAN_WP_SALIENCY=data/checkpoints/danger_saliency/saliency_best.pt` +
  `SYLVAN_WP_SPRINT_CRITIC=data/checkpoints/sprint_critic_decont/sprint_best.pt`. Réf vivante = **42
  repas/10 morts** poolé (seeds 1+2, monde varié). `wm_objcentric_kin_haz` = secours requêtes-main.
- **Pré-inscription FAITE** (`docs/design_gate_capacite.md`) : gates falsifiables écrits AVANT.
- **G-pré-swap GRATUIT PASSÉ** : la mesure suit un swap vers une couleur LIBRE (magenta cos 1.0,
  re-lie food→énergie). ⚠️ **Contrainte de monde déclarée** : le swap doit atterrir en espace de
  couleur LIBRE (vers vert=danger ou bleu≈eau = inséparable, cousin de G-sep). Cible = magenta
  (teinte ~0.83). Poser une teinte-CIBLE propre (HSV, garder S/V), pas une grande rotation.

## Le chantier — 3 pièces à construire (cheaper-first, chacune vérifiée avant la suivante)
1. **Swap Godot** (`godot/scripts/world/food_manager.gd`) : `SYLVAN_FOOD_SWAP_TICK=T` +
   `SYLVAN_FOOD_SWAP_HUE=<teinte cible libre>` → à T pas dans chaque vie, poser la teinte-cible sur
   `_albedo` (HSV, garder S/V) et ré-appliquer à tous les items. Opt-in, absent = OFF bit-identique
   (vérifier par empreinte, comme la marge par-requête). Le compteur de pas : incrémenter en tête
   de `try_consume` (appelé chaque tick), remis à zéro dans `reset()`. Δ/teinte = propriété MONDE
   déclarée, jamais ajustée pour faciliter. Ne PAS toucher `main.gd` (garde `hazard_manager` ≥8).
2. **Re-mesure périodique** (l'embryon jour/nuit, dans `serve_planner_command.py`) :
   `SYLVAN_REMEASURE_EVERY=N` → bufferiser par tick (rgbn du rayon touchant le plus proche,
   distance, Δdrive énergie/soif, dégât), et toutes les N pas re-lancer cluster+lien (RÉUTILISER
   `scripts.build_typed_slots` : `stage_a_cluster`, `stage_b_bind` — importables) sur la fenêtre
   récente → MAJ live de `wm.slot_encoder.color_queries` + `query_thr`. **MESURE, zéro gradient.**
   N = période circadienne = constante du CORPS déclarée. OFF = requêtes statiques (bit-identique).
   Bootstrap OK : `try_consume` mange par DISTANCE (pas par perception) → même aveugle, l'entité
   mange ce qu'elle touche → l'événement re-lie. Le critère est un TEMPS-DE-RÉCUPÉRATION.
3. **Juge 2-bras** (adapter `scripts/judge_typed_slots.sh`) : mesurer les repas par fenêtre
   (pré-swap [0,T) vs tardive [T+grâce, fin]).
   - **G-swap-control** : bras STATIQUE (pas de re-mesure) + swap → s'effondre (taux tardif ≤ 0.3×
     pré-swap). Sinon le swap est trop faible → augmenter Δ (monde déclaré), pas relâcher le gate.
   - **⭐ G-capacité (LE BUT)** : bras APPRIS (re-mesure ON) + swap → récupère (taux tardif ≥ 0.6×
     pré-swap) ET ≫ contrôle. 2×24 vies seeds 1+2 poolés. T=700, grâce 200, fenêtre tardive
     [900, fin] (pré-enregistrés).

## Contraintes imposées par les leçons (pas des options)
1. **Une variable à la fois** : l'embryon ne touche QUE la perception (pas les têtes de décision —
   leur ré-entraînement = jour/nuit v1, chantier suivant). Verdict propre.
2. **Re-mesure = MESURE** (cluster+lien, cf build_typed_slots), jamais un fit gradient de requête
   (leçon P6 : le gradient n'identifie pas une requête — jauge/init).
3. **N (période) et la teinte-cible du swap = DÉCLARÉS**, jamais ajustés pour passer un gate.
4. **Marges par-requête `query_thr` = mesurées** (ne pas revenir au 0.55 global).
5. **Budget dur** : les 2 pièces + le pré-gate (déjà passé) + 1 juge 2-bras. Un échec = négatif
   commité + STOP + escalade owner (pas d'enchaînement de tweaks).

## Discipline (non négociable, cf CLAUDE.md)
Critères/KILL écrits (déjà dans design_gate_capacite.md) ; négatif = commité ; collecte
SÉQUENTIELLE ; `/sylvan-kill` + 0 orphelin ; ne JAMAIS stager `godot/scripts/main.gd` ni `ui/` ;
carte `tools/archi_hud/architecture.json` à jour DANS LE MÊME COMMIT ; commits Conventional anglais
sans attribution IA ; README = constat au présent, zéro em-dash/emoji ; PAS d'artifact hébergé.

## Fichiers probablement touchés
`godot/scripts/world/food_manager.gd` (swap opt-in) ; `python/scripts/serve_planner_command.py`
(re-mesure périodique, buffer + appel build_typed_slots) ; `scripts/judge_typed_slots.sh` ou
nouveau `scripts/judge_gate_capacite.sh` (2 bras, fenêtres) ; `docs/design_gate_capacite.md`
(verdicts) ; carte. Réutiliser tel quel : `scripts.build_typed_slots` (la mesure),
`sylvan/models/slot_head.py` (buffer `query_thr` déjà là).

## Si PASS
La re-mesure périodique (embryon jour/nuit) est promue — première capacité « survit à un
changement de monde » prouvée en vies. Le jour/nuit v1 (têtes de décision) hérite d'un squelette
validé. Mettre à jour carte + README + mémoire dans le même commit.
