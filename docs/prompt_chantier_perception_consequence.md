# Prompt — chantier « perception par la CONSÉQUENCE » (dissoudre la clé-couleur)

> Copier-coller comme premier message d'une session fraîche. Écrit le 2026-07-17, à la clôture de
> la session sprint-critic/pureté (juge PASS 45/8, oracle mort, W reclassé préférence du corps,
> main mergé+poussé a1e764f).

---

Sylvan — chantier d'IMPLÉMENTATION : **perception ancrée dans la conséquence vécue**, pour
dissoudre les variables clé-apparence avant tout enrichissement du monde. Nouvelle branche
`feat/perception-consequence` depuis `main`. Venv `env_pytorch_3.12`, CPU, `PYTHONPATH=python`,
racine. Re-checker : orphelins (`pgrep -xc godot`), git log, disque (≥2 G ; actuellement ~24 G).
⚠️ NOUVEAU réflexe : `SYLVAN_RUN_DIR` est RELATIF au projet Godot → les dumps de collecte
s'accumulent dans `godot/data/replay_buffer/` (PAS `data/replay_buffer/`) — purger CE chemin
après chaque collecte.

LIRE D'ABORD : 1) **`docs/design_purete_hjepa.md`** (le doc du chantier : critère officiel owner
« est-ce que ça survit à un changement de monde ? », inventaire clé-apparence, verdicts P1/P2/
P2-bis) ; 2) `docs/design_critique_sprint.md` (la méthode qui a marché : têtes composées, gates,
juge) ; 3) `memory/MEMORY.md` + fin de `memory/sylvan-mode1-build.md` (2026-07-16/17).

## Le problème (critère owner, 2026-07-17)
Le but à terme est un monde ressemblant au VRAI monde — plus de boules de couleur ni de formes
fluo. Le monde VA changer. Or aujourd'hui la perception de l'étage décisionnel est CLÉ-APPARENCE :
1. **`green_points`** (`waypoint_layer.py:95`) : règle codée « danger = vert » (G>R, G>B,
   sat>0.15) — TOUTE la perception danger de l'étage waypoint passe par cette lunette ;
2. **Requêtes-couleur des slots WM** (« bouffe = rouge, eau = bleue, danger = vert »,
   `build_hazard_slot.py`) — la perception ressource entière ;
3. **`green_margin=1.0` / `tangent_margin=1.4`** : géométrie des piliers connue d'avance ;
4. ⚠️ les têtes APPRISES elles-mêmes sont contaminées EN ENTRÉE : dg1/dg2 (features de douleur̂
   AUC 0.894 et P̂mort AUC 0.839) sont des distances aux points VERTS. Labels purs (vécus),
   lunette impure.
Si l'apparence du danger change, l'étage devient aveugle SANS erreur visible. La fondation, elle,
est prête : la rétine est du RGB brut (144-d), le latent WM est général, tous les labels sont vécus.

## Le chantier (scope tranché : le DANGER d'abord)
**« Dangereux » = ce qui a précédé mes dégâts.** Apprendre une tête de saillance-danger sur la
RÉTINE BRUTE (144-d), auto-supervisée par les couples percept→dégâts vécus, puis remplacer
`green_points` par sa lecture dans l'étage waypoint (proposeur tangent + features dg1/dg2 des
têtes, ré-entraînées dé-contaminées). Les marges 1.0/1.4 doivent tomber avec (la saillance apprise
sait où le danger mord). Le volet « nourrissant = ce qui a soulagé le drive » (requêtes-couleur des
slots WM) est un chantier SÉPARÉ à licencier ensuite — ne pas tout faire d'un coup (§4).

## Ce qui existe déjà (ne pas re-découvrir, ne pas re-collecter a priori)
- **Corpus** : 10 runs instrumentés (`critic_kin_{g24as1,g24as2,g24bs1,g24bs2,spx3,spx4,judge1,
  judge2,pure1,pure2}`) : BC gzippé avec `wm.retina0` (144-d) + santé/drives PAR TICK,
  `decisions.jsonl` (feats+intr+drives), `godot.log`. 12 306 décisions, 207 morts-danger,
  chaque morsure = un couple (rétine, dégâts) daté au tick. Loaders : `train_sprint_critic.py`
  (`load_corpus`, `_open_text` gz, `pursuit_end`, `_fit_bce`, CV-4 par vie).
- **Réfs vivantes du juge** : remise-capée **45 repas / 8 morts-danger** poolés (2×24 vies,
  seeds 1+2) ; géométrie 34/11. Harnais : `scripts/judge_sprint_critic_v2.sh` (bras) +
  `diagnostics/diag_hazard_gate.py` (parse). Bruit : ±5 repas par 24-total → gates POOLÉS.
- **Config vivante** : monde v2 (`SYLVAN_HAZARD_COUNT=1 ENGULF_P=0.5 HEALTH_REGEN=0.05`), WM
  `wm_objcentric_kin_haz`, `SYLVAN_WAYPOINT=1`, `SYLVAN_WP_SPRINT_CRITIC=data/checkpoints/
  sprint_critic/sprint_best.pt` (composed_v1, juge PASS).
- **Têtes bankées** : douleur̂ v3 (0.894), P̂mort (0.839), P̂repas — toutes sur le contrat 14-d
  `sprint_inputs` ; leur ré-entraînement dé-contaminé = partie du chantier.

## Contraintes imposées par les leçons (pas des options)
1. **Comportement-préservant dans le monde ACTUEL** : dans le monde-jouet, le vert EST la vérité —
   la saillance apprise doit retrouver ce que la règle voyait, puis le juge closed-loop doit tenir
   la parité avec la réf vivante (≥40 repas ET ≤10 morts poolés). Un retrait qui coûte du forage ou
   des morts n'est pas une purification (leçon P1 : les hints étaient porteurs — 3 repas sans eux).
2. **Positions du hazard des logs = ORACLE D'ÉVALUATION SEULEMENT** (mesure, jamais entrée ni label
   d'entraînement) — les labels d'entraînement sont les dégâts VÉCUS.
3. **W=25 et l'hystérésis NE BOUGENT PAS** (préférence du corps + consistance, verdicts P2/P2-bis).
4. **Diag gratuit AVANT tout train** : compter les événements-dégâts exploitables (morsures avec
   rétine au tick), leur diversité angulaire/distance ; si le contraste manque → collecte ε d'abord.
5. Budget dur : 1 train + 1 re-train diagnostiqué par tête ; gates offline pré-enregistrés AVANT
   (falsifiables : localisation vs oracle-éval, parité des features dg reconstruites, G-consist) ;
   juge payé SEULEMENT si offline passe ; KILL précoce seed 1 < 14 repas.

## Discipline (non négociable, cf CLAUDE.md)
Critères/KILL écrits avant ; négatif = commité ; collecte SÉQUENTIELLE ; `/sylvan-kill` + 0
orphelin ; ne JAMAIS stager `godot/scripts/main.gd` ni `ui/` (vérifier `grep -c hazard_manager
godot/scripts/main.gd` ≥ 8) ; carte `architecture.json` à jour DANS LE MÊME COMMIT ; commits
Conventional anglais sans attribution IA ; README = constat au présent, zéro em-dash/emoji.

## Fichiers probablement touchés
Nouveau `python/scripts/train_danger_saliency.py` (gabarit : train_sprint_critic/_fit_bce) ;
`python/sylvan/control/waypoint_layer.py` (`green_points` → lecture saillance, opt-in défaut OFF,
bit-identique OFF) ; `python/scripts/train_waypoint_pain.py` + `train_sprint_critic.py`
(features dg dé-contaminées, re-train des têtes) ; `diagnostics/diag_saliency_*.py` (gratuits) ;
`docs/design_purete_hjepa.md` (section chantier + gates) ; carte.
