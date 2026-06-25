# CLAUDE.md — Sylvan (règles ops, chargées chaque session)

> Projet : **ALife émergente dans un world-model type JEPA** (Godot + PyTorch CPU). L'entité doit
> décider elle-même (faim → chercher → aller vers la bouffe → survivre) via planification dans un
> WM appris. La locomotion est un **prérequis**, pas le but.
>
> **Lire en premier** : `ETAT_DES_LIEUX.md` (handoff/état courant) puis la mémoire auto
> (`memory/MEMORY.md` + `sylvan-rearchitecture.md` = journal détaillé,
> `sylvan-locomotion-rl-knowledge.md` = LA saga moteur, à lire avant tout retrain virage).

## ⭐ PRINCIPE DE TRAVAIL N°1 — COMPRENDRE AVANT DE LANCER (anti-boucle)
**Préférer TOUJOURS prendre le temps de comprendre et d'investiguer ce qui ne va pas, plutôt que
de perdre des heures à enchaîner des entraînements en croisant les doigts.** Le projet a déjà brûlé
~15 runs moteur ratés (résidu3→9, turn1→3, SAC…) parce qu'on lançait en espérant. Un run = des heures.

Discipline obligatoire avant TOUT entraînement long :
1. **Diagnostiquer la cause-racine d'abord, avec des tests GRATUITS** (sans entraîner). Les meilleurs
   moments du projet l'ont fait : « 3 tests gratuits » du foraging, YAW_TORQUE→1000°/s, pics-pattes→180°/s,
   mesure °/s par côté. Un test gratuit qui localise le goulot vaut mieux qu'un run qui le devine.
2. **Écrire des critères de SUCCÈS et de KILL falsifiables AVANT de lancer.** Pas « on verra si ça
   s'améliore ». Le *return* qui monte ne prouve RIEN (dans la saga il montait pendant que le virage se
   dégradait) → mesurer le BUT directement (°/s par côté, fwd-en-tournant, foraging), pas le proxy.
3. **Gater le cher derrière le pas-cher** : ne payer l'étape coûteuse (recollecte WM, foraging) QUE si
   l'étape précédente passe une mesure cheap pré-enregistrée.
4. **Budget dur** : un run raté = un négatif INFORMATIF → STOP + escalade. NE PAS enchaîner un tweak
   sans une nouvelle hypothèse falsifiable justifiée par un test gratuit.
5. **Tuer tôt** un run qui diverge (kl↗ + fwd_vel↘) — ne pas le laisser finir « au cas où ».

## ⭐ PRINCIPE DE TRAVAIL N°2 — NE PAS MASQUER LE VRAI PROBLÈME (pas de fausse solution)
**Ne JAMAIS « régler » un symptôme en relâchant le critère de succès ou en gonflant une tolérance pour
cacher une lacune de capacité.** Exemple concret (2026-06-18) : l'entité cale à ~1 m de la bouffe sans
closer → proposer d'**agrandir le rayon de capture** (la « bouche ») = fausse solution : ça maquille
qu'elle ne sait pas s'approcher PRÉCISÉMENT, ça ne le corrige pas. Déplacer les poteaux ≠ marquer.
Règles :
1. **Le problème, c'est la CAPACITÉ manquante, pas la métrique.** Si l'entité ne sait pas faire X
   précisément, l'objectif est de COMPRENDRE et CORRIGER X — pas d'élargir le seuil pour que « ça passe ».
2. **Garder le critère HONNÊTE et fixe.** On mesure toujours la vraie chose (approche précise, °/s par
   côté, fwd-en-tournant). Un réglage d'environnement qui rend le succès « plus facile » sans améliorer
   la capacité est un AUTODÉCEPTION — interdit comme « solution », tolérable seulement comme *sonde* explicite.
3. **Ne pas conclure « c'est le plafond X » pour pouvoir s'arrêter, surtout si une autre mesure le CONTREDIT.**
   (Ex. : « c'est le plafond moteur » alors qu'en A→B il closait à 0.04 m → le moteur SAIT closer → la vraie
   cause est ailleurs et reste à trouver.) Une conclusion qui arrange = suspecte → re-vérifier contre les données.
4. **Si le vrai fix est cher/hors-scope, le DIRE explicitement** (« corriger ça = retrain moteur, hors scope »),
   ne pas le déguiser en réglage anodin. Le owner tranche en connaissance de cause.

## ⭐ PRINCIPE DE TRAVAIL N°3 — SÉPARER LE SUBSTRAT DES PULSIONS (ne pas câbler une tâche dans le WM)
**Le but est une ARCHITECTURE GÉNÉRALE (entité intelligente), PAS faire passer la ressource du moment.** Prendre
du recul sur la bouffe : si l'entité galère sur la bouffe à cause de l'archi, elle galèrera PARTOUT. Ne JAMAIS
changer l'archi juste pour qu'« une ressource passe ». Couches à NE PAS confondre :
- **WM = SUBSTRAT LENT** = perception + dynamique, GÉNÉRAL, entraîné RAREMENT (et SEULEMENT si NOUVELLE
  perception/un nouveau sens). Son latent doit être RICHE + à imagination FIDÈLE → il porte TOUTE la perception
  (rouge=bouffe, bleu=eau, obstacles…) sans rien de spécifique-ressource.
- **TÊTES de VALEUR = COUCHE RAPIDE** = une par pulsion, au-dessus du latent, ré-entraînées la nuit (cycle de vie)
  sur l'expérience vécue. Ajouter une pulsion dont la perception est déjà dans le latent = **juste une tête, WM INTACT**.
- **DRIVE** (avoir faim/soif) = propriété du CORPS, définie une fois à la conception (comme l'évolution câble les
  pulsions) ; le LIEN « perception→soulage le drive » s'APPREND (la nuit, idéalement auto-supervisé sur le drive vécu).

🚨 **SIGNAL D'ALERTE = RACCOURCI INTERDIT** : si un fix exige de **ré-entraîner le WM pour UNE pulsion précise**
(ex. `--w-food` qui force la bouffe DANS le latent), c'est le MAUVAIS ÉTAGE → ça fabrique une archi qui se refond à
chaque besoin (axée-ressource), exactement la fausse solution du §2. Le bon réflexe : enrichir le WM **une fois**
(latent riche+fidèle), puis chaque pulsion = une tête. Avant tout `--w-<ressource>` sur le WM, se demander : « est-ce
que le latent ne porte pas DÉJÀ cette perception ? » — si oui, le forçage est un raccourci à refuser.

## ⭐ PRINCIPE DE TRAVAIL N°4 — AVANCER ÉTAPE PAR ÉTAPE, CHAQUE ÉTAPE SOLIDE AVANT LA SUIVANTE
**Un grand principe du projet est d'avancer étape par étape. Avant de passer à l'étape suivante, il faut être sûr que
l'étape précédente soit ROBUSTE, FONCTIONNELLE, PÉRENNE, et surtout JEPA PUR.** Sinon les briques posées plus tard
tombent / deviennent impossibles (mémoire, curiosité, hiérarchie reposent sur un substrat sain). Conséquences pratiques :
1. Ne PAS empiler une nouvelle couche sur un étage non-pur ou non-validé (ex. : ne pas bâtir la mémoire sur un slot codé-main).
2. « Validé » = falsifiable : robuste (multi-seed/conditions), fonctionnel (mesuré au BUT, ≥ baseline — JAMAIS échanger
   robustesse contre pureté), pérenne (module propre/sauvegardé/commité), pur (zéro oracle/hack dans la boucle).
3. Gater le cher derrière un test gratuit décisif AVANT de construire (cf §1), et ne promouvoir que si pur ET ≥ baseline.

## Corps actuel = HEXAPODE (depuis 2026-06-17)
- **Dims** : proprio=**132**, action=**18**, obs policy=**144** (132+vision12), obs WM=**145** (132+radar12+énergie1).
- Ne JAMAIS changer ces dims sans synchroniser `constants.py`, `observation_builder.gd`, `sylvan_agent.gd`, `symmetry.py`.
- **Checkpoints vivants** : base motrice `data/checkpoints/hexapod_v2/policy_best.pt` ;
  **FORAGER VIVANT = SLOT-PLANNER PUR** : WM **PURIFIÉ `data/checkpoints/wm_rich_fidele_sym_jepa/wm_best.pt`**
  (promu 2026-06-25 : reconstruction droppée `--w-proprio/radar 0` → JEPA principe n°1 « prédire la repr., pas
  reconstruire l'entrée » ; warm-start de `wm_rich_fidele_sym`, recette identique sinon ; eff_rank 21>13, transport slot
  +0.65 préservé, engagement 15/16>13/16, foraging survie méd 1100≥1040 = parité gratuite ; trainer `train_wm_jepa_pur.sh`,
  ancien non-purifié = `wm_rich_fidele_sym`) + **SLOT object-centric AUTO-SUPERVISÉ** `data/checkpoints/slot_head/slot_best.pt` (perception 100%
  label-free, ZÉRO oracle de position) → `bash run_forage_purslot.sh` (serveur `--slot-head`). Le slot = coordonnée ego
  apprise SANS label (consistance de transport Rot(+Δyaw) + saillance perceptuelle ; module `slot_head.py`, trainer
  `train_slot_head.py`), transportée par la displacement-head. Engage l'arrière (re-gate 13/16 ≈ retina_head 14/16, arrière
  2/4 préservé), foraging méd 915 ≈ retina_head 860, précis (bearing 4.9° < retina_head 8.4°). NB **PAS « full-latent »** :
  slot = coordonnée explicite apprise (le pur-latent-valeur `plan_latent` était lossy, perdait l'objet). Étape précédente
  `retina_head` (SUPERVISÉ-oracle) = superseded mais valide (`run_forage_retina.sh`). Secours ultime = oracle
  `wm_command_hex_v2` + `run_forage_hex.sh`. Design/preuves : `docs/design_wm_factorise.md` + `docs/plan_wm_objectcentric_pur.md`,
  gates `diag_nav_ab_purslot.sh` (engagement) + `forage_ab_purslot.sh` (survie). **Coût planner = `-min_dist` PUR
  (heading_weight=0 par défaut depuis 2026-06-25)** : le slot précis (4.9°) rend le « how-to-hint » `heading_weight`
  INUTILE — A/B `forage_ab_hw.sh` : hw=0 ≥ hw=2 (engagement 13/16 arrière 3/4≥2/4 ; foraging méd 1040 ≥ 860). Un hack
  flaggé de moins dans le chemin vivant. (Niveau 2 possible = remplacer `-min_dist` par un CRITIQUE APPRIS sur le slot.)
- **Régime PROPRE de l'hexapode** (TOUJOURS le servir/collecter ainsi, sinon il dérive) :
  `SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 SYLVAN_CPG_PERIOD=0.5`.
  Plage vx propre ~0.5-0.75 (en dessous il dérive). Tourne ~25-50°/s (le « mur 15°/s » est cassé).

## Environnement / lancement
- venv : **`env_pytorch_3.12/bin/python`** (à la racine). CPU OBLIGATOIRE (GPU AMD = HIP crash sur torch/ROCm).
- Toujours depuis la **racine** avec **`PYTHONPATH=python`** et **`GODOT_BIN="$(pwd)/tools/godot/godot"`**.
- Godot headless : `./tools/godot/godot --path godot --headless`.

## RÈGLES qui ont déjà coûté cher (respecter absolument)
1. **PPO : `--lr 1e-4`** (le défaut de `train_ppo` est 3e-4 = INSTABLE → divergence, kl qui explose). Toujours passer `--lr 1e-4`.
2. **Tuer un entraînement** : `pkill -9 -f train_ppo` NE suffit PAS — les **8 workers `serve_ppo_collect` survivent**
   (orphelins busy-spinning → thrash la box). Faire : `pkill -9 -f serve_ppo_collect ; pkill -9 -f sylvan_pool ;
   pkill -9 -f 'godot --path godot'` PUIS **vérifier** `pgrep -af serve_ppo_collect` = 0, sinon `kill -9 <PID>`.
   WM : `pkill -9 -f train_wm_command` (workers DataLoader).
3. **Lancer un train en background = la commande python SEULE** (un préambule `kill`/`pkill` dans la commande
   backgroundée la fait exit1). Tuer les orphelins AVANT, dans une commande séparée.
4. **Piège shell** : une commande avec env multi-lignes a besoin de `\` en fin de ligne, sinon chaque ligne est
   traitée comme une commande vide et l'env n'atteint PAS le python. → **préférer un script `.sh` avec `export`**.
5. **JAMAIS de glob large avec `rm`** (`*cr*`, `*tmp*`…) : un `rm -rf '*cr*'` a supprimé TOUT le code le 2026-06-16
   (`scripts`/`critic` contiennent "cr"). Toujours scoper (`rm -rf data/replay_buffer/foo*`). Un hook bloque les cas dangereux.
6. **Ne PAS juger un entraînement avant convergence** (WM long-horizon mauvais à mi-parcours puis se corrige).
   Mais **tuer tôt si divergence claire** : fwd_vel qui s'effondre + kl > ~1 qui monte.

## Commandes canoniques (scripts racine — les utiliser/les adapter)
- **Voir la base motrice** : `bash voir_salamandre_cerveau.sh 0.7` (sert hexapod_v2 ; arg = omega).
- **Test foraging headless** : `bash run_forage_hex.sh [eat_radius=1.0] [horizon=80] [episodes=12]` (planner+WM v2+résidu, survie via `[Godot] Episode|Step|Energy`).
- **Re-collecter le WM** (régime propre) : `bash collect_wm_hex_v2.sh <run-prefix> <episodes> <seed>`.
- **Entraîner le WM** : `PYTHONPATH=python env_pytorch_3.12/bin/python -m scripts.train_wm_command --runs data/replay_buffer/wm_hex_v2_a data/replay_buffer/wm_hex_v2_b --out data/checkpoints/<name> --epochs 20` ; éval : `scripts.eval_wm_command --checkpoint .../wm_best.pt --horizons 50 80 100 150`.
- **Entraîner le résidu PPO** : `bash train_hexapod_omega.sh` (gabarit : warm-start hexapod_v2, régime propre, curriculum ω, symétrie, **lr 1e-4**). Curriculum ω = `SYLVAN_CMD_CURRIC=1` + `--cmd-wmax-start/end/cycles` (PAS `SAMPLE_CMD`). Symétrie command-mode = `--sym-coef` + `--mirror-augment` + `SYLVAN_MIRROR_COMMAND=1`.

## Pipeline (3 couches)
JEPA (planner MPC commande-space, foraging) → sur RÉSIDU PPO borné (équilibre/perf) → sur CPG codé-main (marche+virage par construction). Le WM/planner planifient en (vx,ω) ; "bouffe" vit SEULEMENT dans le coût du planner (agnosticité, BLUEPRINT §14).

## ⭐ CARTE VIVANTE DE L'ARCHI (Archi-HUD) — LA TENIR À JOUR (obligatoire)
`tools/archi_hud/architecture.json` est la **source de vérité** de l'état de l'archi (modules, état
pur/partiel/échafaudage/manquant, rôle JEPA, ce qu'ils sont/apportent, ancre code, focus). Visualisation :
`bash voir_archi.sh` (carte cliquable ; s'anime en live pendant un run lancé via `bash run_forage_hud.sh`).
**RÈGLE : dès qu'une décision change la CONCEPTION, l'IMPLÉMENTATION ou les décisions générales d'un module
(ex. un étage passe pur↔échafaudage, une impureté est résorbée, un module manquant est construit, le focus
change, une preuve/limite évolue) → METTRE À JOUR `architecture.json` DANS LE MÊME COMMIT** (champs `etat`,
`etat_detail`, `role`/`comment`/`apporte`, `limites`, `preuves`, `code`, `focus`). Le validateur
`env_pytorch_3.12/bin/python tools/archi_hud/validate_architecture.py` garde la carte honnête (états/clés/ancres) ;
`voir_archi.sh` refuse de servir une carte invalide. Ne JAMAIS laisser la carte mentir (cf §2 ne-pas-masquer).
