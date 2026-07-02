# Mode-1 — prompt de démarrage (à coller dans une nouvelle session)

> Handoff écrit le 2026-06-26 à la fin de la session « tester la vie ». Colle le bloc ci-dessous
> (idéalement précédé de `/superpowers:brainstorming` pour cadrer le design avant de coder).

---

Projet Sylvan (ALife émergente dans un world-model JEPA, Godot + PyTorch CPU, hexapode).

**LIS D'ABORD, dans l'ordre** : `ETAT_DES_LIEUX.md`, puis `CLAUDE.md` (règles ops + les 4 principes de travail),
puis la mémoire auto `memory/MEMORY.md` et **surtout `memory/sylvan-second-drive-arbitration.md`** (fil actif :
tout le contexte multi-pulsions + le mapping complet de la voie apprise + la décision Mode-1). Regarde aussi la
carte vivante : `bash voir_archi.sh` (focus = `drives` ; nœuds `cout_planner`/`drives`/`critique_appris`/`world_model`).
**Ne lance RIEN avant d'avoir compris l'état.**

**OÙ ON EN EST (2026-06-26) :**
- Le corps (CPG hexapode + résidu PPO `hexapod_v2`), la perception (slot label-free DANS le WM `wm_objcentric_s1`)
  et le planner JEPA (MPC = recherche en espace commande (vx,ω) sur WM gelé) sont **vivants** : la créature
  perçoit → planifie → navigue → mange.
- **MONO-pulsion (faim) = RÉSOLU** (survie 3000/3000, 10/12).
- **MULTI-pulsions (faim+soif) = amélioré + banké** (`survival_weight=300` = foresight de survie *designed*,
  validé multi-seed 3/3, ~+15 %) **MAIS encore myope** (meurt ~la moitié du temps) → le coût **DESIGNED plafonne**.
- La **décision** est encore **codée-main** (le coût du planner). Le but = qu'elle l'**APPRENNE**.
- La voie apprise **incrémentale** a été **entièrement cartographiée par 5 gates gratuits** et est **BLOQUÉE par
  2 murs** : (1) **OFF-POLICY** — la politique myope ne mange jamais quand elle a faim → pas de signal dans le vécu ;
  (2) **PERCEPTION COURTE-PORTÉE imprécise du WM** — la bouffe au contact = 14 % capté vs 46 % en perception parfaite.
  (chiffres + détails dans `sylvan-second-drive-arbitration.md`.)

**LE CHANTIER = MODE-1** : une **POLITIQUE APPRISE** (RL + exploration) qui apprend à **gérer plusieurs pulsions et
survivre**, end-to-end. C'est la **seule voie robuste qui contourne LES DEUX murs** : l'exploration règle l'off-policy
(elle ESSAIE d'aller manger même quand le coût actuel ne l'y pousse pas), et l'end-to-end apprend **sa propre
représentation** (plus besoin d'extraire une eat-dynamics explicite d'une perception imprécise). C'est le vrai
« elle se gère elle-même », explicitement déféré « en dernier » dans le projet → on le démarre **maintenant, délibérément**.

**CE QU'IL FAUT FAIRE (dans l'ordre, discipline §1 OBLIGATOIRE) :**
1. **BRAINSTORM + DESIGN d'abord** (skill brainstorming). Choix à trancher avec l'owner :
   - **Action** : la politique sort-elle des commandes (vx,ω) — même espace que le planner, réutilise résidu+CPG —
     ou autre chose ?
   - **Observation** : proprio + rétine/slot + niveaux de drive (énergie, soif) ?
   - **Récompense** : survie/homéostasie (rester en vie, énergie+soif > 0 ; drain « de vie » ~0.05). Sparse ou shaped ?
   - **Model-free** (PPO sur la politique) **vs model-based** (Dreamer-like sur le WM) ?
   - **Rapport au planner-MPC existant** : remplacement ? **warm-start par behavioral-cloning du planner** (ne pas
     repartir de zéro) ? planner en fallback ?
   - **Exploration** : comment GARANTIR qu'elle essaie d'aller manger (le point qui débloque l'off-policy) ?
2. **Écrire des critères de SUCCÈS et de KILL falsifiables AVANT tout entraînement.**
3. **UN GATE GRATUIT décisif de faisabilité AVANT tout run long.** Idées : un BC warm-start du planner donne-t-il une
   politique qui survit ≥ baseline ? un mini-run RL court montre-t-il un gradient de survie ? la perception
   courte-portée bloque-t-elle même une politique apprise (tester le closing sur la bouffe) ?
4. **SEULEMENT si le gate passe** → entraînement, mesuré au BUT (survie multi-pulsions ; **à battre : médiane ~2300**),
   gaté derrière le pas-cher.

**DISCIPLINE (les leçons qui ont payé toute la dernière session)** : diagnostiquer GRATUITEMENT d'abord, critères
falsifiables avant de lancer, gater le cher derrière le pas-cher, un run raté = négatif informatif → STOP + escalade
(pas d'enchaînement de tweaks à l'aveugle). Tenir `architecture.json` à jour (nœud `critique_appris` ou un nouveau
nœud `mode_1`). Ne rien promouvoir sans gate closed-loop ≥ baseline.

**OPS** : venv `env_pytorch_3.12/bin/python`, **CPU OBLIGATOIRE**, depuis la racine avec `PYTHONPATH=python` et
`GODOT_BIN`. PPO **`--lr 1e-4`** (le défaut 3e-4 diverge). Tuer un train : `pkill -9 -f serve_ppo_collect` +
`pkill -9 -f train_ppo` + `pkill -9 -f 'godot --path godot'` PUIS **vérifier** 0 restant. Lancer un train en
background = la commande python SEULE (un préambule kill la fait exit1). Régime propre hexapode = `SYLVAN_CPG=1
SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 SYLVAN_CPG_PERIOD=0.5`.
Scripts survie réutilisables : `baseline_survie_vie.sh` (mono), `baseline_multidrive_slot.sh` (multi ; `SEED` +
`SYLVAN_PLANNER_SURVIVAL_W` réglables). Godot tourne en arrière-plan via `run_in_background` (1 godot + 1 serveur, pas
le 8× de la collecte PPO).

**Commence par** : me proposer le **design Mode-1** (brainstorm), puis **UN gate gratuit décisif de faisabilité AVANT
tout entraînement**. Pose-moi tes questions si un choix m'appartient.
