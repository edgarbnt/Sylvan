---
description: Boucle d'entraînement Sylvan (hexapode) — diagnostiquer GRATUITEMENT d'abord, lancer un run informé, mesurer le BUT, gater le coûteux. Anti-boucle.
argument-hint: "[objectif du run / hypothèse à tester]"
model: opus
---

Tu pilotes un cycle d'entraînement pour **Sylvan** (Godot + PyTorch CPU, world-model JEPA sur un
corps **hexapode**). Lis d'abord `CLAUDE.md` (règles ops + PRINCIPE N°1 anti-boucle), `ETAT_DES_LIEUX.md`
(état courant) et `memory/sylvan-locomotion-rl-knowledge.md` (LA saga moteur — ~15 runs ratés, à ne pas répéter).

## Règle d'or (PRINCIPE N°1)
**Comprendre/diagnostiquer AVANT de lancer. Pas de runs « en croisant les doigts ».** Un run = des heures.
Le *return* qui monte ne prouve RIEN (il montait pendant que le virage se dégradait, dans la saga).

## La boucle
1. **DIAGNOSTIQUER gratuitement (sans entraîner).** Quel est le goulot EXACT ? Le localiser par des tests
   cheap : mesure °/s par côté (régime propre), test foraging, sonde open-loop du WM/planner, probe physique.
   Les meilleurs acquis du projet viennent de tests gratuits, pas de runs devinés.
2. **ÉCRIRE les critères AVANT de lancer** : SUCCÈS falsifiable (mesuré au BUT : °/s symétrique, fwd-en-tournant,
   foraging) + triggers KILL (kl>0.5 soutenu, fwd_vel↘ sous seuil, chutes>10%). Les annoncer à l'owner.
3. **LANCER UN run informé** (warm-start, régime propre, **`--lr 1e-4`**). Voir le gabarit `train_hexapod_omega.sh`.
   Toujours en background, la commande python SEULE (cf CLAUDE.md). Curriculum ω = `SYLVAN_CMD_CURRIC=1` +
   `--cmd-wmax-start/end/cycles`. Symétrie = `--sym-coef --mirror-augment` + `SYLVAN_MIRROR_COMMAND=1`.
4. **SURVEILLER une fenêtre prédictive** (là où ça casse d'habitude) : si un trigger KILL saute → **tuer tôt**
   (`pkill -9 -f serve_ppo_collect` + vérifier 0 orphelin, cf CLAUDE.md). Ne pas laisser finir « au cas où ».
5. **MESURER LE BUT à convergence**, pas le return : °/s par côté + fwd-en-tournant (outil cheap), puis foraging.
6. **GATER le coûteux** : ne recollecter le WM / re-tester le foraging QUE si la base passe les critères moteur.
7. **BUDGET DUR** : un run raté = négatif INFORMATIF → STOP + escalade (planner-side, ou décision structurelle owner).
   NE PAS enchaîner un tweak sans nouvelle hypothèse falsifiable justifiée par un test gratuit.

## Diagnostic du log PPO
`grep '\[PPO\] iter' <log>` → `return / fall% / fwd_vel / kl / std / sym`. Sain = kl<0.1 stable, std qui s'annèle
doucement, fall 0%, return/score qui montent. Divergence = kl qui explose (>1) + fwd_vel qui s'effondre. Le best
checkpoint est promu sur `--best-metric stable_fwd` (jamais `return` seul).

## STOP et demander un VISUEL à l'owner quand
- un run atteint ses critères de succès (vérifier visuellement avant de banker — anti reward-hacking),
- un nouveau régime/pattern dégénéré apparaît,
- avant toute refonte profonde de reward ou décision structurelle (corps, off-policy…).
Dire quoi lancer : `bash voir_salamandre_cerveau.sh <omega>` (base) ou `bash run_forage_hex.sh` (foraging).

## Garde-fous
- Garder le cœur JEPA intact (encodeur → WM commande-space → planner). Le CPG+résidu est le moteur, pas le but.
- Dims hexapode FIXES : proprio=132, action=18 (cf CLAUDE.md). Ne pas les changer sans tout synchroniser.
- Rapporter fidèlement : si ça plafonne/diverge, le dire avec les chiffres.
