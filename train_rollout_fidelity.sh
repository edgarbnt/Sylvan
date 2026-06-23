#!/bin/zsh
# CHANTIER ARCHI (owner 2026-06-19) — RÉPARER L'IMAGINATION du WM SANS sacrifier la richesse JEPA.
# Diagnostic prouvé : VICReg enrichit le latent (eff_rank↑, bon pour LIRE) mais le rêve open-loop dérive de la
# trajectoire réelle dès t=1 (cos 0.59) car le prédicteur n'est entraîné qu'à 1 pas (exposure-bias en espace riche).
# FIX = nouvelle perte --w-rollout : aligne le rollout open-loop sur la trajectoire latente RÉELLE (teacher-forced,
# stop-grad) → le prédicteur apprend à imaginer fidèlement sur l'horizon. VICReg GARDÉ (LeCun a raison ; on corrige
# NOTRE archi, pas VICReg). But large = un WM qui imagine de façon stable ET riche → meilleur pour TOUTE planif.
#
# Setup : warm-start retina_jepa_v2 (riche-MAIS-cassé : eff_rank 14, rêve 0.18 m) → on tente de le RÉPARER.
# Recette JEPA identique (cosine + VICReg 1/1) + --w-rollout. Données babbling propre (dynamique pure).
#
# CRITÈRES PRÉ-ÉCRITS (CLAUDE.md §1) — mesurés par diag_dream_disp (path + eff_rank + fidélité rêve↔réel) :
#   SUCCÈS (chantier validé) : eff_rank reste RICHE (≥ ~10) ET path ≥ ~0.45 m ET fidélité rêve↔réel@40 ≥ ~0.85
#     → richesse + imagination fidèle COEXISTENT → "JEPA bien fait", compromis résolu à la racine.
#   ÉCHEC-A (w-rollout écrase la richesse) : eff_rank chute < ~6 → le rollout-loss tire le latent vers le régime
#     pauvre → baisser w_rollout / repenser (le but n'est PAS de re-appauvrir).
#   ÉCHEC-B (predicteur figé non réparable en warm-start) : path reste < 0.35 m → relancer FROM-SCRATCH
#     (cosine+VICReg+w-rollout, 20 ep) au lieu du warm-start.
#
# Lancer backgroundé (orphelins tués À PART avant) : nohup zsh train_rollout_fidelity.sh > /tmp/rollfix.log 2>&1 &
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
export PYTHONPATH=python
OUT=data/checkpoints/wm_rollout_fix

echo "############### TRAIN : warm jepa_v2 + cosine + VICReg(1,1) + --w-rollout 3.0 (fidélité du rêve) ###############"
SYLVAN_WM_USE_RETINA=1 \
./env_pytorch_3.12/bin/python -m scripts.train_wm_command \
  --runs data/replay_buffer/retina_wm_a data/replay_buffer/retina_wm_b \
  --out $OUT \
  --init-from data/checkpoints/wm_command_hex_retina_jepa_v2/wm_best.pt \
  --epochs 10 --stride 8 --batch-size 48 --lr 1e-4 \
  --latent-loss cosine --vicreg-var 1.0 --vicreg-cov 1.0 \
  --w-rollout 3.0
echo "train exit=$?"

echo "############### VERDICT : WM RÉPARÉ (chantier) ###############"
SYLVAN_WM_USE_RETINA=1 ./env_pytorch_3.12/bin/python diag_dream_disp.py $OUT/wm_best.pt 1 2>&1 | grep -v -i warning
echo "############### BASELINE (retina_jepa_v2, riche-MAIS-cassé) pour comparaison ###############"
SYLVAN_WM_USE_RETINA=1 ./env_pytorch_3.12/bin/python diag_dream_disp.py data/checkpoints/wm_command_hex_retina_jepa_v2/wm_best.pt 1 2>&1 | grep -v -i warning
echo "ROLLFIX_DONE"
