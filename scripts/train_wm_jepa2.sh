#!/bin/zsh
# JEPA-ification ÉTAPE 2 — SHIFT RECONSTRUCTION → LATENT (2026-06-18).
# Étape 1 (train_wm_jepa.sh) a dé-collapsé la rep (eff_rank 6→26) en gardant l'anti-collapse ON.
# Étape 2 = déplacer le VRAI travail prédictif vers la voie latente : on DROP la reconstruction
# d'entrée (proprio + radar = la voie générative que LeCun rejette) et on MONTE le poids latent.
# On GARDE en ancres les lectures ABSTRAITES dont le planner a besoin (displacement, énergie, done)
# — ce ne sont PAS de la reconstruction d'entrée. On GARDE l'anti-collapse (cosine + VICReg), sinon
# le drop de reconstruction ré-effondre la rep. Mêmes données, from-scratch, → NOUVEAU dossier.
# C'est ICI que Sylvan devient FONCTIONNELLEMENT JEPA (prédire SUR le latent, plus reconstruire).
# Critères : docs/jepa_step2_criteria.md
# Lancer (orphelins tués AVANT, séparément) :
#   pkill -9 -f train_wm_command ; nohup zsh train_wm_jepa2.sh > /tmp/wm_jepa2.log 2>&1 &
cd /home/edgarbrunet/Documents/PERSO/SylvanV1
export PYTHONPATH=python
./env_pytorch_3.12/bin/python -m scripts.train_wm_command \
  --runs data/replay_buffer/wm_hex_v2_a data/replay_buffer/wm_hex_v2_b \
  --out data/checkpoints/wm_command_hex_v3_jepa2 \
  --epochs 20 --lr 1e-4 \
  --latent-loss cosine \
  --vicreg-var 1.0 --vicreg-cov 1.0 --vicreg-gamma 1.0 \
  --w-radar 0.0 --w-proprio 0.0 --w-latent 5.0
