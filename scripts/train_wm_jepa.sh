#!/bin/zsh
# JEPA-ification ÉTAPE 1 — DÉ-COLLAPSE (2026-06-18). Diag gratuit (diagnostics/diag_jepa.py) : la représentation
# est effondrée (eff_rank latent ~6/128, cible encodeur ~1.5/128) → la perte `latent` est VACANTE
# (trivialement satisfaite par une rep quasi-constante) → on n'est PAS en JEPA, c'est un Dreamer.
# Ce run NE déplace PAS encore les poids vers le latent (ça = étape 2). Il ne fait QU'UNE chose :
# casser l'effondrement via l'anti-collapse DÉJÀ codé (VICReg var+cov sur les latents RSSM + perte
# latente COSINE anti-shrink). Mêmes données que hex_v2, from-scratch, poids de perte par défaut.
# Sortie = NOUVEAU dossier (hex_v2 validé reste intact). Critères SUCCÈS/KILL : docs/jepa_anticollapse_criteria.md
# Lancer (backgroundé) = cette commande SEULE (tuer les orphelins AVANT, séparément) :
#   pkill -9 -f train_wm_command ; pgrep -af train_wm_command   # vérifier = 0
#   nohup zsh train_wm_jepa.sh > /tmp/wm_jepa.log 2>&1 &
# Suivre eff_rank live : grep eff_rank /tmp/wm_jepa.log
cd /home/edgarbrunet/Documents/PERSO/SylvanV1
export PYTHONPATH=python
./env_pytorch_3.12/bin/python -m scripts.train_wm_command \
  --runs data/replay_buffer/wm_hex_v2_a data/replay_buffer/wm_hex_v2_b \
  --out data/checkpoints/wm_command_hex_v3_jepa \
  --epochs 20 --lr 1e-4 \
  --latent-loss cosine \
  --vicreg-var 1.0 --vicreg-cov 1.0 --vicreg-gamma 1.0
