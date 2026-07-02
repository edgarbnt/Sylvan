#!/bin/zsh
# Voir la SALAMANDRE bouger (CPG pur, AUCUN serveur requis — pas encore entraînée).
# Usage :  bash scripts/voir_salamandre.sh 0.6    (1er arg = omega : 0 = tout droit, 0.6/-0.6 = tourne)
#          bash scripts/voir_salamandre.sh 0       (marche droite)
# Une FENÊTRE Godot s'ouvre. Ferme-la pour arrêter. Le virage se fait par flexion de la colonne.
cd /home/edgarbrunet/Documents/PERSO/SylvanV1
OM=${1:-0.6}
SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.0 SYLVAN_CMD_VX=0.3 SYLVAN_CMD_OMEGA=$OM SYLVAN_CPG_PERIOD=0.5 \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=20 SYLVAN_MAX_EPISODE_STEPS=600 SYLVAN_SEED=1 SYLVAN_DISABLE_HOMEOSTASIS=1 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 \
SYLVAN_RUN_DIR=data/replay_buffer/view \
./tools/godot/godot --path godot
