#!/bin/zsh
# Collecte WM RÉTINE SCRIPTÉE en ACQUISITIONS (clé de voûte 3b, 2026-06-23).
# Diffère du babbling raté (collect_wm_turning.sh, retina_wm_a = 1.0 acquisition/ép) : ici ROTATION SOUTENUE
# d'un seul signe par longs blocs (SYLVAN_WM_TURN_SCRIPT=1) → l'agent ORBITE (arc R=vx/ω≈1 m) → son cap balaye
# 360° en continu → chaque cible (placées à 360°, FC=8) traverse derrière→devant à chaque tour = l'ÉVÉNEMENT
# d'acquisition que le rêve doit apprendre. Le signe alterne par bloc (symétrie G/D).
# RÉGIME MOTEUR PROPRE : vx fixé au milieu de [0.55,0.75], |ω|=0.6 (plage que le corps exécute proprement).
# Usage: zsh collect_wm_turn_script.sh <run-prefix> <episodes> <seed>
cd /home/edgarbrunet/Documents/PERSO/SylvanV1
PREFIX=${1:-wm_turn_script_smoke}
EPS=${2:-2}
SEED=${3:-23}
pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
export GODOT_BIN="$(pwd)/tools/godot/godot"
export PYTHONPATH=python
export SYLVAN_WM_COLLECT=1 SYLVAN_WM_TURN_SCRIPT=1
export SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0
export SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 SYLVAN_CPG_PERIOD=0.5
export SYLVAN_WM_VX_MIN=0.55 SYLVAN_WM_VX_MAX=0.75 SYLVAN_WM_WMAX=0.6   # régime PROPRE (idem planner)
export SYLVAN_FOOD_COUNT=${FC:-8}                                       # bouffe à 360° → cibles à tous azimuts
export SYLVAN_EAT_RADIUS=1.0
./env_pytorch_3.12/bin/python -m scripts.collect_wm_data \
  --checkpoint data/checkpoints/hexapod_v2/policy_best.pt \
  --run-prefix "$PREFIX" --episodes "$EPS" --max-steps 600 --seed "$SEED"
