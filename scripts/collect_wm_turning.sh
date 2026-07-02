#!/bin/zsh
# Collecte WM RÉTINE riche en VIRAGES-ACQUISITION (2026-06-21) — pour combler le mur d'engagement :
# le WM-rétine n'a JAMAIS vu de babbling (les données retina_eat/forage venaient du planner-coordonnées qui
# fait TOUJOURS face à la bouffe → la dynamique « je tourne → la cible entre dans le champ » est hors-distribution).
# Ici : BABBLING (commandes (vx,ω) par morceaux, INDÉPENDANTES de la bouffe) + bouffe à 360° + rétine loggée
# → l'agent tourne pendant que des cibles sont à tous les azimuts → couverture du virage-acquisition.
# RÉGIME MOTEUR PROPRE PRÉSERVÉ : vx 0.55-0.75, |ω|≤0.6 (la plage que le planner utilise — PAS plus, sinon on
# collecte un régime que le corps n'exécute pas, cf leçon hexapode). Le gain vient du babbling, pas d'un ω violent.
# Usage: zsh collect_wm_turning.sh <run-prefix> <episodes> <seed>
cd /home/edgarbrunet/Documents/PERSO/SylvanV1
PREFIX=${1:-wm_turn_retina_smoke}
EPS=${2:-2}
SEED=${3:-23}
pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
export GODOT_BIN="$(pwd)/tools/godot/godot"
export PYTHONPATH=python
export SYLVAN_WM_COLLECT=1 SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0
export SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 SYLVAN_CPG_PERIOD=0.5
export SYLVAN_WM_VX_MIN=0.55 SYLVAN_WM_VX_MAX=0.75 SYLVAN_WM_WMAX=0.6   # régime PROPRE (idem planner)
export SYLVAN_FOOD_COUNT=${FC:-8}                                       # bouffe à 360° → cibles à tous azimuts
export SYLVAN_EAT_RADIUS=1.0
./env_pytorch_3.12/bin/python -m scripts.collect_wm_data \
  --checkpoint data/checkpoints/hexapod_v2/policy_best.pt \
  --run-prefix "$PREFIX" --episodes "$EPS" --max-steps 600 --seed "$SEED"
