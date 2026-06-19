#!/bin/zsh
# Phase 5a A->B NAVIGATION PROBE (free diagnostic, NO training).
# Single FIXED pellet at a controlled azimuth + fixed distance, homeostasis OFF
# (DISABLE_HOMEOSTASIS=1 -> pellet never eaten/respawned, no starvation-death) so the
# target stays pinned and we measure the CLOSEST APPROACH per azimuth. Goal: quantify
# WHICH azimuths the planner fails to ENGAGE (the bimodal A->B failure tail).
# Planner = wm_command_hex_v2 + hexapod_v2, horizon 160 (best A->B), clean hexapod regime.
# Usage: bash diag_nav_ab.sh [dist=4.0] [eps_per_angle=2] [max_steps=900]
set +e
DIST=${1:-4.0}
NEP=${2:-2}
MS=${3:-900}
HW=${4:-1.0}   # planner heading-alignment weight (A→B engagement); 0 = original cost
ANGLES=(0 45 90 135 180 225 270 315)
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1
cd "$ROOT"
pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
rm -f /tmp/nav_ab_*.log
echo "heading_weight=$HW"

WM=${WM:-data/checkpoints/wm_command_hex_v2/wm_best.pt}   # override: WM=... bash diag_nav_ab.sh ...
echo "WM=$WM"
SYLVAN_PLANNER_HEADING_W=$HW \
PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
  --wm "$WM" \
  --residual data/checkpoints/hexapod_v2/policy_best.pt \
  --host 127.0.0.1 --port 6052 --horizon 160 --replan-every 10 > /tmp/nav_ab_planner.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ':6052' && break; sleep 1; done

for A in $ANGLES; do
  echo ">>> azimuth=${A}deg dist=${DIST}m"
  SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
  SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 \
  SYLVAN_DISABLE_HOMEOSTASIS=1 \
  SYLVAN_FOOD_COUNT=1 SYLVAN_FOOD_ANGLE_DEG=$A SYLVAN_FOOD_MIN_RADIUS=$DIST SYLVAN_FOOD_SPAWN_RADIUS=$DIST \
  SYLVAN_EAT_RADIUS=0.5 \
  SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$NEP SYLVAN_MAX_EPISODE_STEPS=$MS SYLVAN_SEED=1 \
  SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=6052 \
  SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
  SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 \
  SYLVAN_RUN_DIR=data/replay_buffer/nav_ab \
  ./tools/godot/godot --path godot --headless > /tmp/nav_ab_${A}.log 2>&1
done
kill -9 $SRV 2>/dev/null
echo "DONE dist=${DIST} angles=${ANGLES} -> /tmp/nav_ab_*.log"
