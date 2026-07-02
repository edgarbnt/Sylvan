#!/bin/zsh
# Mesure du virage open-loop : résidu servi (6041) + commande FIXE (vx=0.4, omega=arg2), homéostasie OFF.
# Usage: bash scripts/measure_turn.sh <yawlat> [omega] [turnk] [turnamp]   → log /tmp/turn_y<yawlat>_o<omega>.log
cd /home/edgarbrunet/Documents/PERSO/SylvanV1
YL=${1:-0.0}; OM=${2:-0.6}; TK=${3:-0.6}; TA=${4:-0.0}
LOG=/tmp/turn_y${YL}_o${OM}.log
SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_CMD_VX=0.4 SYLVAN_CMD_OMEGA=$OM SYLVAN_CPG_PERIOD=0.5 \
SYLVAN_CPG_TURNK=$TK SYLVAN_CPG_TURNAMP=$TA SYLVAN_CPG_YAWLAT=$YL \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=5 SYLVAN_MAX_EPISODE_STEPS=300 SYLVAN_SEED=1 SYLVAN_DISABLE_HOMEOSTASIS=1 \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=6041 \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 \
SYLVAN_RUN_DIR=data/replay_buffer/turn \
./tools/godot/godot --path godot --headless > "$LOG" 2>&1
echo "done -> $LOG"
