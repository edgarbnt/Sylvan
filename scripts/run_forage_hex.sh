#!/bin/zsh
# Phase 5 foraging test on the HEXAPOD (2026-06-17): planner server (wm_command_hex_v1 + hexapod_v2)
# + the hexapod motor base, homeostasis ON, multi-pellets. Mirrors run_forage_eat.sh but on the new body.
# IMPORTANT: the Godot env params MUST match the WM-collection env (TURN_FADE=0, FOOT_FRICTION=7,
# CPG_SPEEDCAD=0.6, RESIDUAL_GAIN=0.4) so the command->motion map the planner sees == what the WM learned.
# No quad turn_k/turn_amp/period overrides (hexapod tripod CPG uses its own defaults).
# Usage: bash scripts/run_forage_hex.sh [eat_radius=1.0] [horizon=80] [num_episodes=12]
set +e
ER=${1:-1.0}
HZ=${2:-80}
NEP=${3:-12}
HW=${4:-2.0}   # planner heading-alignment weight (A→B engagement fix, 2026-06-18); 0 = original cost
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1
cd "$ROOT"
# clean any orphan planner/godot first
pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
echo "heading_weight=$HW horizon=$HZ"

SYLVAN_PLANNER_HEADING_W=$HW \
PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
  --wm data/checkpoints/wm_command_hex_v2/wm_best.pt \
  --residual data/checkpoints/hexapod_v2/policy_best.pt \
  --host 127.0.0.1 --port 6051 --horizon $HZ --replan-every 10 > /tmp/planner_hex.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ':6051' && break; sleep 1; done

SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_EAT_RADIUS=$ER \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$NEP SYLVAN_MAX_EPISODE_STEPS=1500 SYLVAN_SEED=1 SYLVAN_FOOD_COUNT=${FC:-6} \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=6051 \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 \
SYLVAN_RUN_DIR=data/replay_buffer/forage_hex \
./tools/godot/godot --path godot --headless > /tmp/forage_hex.log 2>&1
kill -9 $SRV 2>/dev/null
echo "done eat_radius=$ER horizon=$HZ episodes=$NEP -> /tmp/forage_hex.log"
