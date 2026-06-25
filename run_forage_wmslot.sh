#!/bin/zsh
# STEP 3 — foraging avec le SLOT DANS LE WM (out["slot"]), SANS --slot-head : la perception+permanence vient du WM
# (wm_objcentric_s1). À comparer à run_forage_purslot.sh (échafaudage codé-main) : survie ≥ ~1040 pour promouvoir.
# Usage: bash run_forage_wmslot.sh [eat_radius=1.0] [horizon=160] [episodes=12]
set +e
ER=${1:-1.0}; HZ=${2:-160}; NEP=${3:-12}
WM=${WM_CKPT:-data/checkpoints/wm_objcentric_s1/wm_best.pt}
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
export SYLVAN_PLANNER_HEADING_W=${SYLVAN_PLANNER_HEADING_W:-0.0}
echo "WM-SLOT forage WM=$WM eat_radius=$ER horizon=$HZ episodes=$NEP heading_w=$SYLVAN_PLANNER_HEADING_W"
PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
  --wm "$WM" --residual data/checkpoints/hexapod_v2/policy_best.pt \
  --host 127.0.0.1 --port 6052 --horizon $HZ --replan-every 10 > /tmp/planner_wmslot.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ':6052' && break; sleep 1; done

SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 SYLVAN_EAT_RADIUS=$ER \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$NEP SYLVAN_MAX_EPISODE_STEPS=1500 SYLVAN_SEED=1 SYLVAN_FOOD_COUNT=${FC:-6} \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=6052 \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 \
SYLVAN_RUN_DIR=data/replay_buffer/forage_wmslot \
./tools/godot/godot --path godot --headless > /tmp/forage_wmslot.log 2>&1
kill -9 $SRV 2>/dev/null
echo "done -> /tmp/forage_wmslot.log"
