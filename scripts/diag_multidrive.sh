#!/bin/zsh
# MULTI-PULSIONS EN SURVIE LIBRE (2026-06-18) — la vraie démo ALife.
# Faim + soif drainent ENSEMBLE (métabolisme de vie 0.05), bouffe ROUGE + eau BLEUE, eat/drink
# radius 1.0 (HONNÊTE). Question : jongle-t-il entre les deux pour survivre, ou laisse-t-il une
# pulsion crasher ? Mesure : survie + repas + boissons + min(énergie) min(soif) (équilibre-t-il ?).
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
PORT=6072
pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'tools/godot/godot' 2>/dev/null; sleep 1
rm -f /tmp/multi_*.log
SYLVAN_PLANNER_HEADING_W=2.0 SYLVAN_PLANNER_URGENCY_W=6.0 \
PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
  --wm data/checkpoints/wm_command_hex_v2/wm_best.pt --residual data/checkpoints/hexapod_v2/policy_best.pt \
  --host 127.0.0.1 --port $PORT --horizon 80 --replan-every 10 > /tmp/multi_srv.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done
SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_EAT_RADIUS=1.0 SYLVAN_DRINK_RADIUS=1.0 \
SYLVAN_FOOD_COUNT=5 SYLVAN_WATER_COUNT=5 SYLVAN_ENERGY_DRAIN=0.05 SYLVAN_THIRST_DRAIN=0.05 \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=10 SYLVAN_MAX_EPISODE_STEPS=3000 SYLVAN_SEED=1 \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$PORT \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 SYLVAN_RUN_DIR=data/replay_buffer/multidrive \
./tools/godot/godot --path godot --headless > /tmp/multi_free.log 2>&1
kill -9 $SRV 2>/dev/null
echo MULTI_DONE
