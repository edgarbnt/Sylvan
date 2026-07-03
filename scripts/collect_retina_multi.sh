#!/bin/zsh
# Collecte MULTI-RESSOURCE pour le slot-2 (chantier pureté 2026-07-04) : monde 5 bouffes + 5 eaux,
# SYLVAN_WM_COLLECT=1 → logge wm.retina0 + wm.torso0 + wm.food_rel0 + wm.water_rel0 (labels ÉVAL
# seulement, l'entraînement du slot reste label-free). Pilote = la meilleure config connue
# (coords rétine-argmin + coût survie, 8f9ff54/3cf3204) → distribution riche en approches ET
# consommations des DEUX ressources. Usage: bash scripts/collect_retina_multi.sh [episodes=60] [seed=31]
set +e
NEP=${1:-60}; SEED=${2:-31}
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
rm -rf data/replay_buffer/retina_multi_a

SYLVAN_PLANNER_HEADING_W=2.0 SYLVAN_PLANNER_URGENCY_W=6.0 \
SYLVAN_MULTI_FOOD_SLOT=0 SYLVAN_PLANNER_COST=survival SYLVAN_PLANNER_DRAIN=0.0005 SYLVAN_PLANNER_RESTORE=0.4 \
PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
  --wm data/checkpoints/wm_objcentric_s1/wm_best.pt \
  --residual data/checkpoints/hexapod_v2/policy_best.pt \
  --host 127.0.0.1 --port 6053 --horizon 80 --replan-every 10 > /tmp/collect_multi_srv.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ':6053' && break; sleep 1; done

SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 SYLVAN_WM_COLLECT=1 \
SYLVAN_EAT_RADIUS=1.0 SYLVAN_DRINK_RADIUS=1.0 SYLVAN_ENERGY_DRAIN=0.05 SYLVAN_THIRST_DRAIN=0.05 \
SYLVAN_FOOD_COUNT=5 SYLVAN_WATER_COUNT=5 \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$NEP SYLVAN_MAX_EPISODE_STEPS=1500 SYLVAN_SEED=$SEED \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=6053 \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 \
SYLVAN_RUN_DIR=data/replay_buffer/retina_multi_a \
./tools/godot/godot --path godot --headless > /tmp/collect_multi.log 2>&1
kill -9 $SRV 2>/dev/null
echo "done -> data/replay_buffer/retina_multi_a"
ls data/replay_buffer/retina_multi_a 2>/dev/null | wc -l