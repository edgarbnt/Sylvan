#!/bin/zsh
# ÉTAGE 1bis — collecter des données FORAGING (pilotées par l'ORACLE) AVEC log rétine, pour ré-entraîner la
# tête sur la DISTRIBUTION DE DÉPLOIEMENT (≠ babbling). Oracle pilote (food_xz_from_radar) ; SYLVAN_WM_COLLECT=1
# logge retina0 + food_rel0 (label position vraie). Drain « de vie » 0.05 → épisodes longs = beaucoup de data
# d'approche/repas (la vraie distribution closed-loop). Usage: bash collect_forage_retina.sh [episodes=12] [seed=31]
set +e
NEP=${1:-12}; SEED=${2:-31}
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
rm -rf data/replay_buffer/retina_forage

SYLVAN_PLANNER_HEADING_W=2.0 \
PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
  --wm data/checkpoints/wm_command_hex_v2/wm_best.pt \
  --residual data/checkpoints/hexapod_v2/policy_best.pt \
  --host 127.0.0.1 --port 6053 --horizon 80 --replan-every 10 > /tmp/planner_collect.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ':6053' && break; sleep 1; done

SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_WM_COLLECT=1 SYLVAN_EAT_RADIUS=1.0 SYLVAN_ENERGY_DRAIN=0.05 \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$NEP SYLVAN_MAX_EPISODE_STEPS=1500 SYLVAN_SEED=$SEED SYLVAN_FOOD_COUNT=6 \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=6053 \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 \
SYLVAN_RUN_DIR=data/replay_buffer/retina_forage \
./tools/godot/godot --path godot --headless > /tmp/collect_forage.log 2>&1
kill -9 $SRV 2>/dev/null
echo "done -> data/replay_buffer/retina_forage"; ls data/replay_buffer/retina_forage 2>/dev/null | head
