#!/bin/zsh
# VOIR le forager CINÉMATIQUE (pivot corps différentiel) dans une FENÊTRE (temps réel).
# Le corps glisse en (vx, omega) via le cœur cinématique (SYLVAN_KINEMATIC), le planner WM le pilote
# vers bouffe/eau. Maillage = hexapode actuel (pattes figées, loup pas encore câblé). Ferme la fenêtre
# pour arrêter (serveur tué automatiquement). Usage: bash scripts/voir_kinematic.sh
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
export GODOT_BIN="$(pwd)/tools/godot/godot"
WM=${WM_CKPT:-data/checkpoints/wm_objcentric_kin/wm_best.pt}
PORT=${PORT:-6191}

env SYLVAN_PLANNER_HEADING_W=2.0 SYLVAN_PLANNER_URGENCY_W=6.0 \
    SYLVAN_PLANNER_COST=survival SYLVAN_PLANNER_DRAIN=0.0005 SYLVAN_PLANNER_RESTORE=0.4 \
    SYLVAN_PLANNER_FAR_ALIGN=1 SYLVAN_PLANNER_ALIGN_GAIN=60 \
    PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
    --wm "$WM" --residual data/checkpoints/hexapod_v2/policy_best.pt \
    --host 127.0.0.1 --port $PORT --horizon 80 --replan-every 10 > /tmp/voir_kin_srv.log 2>&1 &
SRV=$!
trap "kill -9 $SRV 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null" EXIT INT TERM
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done
echo "[voir-kin] serveur planner prêt (WM=$WM). Une fenêtre Godot va s'ouvrir — ferme-la pour arrêter."

# Fenêtré (PAS de --headless) => temps réel, regardable. Monde multi-ressource à distances variées.
env SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
SYLVAN_KINEMATIC=1 SYLVAN_KIN_SPEED=${KIN_SPEED:-0.8} SYLVAN_KIN_TURN=${KIN_TURN:-1.5} \
SYLVAN_WOLF=1 SYLVAN_WOLF_SCALE=${WOLF_SCALE:-0.4} SYLVAN_WOLF_YAW=${WOLF_YAW:-0.0} SYLVAN_WOLF_Y=${WOLF_Y:--0.30} \
SYLVAN_WOLF_LIGHTEN=${WOLF_LIGHTEN:-0.35} \
SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 SYLVAN_EAT_RADIUS=1.0 SYLVAN_DRINK_RADIUS=1.0 \
SYLVAN_FOOD_COUNT=3 SYLVAN_WATER_COUNT=3 SYLVAN_ENERGY_DRAIN=0.05 SYLVAN_THIRST_DRAIN=0.05 \
SYLVAN_INIT_ENERGY=70 SYLVAN_INIT_THIRST=70 \
SYLVAN_FOOD_MIN_RADIUS=2.0 SYLVAN_FOOD_SPAWN_RADIUS=6.0 SYLVAN_WATER_MIN_RADIUS=2.0 SYLVAN_WATER_SPAWN_RADIUS=6.0 \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=10 SYLVAN_MAX_EPISODE_STEPS=3000 SYLVAN_SEED=1 \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$PORT \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 SYLVAN_RUN_DIR=data/replay_buffer/view_kin \
./tools/godot/godot --path godot
