#!/bin/zsh
# Sonde n°1 (debug arbitrage vs moteur) : collecte APPARIÉE mono-drive vs multi-drive, tout IDENTIQUE
# sauf l'eau, pour mesurer P(atteindre une ressource EN VUE) à distance égale. mono = bouffe SEULE,
# soif GELÉE (THIRST_DRAIN=0) → ZÉRO arbitrage possible → plancher de nav/commitment PUR. multi =
# bouffe+eau, soif qui draine → arbitrage réel. Écart à 2 m = coût du FLOTTEMENT d'arbitrage.
# Monde SANS hazard (isoler l'arbitrage). Parallèle-safe : port + run-dir uniques, tue SEULEMENT son
# serveur (jamais de pkill global — deux instances coexistent).
# Usage : PORT=62xx bash scripts/collect_reachprobe.sh <mono|multi> [seed=5] [ep=16]
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
MODE=${1:-mono}; SEED=${2:-5}; NEP=${3:-16}
PORT=${PORT:-6250}; DELTA=${DELTA:-0}; TAG="reach_${MODE}_s${SEED}_d${DELTA}"
OUT="data/replay_buffer/critic_kin_${TAG}"
export GODOT_BIN="$(pwd)/tools/godot/godot"
if [[ "$MODE" == "mono" ]]; then WC=0; TD=0; else WC=1; TD=0.05; fi
rm -rf "$OUT"
echo "=== REACHPROBE $MODE : ep=$NEP seed=$SEED port=$PORT (WC=$WC thirst_drain=$TD, no hazard) ==="

env SYLVAN_PLANNER_HEADING_W=2.0 SYLVAN_PLANNER_URGENCY_W=6.0 \
    SYLVAN_PLANNER_COST=survival SYLVAN_PLANNER_DRAIN=0.0005 SYLVAN_PLANNER_RESTORE=0.4 \
    SYLVAN_PLANNER_FAR_ALIGN=1 SYLVAN_PLANNER_ALIGN_GAIN=60 \
    SYLVAN_PLANNER_COMMIT_DELTA=$DELTA \
    SYLVAN_PLANNER_CRITIC=data/checkpoints/survival_critic_kin/critic_best.pt \
    SYLVAN_BC_LOG="$OUT" \
    PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
    --wm data/checkpoints/wm_objcentric_kin/wm_best.pt --residual data/checkpoints/hexapod_v2/policy_best.pt \
    --host 127.0.0.1 --port $PORT --horizon 80 --replan-every 10 > /tmp/reach_srv_${TAG}.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done

env SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
SYLVAN_KINEMATIC=1 SYLVAN_KIN_SPEED=0.8 SYLVAN_KIN_TURN=1.5 \
SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 SYLVAN_EAT_RADIUS=1.0 SYLVAN_DRINK_RADIUS=1.0 \
SYLVAN_FOOD_COUNT=1 SYLVAN_WATER_COUNT=$WC SYLVAN_ENERGY_DRAIN=0.05 SYLVAN_THIRST_DRAIN=$TD \
SYLVAN_INIT_ENERGY=70 SYLVAN_INIT_THIRST=70 \
SYLVAN_FOOD_MIN_RADIUS=2.0 SYLVAN_FOOD_SPAWN_RADIUS=8.0 SYLVAN_FOOD_RESPAWN_MIN=2.0 SYLVAN_FOOD_RESPAWN_MAX=8.0 \
SYLVAN_WATER_MIN_RADIUS=2.0 SYLVAN_WATER_SPAWN_RADIUS=8.0 SYLVAN_WATER_RESPAWN_MIN=2.0 SYLVAN_WATER_RESPAWN_MAX=8.0 \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$NEP SYLVAN_MAX_EPISODE_STEPS=3000 SYLVAN_SEED=$SEED \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$PORT \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 SYLVAN_RUN_DIR=data/replay_buffer/reachtmp_${TAG} \
./tools/godot/godot --path godot --headless > /tmp/reach_godot_${TAG}.log 2>&1
kill -9 $SRV 2>/dev/null
rm -rf "data/replay_buffer/reachtmp_${TAG}"
echo "reach -> $OUT ($(wc -l < "$OUT/ep_0000.jsonl" 2>/dev/null) ticks)"
echo "ALL_DONE_REACH_${MODE}"
