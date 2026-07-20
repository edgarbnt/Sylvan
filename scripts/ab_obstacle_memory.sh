#!/bin/zsh
# A/B MÉMOIRE en MONDE-MUR (docs/design_memoire_spatiale.md, G0 obstacle PASS-modeste) : la mémoire
# spatiale (SlotMemory) récupère-t-elle le forage perdu à cause de l'occlusion ? Monde IDENTIQUE par
# seed (food-only + mur solide, corps cinématique, COST=mindist), seule différence = la mémoire.
# OFF = agent naïf (perd la bouffe occultée, erre — finding G3-gelé). ON = belief persistant
# (--slot-memory + egomotion) → le planner continue de viser la bouffe mémorisée derrière le mur.
# Métrique = REPAS (sauts d'énergie) sur le BC_LOG, même parseur pour les deux bras.
# Parallèle-safe : port + run-dir uniques, tue SEULEMENT son serveur.
# Pré-inscrit : PASS = repas(ON) > repas(OFF) au-delà du bruit inter-seed ; KILL = ON < OFF (la
# mémoire fait CHASSER des fantômes = pire). Attentes TEMPÉRÉES (occlusion à ~7 m, portée-capée).
# Usage : PORT=62xx bash scripts/ab_obstacle_memory.sh <off|on> [seed=1] [ep=16]
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
ARM=${1:-off}; SEED=${2:-1}; NEP=${3:-16}
PORT=${PORT:-6240}; TAG="obmem_${ARM}_s${SEED}"
OUT="data/replay_buffer/${TAG}"; RUNDIR="data/replay_buffer/${TAG}_run"
WM=data/checkpoints/wm_objcentric_kin/wm_best.pt
export GODOT_BIN="$(pwd)/tools/godot/godot"
MEM=""
[[ "$ARM" == "on" ]] && MEM="--slot-memory --egomotion-head data/checkpoints/egomotion_head/best.pt"
rm -rf "$OUT" "$RUNDIR"
echo "=== A/B OBSTACLE-MÉMOIRE $ARM : seed=$SEED port=$PORT (food-only + mur solide) mem='$MEM' ==="

env SYLVAN_PLANNER_HEADING_W=2.0 SYLVAN_PLANNER_COST=mindist SYLVAN_CMD_EXPLORE_STD=0 \
    SYLVAN_BC_LOG="$OUT" \
    PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
    --wm "$WM" --residual data/checkpoints/hexapod_v2/policy_best.pt $MEM \
    --host 127.0.0.1 --port $PORT --horizon 80 --replan-every 10 > /tmp/obmem_srv_${TAG}.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done

env SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
SYLVAN_KINEMATIC=1 SYLVAN_KIN_SPEED=0.8 SYLVAN_KIN_TURN=1.5 \
SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 SYLVAN_EAT_RADIUS=1.0 \
SYLVAN_FOOD_COUNT=1 SYLVAN_ENERGY_DRAIN=0.05 SYLVAN_INIT_ENERGY=80 \
SYLVAN_FOOD_MIN_RADIUS=2.5 SYLVAN_FOOD_SPAWN_RADIUS=5.0 SYLVAN_FOOD_RESPAWN_MIN=2.5 SYLVAN_FOOD_RESPAWN_MAX=5.0 \
SYLVAN_OBSTACLE_COUNT=1 SYLVAN_OBSTACLE_FRAC=0.5 SYLVAN_OBSTACLE_HALFWIDTH=0.6 SYLVAN_OBSTACLE_SOLID=1 \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$NEP SYLVAN_MAX_EPISODE_STEPS=1500 SYLVAN_SEED=$SEED \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$PORT \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 SYLVAN_RUN_DIR="$RUNDIR" \
./tools/godot/godot --path godot --headless > /tmp/obmem_godot_${TAG}.log 2>&1
kill -9 $SRV 2>/dev/null
rm -rf "$RUNDIR"
grep -m1 'MÉMOIRE SPATIALE active\|AVERTISSEMENT' /tmp/obmem_srv_${TAG}.log
echo "corpus -> $OUT ($(wc -l < "$OUT/ep_0000.jsonl" 2>/dev/null) ticks)"
echo "ALL_DONE_OBMEM_${ARM}_s${SEED}"
