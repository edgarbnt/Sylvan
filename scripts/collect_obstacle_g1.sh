#!/bin/zsh
# G1 (chantier CANAL OBSTACLE, docs/design_obstacle_affordance.md) — collecte du corpus de VIABILITÉ.
# Monde FOOD-ONLY + OBSTACLE bloquant ON, corps CINÉMATIQUE. But : un corpus où (1) le corps s'ARRÊTE
# contre le mur (déplacement réalisé < commandé), (2) la bouffe reste PERCEPTIBLE autour du mur
# (non-occlusion), (3) l'obstacle forme un cluster couleur SÉPARABLE. Mesuré offline par
# diagnostics/diag_obstacle_g1.py (AUCUN entraînement).
#
# WM SINGLE-FOOD (wm_objcentric_kin, requête ROUGE) → l'obstacle CYAN ne fire AUCUN slot → l'agent
# l'ignore perceptuellement, fonce vers la bouffe et le PERCUTE (l'évitement n'existe pas encore : c'est
# ce que voie B apprendra à G2/G3). COST=mindist = nav propre (pas de critique/survie nécessaire à G1).
# REQUIERT le hook local main.gd (obstacle_manager.begin_episode) — non stagé, comme hazard.
# Usage: [PORT=61xx] bash scripts/collect_obstacle_g1.sh [ep=16] [seed=1] [tag=g1] [OBSTACLE_COUNT=1]
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
NEP=${1:-16}; SEED=${2:-1}; TAG=${3:-g1}; OBST=${4:-1}
PORT=${PORT:-6231}
WM=${WM_CKPT:-data/checkpoints/wm_objcentric_kin/wm_best.pt}
OUT="data/replay_buffer/obstacle_${TAG}"          # BC_LOG (serveur) : retina0 + cmd
RUNDIR="data/replay_buffer/obstacle_${TAG}_run"   # rollout writer (Godot) : proprio + retina0 + cmd + torso0 (LA source G1)
export GODOT_BIN="$(pwd)/tools/godot/godot"

pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
rm -rf "$OUT" "$RUNDIR"
echo "=== CORPUS OBSTACLE G1 tag=$TAG : ep=$NEP seed=$SEED port=$PORT obstacle_count=$OBST (food-only, mindist) ==="

# Serveur : nav food-only COST=mindist (pas de critique), log des replans -> corpus (retina0/cmd/torso0).
env SYLVAN_PLANNER_HEADING_W=${HW:-2.0} SYLVAN_PLANNER_COST=${COST:-mindist} \
    SYLVAN_CMD_EXPLORE_STD=0 \
    SYLVAN_BC_LOG="$OUT" \
    PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
    --wm "$WM" --residual data/checkpoints/hexapod_v2/policy_best.pt \
    --host 127.0.0.1 --port $PORT --horizon ${HORIZON:-80} --replan-every 10 > /tmp/obstacle_srv_${TAG}.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done

# Corps CINÉMATIQUE (0.8), monde FOOD-ONLY, obstacle bloquant sur le trajet spawn→bouffe.
env SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
SYLVAN_KINEMATIC=1 SYLVAN_KIN_SPEED=${KIN_SPEED:-0.8} SYLVAN_KIN_TURN=${KIN_TURN:-1.5} \
SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 SYLVAN_EAT_RADIUS=1.0 \
SYLVAN_FOOD_COUNT=1 SYLVAN_ENERGY_DRAIN=0.05 SYLVAN_INIT_ENERGY=${INIT_E:-80} \
SYLVAN_FOOD_MIN_RADIUS=${RMIN:-2.5} SYLVAN_FOOD_SPAWN_RADIUS=${RMAX:-5.0} SYLVAN_FOOD_RESPAWN_MIN=${RMIN:-2.5} SYLVAN_FOOD_RESPAWN_MAX=${RMAX:-5.0} \
SYLVAN_OBSTACLE_COUNT=$OBST SYLVAN_OBSTACLE_FRAC=${OFRAC:-0.5} SYLVAN_OBSTACLE_HALFWIDTH=${OHW:-0.6} SYLVAN_OBSTACLE_SOLID=${OSOLID:-1} \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$NEP SYLVAN_MAX_EPISODE_STEPS=${STEPS:-1500} SYLVAN_SEED=$SEED \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$PORT \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 SYLVAN_RUN_DIR="$RUNDIR" \
./tools/godot/godot --path godot --headless > /tmp/obstacle_free_${TAG}.log 2>&1
kill -9 $SRV 2>/dev/null
pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null
echo "corpus BC_LOG -> $OUT | run_dir (torso0) -> $RUNDIR ($(ls "$RUNDIR"/episode_*.jsonl 2>/dev/null | wc -l) épisodes)"
echo "ALL_DONE_OBSTACLE_${TAG}"
