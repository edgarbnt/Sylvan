#!/bin/zsh
# Corpus d'entraînement du CRITIQUE, collecté sur le corps CINÉMATIQUE (pivot 2026-07-07).
# But : fabriquer enfin la classe positive MANQUANTE du critique = "pulsion urgente + ressource LOIN
# -> ATTEINTE -> survie", que le corps legged ne produisait jamais (plafond d'imitation). Le corps
# cinématique + l'échafaudage de cap l'atteignent (far-food ~75%). Monde ÉPARS 1+1 à distances VARIÉES
# (proche 2m ET loin 8m) -> arbitrage faim/soif réel AVEC poursuites lointaines réussies.
# Le serveur logue les replans (SYLVAN_BC_LOG) au format lu par train_survival_critic.load().
# Usage: [PARALLEL=1 PORT=61xx] bash scripts/collect_critic_corpus_kin.sh [ep=20] [seed=1] [tag=a]
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
NEP=${1:-20}; SEED=${2:-1}; TAG=${3:-a}
PORT=${PORT:-6201}
WM=${WM_CKPT:-data/checkpoints/wm_objcentric_kin/wm_best.pt}
OUT="data/replay_buffer/critic_kin_${TAG}"
export GODOT_BIN="$(pwd)/tools/godot/godot"

[[ -z "$PARALLEL" ]] && { pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1; }
rm -rf "$OUT"
echo "=== CORPUS CRITIQUE (cinématique) tag=$TAG : ep=$NEP seed=$SEED port=$PORT (épars 1+1, distances variées) ==="

# Serveur : coût survie + échafaudage ON (pour réussir le loin) + log des replans -> corpus critique.
# Perception food/eau symétrique par construction (2026-07-07, hygiène train=déploiement) : plus de
# flag, plus de hack "eau garde sa dernière position" nulle part dans le codebase.
env SYLVAN_PLANNER_HEADING_W=2.0 SYLVAN_PLANNER_URGENCY_W=6.0 \
    SYLVAN_PLANNER_COST=${COST:-survival} SYLVAN_PLANNER_DRAIN=0.0005 SYLVAN_PLANNER_RESTORE=0.4 \
    SYLVAN_PLANNER_FAR_ALIGN=${FAR_ALIGN:-1} SYLVAN_PLANNER_ALIGN_GAIN=60 \
    SYLVAN_PLANNER_CRITIC=${CRITIC:-data/checkpoints/survival_critic_kin/critic_best.pt} \
    SYLVAN_BC_LOG="$OUT" \
    PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
    --wm "$WM" --residual data/checkpoints/hexapod_v2/policy_best.pt \
    --host 127.0.0.1 --port $PORT --horizon 80 --replan-every 10 > /tmp/critic_srv_${TAG}.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done

# Corps CINÉMATIQUE (0.8), monde ÉPARS 1+1, ressources à distances VARIÉES (2 à 8 m).
env SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
SYLVAN_KINEMATIC=1 SYLVAN_KIN_SPEED=${KIN_SPEED:-0.8} SYLVAN_KIN_TURN=${KIN_TURN:-1.5} \
SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 SYLVAN_EAT_RADIUS=1.0 SYLVAN_DRINK_RADIUS=1.0 \
SYLVAN_FOOD_COUNT=${FC:-1} SYLVAN_WATER_COUNT=${WC:-1} SYLVAN_ENERGY_DRAIN=0.05 SYLVAN_THIRST_DRAIN=0.05 \
SYLVAN_INIT_ENERGY=${INIT_E:-70} SYLVAN_INIT_THIRST=${INIT_T:-70} \
SYLVAN_FOOD_MIN_RADIUS=${RMIN:-2.0} SYLVAN_FOOD_SPAWN_RADIUS=${RMAX:-8.0} SYLVAN_FOOD_RESPAWN_MIN=${RMIN:-2.0} SYLVAN_FOOD_RESPAWN_MAX=${RMAX:-8.0} \
SYLVAN_WATER_MIN_RADIUS=${RMIN:-2.0} SYLVAN_WATER_SPAWN_RADIUS=${RMAX:-8.0} SYLVAN_WATER_RESPAWN_MIN=${RMIN:-2.0} SYLVAN_WATER_RESPAWN_MAX=${RMAX:-8.0} \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$NEP SYLVAN_MAX_EPISODE_STEPS=3000 SYLVAN_SEED=$SEED \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$PORT \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 SYLVAN_RUN_DIR=data/replay_buffer/critic_tmp_${TAG} \
./tools/godot/godot --path godot --headless > /tmp/critic_free_${TAG}.log 2>&1
kill -9 $SRV 2>/dev/null
[[ -z "$PARALLEL" ]] && { pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; }
echo "corpus -> $OUT ($(wc -l < "$OUT/ep_0000.jsonl" 2>/dev/null) replans loggés)"
echo "ALL_DONE_CRITIC_${TAG}"
