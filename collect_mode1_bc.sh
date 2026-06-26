#!/usr/bin/env bash
# collect_mode1_bc.sh — collecte données BC (behavioral-cloning) pour Mode-1.
#
# Le planner multi-pulsions (expert) drive l'environnement en survie autonome ;
# serve_planner_command.py logge chaque (obs, cmd) en JSONL via SYLVAN_BC_LOG.
# Un fichier ep_XXXX.jsonl par épisode (rotation sur reset) → contrat Task 3.
#
# Format par ligne :
#   {"obs":{"proprio":[132],"energy":float,"thirst":float},
#    "wm" :{"retina0":[144],"cmd":[vx,omega]}}
#
# Usage: bash collect_mode1_bc.sh [prefix=mode1_bc_a] [episodes=40] [seed=1]
# Tuer : pkill -9 -f serve_planner_command ; pkill -9 -f 'godot --path godot'

set +e   # pkill renvoie 1 si rien à tuer — ne pas interrompre le script

export GODOT_BIN="$(pwd)/tools/godot/godot"
export PYTHONPATH=python

PREFIX="${1:-mode1_bc_a}"
EPISODES="${2:-40}"
SEED="${3:-1}"
PORT=6075
WM="${WM_CKPT:-data/checkpoints/wm_objcentric_s1/wm_best.pt}"
BC_DIR="data/replay_buffer/${PREFIX}"

# Tuer toute instance précédente
pkill -9 -f serve_planner_command 2>/dev/null
pkill -9 -f 'godot --path godot'   2>/dev/null
sleep 1

mkdir -p "$BC_DIR"

echo "=== BC COLLECT : prefix=${PREFIX}  episodes=${EPISODES}  seed=${SEED} ==="
echo "    WM=${WM}  PORT=${PORT}  BC_DIR=${BC_DIR}"

# ── Lancer le serveur planner avec BC logger actif ──────────────────────────
SYLVAN_BC_LOG="${BC_DIR}" \
SYLVAN_PLANNER_HEADING_W=2.0 \
SYLVAN_PLANNER_URGENCY_W=6.0 \
./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
    --wm "$WM" \
    --residual data/checkpoints/hexapod_v2/policy_best.pt \
    --host 127.0.0.1 --port $PORT \
    --horizon 80 --replan-every 10 \
    > /tmp/bc_srv.log 2>&1 &
SRV=$!

# Attendre que le port soit ouvert (max 60 s)
echo -n "Attente serveur port ${PORT} ..."
for i in $(seq 1 60); do
    ss -ltn 2>/dev/null | grep -q ":${PORT}" && break
    sleep 1
done
echo " OK (pid=${SRV})"

# ── Lancer Godot : régime propre + multi-pulsions + rétine ──────────────────
SYLVAN_CPG=1                         \
SYLVAN_RESIDUAL_GAIN=0.4             \
SYLVAN_TURN_FADE=0                   \
SYLVAN_FOOT_FRICTION=7               \
SYLVAN_CPG_SPEEDCAD=0.6              \
SYLVAN_CPG_PERIOD=0.5                \
SYLVAN_CPG_PLANNER=1                 \
SYLVAN_RETINA_PLANNER=1              \
SYLVAN_WM_USE_RETINA=1               \
SYLVAN_EAT_RADIUS=1.0                \
SYLVAN_DRINK_RADIUS=1.0              \
SYLVAN_FOOD_COUNT=5                  \
SYLVAN_WATER_COUNT=8                 \
SYLVAN_ENERGY_DRAIN=0.05             \
SYLVAN_THIRST_DRAIN=0.05             \
SYLVAN_COLLECT=1                     \
SYLVAN_NUM_EPISODES="$EPISODES"      \
SYLVAN_MAX_EPISODE_STEPS=3000        \
SYLVAN_SEED="$SEED"                  \
SYLVAN_COLLECTOR_MODE=policy_server  \
SYLVAN_POLICY_HOST=127.0.0.1         \
SYLVAN_POLICY_PORT=$PORT             \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 \
SYLVAN_POLICY_EXPLORATION_STD_FINAL=0   \
SYLVAN_REFLEX_STRENGTH=0             \
SYLVAN_ASSIST_RATIO=0                \
SYLVAN_RUN_DIR="data/replay_buffer/${PREFIX}_run" \
"$GODOT_BIN" --path godot --headless > /tmp/bc_godot.log 2>&1

kill -9 $SRV 2>/dev/null

# ── Compte et résumé ─────────────────────────────────────────────────────────
echo "=== TRANSITIONS BC COLLECTÉES ==="
NFILES=$(ls -1 "${BC_DIR}"/*.jsonl 2>/dev/null | wc -l || echo 0)
NROWS=$(cat  "${BC_DIR}"/*.jsonl 2>/dev/null | wc -l  || echo 0)
echo "  Fichiers (épisodes) = ${NFILES}   Transitions = ${NROWS}"
echo "  Répertoire : ${BC_DIR}"
echo "ALL_DONE_BC"
