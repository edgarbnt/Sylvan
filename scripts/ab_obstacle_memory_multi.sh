#!/bin/zsh
# A/B MÉMOIRE MULTI-DRIVE en MONDE-MUR (suite du PASS food-only, docs/design_memoire_spatiale.md).
# Question de PROMOTION : la mémoire tient-elle dans la config MULTI-DRIVE (bouffe + eau), là où
# l'arbitrage ajoute sa propre difficulté, et pas seulement en food-only ?
# OFF = agent naïf. ON = MultiSlotMemory (SYLVAN_SLOT_MEMORY2=1, PAR ressource, invalidate à la
# consommation, âge géométrique 500 pas) sur WM slot-2 (wm_objcentric_kin). ZÉRO retrain.
# Monde IDENTIQUE par seed (bouffe + eau + mur solide), seule différence = la mémoire.
# Coût planner = survival (le multi-drive vivant), pas mindist (mono-cible).
# Métrique = CONSOMMATIONS (repas + boissons) sur le BC_LOG, même parseur pour les deux bras.
# Parallèle-safe : port + run-dir uniques, tue SEULEMENT son serveur.
# Pré-inscrit : PASS = conso(ON) > conso(OFF) sur les 2 seeds ; KILL = ON < OFF (fantômes).
# Usage : PORT=62xx bash scripts/ab_obstacle_memory_multi.sh <off|on> [seed=1] [ep=16]
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
ARM=${1:-off}; SEED=${2:-1}; NEP=${3:-16}
# RE-MESURE CIBLEE (2026-07-21) : FA = echafaudage far-target. Il SUPPRIME l atteinte lointaine
# (seed 1 : 4-6 m 38 -> 64 % sans lui ; poole 2 seeds : 47,4 -> 69,9 %) ; toutes les mesures
# memoire precedentes le portaient. ATTENTION : en MONDE-MUR il est au contraire PORTEUR
# (sans lui l entite est immobile 79 % des ticks) -> l effet est DEPENDANT DU MONDE.
# On re-mesure a FA=0 (condition PROPRE). JUGE PRINCIPAL = courbe d atteinte (n en milliers),
# PAS les consommations (n en dizaines : 26 vs 32 = 1,2 sigma, sous-puissant).
FA=${FA:-1}
PORT=${PORT:-6250}; TAG="obmemM_${ARM}_s${SEED}_fa${FA}"
OUT="data/replay_buffer/${TAG}"; RUNDIR="data/replay_buffer/${TAG}_run"
WM=data/checkpoints/wm_objcentric_kin/wm_best.pt     # slot_resources=2 (bouffe+eau)
export GODOT_BIN="$(pwd)/tools/godot/godot"
MEM2=0; [[ "$ARM" == "on" ]] && MEM2=1
rm -rf "$OUT" "$RUNDIR"
echo "=== A/B OBSTACLE-MÉMOIRE MULTI $ARM : seed=$SEED port=$PORT (bouffe+eau+mur) SLOT_MEMORY2=$MEM2 ==="

env SYLVAN_PLANNER_HEADING_W=2.0 SYLVAN_PLANNER_URGENCY_W=6.0 \
    SYLVAN_PLANNER_COST=survival SYLVAN_PLANNER_DRAIN=0.0005 SYLVAN_PLANNER_RESTORE=0.4 \
    SYLVAN_PLANNER_FAR_ALIGN=$FA SYLVAN_PLANNER_ALIGN_GAIN=60 \
    SYLVAN_SLOT_MEMORY2=$MEM2 \
    SYLVAN_CMD_EXPLORE_STD=0 SYLVAN_BC_LOG="$OUT" \
    PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
    --wm "$WM" --residual data/checkpoints/hexapod_v2/policy_best.pt \
    --host 127.0.0.1 --port $PORT --horizon 80 --replan-every 10 > /tmp/obmemM_srv_${TAG}.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done

env SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
SYLVAN_KINEMATIC=1 SYLVAN_KIN_SPEED=0.8 SYLVAN_KIN_TURN=1.5 \
SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 SYLVAN_EAT_RADIUS=1.0 SYLVAN_DRINK_RADIUS=1.0 \
SYLVAN_FOOD_COUNT=1 SYLVAN_WATER_COUNT=1 SYLVAN_ENERGY_DRAIN=0.05 SYLVAN_THIRST_DRAIN=0.05 \
SYLVAN_INIT_ENERGY=70 SYLVAN_INIT_THIRST=70 \
SYLVAN_FOOD_MIN_RADIUS=2.5 SYLVAN_FOOD_SPAWN_RADIUS=5.0 SYLVAN_FOOD_RESPAWN_MIN=2.5 SYLVAN_FOOD_RESPAWN_MAX=5.0 \
SYLVAN_WATER_MIN_RADIUS=2.5 SYLVAN_WATER_SPAWN_RADIUS=5.0 SYLVAN_WATER_RESPAWN_MIN=2.5 SYLVAN_WATER_RESPAWN_MAX=5.0 \
SYLVAN_OBSTACLE_COUNT=1 SYLVAN_OBSTACLE_FRAC=0.5 SYLVAN_OBSTACLE_HALFWIDTH=0.6 SYLVAN_OBSTACLE_SOLID=1 \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$NEP SYLVAN_MAX_EPISODE_STEPS=1500 SYLVAN_SEED=$SEED \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$PORT \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 SYLVAN_RUN_DIR="$RUNDIR" \
./tools/godot/godot --path godot --headless > /tmp/obmemM_godot_${TAG}.log 2>&1
kill -9 $SRV 2>/dev/null
rm -rf "$RUNDIR"
grep -m1 'MÉMOIRE SPATIALE MULTI\|MÉMOIRE SPATIALE active\|AVERTISSEMENT' /tmp/obmemM_srv_${TAG}.log
echo "corpus -> $OUT ($(wc -l < "$OUT/ep_0000.jsonl" 2>/dev/null) ticks)"
echo "ALL_DONE_OBMEMM_${ARM}_s${SEED}"
