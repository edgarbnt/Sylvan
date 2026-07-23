#!/bin/bash
# Corpus critique sur le MONDE CONSÉQUENT (bosquets_v3_perish, levier périssable validé 33%).
# But : gate GRATUIT — la cible résidu-de-repas a-t-elle de la variance INTRA-état classable par un
# trait-slot ? (le survival_critic historique a pour cible la survie, MORTE ici car saturée).
# Log BC au régime VÉCU (replan-60). Run complet -> /tmp (jeté), on ne garde que le log BC.
# Usage: bash scripts/collect_critic_bosquets.sh [ep=20] [seed=1] [tag=a] [port=6440]
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
NEP=${1:-20}; SEED=${2:-1}; TAG=${3:-a}; PORT=${4:-6440}
WM=data/checkpoints/wm_objcentric_kin/wm_best.pt
OUT="data/replay_buffer/critic_bosq_${TAG}"; rm -rf "$OUT"
WE=$(PYTHONPATH=python ./env_pytorch_3.12/bin/python -m sylvan.world --preset bosquets_v3_perish --env | sed 's/^export //' | tr '\n' ' ')
FOV=$(echo "$WE" | tr ' ' '\n' | grep '^SYLVAN_RETINA_FOV_DEG=' | cut -d= -f2)
export GODOT_BIN="$(pwd)/tools/godot/godot"
pkill -9 -f "serve_planner_command.*$PORT" 2>/dev/null; sleep 1
env SYLVAN_PLANNER_HEADING_W=2.0 SYLVAN_PLANNER_TURN_RATE=0.015 SYLVAN_PLANNER_URGENCY_W=6.0 \
    SYLVAN_PLANNER_COST=survival SYLVAN_PLANNER_DRAIN=0.0005 SYLVAN_PLANNER_RESTORE=0.4 \
    SYLVAN_RETINA_FOV_DEG=$FOV SYLVAN_BC_LOG="$OUT" \
    PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
    --wm "$WM" --residual data/checkpoints/hexapod_v2/policy_best.pt \
    --host 127.0.0.1 --port $PORT --horizon 80 --replan-every 60 \
    --egomotion-head data/checkpoints/egomotion_head/best.pt --slot-memory > /tmp/critbosq_srv_${TAG}.log 2>&1 &
SRV=$!; for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done
env $WE SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 \
  SYLVAN_CPG_SPEEDCAD=0.6 SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 \
  SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$NEP SYLVAN_MAX_EPISODE_STEPS=3000 SYLVAN_SEED=$SEED \
  SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$PORT \
  SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
  SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 SYLVAN_RUN_DIR=/tmp/critbosq_run_${TAG} \
  ./tools/godot/godot --path godot --headless > /tmp/critbosq_${TAG}.log 2>&1
kill -9 $SRV 2>/dev/null; pkill -9 -f "serve_planner_command.*$PORT" 2>/dev/null
rm -rf /tmp/critbosq_run_${TAG}
NL=$(wc -l < "$OUT/ep_0000.jsonl" 2>/dev/null || echo 0)
echo "corpus BC -> $OUT ($NL replans loggés, ep0)"
du -sh "$OUT" 2>/dev/null
echo "ALL_DONE_CRITBOSQ" > /tmp/critcol_done.txt
