#!/bin/zsh
# Served-checkpoint eval at a FIXED command (turn-fade OFF, residual 0.3). Measures whether fwd_v stays
# high WHILE yaw changes (curving run) vs collapses (pivot/frozen). Parses [Godot] Yaw/fwd_v/disp.
# Usage: tools/eval_ckpt.sh LABEL CKPT VX OMEGA
cd /home/edgarbrunet/Documents/PERSO/SylvanV1
LABEL=$1; CKPT=$2; VX=${3:-0.4}; OM=${4:-0.0}
PORT=6071; PF=$(mktemp); AF=$(mktemp); LOG=/tmp/eval_${LABEL}.log
( cd python && ../env_pytorch_3.12/bin/python -m scripts.serve_ppo_collect \
    --checkpoint "../$CKPT" --host 127.0.0.1 --port $PORT --seed 1 \
    --port-file "$PF" --ack-file "$AF" > /tmp/eval_srv_${LABEL}.log 2>&1 ) &
SRV=$!
for i in $(seq 1 40); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done
trap "kill -9 $SRV 2>/dev/null" EXIT INT TERM
SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=${RGAIN:-0.3} SYLVAN_TURN_FADE=0 \
SYLVAN_CPG_TURNAMP=${TURNAMP:-0.8} SYLVAN_CPG_SPINETURN=${SPINETURN:-1.5} \
SYLVAN_CMD_VX=$VX SYLVAN_CMD_OMEGA=$OM SYLVAN_CPG_PERIOD=0.5 \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=1 SYLVAN_MAX_EPISODE_STEPS=400 SYLVAN_SEED=${SEED:-1} \
SYLVAN_DISABLE_HOMEOSTASIS=1 SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$PORT \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_RUN_DIR=data/replay_buffer/eval \
./tools/godot/godot --path godot --headless > "$LOG" 2>&1
kill -9 $SRV 2>/dev/null
python3 tools/meas_parse.py "$LABEL" "$LOG" "$VX" "$OM"
rm -rf data/replay_buffer/eval 2>/dev/null
