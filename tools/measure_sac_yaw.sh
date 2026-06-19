#!/usr/bin/env bash
# Measure deterministic sustained yaw of a fully-learned SAC checkpoint, both turn directions.
CKPT="${1:-data/checkpoints/sac_turn_fl/policy_best.pt}"
PORT="${2:-6071}"
cd "$(dirname "$0")/.."
pkill -9 -f serve_sac_collect 2>/dev/null || true
sleep 1
# serve_sac_collect needs port-file/ack-file; use them
PF=$(mktemp); AF=$(mktemp)
( cd python && ../env_pytorch_3.12/bin/python -m scripts.serve_sac_collect \
    --checkpoint "../$CKPT" --host 127.0.0.1 --port "$PORT" --seed 1 \
    --port-file "$PF" --ack-file "$AF" --deterministic > /tmp/sacsrv.log 2>&1 ) &
for i in $(seq 1 40); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done
for OM in 0.8 -0.8; do
  SYLVAN_CPG=1 SYLVAN_LEARNED=1 SYLVAN_LEARNED_ACTION_SCALE=0.8 SYLVAN_LEARNED_BLEND=1.0 \
  SYLVAN_CMD_VX=0.3 SYLVAN_CMD_OMEGA=$OM SYLVAN_CPG_PERIOD=0.5 \
  SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=2 SYLVAN_MAX_EPISODE_STEPS=300 SYLVAN_SEED=1 SYLVAN_DISABLE_HOMEOSTASIS=1 \
  SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$PORT \
  SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 \
  SYLVAN_RUN_DIR=data/replay_buffer/saceval timeout 70 ./tools/godot/godot --path godot --headless > /tmp/sac_yaw_$OM.log 2>&1 || true
done
pkill -9 -f serve_sac_collect 2>/dev/null || true
