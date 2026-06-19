#!/bin/zsh
# Measure yaw drift / turn of hexapod_v2 for a given (vx, omega, extra-env). 3 seeds-ish via 3 episodes.
cd /home/edgarbrunet/Documents/PERSO/SylvanV1
LABEL=$1; VX=$2; OM=$3; EXTRA=$4
pkill -9 -f serve_ppo_collect 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
PF=$(mktemp); AF=$(mktemp)
( cd python && ../env_pytorch_3.12/bin/python -m scripts.serve_ppo_collect \
    --checkpoint ../data/checkpoints/hexapod_v2/policy_best.pt --host 127.0.0.1 --port 6071 --seed 1 \
    --port-file "$PF" --ack-file "$AF" > /tmp/sps_$LABEL.log 2>&1 ) &
SRV=$!
for i in $(seq 1 40); do ss -ltn 2>/dev/null | grep -q ':6071' && break; sleep 1; done
env $EXTRA SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
  SYLVAN_CMD_VX=$VX SYLVAN_CMD_OMEGA=$OM \
  SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=3 SYLVAN_MAX_EPISODE_STEPS=400 SYLVAN_SEED=1 SYLVAN_DISABLE_HOMEOSTASIS=1 \
  SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=6071 \
  SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
  SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 SYLVAN_RUN_DIR=data/replay_buffer/straighttest \
  ./tools/godot/godot --path godot --headless > /tmp/straight_$LABEL.log 2>&1
kill -9 $SRV 2>/dev/null
echo "=== $LABEL (vx=$VX om=$OM $EXTRA): Yaw final + disp par épisode ==="
grep '\[Godot\] Episode' /tmp/straight_$LABEL.log | grep 'Step 400 ' | sed -n 's/.*Episode \([0-9]*\).*Yaw: \(-*[0-9]*\) .*disp: \([0-9.]*\).*/  ep\1  yaw=\2°  disp=\3m/p'
