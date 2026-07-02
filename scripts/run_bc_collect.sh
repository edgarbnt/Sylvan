#!/bin/zsh
# Collect a behavior-cloning dataset of the GOOD gait (CPG + residual7), command-sampled, logging
# (obs, applied) per step via SYLVAN_BC_COLLECT=1. Serves residual7, runs N headless episodes, cleans up.
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1
N=${1:-40}
cd "$ROOT/python"
../env_pytorch_3.12/bin/python -m scripts.serve_ppo_visual \
  --checkpoint ../data/checkpoints/ppo_cpg_residual7/policy_best.pt --port 6041 > /tmp/bc_srv.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ':6041' && break; sleep 1; done
cd "$ROOT"
rm -rf godot/data/replay_buffer/bc_data 2>/dev/null
SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_CMD_VX=0.5 SYLVAN_CMD_OMEGA=0 SYLVAN_CPG_PERIOD=0.5 \
SYLVAN_CPG_TURNK=0.6 SYLVAN_CPG_TURNAMP=0 SYLVAN_CPG_SAMPLE_CMD=1 SYLVAN_CPG_SAMPLE_WMAX=0.6 SYLVAN_BC_COLLECT=1 \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$N SYLVAN_MAX_EPISODE_STEPS=400 SYLVAN_SEED=1 SYLVAN_DISABLE_HOMEOSTASIS=1 \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=6041 \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 \
SYLVAN_RUN_DIR=data/replay_buffer/bc_data \
./tools/godot/godot --path godot --headless > /tmp/bc_collect.log 2>&1
kill -9 $SRV 2>/dev/null
echo "done: $(find godot/data/replay_buffer/bc_data -name '*.jsonl' | wc -l) episodes"
