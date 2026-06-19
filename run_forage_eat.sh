#!/bin/zsh
# One-shot foraging test at a given eat_radius (arg1, default 1.0): start planner server (v2 WM +
# residual7), run 12 headless episodes, then kill the server. Logs to /tmp/forage_eatER.log.
set +e
ER=${1:-1.0}
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1
cd "$ROOT/python"
../env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
  --wm ../data/checkpoints/wm_command_v2/wm_best.pt \
  --residual ../data/checkpoints/ppo_cpg_residual7/policy_best.pt \
  --host 127.0.0.1 --port 6051 --horizon 60 --replan-every 10 > /tmp/planner_eat.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ':6051' && break; sleep 1; done
cd "$ROOT"
SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_CMD_VX=0.5 SYLVAN_CMD_OMEGA=0 SYLVAN_CPG_PERIOD=0.5 \
SYLVAN_CPG_TURNK=0.6 SYLVAN_CPG_TURNAMP=0 SYLVAN_CPG_PLANNER=1 SYLVAN_EAT_RADIUS=$ER \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=12 SYLVAN_MAX_EPISODE_STEPS=1500 SYLVAN_SEED=1 SYLVAN_FOOD_COUNT=6 \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=6051 \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 \
SYLVAN_RUN_DIR=data/replay_buffer/forage_eat \
./tools/godot/godot --path godot --headless > /tmp/forage_eat${ER}.log 2>&1
kill -9 $SRV 2>/dev/null
echo "done eat_radius=$ER -> /tmp/forage_eat${ER}.log"
