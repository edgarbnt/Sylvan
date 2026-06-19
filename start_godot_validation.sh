#!/bin/bash
set -euo pipefail

PORT=50051

echo "Starting Python Policy Server (auto device selection)..."

# Auto mode tries ROCm first, then falls back to CPU if GPU kernels are unstable.
export SYLVAN_TORCH_DEVICE="${SYLVAN_TORCH_DEVICE:-auto}"

# Optional strict CPU mode: SYLVAN_FORCE_CPU_VALIDATION=1 ./start_godot_validation.sh
if [[ "${SYLVAN_FORCE_CPU_VALIDATION:-0}" == "1" ]]; then
  export SYLVAN_TORCH_DEVICE="cpu"
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT

# If an old policy server is still bound, stop it first.
EXISTING_PID="$(lsof -ti tcp:"$PORT" || true)"
if [[ -n "$EXISTING_PID" ]]; then
  kill "$EXISTING_PID" 2>/dev/null || true
  sleep 1
fi

env_pytorch/bin/python -c "
import time
from sylvan.config import SylvanConfig
from sylvan.control.policy_server import serve_policy_controller
from pathlib import Path

config = SylvanConfig()
config.godot.policy_port = $PORT

try:
    with serve_policy_controller(
        config,
        world_model_checkpoint=Path('data/checkpoints/world_model_v0.best.pt'),
        controller_checkpoint=Path('data/checkpoints/controller_v0.pt')
    ) as server:
        print(f'Server running at {server[\"host\"]}:{server[\"port\"]}')
        while True:
            time.sleep(1)
except KeyboardInterrupt:
    print('Shutting down server...')
" &
SERVER_PID=$!

sleep 3

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
  printf "Policy server failed to start on port %s.\n" "$PORT"
  exit 1
fi

echo "Launching Godot Editor..."
export SYLVAN_COLLECT=1
export SYLVAN_COLLECTOR_MODE=policy_server
export SYLVAN_POLICY_HOST=127.0.0.1
export SYLVAN_POLICY_PORT=$PORT
export SYLVAN_RUN_DIR="data/replay_buffer/validation_run"
export SYLVAN_NUM_EPISODES=100
export SYLVAN_MAX_EPISODE_STEPS=512
export SYLVAN_SEED=42
export SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0.0
export SYLVAN_POLICY_EXPLORATION_STD_FINAL=0.0

mkdir -p $SYLVAN_RUN_DIR

tools/godot/godot --path godot --editor &
GODOT_PID=$!

wait $GODOT_PID
