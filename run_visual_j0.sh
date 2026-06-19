#!/usr/bin/env bash
# ─── run_visual_j0.sh ─────────────────────────────────────────────────────────
# Watch the J0 (model-free PPO, grounded) policy live in Godot.
#
# Unlike run_visual.sh — which loads the OLD frozen world-model controller
# (controller_v0.stable.pt) — this serves the CURRENT J0 checkpoint
# (data/checkpoints/ppo_j0/policy_latest.pt) DETERMINISTICALLY (mean action, no
# exploration noise) and with NO crutch (reflex=0, assist=0), so you see exactly
# what PPO has learned on the real body. Safe to run while training continues:
# the trainer binds an ephemeral port (policy_port=0); this server uses 6007.
#
# Re-run any time to snapshot the latest checkpoint. Override with:
#   J0_CHECKPOINT=path/to/policy.pt ./run_visual_j0.sh
set -euo pipefail

cd "$(dirname "$0")"

PORT="${J0_PORT:-6007}"
# Prefer the PEAK policy (policy_best.pt); fall back to the latest if a run hasn't
# saved a best yet. Override with J0_CHECKPOINT=...
CHECKPOINT="${J0_CHECKPOINT:-}"
if [ -z "${CHECKPOINT}" ]; then
  if [ -f "data/checkpoints/ppo_j0/policy_best.pt" ]; then
    CHECKPOINT="data/checkpoints/ppo_j0/policy_best.pt"
  else
    CHECKPOINT="data/checkpoints/ppo_j0/policy_latest.pt"
  fi
fi

if [ ! -f "${CHECKPOINT}" ]; then
  echo "[J0-visual] checkpoint not found: ${CHECKPOINT} — train J0 first (scripts.train_ppo)."
  exit 1
fi

echo "[J0-visual] Starting deterministic J0 policy server on 127.0.0.1:${PORT}..."
( cd python && python3 -m scripts.serve_ppo_visual --host 127.0.0.1 --port "${PORT}" \
    --checkpoint "../${CHECKPOINT}" ) &
SERVER_PID=$!

cleanup() {
  echo "[J0-visual] Stopping policy server (PID ${SERVER_PID})..."
  kill "${SERVER_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Give the server a moment to bind.
until python3 - <<PY >/dev/null 2>&1
import socket
s = socket.socket(); s.settimeout(0.5)
s.connect(("127.0.0.1", ${PORT})); s.close()
PY
do sleep 0.3; done
echo "[J0-visual] Server ready."

# Host Godot client config — NO crutch, deterministic policy.
export SYLVAN_COLLECT=1
export SYLVAN_NUM_EPISODES="${SYLVAN_NUM_EPISODES:-100}"
export SYLVAN_MAX_EPISODE_STEPS="${SYLVAN_MAX_EPISODE_STEPS:-1000}"
export SYLVAN_SEED="${SYLVAN_SEED:-42}"
export SYLVAN_COLLECTOR_MODE=policy_server
export SYLVAN_RUN_DIR=data/replay_buffer/visual_j0_run
export SYLVAN_POLICY_HOST=127.0.0.1
export SYLVAN_POLICY_PORT="${PORT}"
export SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0.0
export SYLVAN_REFLEX_STRENGTH=0
export SYLVAN_ASSIST_RATIO=0

echo "[J0-visual] Launching Godot (no crutch, deterministic J0 policy)..."
./tools/godot/godot --path godot || true

echo "[J0-visual] Done!"
