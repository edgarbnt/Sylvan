#!/usr/bin/env bash
# ─── run_visual_planner.sh ────────────────────────────────────────────────────
# Watch the Mode-2 WM PLANNER live in Godot (the north-star test: does planning
# in the food/energy-aware world model produce directed navigation toward food —
# WITHOUT the Mode-1 right-turn bias?).
#
# Starts serve_planner_visual (loads wm_quad_v1 + a quad J0 policy, builds the
# WMPlanner with the intrinsic hunger cost) on a TCP port, then launches Godot
# pointed at it. Unlike run_visual_j0.sh (reactive policy), every action here is
# the first step of a short-horizon plan that maximises predicted energy.
#
# Tunables (env vars):
#   PLANNER_WM         world-model ckpt   (default data/checkpoints/wm_quad_v1/world_model_v1.pt)
#   PLANNER_POLICY     proposal policy    (default data/checkpoints/ppo_quad_forage6/policy_best.pt)
#   ENERGY_WEIGHT      hunger cost weight (default 4.0 — higher = stronger food-seeking)
#   PROPOSAL_STD       candidate spread   (default 1.5 — >1 lets it consider LEFT turns the J0 mean avoids)
#   HORIZON / SAMPLES  plan depth/width   (default 12 / 96)
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PLANNER_PORT:-6009}"
WM="${PLANNER_WM:-data/checkpoints/wm_quad_v1/world_model_v1.pt}"
POLICY="${PLANNER_POLICY:-data/checkpoints/ppo_quad_forage6/policy_best.pt}"
ENERGY_WEIGHT="${ENERGY_WEIGHT:-4.0}"
PROPOSAL_STD="${PROPOSAL_STD:-1.5}"
HORIZON="${HORIZON:-12}"
SAMPLES="${SAMPLES:-96}"

for f in "$WM" "$POLICY"; do
  [ -f "$f" ] || { echo "[planner-visual] missing: $f"; exit 1; }
done

echo "[planner-visual] starting planner server on 127.0.0.1:${PORT} (energy_weight=${ENERGY_WEIGHT} proposal_std=${PROPOSAL_STD})..."
( cd python && ../env_pytorch_3.12/bin/python -m scripts.serve_planner_visual \
    --world-model "../${WM}" --policy "../${POLICY}" --host 127.0.0.1 --port "${PORT}" \
    --horizon "${HORIZON}" --num-samples "${SAMPLES}" \
    --energy-weight "${ENERGY_WEIGHT}" --proposal-std-scale "${PROPOSAL_STD}" ) &
SERVER_PID=$!
cleanup() { echo "[planner-visual] stopping server (PID ${SERVER_PID})..."; kill "${SERVER_PID}" >/dev/null 2>&1 || true; }
trap cleanup EXIT

# Wait for the server to bind.
until python3 - <<PY >/dev/null 2>&1
import socket
s = socket.socket(); s.settimeout(0.5); s.connect(("127.0.0.1", ${PORT})); s.close()
PY
do sleep 0.3; done
echo "[planner-visual] server ready."

# Godot client: deterministic, no crutch, pointed at the planner server.
export SYLVAN_COLLECT=1
export SYLVAN_NUM_EPISODES="${SYLVAN_NUM_EPISODES:-50}"
export SYLVAN_MAX_EPISODE_STEPS="${SYLVAN_MAX_EPISODE_STEPS:-1500}"
export SYLVAN_SEED="${SYLVAN_SEED:-42}"
export SYLVAN_COLLECTOR_MODE=policy_server
export SYLVAN_RUN_DIR=data/replay_buffer/visual_planner_run
export SYLVAN_POLICY_HOST=127.0.0.1
export SYLVAN_POLICY_PORT="${PORT}"
export SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0.0
export SYLVAN_REFLEX_STRENGTH=0
export SYLVAN_ASSIST_RATIO=0

echo "[planner-visual] launching Godot (Mode-2 planner)..."
./tools/godot/godot --path godot || true
echo "[planner-visual] done."
