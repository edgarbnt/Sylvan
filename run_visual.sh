#!/usr/bin/env bash
# ─── run_visual.sh ────────────────────────────────────────────────────────────
# Standalone convenience script to run Sylvan visually on the host machine using
# the best trained policy server running in Docker.

set -euo pipefail

# 1. Start the Policy Server in the background inside Docker
echo "[SYLVAN] Starting Policy Server in Docker (CPU mode) on port 6006..."

# We give the container a name so we can easily kill it later
CONTAINER_NAME="sylvan_policy_server_visual"

# Kill any existing container with the same name first
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run -d \
  --name "${CONTAINER_NAME}" \
  -p 6006:6006 \
  -e SYLVAN_TORCH_DEVICE=cpu \
  -v "$(pwd):/workspace" \
  -w /workspace \
  --entrypoint python3 \
  sylvan:latest \
  -c "
import os, pathlib, sys
from sylvan.config import SylvanConfig
from sylvan.control.policy_server import _PolicyInferenceService, _PolicyTCPServer, _PolicyRequestHandler

config = SylvanConfig()
config.godot.policy_host = '0.0.0.0'
config.godot.policy_port = 6006

wm_path = pathlib.Path('data/checkpoints/world_model_v0.stable.pt')
ctrl_path = pathlib.Path('data/checkpoints/controller_v0.stable.pt')

if not wm_path.exists() or not ctrl_path.exists():
    print('[ERROR] Best checkpoints not found under data/checkpoints/. Please train the agent first.')
    sys.exit(1)

svc = _PolicyInferenceService(config, world_model_checkpoint=wm_path, controller_checkpoint=ctrl_path)
server = _PolicyTCPServer(('0.0.0.0', 6006), _PolicyRequestHandler, inference_service=svc)
print('[Python] Server ready on 6006')
server.serve_forever()
"

# 2. Wait for the server to be ready
echo "[SYLVAN] Waiting for Policy Server to be ready..."
sleep 3

# 3. Configure local environment variables for the host's Godot client
export SYLVAN_COLLECT=1
export SYLVAN_NUM_EPISODES=100
export SYLVAN_MAX_EPISODE_STEPS=1000
export SYLVAN_SEED=42
export SYLVAN_COLLECTOR_MODE=policy_server
export SYLVAN_RUN_DIR=data/replay_buffer/visual_run
export SYLVAN_POLICY_HOST=127.0.0.1
export SYLVAN_POLICY_PORT=6006
export SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0.0
export SYLVAN_POLICY_EXPLORATION_STD_FINAL=0.0

# Balance crutches: by DEFAULT we match the actual training regime, which uses
# `--curriculum-cycles 0` (NO crutch) — so the visual shows the real unaided
# policy, exactly as trained. A previous default assumed a 15-cycle/0.85/0.3
# curriculum and silently applied a heavy crutch (reflex ~0.68), making the agent
# never fall in the visual while it fell 100% in training — a misleading mismatch.
# If you DID train with a fading crutch, match it by overriding the curriculum:
#   CURRICULUM_CYCLES=15 REFLEX_INITIAL=0.85 ASSIST_INITIAL=0.3 ./run_visual.sh
# Or force exact crutch values directly:
#   SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 ./run_visual.sh
read -r AUTO_REFLEX AUTO_ASSIST AUTO_CYCLE < <(CURRICULUM_CYCLES="${CURRICULUM_CYCLES:-0}" \
  REFLEX_INITIAL="${REFLEX_INITIAL:-0.85}" ASSIST_INITIAL="${ASSIST_INITIAL:-0.3}" python3 - <<'PY'
import os, glob, re
base = "data/replay_buffer"
cyc = -1
for p in glob.glob(os.path.join(base, "*_cycle_*")):
    if p.endswith("_validation"):
        continue
    m = re.search(r"_cycle_(\d+)$", os.path.basename(p))
    if m:
        cyc = max(cyc, int(m.group(1)))
span = float(os.environ["CURRICULUM_CYCLES"])
ri = float(os.environ["REFLEX_INITIAL"])
ai = float(os.environ["ASSIST_INITIAL"])
def decay(c, span, init):
    if span <= 0 or init <= 0 or c < 0 or c >= span:
        return 0.0
    return init * (1.0 - c / span)
c = cyc if cyc >= 0 else 0
print("%.4f %.4f %d" % (decay(c, span, ri), decay(c, span, ai), c))
PY
)
export SYLVAN_REFLEX_STRENGTH="${SYLVAN_REFLEX_STRENGTH:-$AUTO_REFLEX}"
export SYLVAN_ASSIST_RATIO="${SYLVAN_ASSIST_RATIO:-$AUTO_ASSIST}"

echo "[SYLVAN] Crutch matched to cycle ${AUTO_CYCLE} -> reflex=${SYLVAN_REFLEX_STRENGTH} assist=${SYLVAN_ASSIST_RATIO}"
echo "[SYLVAN] Launching Godot visually on host..."
# Run Godot
./tools/godot/godot --path godot || true

# 4. Clean up background container on exit
echo "[SYLVAN] Stopping background Policy Server..."
docker kill "${CONTAINER_NAME}" >/dev/null 2>&1 || true
docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
echo "[SYLVAN] Done!"
