#!/bin/zsh
# Re-collect WM data on the KINEMATIC differential-drive body (pivot corps différentiel 2026-07-06).
# The body obeys (vx, omega) exactly (SYLVAN_KINEMATIC=1) -> dynamics are linear/deterministic, much
# simpler than the hexapod gait. Same command babbling range as the planner (vx 0.55-0.75, omega +-0.6).
# Body constants (MUST match runtime): kin_speed=0.8 (m/s per vx, sweet-spot 83% far-food), kin_turn=1.5.
# Usage: zsh scripts/collect_wm_kinematic.sh <run-prefix> <episodes> <seed>
cd /home/edgarbrunet/Documents/PERSO/SylvanV1
PREFIX=${1:-wm_kin_smoke}
EPS=${2:-4}
SEED=${3:-7}
pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
export GODOT_BIN="$(pwd)/tools/godot/godot"
export PYTHONPATH=python
export SYLVAN_WM_COLLECT=1 SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.0 SYLVAN_TURN_FADE=0
export SYLVAN_KINEMATIC=1 SYLVAN_KIN_SPEED=${KIN_SPEED:-0.8} SYLVAN_KIN_TURN=${KIN_TURN:-1.5}
export SYLVAN_WM_VX_MIN=0.55 SYLVAN_WM_VX_MAX=0.75 SYLVAN_WM_WMAX=0.6
./env_pytorch_3.12/bin/python -m scripts.collect_wm_data \
  --checkpoint data/checkpoints/hexapod_v2/policy_best.pt \
  --run-prefix "$PREFIX" --episodes "$EPS" --max-steps 400 --seed "$SEED"
