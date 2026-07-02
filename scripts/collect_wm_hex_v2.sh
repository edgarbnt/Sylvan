#!/bin/zsh
# Re-collect WM data on hexapod_v2 in the body's CLEAN regime: vx babbling shifted to [0.55,0.75]
# (was 0.3-0.6 where the body drifts) + CPG_PERIOD=0.5 (the validated hexapod config, omitted in v1).
# Usage: zsh collect_wm_hex_v2.sh <run-prefix> <episodes> <seed>
cd /home/edgarbrunet/Documents/PERSO/SylvanV1
PREFIX=${1:-wm_hex_v2_smoke}
EPS=${2:-4}
SEED=${3:-7}
pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
export GODOT_BIN="$(pwd)/tools/godot/godot"
export PYTHONPATH=python
export SYLVAN_WM_COLLECT=1 SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0
export SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 SYLVAN_CPG_PERIOD=0.5
export SYLVAN_WM_VX_MIN=0.55 SYLVAN_WM_VX_MAX=0.75 SYLVAN_WM_WMAX=0.6
./env_pytorch_3.12/bin/python -m scripts.collect_wm_data \
  --checkpoint data/checkpoints/hexapod_v2/policy_best.pt \
  --run-prefix "$PREFIX" --episodes "$EPS" --max-steps 400 --seed "$SEED"
