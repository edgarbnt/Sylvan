#!/bin/zsh
# Compare cheaply plusieurs manœuvres scriptées → mesure b2f/ép. 2 ép chacune. (clé de voûte 3b)
cd /home/edgarbrunet/Documents/PERSO/SylvanV1
export GODOT_BIN="$(pwd)/tools/godot/godot"
export PYTHONPATH=python
export SYLVAN_WM_COLLECT=1 SYLVAN_WM_TURN_SCRIPT=1
export SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0
export SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 SYLVAN_CPG_PERIOD=0.5
export SYLVAN_WM_WMAX=0.6 SYLVAN_FOOD_COUNT=8 SYLVAN_EAT_RADIUS=1.0

run() {  # name vxmin vxmax block
  local name=$1 vmin=$2 vmax=$3 blk=$4
  pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
  rm -rf data/replay_buffer/$name godot/data/replay_buffer/$name 2>/dev/null
  SYLVAN_WM_VX_MIN=$vmin SYLVAN_WM_VX_MAX=$vmax SYLVAN_WM_TURN_BLOCK=$blk \
    ./env_pytorch_3.12/bin/python -m scripts.collect_wm_data \
    --checkpoint data/checkpoints/hexapod_v2/policy_best.pt \
    --run-prefix "$name" --episodes 2 --max-steps 600 --seed 23 2>&1 | grep -E '\[collect\] DONE|fall_rate'
  ./env_pytorch_3.12/bin/python diagnostics/diag_audit_buf.py $name
}

echo "=== scurve_b20  vx0.65 block20 ===";  run cmp_scurve20  0.65 0.65 20
echo "=== scurve_b40  vx0.65 block40 ===";  run cmp_scurve40  0.65 0.65 40
echo "=== spin_vx02   vx0.20 block150 ==="; run cmp_spin02    0.20 0.20 150
echo "=== spin_vx00   vx0.00 block150 ==="; run cmp_spin00    0.00 0.00 150
echo "DONE"
