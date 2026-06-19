#!/bin/zsh
# Pure-CPG headless measurement (no policy server). Sweeps locomotion params via env and prints a
# compact summary parsed from the [Godot] stdout lines (Yaw / fwd_v / disp).
# Usage: tools/meas_cpg.sh LABEL  (params come from already-exported env: VX OM TURNK TURNAMP
#        SPINETURN SPINEAMP PERIOD STEP KP MSPEED). Steps/seed fixed for comparability.
cd /home/edgarbrunet/Documents/PERSO/SylvanV1
LABEL=${1:-cfg}
VX=${VX:-0.4}; OM=${OM:-0.0}; TURNK=${TURNK:-0.6}; TURNAMP=${TURNAMP:-0.0}
SPINETURN=${SPINETURN:-1.5}; SPINEAMP=${SPINEAMP:-0.0}; SPINESIGN=${SPINESIGN:-1.0}; PERIOD=${PERIOD:-0.5}
STEP=${STEP:-0.6}; KP=${KP:-4.0}; MSPEED=${MSPEED:-8.0}
LOG=/tmp/meas_${LABEL}.log
SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.0 \
SYLVAN_CMD_VX=$VX SYLVAN_CMD_OMEGA=$OM \
SYLVAN_CPG_TURNK=$TURNK SYLVAN_CPG_TURNAMP=$TURNAMP \
SYLVAN_CPG_SPINETURN=$SPINETURN SYLVAN_CPG_SPINEAMP=$SPINEAMP SYLVAN_CPG_SPINESIGN=$SPINESIGN \
SYLVAN_CPG_PERIOD=$PERIOD SYLVAN_CPG_STEP=$STEP \
SYLVAN_KP=$KP SYLVAN_MOTOR_SPEED=$MSPEED \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=1 SYLVAN_MAX_EPISODE_STEPS=400 SYLVAN_SEED=1 \
SYLVAN_DISABLE_HOMEOSTASIS=1 SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 \
SYLVAN_RUN_DIR=data/replay_buffer/meas \
./tools/godot/godot --path godot --headless > "$LOG" 2>&1
python3 tools/meas_parse.py "$LABEL" "$LOG" "$VX" "$OM"
