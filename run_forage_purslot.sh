#!/bin/zsh
# Foraging avec le SLOT AUTO-SUPERVISÉ (label-free, pur) — drop-in pur de retina_head. Re-gate Phase 1 (survie ≥ live).
# Usage: bash run_forage_purslot.sh [eat_radius=1.0] [horizon=160] [episodes=12]
set +e
ER=${1:-1.0}; HZ=${2:-160}; NEP=${3:-12}
SLOT=${SLOT:-data/checkpoints/slot_head/slot_best.pt}
# WM PURIFIÉ (2026-06-25) : reconstruction droppée (--w-proprio/radar 0) → JEPA principe n°1. eff_rank 21>13,
# transport slot +0.65 préservé, engagement 15/16>13/16, foraging survie 1100≥1040. Ancien = wm_rich_fidele_sym.
WM=${WM_CKPT:-data/checkpoints/wm_rich_fidele_sym_jepa/wm_best.pt}
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
echo "PUR-SLOT slot=$SLOT WM=$WM eat_radius=$ER horizon=$HZ episodes=$NEP"

# heading_w=0 par défaut (2026-06-25) : le slot PUR précis (4.9°) rend le « how-to-hint » heading_weight INUTILE.
# A/B validé : hw=0 vs hw=2 → engagement 13/16 (arrière 3/4 ≥ 2/4), foraging méd 1040 ≥ 915. Coût = -min_dist PUR.
export SYLVAN_PLANNER_HEADING_W=${SYLVAN_PLANNER_HEADING_W:-0.0}
echo "heading_w=$SYLVAN_PLANNER_HEADING_W"
PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
  --wm "$WM" --residual data/checkpoints/hexapod_v2/policy_best.pt --slot-head "$SLOT" \
  --host 127.0.0.1 --port 6052 --horizon $HZ --replan-every 10 > /tmp/planner_purslot.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ':6052' && break; sleep 1; done

SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 SYLVAN_EAT_RADIUS=$ER \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$NEP SYLVAN_MAX_EPISODE_STEPS=1500 SYLVAN_SEED=1 SYLVAN_FOOD_COUNT=${FC:-6} \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=6052 \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 \
SYLVAN_RUN_DIR=data/replay_buffer/forage_purslot \
./tools/godot/godot --path godot --headless > /tmp/forage_purslot.log 2>&1
kill -9 $SRV 2>/dev/null
echo "done -> /tmp/forage_purslot.log"
