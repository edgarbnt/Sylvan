#!/bin/zsh
# S1 GATE (design WM factorisé, 2026-06-23) — le planner-SLOT (perception APPRISE retina_head + transport par la
# displacement-head du WM = plan() override_pos) engage-t-il une cible DERRIÈRE ? = la version JEPA-pure du slot.
# F3 prédit oui (slot +0.91/+0.65 vs latent +0.30). Le latent-pur (plan_latent/value-head) échoue (0/4). L'oracle
# (plan/radar) réussit (2/4) — ici on remplace l'oracle par la PERCEPTION APPRISE (retina_head), zéro oracle.
# Single pellet pinné à azimut contrôlé, homeostasis OFF, closest approach par azimut. WM = wm_rich_fidele_sym (rétine).
# Usage: bash scripts/diag_nav_ab_slot.sh [dist=3.0] [eps_per_angle=2] [max_steps=700]
set +e
DIST=${1:-3.0}; NEP=${2:-2}; MS=${3:-700}
ANGLES=(0 45 90 135 180 225 270 315)
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
rm -f /tmp/nav_ab_*.log
WM=${WM:-data/checkpoints/wm_rich_fidele_sym/wm_best.pt}
RH=${RETINA_HEAD:-data/checkpoints/retina_head/head_best.pt}
export SYLVAN_PLANNER_HEADING_W=${SYLVAN_PLANNER_HEADING_W:-2.0}   # engagement (A→B), comme le contrôle oracle
echo "SLOT WM=$WM retina_head=$RH heading_w=$SYLVAN_PLANNER_HEADING_W dist=$DIST"

PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
  --wm "$WM" --residual data/checkpoints/hexapod_v2/policy_best.pt --retina-head "$RH" \
  --host 127.0.0.1 --port 6052 --horizon 160 --replan-every 10 > /tmp/nav_ab_planner.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ':6052' && break; sleep 1; done

for A in $ANGLES; do
  echo ">>> azimuth=${A}deg dist=${DIST}m"
  SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
  SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 \
  SYLVAN_DISABLE_HOMEOSTASIS=1 \
  SYLVAN_FOOD_COUNT=1 SYLVAN_FOOD_ANGLE_DEG=$A SYLVAN_FOOD_MIN_RADIUS=$DIST SYLVAN_FOOD_SPAWN_RADIUS=$DIST \
  SYLVAN_EAT_RADIUS=0.5 \
  SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$NEP SYLVAN_MAX_EPISODE_STEPS=$MS SYLVAN_SEED=1 \
  SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=6052 \
  SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
  SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 \
  SYLVAN_RUN_DIR=data/replay_buffer/nav_ab_slot \
  ./tools/godot/godot --path godot --headless > /tmp/nav_ab_${A}.log 2>&1
done
kill -9 $SRV 2>/dev/null
echo "DONE -> parse: PYTHONPATH=python ./env_pytorch_3.12/bin/python parse_nav_ab.py"
