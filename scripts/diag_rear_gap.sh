#!/bin/zsh
# DIAGNOSTIC GRATUIT du trou ±179° (S2 base déplacement, 2026-06-23) : le slot-planner ferme-t-il une cible
# plein-derrière avec PLUS de temps ? Sépare temps / moteur / planner via la trajectoire (food_d, om, yaw).
# Azimuts ARRIÈRE seulement, max_steps 1300 (vs 700 en S1). WM clé de voûte + retina_head (forager promu).
cd /home/edgarbrunet/Documents/PERSO/SylvanV1
set +e
ANGLES=(180 225 270 315)   # bearings initiaux ≈ -92, -137, +179 (plein derrière), +135
DIST=3.0; MS=1300; NEP=1
pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
rm -f /tmp/rear_*.log
WM=data/checkpoints/wm_rich_fidele_sym/wm_best.pt
RH=data/checkpoints/retina_head/head_best.pt
export SYLVAN_PLANNER_HEADING_W=2.0
echo "REAR GAP DIAG WM=$WM dist=$DIST max_steps=$MS"
PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
  --wm "$WM" --residual data/checkpoints/hexapod_v2/policy_best.pt --retina-head "$RH" \
  --host 127.0.0.1 --port 6052 --horizon 160 --replan-every 10 > /tmp/rear_planner.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ':6052' && break; sleep 1; done
for A in $ANGLES; do
  echo ">>> azimuth=${A}"
  SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
  SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 SYLVAN_DISABLE_HOMEOSTASIS=1 \
  SYLVAN_FOOD_COUNT=1 SYLVAN_FOOD_ANGLE_DEG=$A SYLVAN_FOOD_MIN_RADIUS=$DIST SYLVAN_FOOD_SPAWN_RADIUS=$DIST \
  SYLVAN_EAT_RADIUS=0.5 SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$NEP SYLVAN_MAX_EPISODE_STEPS=$MS SYLVAN_SEED=1 \
  SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=6052 \
  SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
  SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 SYLVAN_RUN_DIR=data/replay_buffer/rear_gap \
  ./tools/godot/godot --path godot --headless > /tmp/rear_${A}.log 2>&1
done
kill -9 $SRV 2>/dev/null
echo "ALL_DONE"
