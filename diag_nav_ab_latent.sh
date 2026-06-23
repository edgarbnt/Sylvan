#!/bin/zsh
# RE-VALIDATION SYMPTÔME (clé de voûte, 2026-06-23) — le forager LATENT-PUR engage-t-il une cible DERRIÈRE ?
# Single pellet PINNÉ à un azimut contrôlé, homeostasis OFF (jamais mangé → cible fixe), on mesure le closest
# approach par azimut. Planner = LATENT-PUR (value_head sur latents RÊVÉS, COORDONNÉES DÉBRANCHÉES, rétine),
# WM = meilleur actuel (wm_rich_fidele_sym), horizon 300 (le rêve fidèle atteint ~1.5 m). Compare front/côté/DERRIÈRE.
# Le symptôme historique = rear jamais engagé (0/8). Question : avec le transport réel +0.30 (pas +0.09), tient-il ?
# Usage: bash diag_nav_ab_latent.sh [dist=3.0] [eps_per_angle=2] [max_steps=700]
set +e
DIST=${1:-3.0}; NEP=${2:-2}; MS=${3:-700}
ANGLES=(0 45 90 135 180 225 270 315)
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
rm -f /tmp/nav_ab_*.log
WM=${WM:-data/checkpoints/wm_rich_fidele_sym/wm_best.pt}
VALUE=${VALUE_CKPT:-data/checkpoints/value_head_food_dream/value_best.pt}
export SYLVAN_VALUE_AGG=${SYLVAN_VALUE_AGG:-mean}
echo "LATENT-PUR WM=$WM value=$VALUE dist=$DIST agg=$SYLVAN_VALUE_AGG"

PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
  --wm "$WM" --residual data/checkpoints/hexapod_v2/policy_best.pt --value-head "$VALUE" \
  --host 127.0.0.1 --port 6052 --horizon 300 --replan-every 10 > /tmp/nav_ab_planner.log 2>&1 &
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
  SYLVAN_RUN_DIR=data/replay_buffer/nav_ab_latent \
  ./tools/godot/godot --path godot --headless > /tmp/nav_ab_${A}.log 2>&1
done
kill -9 $SRV 2>/dev/null
echo "DONE -> parse: PYTHONPATH=python ./env_pytorch_3.12/bin/python parse_nav_ab.py"
