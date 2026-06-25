#!/bin/zsh
# MÉMOIRE SPATIALE (Task 4) — gate A→B engagement avec mémoire ON/OFF + masque d'occlusion.
#
# Clone de diag_nav_ab_wmslot.sh avec deux knobs supplémentaires :
#   MEM=on|off  : active (on) la mémoire spatiale (--egomotion-head + --slot-memory) ou non (off).
#   SYLVAN_OCCLUDE_FOV_DEG : cone frontal de perception (défaut 180°).  360 = 360° (pas d'occlusion).
#
# Protocole identique à diag_nav_ab_wmslot.sh : single pellet pinné, homeostasis off,
# closest approach par azimut, 8 azimuts × NEP épisodes.
# Critère §4 : mémoire ON >= mémoire OFF pour les azimuts arrière (|az|>90°).
#
# NOTE HONNÊTETÉ : occluder la rétine côté serveur présente une rétine hors-distribution au WM
# (entraîné sur 360°). C'est une approximation acceptable pour la gate (l'objet disparaît bien
# du slot_encoder → SlotMemory doit le maintenir). Un cône frontal de prod nécessiterait un
# retrain WM — travail différé, hors scope Task 4.
#
# Usage: bash diag_nav_ab_memory.sh [dist=3.0] [eps_per_angle=2] [max_steps=700]
#          MEM=on  bash diag_nav_ab_memory.sh   → mémoire ON,  cone 180°
#          MEM=off bash diag_nav_ab_memory.sh   → mémoire OFF, cone 180°
#          MEM=on  SYLVAN_OCCLUDE_FOV_DEG=360 bash diag_nav_ab_memory.sh  → mémoire ON, 360°
set +e
echo "START diag_nav_ab_memory"
DIST=${1:-3.0}; NEP=${2:-2}; MS=${3:-700}
MEM=${MEM:-on}
export SYLVAN_OCCLUDE_FOV_DEG=${SYLVAN_OCCLUDE_FOV_DEG:-180}
# défaut = sweep complet (littéral → marche en bash ET zsh) ; override SYLVAN_NAV_ANGLES = word-split (bash, ou zsh via ./)
if [ -n "$SYLVAN_NAV_ANGLES" ]; then ANGLES=($SYLVAN_NAV_ANGLES); else ANGLES=(0 45 90 135 180 225 270 315); fi
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
find /tmp -maxdepth 1 -name 'nav_ab_*.log' -delete 2>/dev/null
WM=${WM:-data/checkpoints/wm_objcentric_s1/wm_best.pt}
EGOMOTION_CKPT=${EGOMOTION_CKPT:-data/checkpoints/egomotion_head/best.pt}
export SYLVAN_PLANNER_HEADING_W=${SYLVAN_PLANNER_HEADING_W:-0.0}
echo "=== diag_nav_ab_memory : MEM=$MEM  FOV=${SYLVAN_OCCLUDE_FOV_DEG}°  WM=$WM ==="
echo "    dist=$DIST  eps_per_angle=$NEP  max_steps=$MS  heading_w=$SYLVAN_PLANNER_HEADING_W"

# Construire les flags mémoire selon MEM=on|off
if [[ "$MEM" == "on" ]]; then
    MEM_FLAGS=(--egomotion-head $EGOMOTION_CKPT --slot-memory)
    echo "    [MEMOIRE ON]  egomotion=$EGOMOTION_CKPT + --slot-memory"
else
    MEM_FLAGS=()
    echo "    [MEMOIRE OFF] pas de mémoire (live perception only)"
fi

PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
  --wm "$WM" --residual data/checkpoints/hexapod_v2/policy_best.pt \
  --host 127.0.0.1 --port 6052 --horizon 160 --replan-every 10 \
  "${MEM_FLAGS[@]}" > /tmp/nav_ab_planner.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ':6052' && break; sleep 1; done
echo "serve up=$(ss -ltn 2>/dev/null | grep -c ':6052') | planner log:"; head -6 /tmp/nav_ab_planner.log 2>&1

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
  SYLVAN_RUN_DIR=data/replay_buffer/nav_ab_memory \
  ./tools/godot/godot --path godot --headless > /tmp/nav_ab_${A}.log 2>&1
done
kill -9 $SRV 2>/dev/null
echo "=== PARSE (engagement par bucket) ==="
PYTHONPATH=python ./env_pytorch_3.12/bin/python parse_nav_ab.py
echo "ALL_DONE_NAV_AB_MEMORY"
