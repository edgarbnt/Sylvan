#!/bin/zsh
# ÉTAGE 1 RÉTINE — foraging avec PERCEPTION APPRISE (la tête 🅐 remplace l'oracle food_xz_from_radar).
# WM INCHANGÉ (wm_command_hex_v2) : seul le canal de LOCALISATION de la bouffe devient appris (rayons
# couleur bruts → tête → position). Test CHEAP (zéro retrain) du BUT (manger) avant l'étage 2 (retrain WM).
# Compare à run_forage_hex.sh (même config, oracle). Usage: bash scripts/run_forage_retina.sh [eat_radius=1.0] [horizon=80] [episodes=12] [head_ckpt]
set +e
ER=${1:-1.0}; HZ=${2:-160}; NEP=${3:-12}
HEAD=${4:-data/checkpoints/retina_head/head_best.pt}
# FORAGER VIVANT PROMU (2026-06-23) = SLOT-PLANNER : WM-rétine clé de voûte wm_rich_fidele_sym + retina_head →
# perception apprise (slot, coord explicite) transportée par la displacement-head. Engage l'arrière (S1 14/16 >
# oracle 10/16), survie foraging méd 1045 > oracle 610. (override WM_CKPT=... pour un autre WM.)
WM=${WM_CKPT:-data/checkpoints/wm_rich_fidele_sym/wm_best.pt}
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
echo "RÉTINE head=$HEAD WM=$WM eat_radius=$ER horizon=$HZ episodes=$NEP"

SYLVAN_PLANNER_HEADING_W=2.0 \
PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
  --wm "$WM" \
  --residual data/checkpoints/hexapod_v2/policy_best.pt \
  --retina-head "$HEAD" \
  --host 127.0.0.1 --port 6052 --horizon $HZ --replan-every 10 > /tmp/planner_retina.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ':6052' && break; sleep 1; done

SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 SYLVAN_EAT_RADIUS=$ER \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$NEP SYLVAN_MAX_EPISODE_STEPS=1500 SYLVAN_SEED=1 SYLVAN_FOOD_COUNT=${FC:-6} \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=6052 \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 \
SYLVAN_RUN_DIR=data/replay_buffer/forage_retina \
./tools/godot/godot --path godot --headless > /tmp/forage_retina.log 2>&1
kill -9 $SRV 2>/dev/null
echo "=== survie par épisode (dernier Step de chaque épisode) ==="
grep -a "\[Godot\] Episode" /tmp/forage_retina.log | awk '{for(i=1;i<=NF;i++){if($i=="Episode")e=$(i+1);if($i=="Step")s=$(i+1)}; key=e; last[key]=s} END{for(k in last) print "ep"k" steps="last[k]}' | sort -t'=' -k2 -n
echo "=== planner errors? ==="; grep -a -iE "error|traceback" /tmp/planner_retina.log | head
echo "done -> /tmp/forage_retina.log + /tmp/planner_retina.log"
