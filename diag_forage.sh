#!/bin/zsh
# DIAGNOSTIC GRATUIT du goulot de FORAGING (2026-06-18, zéro entraînement).
# Balaye le nombre de pastilles (1/3/6/12) sur la base VIVANTE (wm_command_hex_v2 + hexapod_v2),
# homéostasie ON (vraie survie). Discrimine :
#   H1 saturation radar  -> le foraging EMPIRE quand on ajoute des pastilles
#   H2 approche terminale -> food_d cale juste au-dessus de eat_radius sans le franchir
#   H3 ré-acquisition     -> long trou après chaque repas
# Métrique : survie médiane + nb de repas (sauts d'énergie) + trajectoires food_d (parser).
# Usage : bash diag_forage.sh [eps_par_count=8]
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
NEP=${1:-8}
PORT=6068
pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'tools/godot/godot' 2>/dev/null; sleep 1
rm -f /tmp/forage_c*.log

SYLVAN_PLANNER_HEADING_W=2.0 \
PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
  --wm data/checkpoints/wm_command_hex_v2/wm_best.pt \
  --residual data/checkpoints/hexapod_v2/policy_best.pt \
  --host 127.0.0.1 --port $PORT --horizon 80 --replan-every 10 > /tmp/forage_diag_srv.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done

for FC in 1 3 6 12; do
  echo ">>> FOOD_COUNT=$FC"
  SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
  SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_EAT_RADIUS=1.0 SYLVAN_FOOD_COUNT=$FC \
  SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$NEP SYLVAN_MAX_EPISODE_STEPS=1500 SYLVAN_SEED=1 \
  SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$PORT \
  SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
  SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 SYLVAN_RUN_DIR=data/replay_buffer/forage_diag \
  ./tools/godot/godot --path godot --headless > /tmp/forage_c${FC}.log 2>&1
done
kill -9 $SRV 2>/dev/null
echo "FORAGE_DIAG_DONE"
