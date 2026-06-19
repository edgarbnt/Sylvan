#!/bin/zsh
# VOIR LE CERVEAU EN ACTION — foraging, fenêtre Godot. Planner (MPC) + World-Model + résidu + CPG.
# L'entité a faim (énergie qui draine) → perçoit la bouffe (pastilles ROUGES) → imagine/planifie dans
# son World-Model → navigue → mange → survit. Ferme la fenêtre pour arrêter.
#
# Par défaut = la BASE VALIDÉE (wm_command_hex_v2), bouffe seule = foraging fiable (mange ~plusieurs fois).
# Options :
#   bash voir_forage_jepa.sh                 # base validée, bouffe seule (RECO, fiable)
#   bash voir_forage_jepa.sh jepa            # WM JEPA v3_jepa2 (meilleure imagination, foraging encore brut)
#   bash voir_forage_jepa.sh v2 water        # + EAU (bleu) = arbitrage faim/soif (EXPÉRIMENTAL, survie pas tunée)
cd /home/edgarbrunet/Documents/PERSO/SylvanV1
case "${1:-v2}" in
  jepa) WM=data/checkpoints/wm_command_hex_v3_jepa2/wm_best.pt ;;
  *)    WM=data/checkpoints/wm_command_hex_v2/wm_best.pt ;;
esac
WATER=${2:-}
PORT=6065
pkill -9 -f 'serve_planner_command' 2>/dev/null; sleep 1
echo "[voir] WM = $WM | eau = ${WATER:-non}"
SYLVAN_PLANNER_HEADING_W=2.0 \
PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
  --wm "$WM" --residual data/checkpoints/hexapod_v2/policy_best.pt \
  --host 127.0.0.1 --port $PORT --horizon 80 --replan-every 10 > /tmp/voir_jepa_srv.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done
trap "kill -9 $SRV 2>/dev/null" EXIT INT TERM
WATER_ENV=""
[ -n "$WATER" ] && WATER_ENV="SYLVAN_WATER_COUNT=4 SYLVAN_DRINK_RADIUS=1.0"
env SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_EAT_RADIUS=1.0 SYLVAN_FOOD_COUNT=6 $WATER_ENV \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=20 SYLVAN_MAX_EPISODE_STEPS=2500 SYLVAN_SEED=1 \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$PORT \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 \
SYLVAN_RUN_DIR=data/replay_buffer/view_jepa \
./tools/godot/godot --path godot
