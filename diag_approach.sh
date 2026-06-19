#!/bin/zsh
# DIAGNOSTIC HONNÊTE de l'APPROCHE PRÉCISE (2026-06-18, gratuit, rayon de capture GARDÉ à 1.0).
# Question : pourquoi il ferme à 0.04 m en A→B (cible fixe) mais cale à ~1 m en foraging ?
# Une SEULE pastille à azimut CONTRÔLÉ + distance modérée, homéostasie ON (donc il PEUT manger),
# eat_radius=1.0 (critère NON relâché). On mesure : mange-t-il (sauts d'énergie) + min food_d.
#   - mange une cible bien placée -> précision OK, faiblesse foraging = survie/spawn (pas la précision)
#   - cale même là -> vraie lacune d'approche terminale (à corriger, pas à cacher)
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
PORT=6070
pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'tools/godot/godot' 2>/dev/null; sleep 1
rm -f /tmp/appr_*.log
SYLVAN_PLANNER_HEADING_W=2.0 PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
  --wm data/checkpoints/wm_command_hex_v2/wm_best.pt --residual data/checkpoints/hexapod_v2/policy_best.pt \
  --host 127.0.0.1 --port $PORT --horizon 80 --replan-every 10 > /tmp/appr_srv.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done
# azimuts monde : 90≈devant(brg0), 0≈droite(brg+91), 180≈gauche(brg-91). distance fixe 2.5 m.
for A in 90 0 180; do
  echo ">>> azimut monde=$A dist=2.5m"
  SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
  SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_EAT_RADIUS=1.0 \
  SYLVAN_FOOD_COUNT=1 SYLVAN_FOOD_ANGLE_DEG=$A SYLVAN_FOOD_MIN_RADIUS=2.5 SYLVAN_FOOD_SPAWN_RADIUS=2.5 \
  SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=6 SYLVAN_MAX_EPISODE_STEPS=1200 SYLVAN_SEED=1 \
  SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$PORT \
  SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
  SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 SYLVAN_RUN_DIR=data/replay_buffer/appr \
  ./tools/godot/godot --path godot --headless > /tmp/appr_${A}.log 2>&1
done
kill -9 $SRV 2>/dev/null
echo APPR_DONE
