#!/bin/zsh
# 2ᵉ PULSION — DIAGNOSTIC D'ARBITRAGE ÉMERGENT (gratuit, zéro entraînement).
# 1 bouffe + 1 eau à des positions FIXES, niveaux énergie/soif FORCÉS (SYLVAN_INIT_*),
# homéostasie OFF (niveaux constants → urgence constante, pastilles plantées). On mesure
# vers QUELLE ressource l'agent navigue (min food_d vs min water_d). L'arbitrage doit ÉMERGER
# de l'urgence : affamé→bouffe, assoiffé→eau, et l'urgence doit battre la proximité.
# Convention A→B : angle monde 0 ≈ bearing +91° (droite), 180 ≈ -91° (gauche).
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
HW=${1:-2.0}; UW=${2:-6.0}; MS=${3:-700}
pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'tools/godot/godot' 2>/dev/null; sleep 1
rm -f /tmp/arb_*.log

SYLVAN_PLANNER_HEADING_W=$HW SYLVAN_PLANNER_URGENCY_W=$UW \
PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
  --wm data/checkpoints/wm_command_hex_v2/wm_best.pt \
  --residual data/checkpoints/hexapod_v2/policy_best.pt \
  --host 127.0.0.1 --port 6053 --horizon 160 --replan-every 10 > /tmp/arb_planner.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ':6053' && break; sleep 1; done

# scénario: nom  E  T  food_ang food_r  water_ang water_r  attendu
run() {
  local name=$1 E=$2 T=$3 FA=$4 FR=$5 WA=$6 WR=$7 exp=$8
  echo ">>> $name (E=$E T=$T) food@${FA}/${FR}m water@${WA}/${WR}m EXPECT=$exp"
  SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
  SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_DISABLE_HOMEOSTASIS=1 \
  SYLVAN_INIT_ENERGY=$E SYLVAN_INIT_THIRST=$T \
  SYLVAN_FOOD_COUNT=1 SYLVAN_FOOD_ANGLE_DEG=$FA SYLVAN_FOOD_MIN_RADIUS=$FR SYLVAN_FOOD_SPAWN_RADIUS=$FR \
  SYLVAN_WATER_COUNT=1 SYLVAN_WATER_ANGLE_DEG=$WA SYLVAN_WATER_MIN_RADIUS=$WR SYLVAN_WATER_SPAWN_RADIUS=$WR \
  SYLVAN_EAT_RADIUS=0.6 SYLVAN_DRINK_RADIUS=0.6 \
  SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=2 SYLVAN_MAX_EPISODE_STEPS=$MS SYLVAN_SEED=1 \
  SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=6053 \
  SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
  SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 SYLVAN_RUN_DIR=data/replay_buffer/arb \
  ./tools/godot/godot --path godot --headless > /tmp/arb_${name}.log 2>&1
}

run hungry_FR    15 100   0 3.0   180 3.0  FOOD    # affamé, bouffe droite / eau gauche
run hungry_FL    15 100 180 3.0     0 3.0  FOOD    # affamé, miroir (bouffe gauche)
run thirsty_FR  100  15   0 3.0   180 3.0  WATER   # assoiffé, eau gauche
run thirsty_FL  100  15 180 3.0     0 3.0  WATER   # assoiffé, miroir (eau droite)
run thirst_vs_prox 60 15   0 2.0   180 5.0  WATER  # assoiffé : eau LOIN(5) vs bouffe PROCHE(2) → EAU
run hunger_vs_prox 15 60   0 5.0   180 2.0  FOOD   # affamé : bouffe LOIN(5) vs eau PROCHE(2) → BOUFFE

kill -9 $SRV 2>/dev/null
echo "ARB_DONE heading_w=$HW urgency_w=$UW"
