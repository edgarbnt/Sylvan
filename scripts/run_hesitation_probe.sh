#!/bin/zsh
# SONDE HÉSITATION (2026-07-03, post re-A/B) : deux runs COURTS avec le coût designed (live) et
# l'instrumentation CIBLE-DU-PLANNER (plan.target dans le log BC) pour trancher :
#   - run 5+5 (monde standard) → hésitation VRAIE (cible planner) vs inférence H0 (mêmes fichiers)
#     → quantifie la part ARTEFACT du « 87% avortées » (plus-proche-rayon qui change d'identité).
#   - run 1+1 (1 bouffe + 1 eau) → monde où le confound d'identité n'existe PAS par construction.
# Analyse : diagnostics/diag_plan_target_switches.py (+ diag_forage_hesitation pour la comparaison).
# Usage: bash scripts/run_hesitation_probe.sh [episodes=8] [max_steps=3000] [seed=1]
#        COST=survival bash scripts/run_hesitation_probe.sh …   → coût survie (tags *_surv)
set +e
NEP=${1:-8}; MS=${2:-3000}; SEED=${3:-1}
COST=${COST:-designed}
WM=${WM_CKPT:-data/checkpoints/wm_objcentric_s1/wm_best.pt}
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
PORT=6074
SUFFIX=""
COST_ENV=()
if [[ "$COST" == "survival" ]]; then
  SUFFIX="_surv"
  COST_ENV=(SYLVAN_PLANNER_COST=survival SYLVAN_PLANNER_DRAIN=0.0005 SYLVAN_PLANNER_RESTORE=0.4)
fi

run_world() {  # $1 = 55|11 ; $2 = food_count ; $3 = water_count
  local tag=$1$SUFFIX fc=$2 wc=$3
  pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
  rm -rf "data/replay_buffer/hesit_probe_${tag}"
  echo "=== SONDE $tag : food=$fc water=$wc episodes=$NEP max_steps=$MS seed=$SEED (coût $COST) ==="
  env SYLVAN_PLANNER_HEADING_W=2.0 SYLVAN_PLANNER_URGENCY_W=6.0 \
      SYLVAN_BC_LOG="data/replay_buffer/hesit_probe_${tag}" "${COST_ENV[@]}" \
      PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
      --wm "$WM" --residual data/checkpoints/hexapod_v2/policy_best.pt \
      --host 127.0.0.1 --port $PORT --horizon 80 --replan-every 10 > /tmp/hesit_srv_${tag}.log 2>&1 &
  local SRV=$!
  for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done
  SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
  SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 SYLVAN_EAT_RADIUS=1.0 SYLVAN_DRINK_RADIUS=1.0 \
  SYLVAN_FOOD_COUNT=$fc SYLVAN_WATER_COUNT=$wc SYLVAN_ENERGY_DRAIN=0.05 SYLVAN_THIRST_DRAIN=0.05 \
  SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$NEP SYLVAN_MAX_EPISODE_STEPS=$MS SYLVAN_SEED=$SEED \
  SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$PORT \
  SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
  SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 SYLVAN_RUN_DIR=data/replay_buffer/hesit_tmp \
  ./tools/godot/godot --path godot --headless > /tmp/hesit_free_${tag}.log 2>&1
  kill -9 $SRV 2>/dev/null
}

run_world 55 5 5
run_world 11 1 1
pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null

echo ""
for tag in 55$SUFFIX 11$SUFFIX; do
  echo "=== ANALYSE $tag — hésitation VRAIE (cible planner) ==="
  PYTHONPATH=python ./env_pytorch_3.12/bin/python diagnostics/diag_plan_target_switches.py \
    --files data/replay_buffer/hesit_probe_${tag}/ep_0000.jsonl
  echo "--- comparaison : inférence rétine H0 (mêmes fichiers) ---"
  PYTHONPATH=python ./env_pytorch_3.12/bin/python diagnostics/diag_forage_hesitation.py \
    --files data/replay_buffer/hesit_probe_${tag}/ep_0000.jsonl | grep -E "excess-switches|AVORTÉES"
  echo "--- survie (log Godot) ---"
  PYTHONPATH=python ./env_pytorch_3.12/bin/python - <<PY
import re, statistics as st
pat = re.compile(r'Episode (\d+) \| Step (\d+) .* Energy: ([\d.]+) \| Thirst: ([\d.]+)')
eps = {}
for line in open('/tmp/hesit_free_${tag}.log'):
    m = pat.search(line)
    if m: eps.setdefault(int(m.group(1)), []).append((int(m.group(2)), float(m.group(3)), float(m.group(4))))
surv = [sorted(v)[-1][0] for v in eps.values()]
meals = sum(sum(1 for i in range(1,len(sorted(v))) if sorted(v)[i][1]-sorted(v)[i-1][1] > 5) for v in eps.values())
drinks = sum(sum(1 for i in range(1,len(sorted(v))) if sorted(v)[i][2]-sorted(v)[i-1][2] > 5) for v in eps.values())
if surv: print(f"survie méd={st.median(surv):.0f} moy={st.mean(surv):.0f} | repas={meals} boissons={drinks} | n={len(surv)}")
PY
  echo ""
done
echo "ALL_DONE_HESIT_PROBE"
