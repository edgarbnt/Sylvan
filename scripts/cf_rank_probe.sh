#!/bin/bash
# JUGE DE CLASSEMENT DU CRITIQUE — contrefactuels RÉELS dans Godot.
#
# POURQUOI. Toute mesure du RANG des candidats hors-ligne est morte (le simulateur maison est 6x
# trop facile, docs/prereg_gate_critique_vecu.md). La seule mesure vraie : rejouer le monde GELÉ
# jusqu'à un tick de replan `k`, y forcer UN candidat via SYLVAN_CF_TICK/SYLVAN_CF_CMD (hook
# main.gd), laisser le planner reprendre, et compter les repas sur la suite. Le déterminisme du
# corps cinématique garantit que chaque candidat repart du MÊME état à `k`.
#
# CE QUE FAIT CE SCRIPT (validation à UN état) : pour un tick `k` donné, lance Godot une fois par
# candidat de la grille et rapporte les repas de fin d'épisode. Deux contrôles décisifs :
#   1. DÉTERMINISME : deux runs SANS contrefactuel donnent le MÊME nombre de repas.
#   2. VARIATION : les repas DIFFÈRENT entre candidats (sinon le tick k n'engage rien, ou le hook
#      est inerte, et le juge ne mesure rien).
#
# Usage: bash scripts/cf_rank_probe.sh <tick_k>     (defaut 600)
set +e
K=${1:-600}
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT" || exit 1
WM=data/checkpoints/wm_objcentric_kin/wm_best.pt
SEED=${SEED:-7}
PORT=${PORT:-6199}

WORLD_ENV=$(PYTHONPATH=python ./env_pytorch_3.12/bin/python -m sylvan.world --preset bosquets_v2 --env | sed 's/^export //' | tr '\n' ' ')
FOV=$(echo "$WORLD_ENV" | tr ' ' '\n' | grep '^SYLVAN_RETINA_FOV_DEG=' | cut -d= -f2)

# un serveur planner partagé (mémoire ON, comme la config vivante)
SYLVAN_PLANNER_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
SYLVAN_RETINA_FOV_DEG=$FOV SYLVAN_PLANNER_HEADING_W=2.0 SYLVAN_PLANNER_TURN_RATE=0.015 \
SYLVAN_PLANNER_URGENCY_W=6.0 SYLVAN_PLANNER_COST=survival \
SYLVAN_PLANNER_DRAIN=0.0005 SYLVAN_PLANNER_RESTORE=0.4 \
PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
  --wm "$WM" --residual data/checkpoints/hexapod_v2/policy_best.pt \
  --host 127.0.0.1 --port $PORT --horizon 80 --replan-every 60 \
  --egomotion-head data/checkpoints/egomotion_head/best.pt --slot-memory > /tmp/cf_srv.log 2>&1 &
SRV=$!
for _i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done

run_one() {   # $1 = etiquette, $2 = SYLVAN_CF_TICK ("" pour aucun), $3 = cmd
  local tag=$1 cf_tick=$2 cf_cmd=$3
  local rundir=/tmp/cf_${tag}
  rm -rf "$rundir"; mkdir -p "$rundir"
  local CF=""
  [ -n "$cf_tick" ] && CF="SYLVAN_CF_TICK=$cf_tick SYLVAN_CF_CMD=$cf_cmd"
  env $CF $WORLD_ENV \
    SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 \
    SYLVAN_CPG_SPEEDCAD=0.6 SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 \
    SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=1 SYLVAN_SEED=$SEED \
    SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$PORT \
    SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
    SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 SYLVAN_RUN_DIR="$rundir" \
    ./tools/godot/godot --path godot --headless > /tmp/cf_${tag}.log 2>&1
  # repas de la vie qui CONTIENT le tick k : dernier compteur meals avant mort/troncature
  grep -oE "meals.: [0-9]+" /tmp/cf_${tag}.log | tail -1 | grep -oE "[0-9]+$" || \
    grep -oE "repas=[0-9]+" /tmp/cf_${tag}.log | tail -1 | grep -oE "[0-9]+$" || echo "NA"
}

echo "=== CONTRÔLE 1 — DÉTERMINISME (2 runs sans contrefactuel) ==="
d1=$(run_one det1 "" ""); d2=$(run_one det2 "" "")
echo "  run A: $d1 repas | run B: $d2 repas -> $([ "$d1" = "$d2" ] && echo DETERMINISTE || echo '!! NON-DETERMINISTE')"

echo "=== CONTRÔLE 2 — VARIATION entre candidats au tick k=$K ==="
echo "  vx    om    repas"
VXS="0.55 0.65 0.75"; OMS="-0.6 -0.4 -0.2 0.0 0.2 0.4 0.6"
i=0
for vx in $VXS; do for om in $OMS; do
  m=$(run_one c${i} "$K" "${vx},${om}")
  printf "  %.2f  %+.1f   %s\n" "$vx" "$om" "$m"
  i=$((i+1))
done; done

kill -9 $SRV 2>/dev/null
echo "ALL_DONE_CF"
