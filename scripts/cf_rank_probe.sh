#!/bin/bash
# JUGE DE CLASSEMENT DU CRITIQUE — contrefactuels RÉELS dans Godot, rejeu DÉTERMINISTE.
#
# POURQUOI. Toute mesure du RANG des candidats hors-ligne est morte (simulateur 6x trop facile,
# docs/prereg_gate_critique_vecu.md). Seule mesure vraie : rejouer le monde GELÉ jusqu'à un tick de
# replan k, y forcer UN candidat (hook SYLVAN_CF_TICK/CF_CMD, main.gd), laisser le planner reprendre,
# compter les repas. Le déterminisme (payé 2026-07-23) garantit que chaque candidat repart du MÊME
# état à k ; sans lui la variation entre candidats est du bruit de rejeu.
#
# 🔒 LES TROIS CONDITIONS DU DÉTERMINISME (toutes nécessaires, mesurées) :
#   1. SYLVAN_SEED fixé -> Godot seede son RNG global (gait_phase -> proprio reproductible).
#   2. mono-thread torch (THREADS=1, OMP/MKL=1) -> pas de bascule d'argmax sur candidats ex-aequo.
#   3. un SERVEUR FRAIS PAR RUN -> la mémoire spatiale ne fuit pas d'un run à l'autre.
#
# CE QUE FAIT CE SCRIPT (validation à UN état, tick k) : lance Godot une fois par candidat + 2 runs
# de contrôle sans contrefactuel. Deux contrôles bloquants :
#   1. DÉTERMINISME : les 2 runs sans CF donnent le MÊME nombre de repas (sinon tout est bruit).
#   2. VARIATION : les repas DIFFÈRENT entre candidats (sinon le tick k n'engage rien).
#
# Usage: bash scripts/cf_rank_probe.sh <tick_k>     (defaut 600)
set +e
K=${1:-600}
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT" || exit 1
WM=data/checkpoints/wm_objcentric_kin/wm_best.pt
SEED=${SEED:-7}
BASEPORT=${BASEPORT:-6220}

WORLD_ENV=$(PYTHONPATH=python ./env_pytorch_3.12/bin/python -m sylvan.world --preset bosquets_v2 --env | sed 's/^export //' | tr '\n' ' ')
FOV=$(echo "$WORLD_ENV" | tr ' ' '\n' | grep '^SYLVAN_RETINA_FOV_DEG=' | cut -d= -f2)

# UN RUN = un serveur FRAIS + un Godot, avec le contrefactuel optionnel. Rend les repas (sauts
# d'énergie > 5) comptés dans le log Godot. Le serveur est tué à la fin du run -> état propre.
run_one() {   # $1=etiquette  $2=SYLVAN_CF_TICK ("" si aucun)  $3=cmd "vx,om"  $4=port
  local tag=$1 cf_tick=$2 cf_cmd=$3 port=$4
  SYLVAN_PLANNER_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  SYLVAN_RETINA_FOV_DEG=$FOV SYLVAN_PLANNER_HEADING_W=2.0 SYLVAN_PLANNER_TURN_RATE=0.015 \
  SYLVAN_PLANNER_URGENCY_W=6.0 SYLVAN_PLANNER_COST=survival \
  SYLVAN_PLANNER_DRAIN=0.0005 SYLVAN_PLANNER_RESTORE=0.4 \
  PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
    --wm "$WM" --residual data/checkpoints/hexapod_v2/policy_best.pt \
    --host 127.0.0.1 --port $port --horizon 80 --replan-every 60 \
    --egomotion-head data/checkpoints/egomotion_head/best.pt --slot-memory > /tmp/cfp_srv_${tag}.log 2>&1 &
  local SRV=$!
  local ok=0
  for _i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$port" && { ok=1; break; }; sleep 1; done
  if [ "$ok" = 0 ]; then kill -9 $SRV 2>/dev/null; echo "NA"; return; fi
  local rundir=/tmp/cfp_${tag}; rm -rf "$rundir"; mkdir -p "$rundir"
  local CF=""
  [ -n "$cf_tick" ] && CF="SYLVAN_CF_TICK=$cf_tick SYLVAN_CF_CMD=$cf_cmd"
  env $CF $WORLD_ENV \
    SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 \
    SYLVAN_CPG_SPEEDCAD=0.6 SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 \
    SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=1 SYLVAN_SEED=$SEED \
    SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$port \
    SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
    SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 SYLVAN_RUN_DIR="$rundir" \
    ./tools/godot/godot --path godot --headless > /tmp/cfp_${tag}.log 2>&1
  kill -9 $SRV 2>/dev/null
  # repas = sauts d'énergie > 5 dans le log par-pas
  PYTHONPATH=python ./env_pytorch_3.12/bin/python - "$tag" <<'PY'
import re,sys
P=re.compile(r"Step (\d+) \| Energy: ([\d.]+)")
rows=[]
for l in open(f"/tmp/cfp_{sys.argv[1]}.log"):
    m=P.search(l)
    if m: rows.append((int(m.group(1)),float(m.group(2))))
rows.sort()
print(sum(1 for i in range(1,len(rows)) if rows[i][1]-rows[i-1][1]>5) if rows else "NA")
PY
  rm -rf "$rundir"
}

echo "=== CONTRÔLE 1 — DÉTERMINISME (2 runs sans contrefactuel) ==="
d1=$(run_one det1 "" "" $((BASEPORT+0)))
d2=$(run_one det2 "" "" $((BASEPORT+1)))
echo "  run A: $d1 repas | run B: $d2 repas -> $([ "$d1" = "$d2" ] && echo DETERMINISTE || echo '!! NON-DETERMINISTE (ne pas lire le controle 2)')"

echo "=== CONTRÔLE 2 — VARIATION entre candidats au tick k=$K ==="
echo "  vx    om    repas"
VXS="0.55 0.65 0.75"; OMS="-0.6 -0.4 -0.2 0.0 0.2 0.4 0.6"
i=0
for vx in $VXS; do for om in $OMS; do
  m=$(run_one c${i} "$K" "${vx},${om}" $((BASEPORT+2+i)))
  printf "  %.2f  %+.1f   %s\n" "$vx" "$om" "$m"
  i=$((i+1))
done; done
echo "ALL_DONE_CF"
