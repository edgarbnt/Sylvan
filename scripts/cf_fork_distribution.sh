#!/bin/bash
# DISTRIBUTION DE CONSÉQUENCE — quelle FRACTION des décisions engage vraiment l'avenir ?
#
# POURQUOI. Deux forks ont donné deux réponses opposées (seed 3 : engager 120 ticks coûte le repas ;
# seed 1 : non). Un fork ne suffit pas. Ici on échantillonne beaucoup de points de décision (ticks
# fixes × plusieurs graines) et on mesure la DISTRIBUTION : à chaque point, forcer le PIRE choix tenu
# 240 ticks (engagement généreux, pour donner sa chance à la conséquence) change-t-il les repas dans
# la fenêtre [t, t+800] par rapport à la référence (le planner de lui-même) ?
#   taux de conséquence >= ~15 % -> un critique a une place.
#   taux <= ~5 %                 -> non ; la plupart des décisions sont récupérables.
#
# Rejeu DÉTERMINISTE (seed + mono-thread + serveur frais/run). Épisode tronqué à max(TICKS)+W.
# Usage: bash scripts/cf_fork_distribution.sh
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT" || exit 1
SEEDS="1 3 5 6 8 9"
TICKS="600 1200 1800"
HOLD=240; W=800; MAXS=2600; BP=6400
PRESET=${PRESET:-bosquets_v2}   # override : PRESET=bosquets_v3_perish bash scripts/cf_fork_distribution.sh
WE=$(PYTHONPATH=python ./env_pytorch_3.12/bin/python -m sylvan.world --preset "$PRESET" --env | sed 's/^export //' | tr '\n' ' ')
FOV=$(echo "$WE" | tr ' ' '\n' | grep '^SYLVAN_RETINA_FOV_DEG=' | cut -d= -f2)

godot_run() {   # $1=tag $2=seed $3=cf_tick("") $4=cmd $5=hold $6=port
  local tag=$1 seed=$2 cft=$3 cmd=$4 hold=$5 port=$6
  SYLVAN_PLANNER_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SYLVAN_RETINA_FOV_DEG=$FOV \
  SYLVAN_PLANNER_HEADING_W=2.0 SYLVAN_PLANNER_TURN_RATE=0.015 SYLVAN_PLANNER_URGENCY_W=6.0 \
  SYLVAN_PLANNER_COST=survival SYLVAN_PLANNER_DRAIN=0.0005 SYLVAN_PLANNER_RESTORE=0.4 \
  PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
    --wm data/checkpoints/wm_objcentric_kin/wm_best.pt --residual data/checkpoints/hexapod_v2/policy_best.pt \
    --host 127.0.0.1 --port $port --horizon 80 --replan-every 60 \
    --egomotion-head data/checkpoints/egomotion_head/best.pt --slot-memory >/dev/null 2>&1 &
  local SRV=$!; for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$port" && break; sleep 1; done
  local rd=/tmp/fd_$tag; rm -rf "$rd"; mkdir -p "$rd"
  local CF=""; [ -n "$cft" ] && CF="SYLVAN_CF_TICK=$cft SYLVAN_CF_CMD=$cmd SYLVAN_CF_HOLD=$hold"
  env $CF $WE SYLVAN_MAX_EPISODE_STEPS=$MAXS SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 \
    SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 \
    SYLVAN_RETINA_PLANNER=1 SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=1 SYLVAN_SEED=$seed \
    SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$port \
    SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
    SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 SYLVAN_RUN_DIR="$rd" \
    ./tools/godot/godot --path godot --headless > /tmp/fd_$tag.log 2>&1
  kill -9 $SRV 2>/dev/null; rm -rf "$rd"
}
meals_win() {   # $1=log $2=k $3=w -> repas dans [k,k+w]
  K=$2 W=$3 PYTHONPATH=python ./env_pytorch_3.12/bin/python - "$1" <<'PY'
import re,sys,os
k=int(os.environ["K"]); w=int(os.environ["W"])
P=re.compile(r"Step (\d+) \| Energy: ([\d.]+)")
rows=sorted((int(m.group(1)),float(m.group(2))) for m in (P.search(l) for l in open(sys.argv[1])) if m)
print(sum(1 for i in range(1,len(rows)) if k<=rows[i][0]<=k+w and rows[i][1]-rows[i-1][1]>5))
PY
}

echo "=== DISTRIBUTION DE CONSÉQUENCE ($PRESET) — pire choix tenu $HOLD ticks, fenêtre $W ==="
echo "  seed  tick  ref  pire  consequent?"
total=0; conseq=0
for seed in $SEEDS; do
  godot_run det_$seed $seed "" "" 0 $((BP++))
  for t in $TICKS; do
    ref=$(meals_win /tmp/fd_det_$seed.log $t $W)
    godot_run cf_${seed}_${t} $seed $t "0.75,-0.6" $HOLD $((BP++))
    cf=$(meals_win /tmp/fd_cf_${seed}_${t}.log $t $W)
    c="non"; if [ "$ref" != "$cf" ]; then c="OUI"; conseq=$((conseq+1)); fi
    total=$((total+1))
    printf "  %4s  %4s  %3s  %4s   %s\n" "$seed" "$t" "$ref" "$cf" "$c"
    rm -f /tmp/fd_cf_${seed}_${t}.log
  done
  rm -f /tmp/fd_det_$seed.log
done
echo
echo "  TAUX DE CONSÉQUENCE : $conseq/$total points"
PYTHONPATH=python ./env_pytorch_3.12/bin/python -c "c=$conseq;t=$total;r=100*c/t if t else 0;print(f'  = {r:.0f} %  ->', 'UN CRITIQUE A UNE PLACE (>=15%)' if r>=15 else ('MARGINAL (5-15%)' if r>=5 else 'NON, decisions recuperables (<5%)'))"
echo "ALL_DONE_FD"
