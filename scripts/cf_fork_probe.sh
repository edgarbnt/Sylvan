#!/bin/bash
# JUGE EN FENÊTRE, AUX FORKS — affine le juge de contrefactuels.
#
# POURQUOI. Le juge sur vie ENTIÈRE dilue l'effet d'une décision : le planner récupère sur ~40
# replans, donc un override unique se noie et le compte de repas (entier) n'a rien vu. Ici on mesure
# les repas dans une FENÊTRE [k, k+W] juste après la décision, à un vrai FORK (replan-boundary ~120
# ticks avant un repas connu). Assez local pour que « aller vers » vs « s'écarter » se voie.
#
# Rejeu DÉTERMINISTE (payé 2026-07-23) : SEED fixé + mono-thread + SERVEUR FRAIS PAR RUN. Épisode
# tronqué à k+W (SYLVAN_MAX_EPISODE_STEPS) : la trajectoire jusqu'à k+W est identique, ~40% plus vite.
#
# MARGE DU CRITIQUE (2026-07-24) : au fork, comparer le MAX sur les 21 candidats au repas que le
# planner analytique obtient de lui-meme (ref). max > ref => il existe une MARGE qu un critique
# pourrait capturer ; max == ref => -min_dist choisit deja le mieux, un critique n a rien a gagner.
# Usage: SEED=5 K=1800 W=800 HOLD=240 PRESET=bosquets_v3_perish bash scripts/cf_fork_probe.sh
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT" || exit 1
SEED=${SEED:-3}; K=${K:-1560}; W=${W:-600}; BASEPORT=${BASEPORT:-6280}
HOLD=${HOLD:-0}; PRESET=${PRESET:-bosquets_v2}
MAXS=$((K+W))
# ⚠️ EXPORT, PAS une affectation inline : un `${VAR:+NOM=1}` issu d'une expansion n'est plus
# reconnu comme une AFFECTATION par bash — il devient un mot de COMMANDE et le serveur meurt sur
# « command not found ». Erreur faite et corrigée le 2026-08-02.
[ -n "${SPRINT:-}" ] && export SYLVAN_PLANNER_SPRINT=1
WORLD_ENV=$(PYTHONPATH=python ./env_pytorch_3.12/bin/python -m sylvan.world --preset "$PRESET" --env | sed 's/^export //' | tr '\n' ' ')
FOV=$(echo "$WORLD_ENV" | tr ' ' '\n' | grep '^SYLVAN_RETINA_FOV_DEG=' | cut -d= -f2)

run_one() {   # $1=tag  $2=cf_tick("")  $3=cmd  $4=port -> repas dans [K, K+W]
  local tag=$1 cf_tick=$2 cf_cmd=$3 port=$4
  SYLVAN_PLANNER_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  SYLVAN_RETINA_FOV_DEG=$FOV SYLVAN_PLANNER_HEADING_W=2.0 SYLVAN_PLANNER_TURN_RATE=0.015 \
  SYLVAN_PLANNER_URGENCY_W=6.0 SYLVAN_PLANNER_COST=survival \
  SYLVAN_PLANNER_DRAIN=0.0005 SYLVAN_PLANNER_RESTORE=0.4 \
  PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
    --wm "${WM_CKPT:-data/checkpoints/wm_objcentric_kin/wm_best.pt}" --residual data/checkpoints/hexapod_v2/policy_best.pt \
    --host 127.0.0.1 --port $port --horizon 80 --replan-every 60 \
    --egomotion-head data/checkpoints/egomotion_head/best.pt --slot-memory > /tmp/ff_srv_${tag}.log 2>&1 &
  local SRV=$!
  local ok=0
  for _i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$port" && { ok=1; break; }; sleep 1; done
  [ "$ok" = 0 ] && { kill -9 $SRV 2>/dev/null; echo "NA"; return; }
  local rundir=/tmp/ff_${tag}; rm -rf "$rundir"; mkdir -p "$rundir"
  local CF=""; [ -n "$cf_tick" ] && CF="SYLVAN_CF_TICK=$cf_tick SYLVAN_CF_CMD=$cf_cmd SYLVAN_CF_HOLD=$HOLD"
  env $CF $WORLD_ENV SYLVAN_MAX_EPISODE_STEPS=$MAXS \
    ${KIN:+SYLVAN_KINEMATIC=1} \
    SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 \
    SYLVAN_CPG_SPEEDCAD=0.6 SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 \
    SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=1 SYLVAN_SEED=$SEED \
    SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$port \
    SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
    SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 SYLVAN_RUN_DIR="$rundir" \
    ./tools/godot/godot --path godot --headless > /tmp/ff_${tag}.log 2>&1
  kill -9 $SRV 2>/dev/null
  K=$K W=$W PYTHONPATH=python ./env_pytorch_3.12/bin/python - "$tag" <<'PY'
import re,sys,os
k=int(os.environ["K"]); w=int(os.environ["W"])
P=re.compile(r"Step (\d+) \| Energy: ([\d.]+)")
rows=sorted((int(m.group(1)),float(m.group(2))) for m in (P.search(l) for l in open(f"/tmp/ff_{sys.argv[1]}.log")) if m)
# repas dans la fenetre [k, k+w]
m=sum(1 for i in range(1,len(rows)) if k<=rows[i][0]<=k+w and rows[i][1]-rows[i-1][1]>5)
print(m)
PY
  rm -rf "$rundir"
}

echo "=== MARGE DU CRITIQUE — $PRESET  seed=$SEED  fork k=$K  fenetre W=$W  hold=$HOLD ==="
echo "=== CONTRÔLE — déterminisme (2 runs sans CF, repas dans la fenêtre) ==="
d1=$(run_one det1 "" "" $((BASEPORT+0))); d2=$(run_one det2 "" "" $((BASEPORT+1)))
# ⚠️ NA==NA affichait « DETERMINISTE » : deux runs MORTS passaient pour un contrôle réussi, et
# les 21 candidats à NA se lisaient comme « aucune marge ». Un contrôle qui valide l'échec est pire
# que pas de contrôle. Corrigé le 2026-08-02.
if [ "$d1" = "NA" ] || [ "$d2" = "NA" ]; then
  echo "  A=$d1  B=$d2 -> ❌ LE SERVEUR N'A PAS DÉMARRÉ — rien n'est mesuré, voir /tmp/ff_srv_det1.log"
  echo "  (ne PAS lire les 21 candidats ci-dessous : ils seront tous NA pour la même raison)"
else
  echo "  A=$d1  B=$d2 -> $([ "$d1" = "$d2" ] && echo DETERMINISTE || echo '!! NON-DET')"
fi
echo "  (repas SANS contrefactuel dans la fenêtre = ce que le planner obtient de lui-même)"

echo "=== 21 candidats au fork k=$K, repas dans [$K,$MAXS] ==="
echo "  vx    om    repas"
VXS="0.55 0.65 0.75"; OMS="-0.6 -0.4 -0.2 0.0 0.2 0.4 0.6"
i=0
for vx in $VXS; do for om in $OMS; do
  m=$(run_one c${i} "$K" "${vx},${om}" $((BASEPORT+2+i)))
  printf "  %.2f  %+.1f   %s\n" "$vx" "$om" "$m"
  i=$((i+1))
done; done
echo
echo "  LECTURE : si le MAX des 21 candidats > ref, un critique a une MARGE a capturer."
echo "ALL_DONE_FF"
