#!/bin/zsh
# CURRICULUM d'amorcage du critique (2026-07-06) : fabriquer la classe positive MANQUANTE
# "pulsion urgente + sa ressource LOIN -> atteinte -> survie", que le corpus (et le professeur
# analytique) ne produit jamais. Monde 2-RESSOURCES avec UNE pulsion DOMINANTE : sa ressource est
# LOIN et son drive BAS ; l'autre est PROCHE et son drive RASSASIE (neutralise). But :
#   - active la BONNE branche du planner (multi-ressource s2+survie = record 2735 ; le chemin
#     single-drive plan_wm_slot ORBITE et ne close pas, teste et rejete) ;
#   - logge plan.food ET plan.water (2 tokens known=1) -> distribution IDENTIQUE au vrai monde
#     (regle le risque de transfert du critique, cf note) ;
#   - genere l'exemplaire positif dans la vraie distribution.
# DRIVE-SYMETRIQUE : TARGET=food (faim dominante) ou TARGET=water (soif dominante).
#
# DOUBLE USAGE (principe n°1) : sert D'ABORD de TEST DE PHYSIQUE (la poursuite lointaine est-elle
# gagnable quand une pulsion est clairement dominante ?) AVANT d'etre des donnees. Mange fiable ->
# gagnable, le mur epars est l'engagement/arbitrage, curriculum = verite. Sinon -> plafond, STOP.
#
# Usage: [PARALLEL=1 PORT=61xx] TARGET=food|water bash scripts/collect_curriculum_farfood.sh [ep=16] [seed=1] [tag=a]
set +e
NEP=${1:-16}; SEED=${2:-1}; TAG=${3:-a}
TARGET=${TARGET:-food}
PORT=${PORT:-6101}
WM=${WM_CKPT:-data/checkpoints/wm_objcentric_s2/wm_best.pt}
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
OUT="data/replay_buffer/curric_${TARGET}_${TAG}"

# pulsion DOMINANTE (loin + basse) vs NEUTRE (proche + rassasiee)
if [[ "$TARGET" == "food" ]]; then
  # INIT_ENERGY/FAR_MIN/FAR_SPAWN parametrables (defauts = curriculum d'origine). NOTE 2026-07-06 :
  # le corps avance ~0.0043 m/pas (verite-terrain buffer WM) -> energie 35 (=~700 pas) plafonne la
  # PORTEE a ~3 m -> bouffe a 5-8 m INATTEIGNABLE meme parfaite (test rigge). Budget equitable =
  # INIT_ENERGY=80 (valeur monde reel) -> ~1600 pas -> portee ~7 m.
  DRIVE_ENV=(SYLVAN_INIT_ENERGY=${INIT_ENERGY:-35} SYLVAN_INIT_THIRST=95 \
             SYLVAN_FOOD_MIN_RADIUS=${FAR_MIN:-5.0} SYLVAN_FOOD_SPAWN_RADIUS=${FAR_SPAWN:-8.0} SYLVAN_FOOD_RESPAWN_MIN=${FAR_MIN:-5.0} SYLVAN_FOOD_RESPAWN_MAX=${FAR_SPAWN:-8.0} \
             SYLVAN_WATER_MIN_RADIUS=1.0 SYLVAN_WATER_SPAWN_RADIUS=2.5 SYLVAN_WATER_RESPAWN_MIN=1.0 SYLVAN_WATER_RESPAWN_MAX=2.5)
  EATCOL="Energy: ([\d.]+)"
else
  DRIVE_ENV=(SYLVAN_INIT_ENERGY=95 SYLVAN_INIT_THIRST=35 \
             SYLVAN_WATER_MIN_RADIUS=5.0 SYLVAN_WATER_SPAWN_RADIUS=8.0 SYLVAN_WATER_RESPAWN_MIN=5.0 SYLVAN_WATER_RESPAWN_MAX=8.0 \
             SYLVAN_FOOD_MIN_RADIUS=1.0 SYLVAN_FOOD_SPAWN_RADIUS=2.5 SYLVAN_FOOD_RESPAWN_MIN=1.0 SYLVAN_FOOD_RESPAWN_MAX=2.5)
  EATCOL="Thirst: ([\d.]+)"
fi

[[ -z "$PARALLEL" ]] && { pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1; }
rm -rf "$OUT"
echo "=== CURRICULUM TARGET=$TARGET tag=$TAG : ep=$NEP seed=$SEED port=$PORT (pulsion dominante loin, l'autre neutralisee proche) ==="

env SYLVAN_PLANNER_HEADING_W=2.0 SYLVAN_PLANNER_URGENCY_W=6.0 \
    SYLVAN_PLANNER_COST=${COST:-survival} SYLVAN_PLANNER_DRAIN=0.0005 SYLVAN_PLANNER_RESTORE=0.4 \
    SYLVAN_PLANNER_FAR_ALIGN=${FAR_ALIGN:-0} SYLVAN_PLANNER_ALIGN_GAIN=${ALIGN_GAIN:-1.0} SYLVAN_PLANNER_ALIGN_MODE=${ALIGN_MODE:-mean} \
    SYLVAN_BC_LOG="$OUT" \
    PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
    --wm "$WM" --residual data/checkpoints/hexapod_v2/policy_best.pt \
    --host 127.0.0.1 --port $PORT --horizon ${HORIZON:-80} --replan-every 10 > /tmp/curric_srv_${TAG}.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done

env SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 SYLVAN_EAT_RADIUS=1.0 SYLVAN_DRINK_RADIUS=1.0 \
SYLVAN_FOOD_COUNT=1 SYLVAN_WATER_COUNT=1 SYLVAN_ENERGY_DRAIN=0.05 SYLVAN_THIRST_DRAIN=0.05 "${DRIVE_ENV[@]}" \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$NEP SYLVAN_MAX_EPISODE_STEPS=3000 SYLVAN_SEED=$SEED \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$PORT \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 SYLVAN_RUN_DIR=data/replay_buffer/curric_tmp_${TAG} \
./tools/godot/godot --path godot --headless > /tmp/curric_free_${TAG}.log 2>&1
kill -9 $SRV 2>/dev/null
[[ -z "$PARALLEL" ]] && { pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; }

echo "--- TEST PHYSIQUE : recuperation depuis loin (par episode, ressource dominante=$TARGET) ---"
PYTHONPATH=python ./env_pytorch_3.12/bin/python - "$TAG" "$EATCOL" <<'PY'
import re, sys, statistics as st
tag, col = sys.argv[1], sys.argv[2]
pat = re.compile(r'Episode (\d+) \| Step (\d+) .* ' + col)
eps = {}
for line in open(f'/tmp/curric_free_{tag}.log'):
    m = pat.search(line)
    if m: eps.setdefault(int(m.group(1)), []).append((int(m.group(2)), float(m.group(3))))
surv, ate = [], 0
for ep in sorted(eps):
    rows = sorted(eps[ep]); surv.append(rows[-1][0])
    meals = sum(1 for i in range(1, len(rows)) if rows[i][1]-rows[i-1][1] > 5)
    if meals > 0: ate += 1
    print(f"Ep{ep:>2}: survie={rows[-1][0]:>5} conso-dominante={meals} {'ATTEINT' if meals else 'MORT-SANS-ATTEINDRE'}")
if surv:
    n = len(surv)
    print(f"\nsurvie med={st.median(surv):.0f} | ep qui ATTEIGNENT la ressource loin >=1x = {ate}/{n} ({100*ate/n:.0f}%)")
    print("GATE PHYSIQUE : atteint>=60% -> poursuite lointaine GAGNABLE (curriculum dit vrai) ;")
    print("               atteint<30% -> plafond PHYSIQUE (ne pas entrainer sur un mensonge).")
PY
echo "ALL_DONE_CURRIC_${TAG}"
