#!/bin/zsh
# BASELINE MULTI-PULSIONS sur le CERVEAU PROMU (wm_objcentric_s1 = slot DANS le WM ; la bouffe est désormais
# localisée par le SLOT appris, plus l'oracle radar — l'eau reste planner-only étage 1). Mêmes réglages que
# l'ancien diag_multidrive.sh (éco de vie 0.05, faim+soif, food/water=5, eat/drink 1.0) → A/B propre vs l'ancien
# cerveau oracle (wm_command_hex_v2, survie méd ~2075). Question ALife : jongle-t-il faim↔soif pour survivre,
# ou laisse-t-il une pulsion crasher ? C'EST le terrain de la boucle jour/nuit (critique appris). Aucun entraînement ici.
# Usage: bash scripts/baseline_multidrive_slot.sh [episodes=10] [max_steps=3000]
set +e
NEP=${1:-10}; MS=${2:-3000}
# ASSAINISSEMENT DE L'INVENTAIRE (2026-07-21) : ce harnais est cite comme la BASELINE VIVANTE
# multi-drive, mais il servait le corps HEXAPODE et le WM hexapode (wm_objcentric_s2) alors que le
# corps CINEMATIQUE est promu depuis le 2026-07-07. Le record cite (med 2735-2820) a donc ete
# mesure sur une config SUPERSEDEE. KIN=1 sert la config reellement promue (corps cinematique +
# wm_objcentric_kin). Defaut KIN=0 = comportement INCHANGE, pour pouvoir reproduire l'ancien chiffre.
KIN=${KIN:-0}
if [[ "$KIN" == "1" ]]; then
  WM=${WM_CKPT:-data/checkpoints/wm_objcentric_kin/wm_best.pt}
  BODY="SYLVAN_KINEMATIC=1 SYLVAN_KIN_SPEED=0.8 SYLVAN_KIN_TURN=1.5"
else
  WM=${WM_CKPT:-data/checkpoints/wm_objcentric_s2/wm_best.pt}
  BODY=""
fi
TAG="bmds_kin${KIN}_s${SEED:-1}"
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
PORT=${PORT:-6072}
# Le pkill GLOBAL d'origine tuait les serveurs des autres runs en cours -> retire
# (parallele-safe : ce script ne tue que SON serveur, via kill -9 $SRV a la fin).
echo "=== BASELINE MULTI-PULSIONS : WM=$WM  ep=$NEP  max_steps=$MS  KIN=$KIN  port=$PORT ==="
echo "=== corps : ${BODY:-hexapode CPG+residu (config SUPERSEDEE depuis 2026-07-07)} ==="
SYLVAN_PLANNER_HEADING_W=2.0 SYLVAN_PLANNER_URGENCY_W=6.0 \
SYLVAN_BC_LOG=data/replay_buffer/${TAG} SYLVAN_PLANNER_COST=${SYLVAN_PLANNER_COST:-survival} SYLVAN_PLANNER_DRAIN=${SYLVAN_PLANNER_DRAIN:-0.0005} SYLVAN_PLANNER_RESTORE=${SYLVAN_PLANNER_RESTORE:-0.4} \
PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
  --wm "$WM" --residual data/checkpoints/hexapod_v2/policy_best.pt \
  --host 127.0.0.1 --port $PORT --horizon 80 --replan-every 10 > /tmp/${TAG}_srv.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done
env $(echo $BODY) SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 SYLVAN_EAT_RADIUS=1.0 SYLVAN_DRINK_RADIUS=1.0 \
SYLVAN_FOOD_COUNT=5 SYLVAN_WATER_COUNT=5 SYLVAN_ENERGY_DRAIN=0.05 SYLVAN_THIRST_DRAIN=0.05 \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$NEP SYLVAN_MAX_EPISODE_STEPS=$MS SYLVAN_SEED=${SEED:-1} \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$PORT \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 SYLVAN_RUN_DIR=data/replay_buffer/${TAG}_run \
./tools/godot/godot --path godot --headless > /tmp/${TAG}_free.log 2>&1
kill -9 $SRV 2>/dev/null
echo "=== SURVIE + ARBITRAGE (par épisode) ==="
FREELOG=/tmp/${TAG}_free.log PYTHONPATH=python ./env_pytorch_3.12/bin/python - <<'PY'
import os, re, statistics as st
eps = {}
pat = re.compile(r'Episode (\d+) \| Step (\d+) .* Energy: ([\d.]+) \| Thirst: ([\d.]+)')
for line in open(os.environ['FREELOG']):
    m = pat.search(line)
    if not m: continue
    ep, s, en, th = int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4))
    eps.setdefault(ep, []).append((s, en, th))
surv, meals_all, drinks_all = [], [], []
for ep in sorted(eps):
    rows = sorted(eps[ep]); last = rows[-1]
    surv.append(last[0])
    meals = sum(1 for i in range(1, len(rows)) if rows[i][1]-rows[i-1][1] > 5)
    drinks = sum(1 for i in range(1, len(rows)) if rows[i][2]-rows[i-1][2] > 5)
    meals_all.append(meals); drinks_all.append(drinks)
    cause = 'PLEIN' if last[0] >= 2999 else ('faim' if last[1] <= 1.0 else ('soif' if last[2] <= 1.0 else 'autre'))
    print(f"Ep{ep:>2}: survie={last[0]:>5} ({cause:5})  repas={meals} boissons={drinks}  E_fin={last[1]:.0f} T_fin={last[2]:.0f}")
if surv:
    print(f"\nSURVIE médiane={st.median(surv):.0f}  moy={st.mean(surv):.0f}  min={min(surv)}  pleins={sum(1 for s in surv if s>=2999)}/{len(surv)}")
    print(f"REPAS méd={st.median(meals_all):.1f}  BOISSONS méd={st.median(drinks_all):.1f}  (jongle = les deux > 0)")
    both = sum(1 for me,dr in zip(meals_all,drinks_all) if me>0 and dr>0)
    print(f"JONGLE faim+soif (repas>0 ET boissons>0) : {both}/{len(surv)} épisodes")
PY
echo "ALL_DONE_BMDS"
