#!/bin/zsh
# PONT Mode-1 <-> Mode-2 sur le regime MULTI-PULSIONS (fork EXACT de baseline_multidrive_slot.sh cote Godot :
# eco de vie 0.05, faim+soif, food/water=5, eat/drink 1.0). SEULE difference = le serveur de politique est le
# PONT (serve_mode1_bridge) au lieu du planner solo. Le pont pilote Mode-1 (reflexe) PAR DEFAUT et DEFERE a
# Mode-2 (planner) quand min(energy,thirst)/100 < seuil.  Compare a M1-solo (~1930) et M2-solo (~2075-2300).
#
#  /!\ SCAFFOLD : le declencheur min_drive<seuil est un PLACEHOLDER code-main (compte-a-rebours de mort),
#      PAS la forme finale (vrai declencheur = incertitude Mode-1 / surprise WM). Voir serve_mode1_bridge.py.
#
# Usage: bash scripts/run_bridge_multidrive.sh [episodes=12] [max_steps=3000] [trigger_thr=0.15]
set +e
NEP=${1:-12}; MS=${2:-3000}; THR=${3:-0.15}
WM=${WM_CKPT:-data/checkpoints/wm_objcentric_s1/wm_best.pt}
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
PORT=6062
# Tuer les orphelins AVANT (commande separee du lancement du serveur -- cf CLAUDE.md ops).
pkill -9 -f serve_mode1_bridg[e] 2>/dev/null; pkill -9 -f serve_planner_comman[d] 2>/dev/null
pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
rm -f /tmp/bridge_srv.log /tmp/bridge_free.log
echo "=== PONT M1<->M2 (slot) : WM=$WM  episodes=$NEP  max_steps=$MS  trigger_thr=$THR ==="

# -- Serveur PONT : MEME regime planner que la baseline (heading_w=2.0, urgency_w=6.0). Commande python SEULE. --
SYLVAN_PLANNER_HEADING_W=2.0 SYLVAN_PLANNER_URGENCY_W=6.0 \
PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_mode1_bridge \
  --wm "$WM" --residual data/checkpoints/hexapod_v2/policy_best.pt \
  --bc-policy data/checkpoints/mode1_bc/policy.pt \
  --host 127.0.0.1 --port $PORT --horizon 80 --replan-every 10 --trigger-thr $THR \
  > /tmp/bridge_srv.log 2>&1 &
SRV=$!
for i in $(seq 1 90); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done

# -- Godot headless : regime multi-pulsions IDENTIQUE a baseline_multidrive_slot.sh, pointe sur le PONT. --
SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 SYLVAN_EAT_RADIUS=1.0 SYLVAN_DRINK_RADIUS=1.0 \
SYLVAN_FOOD_COUNT=5 SYLVAN_WATER_COUNT=5 SYLVAN_ENERGY_DRAIN=0.05 SYLVAN_THIRST_DRAIN=0.05 \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$NEP SYLVAN_MAX_EPISODE_STEPS=$MS SYLVAN_SEED=${SEED:-1} \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$PORT \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 SYLVAN_RUN_DIR=data/replay_buffer/bridge_md \
./tools/godot/godot --path godot --headless > /tmp/bridge_free.log 2>&1
kill -9 $SRV 2>/dev/null

echo "=== SURVIE + ARBITRAGE (par episode) ==="
PYTHONPATH=python ./env_pytorch_3.12/bin/python - <<'PY'
import re, statistics as st
eps = {}
pat = re.compile(r'Episode (\d+) \| Step (\d+) .* Energy: ([\d.]+) \| Thirst: ([\d.]+)')
for line in open('/tmp/bridge_free.log'):
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
    print(f"\nSURVIE mediane={st.median(surv):.0f}  moy={st.mean(surv):.0f}  min={min(surv)}  pleins={sum(1 for s in surv if s>=2999)}/{len(surv)}")
    both = sum(1 for me,dr in zip(meals_all,drinks_all) if me>0 and dr>0)
    print(f"JONGLE faim+soif : {both}/{len(surv)} episodes")
PY
echo "--- DEFER-RATE (depuis les logs du pont) ---"
grep -E '\[bridge\] (episode|GLOBAL|SCAFFOLD)' /tmp/bridge_srv.log
echo "ALL_DONE_BRIDGE"
