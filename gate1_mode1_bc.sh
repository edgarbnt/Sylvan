#!/usr/bin/env bash
# GATE-1 Mode-1 : survie multi-pulsions avec la POLITIQUE BC (serve_mode1) au lieu du planner.
# Fork EXACT de baseline_multidrive_slot.sh : SEUL le serveur change (serve_mode1 au lieu de
# serve_planner_command) -> A/B direct. Env Godot + parsing IDENTIQUES (eco de vie 0.05, food/water=5,
# eat/drink 1.0, retine). A BATTRE : baseline planner mediane ~2300. SUCCES Gate-1 : mediane >= ~2000
# (chemin valide -> debloque le RL). KILL : < 1500 (obs/perception/deploiement insuffisant -> STOP+diag).
# Usage: bash gate1_mode1_bc.sh [episodes=12] [max_steps=3000]
set +e
NEP=${1:-12}; MS=${2:-3000}
RESID=${RESID:-data/checkpoints/hexapod_v2/policy_best.pt}
BC=${BC_CKPT:-data/checkpoints/mode1_bc/policy.pt}
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
PORT=6072
pkill -9 -f serve_mode1 2>/dev/null; pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
rm -f /tmp/gate1_*.log
echo "=== GATE-1 Mode-1 (BC policy) : residual=$RESID  bc=$BC  episodes=$NEP  max_steps=$MS ==="
PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_mode1 \
  --residual "$RESID" --bc-policy "$BC" \
  --host 127.0.0.1 --port $PORT --replan-every 10 > /tmp/gate1_srv.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done
if ! ss -ltn 2>/dev/null | grep -q ":$PORT"; then echo "ERREUR : serveur serve_mode1 non demarre (voir /tmp/gate1_srv.log)"; tail -20 /tmp/gate1_srv.log; kill -9 $SRV 2>/dev/null; exit 1; fi
SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 SYLVAN_WM_USE_RETINA=1 SYLVAN_EAT_RADIUS=1.0 SYLVAN_DRINK_RADIUS=1.0 \
SYLVAN_FOOD_COUNT=5 SYLVAN_WATER_COUNT=5 SYLVAN_ENERGY_DRAIN=0.05 SYLVAN_THIRST_DRAIN=0.05 \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$NEP SYLVAN_MAX_EPISODE_STEPS=$MS SYLVAN_SEED=${SEED:-1} \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$PORT \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 SYLVAN_RUN_DIR=data/replay_buffer/gate1_mode1 \
./tools/godot/godot --path godot --headless > /tmp/gate1_free.log 2>&1
kill -9 $SRV 2>/dev/null
echo "=== SURVIE + ARBITRAGE (par episode) ==="
PYTHONPATH=python ./env_pytorch_3.12/bin/python - <<'PY'
import re, statistics as st
eps = {}
pat = re.compile(r'Episode (\d+) \| Step (\d+) .* Energy: ([\d.]+) \| Thirst: ([\d.]+)')
for line in open('/tmp/gate1_free.log'):
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
    print(f"REPAS med={st.median(meals_all):.1f}  BOISSONS med={st.median(drinks_all):.1f}")
    both = sum(1 for me,dr in zip(meals_all,drinks_all) if me>0 and dr>0)
    print(f"JONGLE faim+soif : {both}/{len(surv)} episodes")
else:
    print("AUCUN episode parse -> wiring KO (voir /tmp/gate1_free.log)")
PY
echo "ALL_DONE_GATE1"
