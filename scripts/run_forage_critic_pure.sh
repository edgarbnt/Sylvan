#!/bin/zsh
# BOUCLE 100% PURE (validée 2026-07-07) : corps cinématique + WM object-centric + VALEUR APPRISE
# (le critique remplace la queue analytique) + perception SYMÉTRIQUE food/eau (le hack "garder la
# dernière position d'eau" est retiré du code, plus de flag — l'eau se comporte comme la bouffe
# partout dans le codebase) + ZÉRO échafaudage de cap. La décision est 100% apprise/sans-oracle
# (readout géométrique du slot mis à part). Forage épars 1+1.
# Résultat validé : forage équilibré (repas ~7-14 vs 2 avec le hack eau ; > échafaudage 8), dense = max.
# Usage: [SEED=5 NEP=12] bash scripts/run_forage_critic_pure.sh
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
NEP=${NEP:-12}; SEED=${SEED:-5}; PORT=${PORT:-6201}
WM=${WM_CKPT:-data/checkpoints/wm_objcentric_kin/wm_best.pt}
CRITIC=${CRITIC:-data/checkpoints/survival_critic_kin/critic_best.pt}
export GODOT_BIN="$(pwd)/tools/godot/godot"
pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
echo "=== BOUCLE PURE : critique appris + eau symétrique + sans échafaudage (épars 1+1, seed $SEED) ==="

# Serveur : COÛT CRITIQUE (valeur apprise) + échafaudage OFF (perception food/eau symétrique = défaut).
env SYLVAN_PLANNER_COST=critic SYLVAN_PLANNER_CRITIC="$CRITIC" \
    SYLVAN_PLANNER_DRAIN=0.0005 SYLVAN_PLANNER_RESTORE=0.4 \
    SYLVAN_PLANNER_FAR_ALIGN=0 \
    PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
    --wm "$WM" --residual data/checkpoints/hexapod_v2/policy_best.pt \
    --host 127.0.0.1 --port $PORT --horizon 80 --replan-every 10 > /tmp/pure_srv.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done

env SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
SYLVAN_KINEMATIC=1 SYLVAN_KIN_SPEED=0.8 SYLVAN_KIN_TURN=1.5 \
SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 SYLVAN_EAT_RADIUS=1.0 SYLVAN_DRINK_RADIUS=1.0 \
SYLVAN_FOOD_COUNT=1 SYLVAN_WATER_COUNT=1 SYLVAN_ENERGY_DRAIN=0.05 SYLVAN_THIRST_DRAIN=0.05 \
SYLVAN_INIT_ENERGY=70 SYLVAN_INIT_THIRST=70 \
SYLVAN_FOOD_MIN_RADIUS=2.0 SYLVAN_FOOD_SPAWN_RADIUS=8.0 SYLVAN_FOOD_RESPAWN_MIN=2.0 SYLVAN_FOOD_RESPAWN_MAX=8.0 \
SYLVAN_WATER_MIN_RADIUS=2.0 SYLVAN_WATER_SPAWN_RADIUS=8.0 SYLVAN_WATER_RESPAWN_MIN=2.0 SYLVAN_WATER_RESPAWN_MAX=8.0 \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$NEP SYLVAN_MAX_EPISODE_STEPS=3000 SYLVAN_SEED=$SEED \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$PORT \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 SYLVAN_RUN_DIR=data/replay_buffer/pure_tmp \
./tools/godot/godot --path godot --headless > /tmp/pure_free.log 2>&1
kill -9 $SRV 2>/dev/null; pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null
rm -rf data/replay_buffer/pure_tmp

echo "--- foraging (repas/boissons/survie par épisode) ---"
PYTHONPATH=python ./env_pytorch_3.12/bin/python - <<'PY'
import re, statistics as st
eps={}
for l in open("/tmp/pure_free.log"):
    m=re.search(r'Episode (\d+) \| Step (\d+) \| Energy: ([\d.]+) \| Thirst: ([\d.]+)',l)
    if m: eps.setdefault(int(m.group(1)),[]).append((int(m.group(2)),float(m.group(3)),float(m.group(4))))
surv=[]; meals=drinks=0
for e in sorted(eps):
    s=sorted(eps[e]); surv.append(s[-1][0])
    meals+=sum(1 for i in range(1,len(s)) if s[i][1]-s[i-1][1]>5)
    drinks+=sum(1 for i in range(1,len(s)) if s[i][2]-s[i-1][2]>5)
if surv: print(f"survie med={st.median(surv):.0f} | REPAS tot={meals} | BOISSONS tot={drinks} | episodes={len(surv)}")
PY
echo "ALL_DONE_PURE"
