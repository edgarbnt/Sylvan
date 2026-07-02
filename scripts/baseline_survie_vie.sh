#!/bin/zsh
# BASELINE D'INTÉGRATION "JOUR 0" — le système ENTIER tel quel (cerveau promu wm_objcentric_s1 = slot DANS le WM,
# résidu hexapod_v2, coût -min_dist pur heading_w=0), sous une ÉCONOMIE DE VIE (drain 0.05, pas la valeur collecte 0.15)
# et un horizon de survie LONG (3000 pas). Mesure : survie + repas par épisode. C'est LE CHIFFRE À BATTRE pour la
# boucle jour→nuit→survie (critique appris). AUCUN entraînement ici — on mesure le point de départ, honnêtement.
# Usage: bash scripts/baseline_survie_vie.sh [eat_radius=1.0] [episodes=16] [max_steps=3000] [drain=0.05]
set +e
ER=${1:-1.0}; NEP=${2:-16}; MS=${3:-3000}; DRAIN=${4:-0.05}
WM=${WM_CKPT:-data/checkpoints/wm_objcentric_s1/wm_best.pt}
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
export SYLVAN_PLANNER_HEADING_W=${SYLVAN_PLANNER_HEADING_W:-0.0}
echo "=== BASELINE VIE : WM=$WM  eat_radius=$ER  episodes=$NEP  max_steps=$MS  drain=$DRAIN  heading_w=$SYLVAN_PLANNER_HEADING_W ==="
PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
  --wm "$WM" --residual data/checkpoints/hexapod_v2/policy_best.pt \
  --host 127.0.0.1 --port 6052 --horizon 160 --replan-every 10 > /tmp/baseline_srv.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ':6052' && break; sleep 1; done

SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 SYLVAN_EAT_RADIUS=$ER \
SYLVAN_ENERGY_DRAIN=$DRAIN \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$NEP SYLVAN_MAX_EPISODE_STEPS=$MS SYLVAN_SEED=1 SYLVAN_FOOD_COUNT=${FC:-6} \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=6052 \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 \
SYLVAN_RUN_DIR=data/replay_buffer/baseline_vie \
./tools/godot/godot --path godot --headless > /tmp/baseline_vie.log 2>&1
kill -9 $SRV 2>/dev/null
echo "=== SURVIE + REPAS (par épisode) ==="
PYTHONPATH=python ./env_pytorch_3.12/bin/python - <<'PY'
import re, statistics as st
eps = {}
for line in open('/tmp/baseline_vie.log'):
    m = re.search(r'Episode (\d+) \| Step (\d+) .* Energy: ([\d.]+)', line)
    if not m: continue
    ep, sstep, en = int(m.group(1)), int(m.group(2)), float(m.group(3))
    eps.setdefault(ep, []).append((sstep, en))
surv, meals_all = [], []
for ep in sorted(eps):
    rows = sorted(eps[ep]); last = rows[-1][0]; surv.append(last)
    meals = sum(1 for i in range(1, len(rows)) if rows[i][1] - rows[i-1][1] > 5)
    meals_all.append(meals)
    print(f"Ep{ep:>2}: survie={last:>5} ({'PLEIN' if last>=2999 else 'mort '})  repas={meals}")
if surv:
    print(f"\nSURVIE   médiane={st.median(surv):.0f}  moy={st.mean(surv):.0f}  min={min(surv)}  max={max(surv)}  pleins={sum(1 for s in surv if s>=2999)}/{len(surv)}")
    print(f"REPAS    médiane={st.median(meals_all):.1f}  total={sum(meals_all)}")
else:
    print("AUCUN épisode parsé")
PY
echo "ALL_DONE_BASELINE_VIE"
