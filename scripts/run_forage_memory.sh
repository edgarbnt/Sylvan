#!/bin/zsh
# MÉMOIRE SPATIALE (Task 4) — foraging avec mémoire ON/OFF + masque d'occlusion paramétrable.
#
# Clone de run_forage_wmslot.sh avec deux knobs supplémentaires :
#   MEM=on|off  : active (on) la mémoire spatiale (--egomotion-head + --slot-memory) ou non (off).
#   SYLVAN_OCCLUDE_FOV_DEG : cone frontal de perception (défaut 180°).  360 = 360° (pas d'occlusion).
#
# NOTE HONNÊTETÉ : occluder la rétine côté serveur présente une rétine hors-distribution au WM
# (entraîné sur 360°). C'est une approximation pour la gate : l'objet disparaît bien du slot_encoder
# → SlotMemory doit le maintenir par dead-reckoning. Un cône frontal de prod nécessiterait un retrain
# WM — travail différé, hors scope Task 4.
#
# Usage : bash scripts/run_forage_memory.sh [eat_radius=1.0] [horizon=160] [episodes=12]
#           MEM=on  bash scripts/run_forage_memory.sh        → mémoire ON,  cone 180°
#           MEM=off bash scripts/run_forage_memory.sh        → mémoire OFF, cone 180°
#           MEM=on  SYLVAN_OCCLUDE_FOV_DEG=360 bash scripts/run_forage_memory.sh  → mémoire ON, 360° (non-régression)
set +e
ER=${1:-1.0}; HZ=${2:-160}; NEP=${3:-12}
MEM=${MEM:-on}
export SYLVAN_OCCLUDE_FOV_DEG=${SYLVAN_OCCLUDE_FOV_DEG:-180}
WM=${WM_CKPT:-data/checkpoints/wm_objcentric_s1/wm_best.pt}
EGOMOTION_CKPT=${EGOMOTION_CKPT:-data/checkpoints/egomotion_head/best.pt}
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
export SYLVAN_PLANNER_HEADING_W=${SYLVAN_PLANNER_HEADING_W:-0.0}
echo "=== run_forage_memory : MEM=$MEM  FOV=${SYLVAN_OCCLUDE_FOV_DEG}°  WM=$WM ==="
echo "    eat_radius=$ER  horizon=$HZ  episodes=$NEP  heading_w=$SYLVAN_PLANNER_HEADING_W"

# Construire les flags mémoire selon MEM=on|off
if [[ "$MEM" == "on" ]]; then
    MEM_FLAGS=(--egomotion-head $EGOMOTION_CKPT --slot-memory)
    echo "    [MEMOIRE ON]  egomotion=$EGOMOTION_CKPT + --slot-memory"
else
    MEM_FLAGS=()
    echo "    [MEMOIRE OFF] pas de mémoire (live perception only)"
fi

PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
  --wm "$WM" --residual data/checkpoints/hexapod_v2/policy_best.pt \
  --host 127.0.0.1 --port 6052 --horizon $HZ --replan-every 10 \
  "${MEM_FLAGS[@]}" > /tmp/planner_memory.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ':6052' && break; sleep 1; done

SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 SYLVAN_EAT_RADIUS=$ER \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$NEP SYLVAN_MAX_EPISODE_STEPS=1500 SYLVAN_SEED=1 SYLVAN_FOOD_COUNT=${FC:-6} \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=6052 \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 \
SYLVAN_RUN_DIR=data/replay_buffer/forage_memory \
./tools/godot/godot --path godot --headless > /tmp/forage_memory.log 2>&1
kill -9 $SRV 2>/dev/null
echo "=== SURVIE (médiane des derniers pas par épisode) ==="
PYTHONPATH=python ./env_pytorch_3.12/bin/python - <<'PY'
import re, statistics as st
eps = {}
for line in open('/tmp/forage_memory.log'):
    m = re.search(r'Episode (\d+) \| Step (\d+) .* Energy: ([\d.]+)', line)
    if not m:
        continue
    ep, sstep, en = int(m.group(1)), int(m.group(2)), float(m.group(3))
    eps.setdefault(ep, []).append((sstep, en))
surv = []
for ep in sorted(eps):
    rows = sorted(eps[ep]); surv.append(rows[-1][0])
    meals = sum(1 for i in range(1, len(rows)) if rows[i][1] - rows[i-1][1] > 5)
    print(f"Ep{ep}: survie={rows[-1][0]:>5} meals={meals}")
if surv:
    print(f"SURVIE MÉDIANE = {st.median(surv):.0f}  (baseline wmslot ~860)")
else:
    print("(aucun épisode parsé)")
PY
echo "ALL_DONE_FORAGE_MEMORY"
