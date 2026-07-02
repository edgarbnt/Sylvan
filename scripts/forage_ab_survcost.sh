#!/bin/zsh
# A/B COÛT SURVIE (Mode-2 refill-aware, gate B0) vs COÛT DESIGNED — multi-pulsions, régime éco de vie.
# OFF = coût designed validé (urgence×distance + heading + survival_weight=300, la baseline ~2300).
# ON  = SYLVAN_PLANNER_COST=survival : score = PAS-VÉCUS SIMULÉS (rollout WM + extension alternance,
#       drain/refill calés sur le régime réel 0.05→0.0005, refill 40→0.4), zéro poids à tuner.
# Chaque bras logge aussi les paires (obs,cmd) via SYLVAN_BC_LOG → re-mesure de l'HÉSITATION (gate H0)
# avec diagnostics/diag_forage_hesitation.py --files <bras>/ep_0000.jsonl.
#
# CRITÈRES PRÉ-ENREGISTRÉS (avant le run, CLAUDE.md §1 ; baseline H0 : avortées 87%, excess méd 2.5) :
#   SUCCÈS : survie médiane ON >= OFF + 200   ET   hésitation ON < OFF (avortées ET excess médian).
#   KILL   : survie médiane ON <  OFF - 200  → négatif informatif, STOP + rediscuter (pas de tweak à l'aveugle).
#   PARTIEL sinon → lire les sous-scores (survie vs hésitation) avant toute suite.
#
# Usage: bash scripts/forage_ab_survcost.sh [episodes=10] [max_steps=3000] [seed=1]
set +e
NEP=${1:-10}; MS=${2:-3000}; SEED=${3:-1}
WM=${WM_CKPT:-data/checkpoints/wm_objcentric_s1/wm_best.pt}
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
PORT=6073

run_arm() {  # $1 = OFF|ON
  local arm=$1
  local extra_env=()
  if [[ "$arm" == "ON" ]]; then
    extra_env=(SYLVAN_PLANNER_COST=survival SYLVAN_PLANNER_DRAIN=0.0005 SYLVAN_PLANNER_RESTORE=0.4)
  fi
  pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
  rm -rf "data/replay_buffer/ab_survcost_${arm}"
  echo "=== BRAS $arm : WM=$WM episodes=$NEP max_steps=$MS seed=$SEED ${extra_env[@]} ==="
  env SYLVAN_PLANNER_HEADING_W=2.0 SYLVAN_PLANNER_URGENCY_W=6.0 \
      SYLVAN_BC_LOG="data/replay_buffer/ab_survcost_${arm}" "${extra_env[@]}" \
      PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
      --wm "$WM" --residual data/checkpoints/hexapod_v2/policy_best.pt \
      --host 127.0.0.1 --port $PORT --horizon 80 --replan-every 10 > /tmp/abs_srv_${arm}.log 2>&1 &
  local SRV=$!
  for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done
  SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
  SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 SYLVAN_EAT_RADIUS=1.0 SYLVAN_DRINK_RADIUS=1.0 \
  SYLVAN_FOOD_COUNT=5 SYLVAN_WATER_COUNT=5 SYLVAN_ENERGY_DRAIN=0.05 SYLVAN_THIRST_DRAIN=0.05 \
  SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$NEP SYLVAN_MAX_EPISODE_STEPS=$MS SYLVAN_SEED=$SEED \
  SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$PORT \
  SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
  SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 SYLVAN_RUN_DIR=data/replay_buffer/abs_tmp \
  ./tools/godot/godot --path godot --headless > /tmp/abs_free_${arm}.log 2>&1
  kill -9 $SRV 2>/dev/null
}

run_arm OFF
run_arm ON
pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null

echo "=== RÉSULTATS A/B (survie) ==="
PYTHONPATH=python ./env_pytorch_3.12/bin/python - <<'PY'
import re, statistics as st
pat = re.compile(r'Episode (\d+) \| Step (\d+) .* Energy: ([\d.]+) \| Thirst: ([\d.]+)')
res = {}
for arm in ("OFF", "ON"):
    eps = {}
    for line in open(f'/tmp/abs_free_{arm}.log'):
        m = pat.search(line)
        if not m: continue
        ep, s, en, th = int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4))
        eps.setdefault(ep, []).append((s, en, th))
    surv, meals_a, drinks_a = [], [], []
    for ep in sorted(eps):
        rows = sorted(eps[ep]); last = rows[-1]
        surv.append(last[0])
        meals = sum(1 for i in range(1, len(rows)) if rows[i][1]-rows[i-1][1] > 5)
        drinks = sum(1 for i in range(1, len(rows)) if rows[i][2]-rows[i-1][2] > 5)
        meals_a.append(meals); drinks_a.append(drinks)
        cause = 'PLEIN' if last[0] >= 2999 else ('faim' if last[1] <= 1.0 else ('soif' if last[2] <= 1.0 else 'autre'))
        print(f"[{arm}] Ep{ep:>2}: survie={last[0]:>5} ({cause:5}) repas={meals} boissons={drinks}")
    if surv:
        res[arm] = st.median(surv)
        both = sum(1 for me, dr in zip(meals_a, drinks_a) if me > 0 and dr > 0)
        print(f"[{arm}] SURVIE méd={st.median(surv):.0f} moy={st.mean(surv):.0f} pleins={sum(1 for s in surv if s>=2999)}/{len(surv)} jongle={both}/{len(surv)}")
if "OFF" in res and "ON" in res:
    d = res["ON"] - res["OFF"]
    verdict = "SUCCÈS(survie)" if d >= 200 else ("KILL" if d <= -200 else "PARTIEL")
    print(f"\nΔ survie médiane ON-OFF = {d:+.0f}  → {verdict} (critères pré-enregistrés en tête de script)")
    print("Hésitation (gate H0, à comparer à la baseline avortées 87% / excess 2.5) :")
    print("  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_forage_hesitation.py \\")
    print("    --files data/replay_buffer/ab_survcost_OFF/ep_0000.jsonl   # puis idem _ON")
PY
echo "ALL_DONE_AB_SURVCOST"
