#!/bin/zsh
# GATE MULTI-SEED CRITIQUE APPRIS (audit LeCun 2026-07-06, grappillage #1).
# Question : le critique appris (SYLVAN_PLANNER_COST=critic) bat-il/égale-t-il les incumbents
# (designed = baseline simple, survival = défaut multi-drive LIVE) ROBUSTEMENT hors bruit ?
# Le re-juge v2 était seed-unique (+585 en 5+5, plat en 1+1) → cette dette multi-seed est ici payée.
#
# DESIGN (justifié, §2 anti-poteaux) :
#   - 5+5 = ARM DISCRIMINANT → 3 seeds × {designed, survival, critic}  (9 arms)
#   - 1+1 = plancher ~1600 pour TOUS les coûts (perception-bound, cf carte) → sanity non-régression,
#           1 seed × {designed, survival, critic}  (3 arms)
#   - SÉQUENTIEL exprès : le parallélisme Godot est la CAUSE du bruit ±360 qui a rendu le seed-unique
#     non-fiable ; on veut un verdict propre, pas rapide.
#
# CRITÈRES PRÉ-ENREGISTRÉS (médiane-sur-seeds en 5+5) :
#   SUCCÈS (promouvoir critic défaut) : critic >= designed dans les 2 mondes, ET 5+5 critic >= designed+200
#                                       sur >=2/3 seeds, ET 1+1 critic >= designed-200 (pas de régression).
#   KILL : 5+5 critic < survival-200 en médiane-sur-seeds → reste optionnel, carte note le négatif.
#   PARTIEL sinon → garder optionnel, documenter (NE PAS bouger les poteaux).
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
NEP=${NEP:-8}; MS=${MS:-3000}; PORT=${PORT:-6081}
WM=${WM_CKPT:-data/checkpoints/wm_objcentric_s2/wm_best.pt}
RES=${RES:-/tmp/gate_critic_results.tsv}
: > "$RES"

run_arm() {  # $1=mode $2=worldtag(55|11) $3=fc $4=wc $5=seed
  local mode=$1 world=$2 fc=$3 wc=$4 seed=$5
  local tag=${mode}_${world}_s${seed}
  local cost_env=()
  case $mode in
    designed) ;;
    survival) cost_env=(SYLVAN_PLANNER_COST=survival SYLVAN_PLANNER_DRAIN=0.0005 SYLVAN_PLANNER_RESTORE=0.4) ;;
    critic)   cost_env=(SYLVAN_PLANNER_COST=critic   SYLVAN_PLANNER_DRAIN=0.0005 SYLVAN_PLANNER_RESTORE=0.4) ;;
  esac
  pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
  rm -rf "data/replay_buffer/gatecrit_${tag}" "data/replay_buffer/gatecrit_tmp_${tag}"
  echo "=== ARM $tag : food=$fc water=$wc seed=$seed mode=$mode episodes=$NEP ==="
  env SYLVAN_PLANNER_HEADING_W=2.0 SYLVAN_PLANNER_URGENCY_W=6.0 \
      SYLVAN_BC_LOG="data/replay_buffer/gatecrit_${tag}" "${cost_env[@]}" \
      PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
      --wm "$WM" --residual data/checkpoints/hexapod_v2/policy_best.pt \
      --host 127.0.0.1 --port $PORT --horizon 80 --replan-every 10 > /tmp/gatecrit_srv_${tag}.log 2>&1 &
  local SRV=$!
  for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done
  SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
  SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 SYLVAN_EAT_RADIUS=1.0 SYLVAN_DRINK_RADIUS=1.0 \
  SYLVAN_FOOD_COUNT=$fc SYLVAN_WATER_COUNT=$wc SYLVAN_ENERGY_DRAIN=0.05 SYLVAN_THIRST_DRAIN=0.05 \
  SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$NEP SYLVAN_MAX_EPISODE_STEPS=$MS SYLVAN_SEED=$seed \
  SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$PORT \
  SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
  SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 SYLVAN_RUN_DIR=data/replay_buffer/gatecrit_tmp_${tag} \
  ./tools/godot/godot --path godot --headless > /tmp/gatecrit_free_${tag}.log 2>&1
  kill -9 $SRV 2>/dev/null
  rm -rf "data/replay_buffer/gatecrit_tmp_${tag}"
  PYTHONPATH=python ./env_pytorch_3.12/bin/python - "$tag" "$mode" "$world" "$seed" "$RES" <<'PY'
import re, statistics as st, sys
tag, mode, world, seed, res = sys.argv[1:6]
pat = re.compile(r'Episode (\d+) \| Step (\d+) .* Energy: ([\d.]+) \| Thirst: ([\d.]+)')
eps = {}
for line in open(f'/tmp/gatecrit_free_{tag}.log'):
    m = pat.search(line)
    if m: eps.setdefault(int(m.group(1)), []).append((int(m.group(2)), float(m.group(3)), float(m.group(4))))
surv = [sorted(v)[-1][0] for v in eps.values()]
meals = sum(sum(1 for i in range(1, len(sorted(v))) if sorted(v)[i][1]-sorted(v)[i-1][1] > 5) for v in eps.values())
drinks = sum(sum(1 for i in range(1, len(sorted(v))) if sorted(v)[i][2]-sorted(v)[i-1][2] > 5) for v in eps.values())
med = st.median(surv) if surv else 0
mean = st.mean(surv) if surv else 0
row = f"{mode}\t{world}\t{seed}\t{med:.0f}\t{mean:.0f}\t{meals}\t{drinks}\t{len(surv)}"
print("   -> " + row)
open(res, "a").write(row + "\n")
PY
}

# 5+5 : arm discriminant, 3 seeds × 3 modes
for seed in 1 2 3; do
  run_arm designed 55 5 5 $seed
  run_arm survival 55 5 5 $seed
  run_arm critic   55 5 5 $seed
done
# 1+1 : plancher, seed 1 × 3 modes
run_arm designed 11 1 1 1
run_arm survival 11 1 1 1
run_arm critic   11 1 1 1

pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null

echo ""
echo "======================= SYNTHÈSE GATE CRITIQUE ======================="
PYTHONPATH=python ./env_pytorch_3.12/bin/python - "$RES" <<'PY'
import statistics as st, sys, collections
rows = [l.split('\t') for l in open(sys.argv[1]) if l.strip()]
by = collections.defaultdict(dict)  # (world) -> mode -> {seed: med}
for mode, world, seed, med, mean, meals, drinks, n in rows:
    by[world].setdefault(mode, {})[int(seed)] = int(med)
def med_over_seeds(d):
    vals = list(d.values()); return st.median(vals) if vals else 0
for world in ("55", "11"):
    if world not in by: continue
    print(f"\n--- monde {world} ---")
    for mode in ("designed", "survival", "critic"):
        d = by[world].get(mode, {})
        if not d: continue
        seeds = " ".join(f"s{s}={v}" for s, v in sorted(d.items()))
        print(f"  {mode:9s} : médiane-sur-seeds={med_over_seeds(d):.0f}  ({seeds})")
    dd = by[world]
    if all(m in dd for m in ("designed","survival","critic")):
        cd = med_over_seeds(dd["critic"]); dg = med_over_seeds(dd["designed"]); sv = med_over_seeds(dd["survival"])
        print(f"  Δ critic-designed = {cd-dg:+.0f} | Δ critic-survival = {cd-sv:+.0f}")
        if world == "55":
            n_beat = sum(1 for s in dd["critic"] if s in dd["designed"] and dd["critic"][s] >= dd["designed"][s]+200)
            print(f"  5+5 : seeds où critic >= designed+200 : {n_beat}/{len(dd['critic'])}")
            if cd < sv-200: print("  VERDICT 5+5 = KILL (critic < survival-200)")
            elif cd >= dg and n_beat >= 2: print("  VERDICT 5+5 = SUCCÈS (sous réserve 1+1 >= designed-200)")
            else: print("  VERDICT 5+5 = PARTIEL")
print("\nRES brut :", sys.argv[1])
PY
echo "ALL_DONE_GATE_CRITIC"
